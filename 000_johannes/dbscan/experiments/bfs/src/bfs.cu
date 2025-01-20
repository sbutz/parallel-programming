#include "device_vector.h"
#include "cuda_helpers.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <class T>
void printArray(T * ary, size_t n) {
    if (n == 0) { std::cout << "[ ]\n"; return; }
    std::cout << "[ " << ary[0];
    for (size_t i = 1; i < n; ++i) std::cout << ", " << ary[i];
    std::cout << " ]\n";
}

using IdxType = unsigned int;
static_assert(sizeof(IdxType) == 4, "");

struct Graph {
    IdxType nVertices;
    IdxType * incidenceLists; // has length nVertices + 1 to simplify code
    IdxType * destinations;
};

__device__ void appendToFrontier(IdxType * cntFrontier, IdxType * frontier, IdxType vertex) {
    IdxType old = atomicAdd(cntFrontier, 1);
    frontier[old] = vertex;
}

__global__ void kernel_bfs(
    Graph graph,
    unsigned int * d_visited,
    unsigned int visitedTag, // must be != 0
    IdxType * cntFrontier,
    IdxType * frontier,
    IdxType * cntNewFrontier,
    IdxType * newFrontier
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    IdxType stride = 1;

    for (IdxType i = tid * stride; i < tid * stride + 1; ++i) {
        if (i < *cntFrontier) {
            IdxType vertex = frontier[i];
            IdxType incidenceListStart = graph.incidenceLists[vertex];
            IdxType incidenceListEnd = graph.incidenceLists[vertex+1];

            for (IdxType j = incidenceListStart; j < incidenceListEnd; ++j) {
                IdxType destination = graph.destinations[j];
                if(!d_visited[destination]) {
                    d_visited[destination] = visitedTag;
                    appendToFrontier(cntNewFrontier, newFrontier, destination);
                }
            }

        }
    }
}


struct ComponentFinder {
    IdxType * d_visited = nullptr;
    IdxType * d_frontierBuffer = nullptr;
    struct {
        IdxType * d_cntFrontier;
        IdxType * d_frontier;
    } frontiers[2];
    IdxType nComponentsFound = 0;
    char currentFrontier = 0;

    ComponentFinder(Graph const * graph, std::size_t maxFrontierSize) {
        // TODO: Should we malloc everything at once?
        CUDA_CHECK(cudaMalloc(&this->d_visited, (std::size_t)graph->nVertices * sizeof(IdxType)))
        CUDA_CHECK(cudaMemset(this->d_visited, 0, (std::size_t)graph->nVertices * sizeof(IdxType)))

        size_t frontierBufferSize = 2 * (1 + (std::size_t)maxFrontierSize);
        CUDA_CHECK(cudaMalloc(&this->d_frontierBuffer, frontierBufferSize * sizeof(IdxType)))
        CUDA_CHECK(cudaMemset(this->d_frontierBuffer, 0, frontierBufferSize * sizeof(IdxType)))
        this->frontiers[0] = { this->d_frontierBuffer, this->d_frontierBuffer + 1 };
        this->frontiers[1] = { this->d_frontierBuffer + frontierBufferSize / 2, this->d_frontierBuffer + frontierBufferSize / 2 + 1 };

    }

    ComponentFinder(ComponentFinder const &) = delete;

    ~ComponentFinder() {
        (void)cudaFree(this->d_frontierBuffer);
        (void)cudaFree(this->d_visited);
    }

    template <typename Callback>
    void findComponent(
        Graph const * graph, IdxType startVertex, IdxType visitedTag,
        Callback && callback = []{}
    ) {
        constexpr int threadsPerBlock = 128;
        IdxType startValues [2] = { 1, startVertex };

        CUDA_CHECK(cudaMemcpy(this->frontiers[this->currentFrontier].d_cntFrontier, &startValues, 2 * sizeof(IdxType), cudaMemcpyHostToDevice))
        CUDA_CHECK(cudaMemset(&this->d_visited[startVertex], visitedTag, sizeof(IdxType)))
        for (;;) {
            CUDA_CHECK(cudaMemset(this->frontiers[!this->currentFrontier].d_cntFrontier, 0, sizeof(IdxType)))
            kernel_bfs <<<
                dim3((graph->nVertices + threadsPerBlock - 1) / threadsPerBlock),
                dim3(threadsPerBlock)
            >>> (
                *graph,
                this->d_visited,
                visitedTag,
                this->frontiers[this->currentFrontier].d_cntFrontier,
                this->frontiers[this->currentFrontier].d_frontier,
                this->frontiers[!this->currentFrontier].d_cntFrontier,
                this->frontiers[!this->currentFrontier].d_frontier
            );
            cudaDeviceSynchronize();

            std::forward<Callback> (callback) ();

            IdxType cntNewFrontier;
            CUDA_CHECK(cudaMemcpy(
                &cntNewFrontier, this->frontiers[!this->currentFrontier].d_cntFrontier, sizeof(IdxType),
                cudaMemcpyDeviceToHost
            ))
            if (!cntNewFrontier) break;
            this->currentFrontier = !this->currentFrontier;
        }  
    }
};

__global__ void kernel_findUnvisited(
    IdxType * outBuffer,
    IdxType * d_visited,
    IdxType nVertices,
    IdxType startPos
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO: This is bad. We should start at startPos and finish early when vertex has been found.
    if (tid < nVertices) {
        if (!d_visited[tid]) {
            outBuffer[0] = 1; // true
            outBuffer[1] = tid;
        }
    }    
}

struct AllComponentsFinder {
    ComponentFinder cf;
    IdxType nextFreeTag;
    IdxType nextStartIndex;
    IdxType * d_resultBuffer;

    AllComponentsFinder(Graph const * graph, std::size_t maxFrontierSize)
    : cf (graph, maxFrontierSize), nextFreeTag(1), nextStartIndex(0) {
        CUDA_CHECK(cudaMalloc(&this->d_resultBuffer, 2 * sizeof(IdxType)))
    }

    AllComponentsFinder(AllComponentsFinder const &) = delete;

    ~AllComponentsFinder() {
        (void)cudaFree(this->d_resultBuffer);
    }

    auto findNextUnvisited(Graph const * graph) {
        constexpr int threadsPerBlock = 128;
        CUDA_CHECK(cudaMemset(this->d_resultBuffer, 0, 2 * sizeof(IdxType)))

        kernel_findUnvisited <<<
            dim3((graph->nVertices + threadsPerBlock - 1) / threadsPerBlock),
            dim3(threadsPerBlock)
        >>> (this->d_resultBuffer, this->cf.d_visited, graph->nVertices, 0);
        cudaDeviceSynchronize();

        IdxType localBuffer [2];
        CUDA_CHECK(cudaMemcpy(localBuffer, this->d_resultBuffer, 2 * sizeof(IdxType), cudaMemcpyDeviceToHost))

        struct Result {
            bool wasFound;
            IdxType idx;
        };
        return Result{!!localBuffer[0], localBuffer[1]};
    }

    template <typename Callback>
    void findAllComponents(Graph const * graph, Callback && callback = []{}) {
        for (;;) {
            auto nextUnvisited = this->findNextUnvisited(graph);
            if (!nextUnvisited.wasFound) return;
            this->cf.findComponent(graph, nextUnvisited.idx, this->nextFreeTag, std::forward<Callback> (callback));
            ++this->nextFreeTag;
        }
    }
};

constexpr IdxType sampleGraphNVertices   = 5;
constexpr auto sampleGraphIncidenceLists = std::array<IdxType, 7>  { 0,    2,    4,       7, 7, 8,    10 }; // has length nVertices + 1 to simplify code
constexpr auto sampleGraphDestinations   = std::array<IdxType, 10> { 1, 3, 4, 5, 1, 3, 5,    3, 4, 5 };

auto tmpFrontierData = std::array<IdxType, sampleGraphDestinations.size()> {};

int main () {
    DeviceVector<IdxType> d_sampleGraphInicidenceLists(sampleGraphIncidenceLists);
    DeviceVector<IdxType> d_sampleGraphDestinations(sampleGraphDestinations);
    IdxType startVertex = 0;

    Graph sampleGraph_d = {
        sampleGraphNVertices,
        d_sampleGraphInicidenceLists.data(),
        d_sampleGraphDestinations.data()
    };

    AllComponentsFinder acf(&sampleGraph_d, sampleGraphDestinations.size());

    acf.findAllComponents(&sampleGraph_d, [&]() {
        IdxType cntNewFrontier = 0;
        CUDA_CHECK(cudaMemcpy(
            &cntNewFrontier,
            acf.cf.frontiers[!acf.cf.currentFrontier].d_cntFrontier,
            sizeof(IdxType),
            cudaMemcpyDeviceToHost
        ))
        CUDA_CHECK(cudaMemcpy(
            tmpFrontierData.data(),
            acf.cf.frontiers[!acf.cf.currentFrontier].d_frontier,
            sampleGraphDestinations.size() * sizeof(IdxType),
            cudaMemcpyDeviceToHost
        ))
        printArray(tmpFrontierData.data(), cntNewFrontier);
    });

    return 0;
}