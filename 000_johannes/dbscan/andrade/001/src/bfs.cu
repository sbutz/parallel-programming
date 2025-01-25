#include "bfs.h"
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


DeviceGraph::DeviceGraph(IdxType nVertices, IdxType lenDestinations, IdxType * d_startIndices, IdxType * d_incidenceAry) {
    g.nVertices = nVertices;
    CUDA_CHECK(cudaMalloc(&g.d_startIndices, (nVertices + 1) * sizeof(IdxType)))
    CUDA_CHECK(cudaMemcpy(g.d_startIndices, d_startIndices, (nVertices + 1) * sizeof(IdxType), cudaMemcpyHostToDevice))
    CUDA_CHECK(cudaMalloc(&g.d_incidenceAry, lenDestinations * sizeof(IdxType)))
    CUDA_CHECK(cudaMemcpy(g.d_incidenceAry, d_incidenceAry, lenDestinations * sizeof(IdxType), cudaMemcpyHostToDevice))
}

DeviceGraph::~DeviceGraph() {
    (void)cudaFree(g.d_incidenceAry);
    (void)cudaFree(g.d_startIndices);
}



static __device__ void appendToFrontier(IdxType * cntFrontier, IdxType * frontier, IdxType vertex) {
    IdxType old = atomicAdd(cntFrontier, 1);
    frontier[old] = vertex;
}

static __global__ void kernel_bfs(
    DNeighborGraph graph,
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
            IdxType incidenceListStart = graph.d_startIndices[vertex];
            IdxType incidenceListEnd = graph.d_startIndices[vertex+1];

            for (IdxType j = incidenceListStart; j < incidenceListEnd; ++j) {
                IdxType destination = graph.d_incidenceAry[j];
                unsigned int destinationVisited = d_visited[destination];
                if(destinationVisited <= 1) {
                    d_visited[destination] = visitedTag;
                    if (destinationVisited == 0) appendToFrontier(cntNewFrontier, newFrontier, destination);
                }
            }

        }
    }
}

ComponentFinder::ComponentFinder(DNeighborGraph const * graph, std::size_t maxFrontierSize) : nVertices(graph->nVertices) {
    // TODO: Should we malloc everything at once?
    CUDA_CHECK(cudaMalloc(&this->d_visited, (std::size_t)graph->nVertices * sizeof(IdxType)))
    CUDA_CHECK(cudaMemset(this->d_visited, 0, (std::size_t)graph->nVertices * sizeof(IdxType)))

    size_t frontierBufferSize = 2 * (1 + (std::size_t)maxFrontierSize);
    CUDA_CHECK(cudaMalloc(&this->d_frontierBuffer, frontierBufferSize * sizeof(IdxType)))
    CUDA_CHECK(cudaMemset(this->d_frontierBuffer, 0, frontierBufferSize * sizeof(IdxType)))
    this->frontiers[0] = { this->d_frontierBuffer, this->d_frontierBuffer + 1 };
    this->frontiers[1] = { this->d_frontierBuffer + frontierBufferSize / 2, this->d_frontierBuffer + frontierBufferSize / 2 + 1 };
}

ComponentFinder::~ComponentFinder() {
    (void)cudaFree(this->d_frontierBuffer);
    (void)cudaFree(this->d_visited);
}

void ComponentFinder::findComponent(
    DNeighborGraph const * graph, IdxType startVertex, IdxType visitedTag,
    void (*callback) (void *), void * callbackData
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

        if (callback) (*callback) (callbackData);

        IdxType cntNewFrontier;
        CUDA_CHECK(cudaMemcpy(
            &cntNewFrontier, this->frontiers[!this->currentFrontier].d_cntFrontier, sizeof(IdxType),
            cudaMemcpyDeviceToHost
        ))
        if (!cntNewFrontier) break;
        this->currentFrontier = !this->currentFrontier;
    }  
}

static __global__ void kernel_markNonCore(
    IdxType * d_visited,
    IdxType * d_d_startIndices,
    IdxType nVertices
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nVertices) {
        if (d_d_startIndices[tid + 1] - d_d_startIndices[tid] == 0) {
            d_visited[tid] = 1;
        }
    }    
}

static __global__ void kernel_findUnvisited(
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

AllComponentsFinder::AllComponentsFinder(DNeighborGraph const * graph, std::size_t maxFrontierSize)
: cf (graph, maxFrontierSize), nextFreeTag(2), nextStartIndex(0) {
    CUDA_CHECK(cudaMalloc(&this->d_resultBuffer, 2 * sizeof(IdxType)))
}


AllComponentsFinder::~AllComponentsFinder() {
    (void)cudaFree(this->d_resultBuffer);
}

std::vector<IdxType> AllComponentsFinder::getComponentTagsVector() const {
    auto res = std::vector<IdxType> (this->cf.nVertices);
    CUDA_CHECK(cudaMemcpy(res.data(), this->cf.d_visited, this->cf.nVertices * sizeof(IdxType), cudaMemcpyDeviceToHost))
    return res;
}

static void markNonCore(AllComponentsFinder * acf, DNeighborGraph const * graph) {
    constexpr int threadsPerBlock = 128;
    kernel_markNonCore <<<
        dim3((graph->nVertices + threadsPerBlock - 1) / threadsPerBlock),
        dim3(threadsPerBlock)    
    >>> (
        acf->cf.d_visited,
        graph->d_startIndices,
        graph->nVertices
    );
    CUDA_CHECK(cudaGetLastError())
}

static auto findNextUnvisited(AllComponentsFinder * acf, DNeighborGraph const * graph) {
    constexpr int threadsPerBlock = 128;
    CUDA_CHECK(cudaMemset(acf->d_resultBuffer, 0, 2 * sizeof(IdxType)))

    kernel_findUnvisited <<<
        dim3((graph->nVertices + threadsPerBlock - 1) / threadsPerBlock),
        dim3(threadsPerBlock)
    >>> (acf->d_resultBuffer, acf->cf.d_visited, graph->nVertices, 0);
    cudaDeviceSynchronize();

    IdxType localBuffer [2];
    CUDA_CHECK(cudaMemcpy(localBuffer, acf->d_resultBuffer, 2 * sizeof(IdxType), cudaMemcpyDeviceToHost))

    struct Result {
        bool wasFound;
        IdxType idx;
    };
    return Result{!!localBuffer[0], localBuffer[1]};
}

void doFindAllComponents(
    AllComponentsFinder * acf, DNeighborGraph const * graph,
    void (*callback) (void *), void * callbackData
) {
    markNonCore(acf, graph);
    for (;;) {
        auto nextUnvisited = findNextUnvisited(acf, graph);
        if (!nextUnvisited.wasFound) return;
        acf->cf.findComponent(graph, nextUnvisited.idx, acf->nextFreeTag, callback, callbackData);
        ++acf->nextFreeTag;
    }
}

