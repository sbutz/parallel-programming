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

struct Frontier {
    IdxType cntFrontier;
    IdxType * frontier;
};

__device__ void appendToFrontier(Frontier * frontier, IdxType vertex) {
    IdxType old = atomicAdd(&frontier->cntFrontier, 1);
    frontier->frontier[old] = vertex;
}

__global__ void kernel_bfs(
    Graph graph,
    bool * d_visited,   // working memory
    Frontier * frontier,
    Frontier * newFrontier
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    IdxType stride = 1;

    for (IdxType i = tid * stride; i < tid * stride + 1; ++i) {
        if (i < frontier->cntFrontier) {
            IdxType vertex = frontier->frontier[i];
            IdxType incidenceListStart = graph.incidenceLists[vertex];
            IdxType incidenceListEnd = graph.incidenceLists[vertex+1];

            for (IdxType j = incidenceListStart; j < incidenceListEnd; ++j) {
                IdxType destination = graph.destinations[j];
                if(!d_visited[destination]) {
                    d_visited[destination] = true;
                    appendToFrontier(newFrontier, destination);
                }
            }

        }
    }
}

IdxType sampleGraphNVertices       = 5;
auto sampleGraphIncidenceLists = std::array<IdxType, 7>  { 0,    2,    4,       7, 7, 8,    10 }; // has length nVertices + 1 to simplify code
auto sampleGraphDestinations   = std::array<IdxType, 10> { 1, 3, 4, 5, 1, 3, 5,    3, 4, 5 };

auto tmpFrontierData = std::array<IdxType, 100> {};
int main () {
    constexpr int threadsPerBlock = 128;

    DeviceVector<IdxType> d_sampleGraphInicidenceLists(sampleGraphIncidenceLists);
    DeviceVector<IdxType> d_sampleGraphDestinations(sampleGraphDestinations);
    DeviceVector<bool> d_visited (sampleGraphNVertices);
    DeviceVector<IdxType> d_frontierData (UninitializedDeviceVectorTag {}, 4 * sampleGraphNVertices);
    DeviceVector<Frontier> d_frontiers (std::array<Frontier, 2> {
        Frontier { 1, d_frontierData.data() },
        Frontier { 0, d_frontierData.data() + 2 * sampleGraphNVertices }
    });
    IdxType startVertex = 0;
    CUDA_CHECK(cudaMemcpy(d_frontierData.data(), &startVertex, sizeof(IdxType), cudaMemcpyHostToDevice));

    Graph sampleGraph_d = {
        sampleGraphNVertices,
        d_sampleGraphInicidenceLists.data(),
        d_sampleGraphDestinations.data()
    };

    int currentFrontier = 0;
    for (;;) {
        CUDA_CHECK(cudaMemset(
            &(d_frontiers.data() + !currentFrontier)->cntFrontier,
            0,
            sizeof(IdxType)
        ))
        kernel_bfs<<<
            dim3((sampleGraphNVertices + threadsPerBlock - 1) / threadsPerBlock),
            dim3(threadsPerBlock)
        >>> (
            sampleGraph_d,
            d_visited.data(),
            &d_frontiers.data()[currentFrontier],
            &d_frontiers.data()[!currentFrontier]
        );
        CUDA_CHECK(cudaMemcpy(
            tmpFrontierData.data(),
            d_frontierData.data() + (!currentFrontier) * 2 * sampleGraphNVertices,
            sampleGraphNVertices * sizeof(IdxType),
            cudaMemcpyDeviceToHost
        ))
        cudaDeviceSynchronize();
        IdxType frontierSize;
        CUDA_CHECK(cudaMemcpy(
            &frontierSize, &d_frontiers.data()[!currentFrontier].cntFrontier, sizeof(IdxType), cudaMemcpyDeviceToHost
        ))

        printArray(tmpFrontierData.data(), frontierSize);
        if (frontierSize == 0) break;

        currentFrontier = !currentFrontier;
    }

    return 0;
}