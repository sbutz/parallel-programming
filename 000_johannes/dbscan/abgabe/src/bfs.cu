#include "bfs.h"
#include "cuda_helpers.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

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

constexpr IdxType sharedFrontierSize = 1u << 13; // 8 * 1024 values -> 4 * 8 * 1024 Bytes = 32 kiB

static __device__ void appendToFrontierShared(
    IdxType * cntSharedFrontier, IdxType * sharedFrontier,
    IdxType * cntFrontier, IdxType * frontier, IdxType vertex
) {
    if (*cntSharedFrontier < sharedFrontierSize) {
        IdxType old = atomicAdd(cntSharedFrontier, 1);
        sharedFrontier[old] = vertex;
    } else {
        IdxType old = atomicAdd(cntFrontier, 1);
        frontier[old] = vertex;
    }
}

static __device__ void copySharedToGlobalFrontier(
    IdxType * startPos,
    IdxType * cntSharedFrontier, IdxType * sharedFrontier,
    IdxType * cntGlobalFrontier, IdxType * globalFrontier
) {
    if (threadIdx.x == 0) *startPos = atomicAdd(cntGlobalFrontier, *cntSharedFrontier);
    __syncthreads();
    IdxType start = *startPos;
    IdxType stride = blockDim.x;
    for (IdxType i = threadIdx.x; i < sharedFrontierSize; i += stride) {
        globalFrontier[start + i] = sharedFrontier[i];
    }
}

static __global__ void kernel_bfs_shared_frontier(
    DNeighborGraph graph,
    unsigned int * d_visited,
    unsigned int visitedTag, // must be != 0
    IdxType * cntFrontier,
    IdxType * frontier,
    IdxType * cntNewFrontier,
    IdxType * newFrontier
) {
    __shared__ IdxType cntSharedFrontier;
    __shared__ IdxType startPos;
    __shared__ IdxType sharedFrontier[sharedFrontierSize];

    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    IdxType stride = 1;

    if (threadIdx.x == 0) cntSharedFrontier = 0;
    __syncthreads();

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
                    if (destinationVisited == 0) appendToFrontierShared(
                        &cntSharedFrontier, sharedFrontier,
                        cntNewFrontier, newFrontier, destination
                    );
                }
            }
        }
    }

    __syncthreads();

    copySharedToGlobalFrontier(
        &startPos,
        &cntSharedFrontier, sharedFrontier,
        cntNewFrontier, newFrontier
    );
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

template <int FrontierPolicyKey> struct FindComponent;

static __global__ void kernel_checkZero(IdxType * ary, IdxType n, int m) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n && ary[tid] > 1) printf("%d %u %d\n", m, tid, ary[tid]);
}

template <>
struct FindComponent<frontierBasicPolicy> {
    static void findComponent(
        ComponentFinder * cf,
        DNeighborGraph const * graph, IdxType startVertex, IdxType visitedTag,
        void (*callback) (void *), void * callbackData
    ) {
        constexpr int threadsPerBlock = 128;
        IdxType startValues [2] = { 1, startVertex };

        CUDA_CHECK(cudaMemcpy(cf->frontiers[cf->currentFrontier].d_cntFrontier, &startValues[0], 2 * sizeof(IdxType), cudaMemcpyHostToDevice))
        CUDA_CHECK(cudaMemcpy(&cf->d_visited[startVertex], &visitedTag, sizeof(IdxType), cudaMemcpyHostToDevice))

        for (;;) {
            CUDA_CHECK(cudaMemset(cf->frontiers[!cf->currentFrontier].d_cntFrontier, 0, sizeof(IdxType)))

            kernel_bfs <<<
                dim3((graph->nVertices + threadsPerBlock - 1) / threadsPerBlock),
                dim3(threadsPerBlock)
            >>> (
                *graph,
                cf->d_visited,
                visitedTag,
                cf->frontiers[cf->currentFrontier].d_cntFrontier,
                cf->frontiers[cf->currentFrontier].d_frontier,
                cf->frontiers[!cf->currentFrontier].d_cntFrontier,
                cf->frontiers[!cf->currentFrontier].d_frontier
            );

            if (callback) (*callback) (callbackData);

            IdxType cntNewFrontier;
            CUDA_CHECK(cudaMemcpy(
                &cntNewFrontier, cf->frontiers[!cf->currentFrontier].d_cntFrontier, sizeof(IdxType),
                cudaMemcpyDeviceToHost
            ))
            if (!cntNewFrontier) break;
            cf->currentFrontier = !cf->currentFrontier;
        }
    }
};

template <>
struct FindComponent<frontierSharedPolicy> {
    static void findComponent(
        ComponentFinder * cf,
        DNeighborGraph const * graph, IdxType startVertex, IdxType visitedTag,
        void (*callback) (void *), void * callbackData
    ) {
        constexpr int threadsPerBlock = 128;
        IdxType startValues [2] = { 1, startVertex };

        CUDA_CHECK(cudaMemcpy(cf->frontiers[cf->currentFrontier].d_cntFrontier, &startValues, 2 * sizeof(IdxType), cudaMemcpyHostToDevice))
        CUDA_CHECK(cudaMemset(&cf->d_visited[startVertex], visitedTag, sizeof(IdxType)))
        for (;;) {
            CUDA_CHECK(cudaMemset(cf->frontiers[!cf->currentFrontier].d_cntFrontier, 0, sizeof(IdxType)))
            kernel_bfs_shared_frontier <<<
                dim3((graph->nVertices + threadsPerBlock - 1) / threadsPerBlock),
                dim3(threadsPerBlock)
            >>> (
                *graph,
                cf->d_visited,
                visitedTag,
                cf->frontiers[cf->currentFrontier].d_cntFrontier,
                cf->frontiers[cf->currentFrontier].d_frontier,
                cf->frontiers[!cf->currentFrontier].d_cntFrontier,
                cf->frontiers[!cf->currentFrontier].d_frontier
            );
            cudaDeviceSynchronize();

            if (callback) (*callback) (callbackData);

            IdxType cntNewFrontier;
            CUDA_CHECK(cudaMemcpy(
                &cntNewFrontier, cf->frontiers[!cf->currentFrontier].d_cntFrontier, sizeof(IdxType),
                cudaMemcpyDeviceToHost
            ))
            if (!cntNewFrontier) break;
            cf->currentFrontier = !cf->currentFrontier;
        }
    }
};


template <int FrontierPolicyKey>
void ComponentFinder::findComponent(
    DNeighborGraph const * graph, IdxType startVertex, IdxType visitedTag,
    void (*callback) (void *), void * callbackData
) {
    FindComponent<FrontierPolicyKey>::findComponent(
        this, graph, startVertex, visitedTag, callback, callbackData
    );
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

static __global__ void kernel_findUnvisitedNaive(
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


template <int FindNextUnvisitedPolicyKey> struct FindNextUnvisitedPolicy;

template <>
struct FindNextUnvisitedPolicy<findNextUnivisitedNaivePolicy> {

    static auto findNextUnvisited(
        IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
    ) {
        constexpr int threadsPerBlock = 128;
        CUDA_CHECK(cudaMemset(d_resultBuffer, 0, 2 * sizeof(IdxType)))

        kernel_findUnvisitedNaive <<<
            dim3((nVertices + threadsPerBlock - 1) / threadsPerBlock),
            dim3(threadsPerBlock)
        >>> (d_resultBuffer, d_visited, nVertices, startIdx);
        cudaDeviceSynchronize();

        IdxType localBuffer [2];
        CUDA_CHECK(cudaMemcpy(localBuffer, d_resultBuffer, 2 * sizeof(IdxType), cudaMemcpyDeviceToHost))

        struct Result {
            bool wasFound;
            IdxType idx;
        };
        return Result{!!localBuffer[0], localBuffer[1]};
    }
};

static __global__ void kernel_findUnvisitedSuccessive(
    IdxType * outBuffer,
    IdxType * d_visited,
    IdxType nVertices,
    IdxType startPos
) {
    constexpr unsigned int wrp = 32;
    constexpr int logWrp = 5;
    constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;
    unsigned int tid = threadIdx.x;
    unsigned int idx = (startPos & ~(wrp - 1)) + tid;

    IdxType contribution;
    for (;;) {
        contribution = idx < startPos || idx >= nVertices || !!d_visited[idx] ?
            maxIdxType : idx;

        #pragma unroll
        for (int delta = 1; delta < wrp; delta <<= 1) {
            auto other = __shfl_down_sync(0xffffffff, contribution, delta);
            if (other < contribution) contribution = other;
        }

        contribution = __shfl_sync(0xffffffff, contribution, 0);

        if ((idx >> logWrp) == (nVertices >> logWrp) || contribution != maxIdxType) break;

        idx += wrp;
    };

    if (tid == 0) *outBuffer = contribution;
}

template <>
struct FindNextUnvisitedPolicy<findNextUnivisitedSuccessivePolicy> {

    static auto findNextUnvisited(
        IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
    ) {
        constexpr int threadsPerBlock = 32;
        constexpr int blocks = 1;
        constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

        IdxType localBuffer;
        kernel_findUnvisitedSuccessive <<<
            dim3(blocks), dim3(threadsPerBlock)
        >>> (d_resultBuffer, d_visited, nVertices, startIdx);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(&localBuffer, d_resultBuffer, sizeof(IdxType), cudaMemcpyDeviceToHost))

        struct Result {
            bool wasFound;
            IdxType idx;
        };
        return Result{localBuffer != maxIdxType, localBuffer};
    }
};



static __global__ void kernel_findUnvisitedSuccessiveSimplified(
    IdxType * outBuffer,
    IdxType * d_visited,
    IdxType nVertices,
    IdxType startPos
) {
    constexpr unsigned int wrp = 32;
    IdxType result = (IdxType)-1;
    for (IdxType strideIdx = startPos / wrp; strideIdx <= ((nVertices - 1) / wrp); ++strideIdx) {
        IdxType idx = strideIdx * wrp + threadIdx.x;
        int unvisitedMask = __ballot_sync(0xffffffff, idx >= startPos && idx < nVertices && !d_visited[idx]);
        if (unvisitedMask != 0) {
            result = strideIdx * wrp + __ffs(unvisitedMask) - 1;
            break;
        }
    }
    if (threadIdx.x == 0) *outBuffer = result;
}

template <>
struct FindNextUnvisitedPolicy<findNextUnivisitedSuccessiveSimplifiedPolicy> {

    static auto findNextUnvisited(
        IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
    ) {
        struct Result {
            bool wasFound;
            IdxType idx;
        };

        constexpr int threadsPerBlock = 32;
        constexpr int blocks = 1;
        constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

        if (startIdx >= nVertices) return Result{false, 0};

        IdxType localBuffer;
        kernel_findUnvisitedSuccessiveSimplified <<<
            dim3(blocks), dim3(threadsPerBlock)
        >>> (d_resultBuffer, d_visited, nVertices, startIdx);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(&localBuffer, d_resultBuffer, sizeof(IdxType), cudaMemcpyDeviceToHost))

        return Result{localBuffer != maxIdxType, localBuffer};
    }
};

static __global__ void kernel_findUnvisitedSuccessiveMultWarp(
    IdxType * outBuffer,
    IdxType * d_visited,
    IdxType nVertices,
    IdxType startPos
) {
    constexpr unsigned int wrp = 32;
    constexpr unsigned int stride = 2 * wrp;
    constexpr unsigned int strideStartMask = ~(stride - 1);
    constexpr int warpsPerBlock = stride / wrp;

    __shared__ unsigned int contributions[warpsPerBlock];

    constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

    //unsigned int stride = blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int wid = threadIdx.x / wrp;
    unsigned int lane = threadIdx.x % wrp;

    IdxType strideStartIdx = (startPos & ~(wrp - 1));

    IdxType contribution;
    for (;;) {
        // ! TODO: this may overflow
        IdxType idx = strideStartIdx + tid;
        int unvisitedMask = __ballot_sync(0xffffffff, idx >= startPos && idx < nVertices && !d_visited[idx]);

        if (lane == 0) contributions[wid] = unvisitedMask ? strideStartIdx + wrp * wid + __ffs(unvisitedMask) - 1 : maxIdxType;

        __syncthreads();

        if (wid == 0) {
            contribution = tid < warpsPerBlock ? contributions[tid] : maxIdxType;

            #pragma unroll
            for (int delta = 1; delta < warpsPerBlock; delta <<= 1) {
                auto other = __shfl_down_sync(0xffffffff, contribution, delta);
                if (other < contribution) contribution = other;
            }

            if (tid == 0) contributions[0] = contribution;
        }

        __syncthreads();

        contribution = contributions[0];

        if (strideStartIdx >= (nVertices & strideStartMask) || contribution != maxIdxType) break;

        strideStartIdx += stride;
    };

    if (tid == 0) *outBuffer = contribution;
}



template <>
struct FindNextUnvisitedPolicy<findNextUnivisitedSuccessiveMultWarpPolicy> {

    static auto findNextUnvisited(
        IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
    ) {
        constexpr int threadsPerBlock = 2 * 32;
        constexpr int blocks = 1;
        constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

        IdxType localBuffer;
        kernel_findUnvisitedSuccessiveMultWarp <<<
            dim3(blocks), dim3(threadsPerBlock)
        >>> (d_resultBuffer, d_visited, nVertices, startIdx);
        CUDA_CHECK(cudaGetLastError())
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(&localBuffer, d_resultBuffer, sizeof(IdxType), cudaMemcpyDeviceToHost))

        struct Result {
            bool wasFound;
            IdxType idx;
        };
        return Result{localBuffer != maxIdxType, localBuffer};
    }
};



template <int FindNextUnvisitedPolicyKey, int FrontierPolicyKey>
void doFindAllComponents(
    FindComponentsProfile * profile,
    AllComponentsFinder * acf, DNeighborGraph const * graph,
    void (*callback) (void *), void * callbackData
) {
    profile->timeMarkNonCore = runAndMeasureCuda(markNonCore, acf, graph);
    profile->timeFindComponents = runAndMeasureCuda([&]{
        IdxType nIterations = 0;
        IdxType startIdx = 1;
        for (;;) {
            auto nextUnvisited = FindNextUnvisitedPolicy<FindNextUnvisitedPolicyKey>::findNextUnvisited(
                acf->d_resultBuffer, acf->cf.d_visited, graph->nVertices, startIdx
            );
            if (!nextUnvisited.wasFound) break;
            acf->cf.findComponent<FrontierPolicyKey>(graph, nextUnvisited.idx, acf->nextFreeTag, callback, callbackData);
            startIdx = nextUnvisited.idx + 1;
            ++acf->nextFreeTag;
            ++nIterations;
        }
    });
}

static void forceInstantiation() __attribute__ ((unused));
static void forceInstantiation() {
    doFindAllComponents<findNextUnivisitedNaivePolicy, frontierBasicPolicy> (nullptr, nullptr, nullptr, nullptr, nullptr);
    doFindAllComponents<findNextUnivisitedSuccessivePolicy, frontierBasicPolicy> (nullptr, nullptr, nullptr, nullptr, nullptr);
    doFindAllComponents<findNextUnivisitedSuccessiveMultWarpPolicy, frontierBasicPolicy> (nullptr, nullptr, nullptr, nullptr, nullptr);
    doFindAllComponents<findNextUnivisitedSuccessiveSimplifiedPolicy, frontierBasicPolicy> (nullptr, nullptr, nullptr, nullptr, nullptr);
    doFindAllComponents<findNextUnivisitedSuccessiveSimplifiedPolicy, frontierSharedPolicy> (nullptr, nullptr, nullptr, nullptr, nullptr);
}
