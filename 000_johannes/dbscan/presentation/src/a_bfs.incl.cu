#include "a_bfs.h"
#include "cuda_helpers.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

constexpr int findNextUnvisitedCoreNaivePolicy = 1;
constexpr int findNextUnvisitedCoreSuccessivePolicy = 2;
constexpr int findNextUnvisitedCoreSuccessiveMultWarpPolicy = 3;
constexpr int findNextUnvisitedCoreSuccessiveSimplifiedPolicy = 4;

constexpr int frontierBasicPolicy = 1;
constexpr int frontierSharedPolicy = 2;
constexpr int graphTexturePolicy = 3;

// ******************************************************************************************************************************
// Auxiliary data structure for BFS
// ******************************************************************************************************************************

struct FrontierData {
    IdxType * d_frontierBuffer = nullptr;
    struct {
        IdxType * d_cntFrontier;
        IdxType * d_frontier;
    } frontiers[2];
    char currentFrontier = 0;
    IdxType maxFrontierSize;
  
    explicit FrontierData(IdxType maxFrontierSize): maxFrontierSize(maxFrontierSize) {
        // TODO: Should we malloc everything at once?
        size_t frontierBufferSize = 2 * (1 + (std::size_t)maxFrontierSize);
        CUDA_CHECK(cudaMalloc(&this->d_frontierBuffer, frontierBufferSize * sizeof(IdxType)))
        CUDA_CHECK(cudaMemset(this->d_frontierBuffer, 0, frontierBufferSize * sizeof(IdxType)))
        this->frontiers[0] = { this->d_frontierBuffer, this->d_frontierBuffer + 1 };
        this->frontiers[1] = { this->d_frontierBuffer + frontierBufferSize / 2, this->d_frontierBuffer + frontierBufferSize / 2 + 1 };
    }
    FrontierData(FrontierData const &) = delete;
    ~FrontierData() {
        (void)cudaFree(this->d_frontierBuffer);
    }
};

static __device__ __forceinline__ void trap() {
    asm("trap;");
}

// ******************************************************************************************************************************
// FindComponent: template struct, FindComponent<FrontierPolicyKey>::findComponent will provide an interface to our BFS
// ******************************************************************************************************************************

template <int FrontierPolicyKey> struct FindComponent;

// ******************************************************************************************************************************
// FindComponent<frontierBasicPolicy>, using
//   kernel_bfs: simple BFS kernel
// ******************************************************************************************************************************

static __device__ void appendToFrontier(IdxType * cntFrontier, IdxType * frontier, IdxType maxFrontierSize, IdxType vertex) {
    IdxType old = atomicAdd(cntFrontier, 1);
    if (old >= maxFrontierSize) trap();
    frontier[old] = vertex;
}

static __global__ void kernel_bfs(
    DNeighborGraph graph,
    unsigned int * d_visited,
    unsigned int visitedTag, // must be != 0
    IdxType * cntFrontier,
    IdxType * frontier,
    IdxType * cntNewFrontier,
    IdxType * newFrontier,
    IdxType maxFrontierSize
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    auto processFrontierEntry = [&] (IdxType i) {
        IdxType vertex = frontier[i];
        IdxType incidenceListStart = graph.d_startIndices[vertex];
        IdxType incidenceListEnd = graph.d_startIndices[vertex+1];

        for (IdxType j = incidenceListStart; j < incidenceListEnd; ++j) {
            IdxType destination = graph.d_incidenceAry[j];
            unsigned int destinationVisited = d_visited[destination];
            if(destinationVisited <= 1) {
                d_visited[destination] = visitedTag;
                if (destinationVisited == 1) appendToFrontier(cntNewFrontier, newFrontier, maxFrontierSize, destination);
            }
        }
    };

    IdxType cnt = *cntFrontier;
    IdxType strideBegin = 0;
    if (cnt > stride) for (; strideBegin < cnt - stride; strideBegin += stride) processFrontierEntry(strideBegin + tid);
    if (tid < cnt - strideBegin) processFrontierEntry(strideBegin + tid);
}

template <>
struct FindComponent<frontierBasicPolicy> {
    static void findComponent(
        std::vector<std::vector<IdxType>> & visitedSteps,
        int nSm,
        IdxType * d_visited,
        FrontierData * fd,
        DNeighborGraph const * graph, IdxType startVertex, IdxType visitedTag
    ) {
        int nBlocks = 16 * nSm;
        constexpr int nThreadsPerBlock = 128;
        IdxType startValues [2] = { 1, startVertex };

        CUDA_CHECK(cudaMemcpy(fd->frontiers[fd->currentFrontier].d_cntFrontier, &startValues[0], 2 * sizeof(IdxType), cudaMemcpyHostToDevice))
        CUDA_CHECK(cudaMemcpy(&d_visited[startVertex], &visitedTag, sizeof(IdxType), cudaMemcpyHostToDevice))

        visitedSteps.push_back(std::vector<IdxType> (graph->nVertices));
        CUDA_CHECK(cudaMemcpy(visitedSteps[visitedSteps.size() - 1].data(), d_visited, graph->nVertices * sizeof(IdxType), cudaMemcpyDeviceToHost))
        for (;;) {
            CUDA_CHECK(cudaMemset(fd->frontiers[!fd->currentFrontier].d_cntFrontier, 0, sizeof(IdxType)))

            kernel_bfs <<<nBlocks, nThreadsPerBlock>>> (
                *graph,
                d_visited,
                visitedTag,
                fd->frontiers[fd->currentFrontier].d_cntFrontier,
                fd->frontiers[fd->currentFrontier].d_frontier,
                fd->frontiers[!fd->currentFrontier].d_cntFrontier,
                fd->frontiers[!fd->currentFrontier].d_frontier,
                fd->maxFrontierSize
            );
            visitedSteps.push_back(std::vector<IdxType> (graph->nVertices));
            CUDA_CHECK(cudaMemcpy(visitedSteps[visitedSteps.size() - 1].data(), d_visited, graph->nVertices * sizeof(IdxType), cudaMemcpyDeviceToHost))    

            IdxType cntNewFrontier;
            CUDA_CHECK(cudaMemcpy(
                &cntNewFrontier, fd->frontiers[!fd->currentFrontier].d_cntFrontier, sizeof(IdxType),
                cudaMemcpyDeviceToHost
            ))
            if (!cntNewFrontier) break;
            fd->currentFrontier = !fd->currentFrontier;
        }
    }
};

// ******************************************************************************************************************************
// FindComponent<frontierSharedPolicy>, using
//   kernel_bfs_shared_frontier: BFS kernel which uses shared memory in building the frontier
// ******************************************************************************************************************************

static __device__ void appendToFrontierShared(
    IdxType * cntSharedFrontier, IdxType * sharedFrontier, unsigned int sharedFrontierSize,
    IdxType * cntFrontier, IdxType * frontier, IdxType maxFrontierSize, IdxType vertex
) {
    if (*cntSharedFrontier < sharedFrontierSize) {
        IdxType old = atomicAdd(cntSharedFrontier, 1);
        sharedFrontier[old] = vertex;
    } else {
        IdxType old = atomicAdd(cntFrontier, 1);
        if (old >= maxFrontierSize) trap();
        frontier[old] = vertex;
    }
}

static __device__ void copySharedToGlobalFrontier(
    IdxType * startPos,
    IdxType * cntSharedFrontier, IdxType * sharedFrontier, unsigned int sharedFrontierSize,
    IdxType * cntGlobalFrontier, IdxType * globalFrontier, IdxType maxFrontierSize
) {
    if (threadIdx.x == 0) *startPos = atomicAdd(cntGlobalFrontier, *cntSharedFrontier);
    __syncthreads();
    if (*startPos >= maxFrontierSize) trap();
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
    IdxType * newFrontier,
    unsigned int sharedFrontierSize,
    IdxType maxFrontierSize
) {
    __shared__ IdxType cntSharedFrontier;
    __shared__ IdxType startPos;
    extern __shared__ IdxType sharedFrontier[];

    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    if (threadIdx.x == 0) cntSharedFrontier = 0;
    __syncthreads();

    auto processFrontierEntry = [&] (IdxType i) {
        IdxType vertex = frontier[i];
        IdxType incidenceListStart = graph.d_startIndices[vertex];
        IdxType incidenceListEnd = graph.d_startIndices[vertex+1];

        for (IdxType j = incidenceListStart; j < incidenceListEnd; ++j) {
            IdxType destination = graph.d_incidenceAry[j];
            unsigned int destinationVisited = d_visited[destination];
            if(destinationVisited <= 1) {
                d_visited[destination] = visitedTag;
                if (destinationVisited == 1) appendToFrontierShared(
                    &cntSharedFrontier, sharedFrontier, sharedFrontierSize,
                    cntNewFrontier, newFrontier, maxFrontierSize,
                    destination
                );
            }
        }
    };

    IdxType cnt = *cntFrontier;
    IdxType strideBegin = 0;
    if (cnt > stride) for (; strideBegin < cnt - stride; strideBegin += stride) processFrontierEntry(strideBegin + tid);
    if (tid < cnt - strideBegin) processFrontierEntry(strideBegin + tid);

    __syncthreads();

    copySharedToGlobalFrontier(
        &startPos,
        &cntSharedFrontier, sharedFrontier, sharedFrontierSize,
        cntNewFrontier, newFrontier, maxFrontierSize
    );
}

template <>
struct FindComponent<frontierSharedPolicy> {
    static void findComponent(
        int nSm,
        IdxType * d_visited,
        FrontierData * fd,
        DNeighborGraph const * graph, IdxType startVertex, IdxType visitedTag
    ) {
        int nBlocks = 2 * nSm;
        unsigned int sharedFrontierSize = (1u << 13) / (nBlocks / nSm); // 8 * 1024 values -> 4 * 8 * 1024 Bytes = 32 kiB
        constexpr int nThreadsPerBlock = 512;

        IdxType startValues [2] = { 1, startVertex };

        CUDA_CHECK(cudaMemcpy(fd->frontiers[fd->currentFrontier].d_cntFrontier, &startValues, 2 * sizeof(IdxType), cudaMemcpyHostToDevice))
        CUDA_CHECK(cudaMemcpy(&d_visited[startVertex], &visitedTag, sizeof(IdxType), cudaMemcpyHostToDevice))
        for (;;) {
            CUDA_CHECK(cudaMemset(fd->frontiers[!fd->currentFrontier].d_cntFrontier, 0, sizeof(IdxType)))
            kernel_bfs_shared_frontier <<<nBlocks, nThreadsPerBlock, sharedFrontierSize * sizeof(IdxType)>>> (
                *graph,
                d_visited,
                visitedTag,
                fd->frontiers[fd->currentFrontier].d_cntFrontier,
                fd->frontiers[fd->currentFrontier].d_frontier,
                fd->frontiers[!fd->currentFrontier].d_cntFrontier,
                fd->frontiers[!fd->currentFrontier].d_frontier,
                sharedFrontierSize,
                fd->maxFrontierSize
            );

            IdxType cntNewFrontier;
            CUDA_CHECK(cudaMemcpy(
                &cntNewFrontier, fd->frontiers[!fd->currentFrontier].d_cntFrontier, sizeof(IdxType),
                cudaMemcpyDeviceToHost
            ))
            if (!cntNewFrontier) break;
            fd->currentFrontier = !fd->currentFrontier;
        }
    }
};

// ******************************************************************************************************************************
// markCore: helper function for initializing clusters array by marking core elements
// ******************************************************************************************************************************

static __global__ void kernel_markCoreUnvisited(
    IdxType * d_visited,
    IdxType * d_d_startIndices,
    IdxType nVertices
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nVertices) {
        if (d_d_startIndices[tid + 1] - d_d_startIndices[tid] != 0) {
            d_visited[tid] = 1;
        }
    }    
}

static void markCoreUnvisited(IdxType * d_visited, DNeighborGraph const * graph) {
    constexpr int nThreadsPerBlock = 128;
    kernel_markCoreUnvisited <<<
        dim3((graph->nVertices + nThreadsPerBlock - 1) / nThreadsPerBlock),
        dim3(nThreadsPerBlock)    
    >>> (
        d_visited,
        graph->d_startIndices,
        graph->nVertices
    );
    CUDA_CHECK(cudaGetLastError())
}

// ******************************************************************************************************************************
// FindNextUnvisitedCore:
//   template struct, FindNextUnvisitedCore<FrontierPolicyKey>::findNextUnvisitedCore finds next unvisited (core) node
// ******************************************************************************************************************************

template <int FindNextUnvisitedCorePolicyKey>
struct FindNextUnvisitedCore {
    struct Result {
        bool wasFound;
        IdxType idx;
    };
    static Result findNextUnvisitedCore(
        IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
    );
};

// ******************************************************************************************************************************
// FindNextUnvisitedCore<findNextUnvisitedCoreNaivePolicy>
//   naive, but simple way of finding next unvisited (core) node
// ******************************************************************************************************************************

static __global__ void kernel_findUnvisitedCoreNaive(
    IdxType * outBuffer,
    IdxType * d_visited,
    IdxType nVertices,
    IdxType startPos
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nVertices) {
        if (d_visited[tid] == 1) {
            outBuffer[0] = 1; // true
            outBuffer[1] = tid;
        }
    }    
}

template <>
auto FindNextUnvisitedCore<findNextUnvisitedCoreNaivePolicy>::findNextUnvisitedCore(
    IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
) -> FindNextUnvisitedCore<findNextUnvisitedCoreNaivePolicy>::Result {
    constexpr int nThreadsPerBlock = 128;
    CUDA_CHECK(cudaMemset(d_resultBuffer, 0, 2 * sizeof(IdxType)))

    kernel_findUnvisitedCoreNaive <<<
        (nVertices + nThreadsPerBlock - 1) / nThreadsPerBlock,
        nThreadsPerBlock
    >>> (d_resultBuffer, d_visited, nVertices, startIdx);

    IdxType localBuffer [2];
    CUDA_CHECK(cudaMemcpy(localBuffer, d_resultBuffer, 2 * sizeof(IdxType), cudaMemcpyDeviceToHost))

    return {!!localBuffer[0], localBuffer[1]};
}

// ******************************************************************************************************************************
// FindNextUnvisitedCore<findNextUnvisitedCoreSuccessivePolicy>
//   start over where you stopped, rather than at the beginning
// ******************************************************************************************************************************

static __global__ void kernel_findUnvisitedCoreSuccessive(
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
        contribution = idx < startPos || idx >= nVertices || d_visited[idx] != 1?
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
auto FindNextUnvisitedCore<findNextUnvisitedCoreSuccessivePolicy>::findNextUnvisitedCore(
    IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
) -> FindNextUnvisitedCore<findNextUnvisitedCoreSuccessivePolicy>::Result {
    constexpr int nThreadsPerBlock = 32;
    constexpr int blocks = 1;
    constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

    IdxType localBuffer;
    kernel_findUnvisitedCoreSuccessive <<<
        blocks, nThreadsPerBlock
    >>> (d_resultBuffer, d_visited, nVertices, startIdx);

    CUDA_CHECK(cudaMemcpy(&localBuffer, d_resultBuffer, sizeof(IdxType), cudaMemcpyDeviceToHost))

    return {localBuffer != maxIdxType, localBuffer};
}

// ******************************************************************************************************************************
// FindNextUnvisitedCore<findNextUnvisitedCoreSuccessiveSimplifiedPolicy>
//   __ballot_sync rather than __shfl_sync
// ******************************************************************************************************************************

static __global__ void kernel_findUnvisitedCoreSuccessiveSimplified(
    IdxType * outBuffer,
    IdxType * d_visited,
    IdxType nVertices,
    IdxType startPos
) {
    constexpr unsigned int wrp = 32;
    IdxType result = (IdxType)-1;
    for (IdxType strideIdx = startPos / wrp; strideIdx <= ((nVertices - 1) / wrp); ++strideIdx) {
        IdxType idx = strideIdx * wrp + threadIdx.x;
        int unvisitedMask = __ballot_sync(0xffffffff, idx >= startPos && idx < nVertices && d_visited[idx] == 1);
        if (unvisitedMask != 0) {
            result = strideIdx * wrp + __ffs(unvisitedMask) - 1;
            break;
        }
    }
    if (threadIdx.x == 0) *outBuffer = result;
}

template <>
auto FindNextUnvisitedCore<findNextUnvisitedCoreSuccessiveSimplifiedPolicy>::findNextUnvisitedCore(
    IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
) -> FindNextUnvisitedCore<findNextUnvisitedCoreSuccessiveSimplifiedPolicy>::Result {
    constexpr int nThreadsPerBlock = 32;
    constexpr int blocks = 1;
    constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

    if (startIdx >= nVertices) return Result{false, 0};

    IdxType localBuffer;
    kernel_findUnvisitedCoreSuccessiveSimplified <<<
        blocks, nThreadsPerBlock
    >>> (d_resultBuffer, d_visited, nVertices, startIdx);

    CUDA_CHECK(cudaMemcpy(&localBuffer, d_resultBuffer, sizeof(IdxType), cudaMemcpyDeviceToHost))

    return Result{localBuffer != maxIdxType, localBuffer};
}

// ******************************************************************************************************************************
// FindNextUnvisitedCore<findNextUnvisitedCoreSuccessiveMultWarpPolicy>
//   __ballot_sync rather than __shfl_sync, employ several warps
// ******************************************************************************************************************************

static __global__ void kernel_findUnvisitedCoreSuccessiveMultWarp(
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
        int unvisitedMask = __ballot_sync(0xffffffff, idx >= startPos && idx < nVertices && d_visited[idx] == 1);

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
auto FindNextUnvisitedCore<findNextUnvisitedCoreSuccessiveMultWarpPolicy>::findNextUnvisitedCore(
    IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
) -> FindNextUnvisitedCore<findNextUnvisitedCoreSuccessiveMultWarpPolicy>::Result {
    constexpr int nThreadsPerBlock = 2 * 32;
    constexpr int blocks = 1;
    constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

    IdxType localBuffer;
    kernel_findUnvisitedCoreSuccessiveMultWarp <<<
        dim3(blocks), dim3(nThreadsPerBlock)
    >>> (d_resultBuffer, d_visited, nVertices, startIdx);
    CUDA_CHECK(cudaGetLastError())

    CUDA_CHECK(cudaMemcpy(&localBuffer, d_resultBuffer, sizeof(IdxType), cudaMemcpyDeviceToHost))

        struct Result {
            bool wasFound;
            IdxType idx;
        };
    return {localBuffer != maxIdxType, localBuffer};
}

// ******************************************************************************************************************************
// findAllComponents: find all the clusters
// ******************************************************************************************************************************

template <int FindNextUnvisitedCorePolicyKey, int FrontierPolicyKey>
static void findAllComponents(
    int nSm,
    IdxType * d_visited,
    FindComponentsProfilingData * profile,
    DNeighborGraph const * graph,
    std::vector<std::vector<IdxType>> & visitedSteps
) {
    FrontierData fd{graph->lenIncidenceAry};

    IdxType nextFreeTag = 2;
    ManagedDeviceArray<IdxType> d_resultBuffer {2};

    CUDA_CHECK(cudaMemset(d_visited, 0, graph->nVertices * sizeof(IdxType)))

    profile->timeMarkCoreUnvisited = runAndMeasureCuda(markCoreUnvisited, d_visited, graph);
    profile->timeFindComponents = runAndMeasureCuda([&]{
        IdxType nIterations = 0;
        IdxType startIdx = 0;
        for (;;) {
            auto nextUnvisitedCore = FindNextUnvisitedCore<FindNextUnvisitedCorePolicyKey>::findNextUnvisitedCore(
                d_resultBuffer.ptr(), d_visited, graph->nVertices, startIdx
            );
            if (!nextUnvisitedCore.wasFound) break;
            FindComponent<FrontierPolicyKey>::findComponent(visitedSteps, nSm, d_visited, &fd, graph, nextUnvisitedCore.idx, nextFreeTag);
            startIdx = nextUnvisitedCore.idx + 1;
            ++nextFreeTag;
            ++nIterations;
        }
        CUDA_CHECK(cudaGetLastError())
    });
}
