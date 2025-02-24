#include "bfs.h"
#include "cuda_helpers.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

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
  
    FrontierData(size_t maxFrontierSize) {
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

// ******************************************************************************************************************************
// FindComponent: template struct, FindComponent<FrontierPolicyKey>::findComponent will provide an interface to our BFS
// ******************************************************************************************************************************

template <int FrontierPolicyKey> struct FindComponent;

// ******************************************************************************************************************************
// FindComponent<graphTexturePolicy>, using
//   kernel_bfs_texture: kernel relying on texture memory for incidence lists
// ******************************************************************************************************************************

static __device__ void appendToFrontier(IdxType * cntFrontier, IdxType * frontier, IdxType vertex) {
    IdxType old = atomicAdd(cntFrontier, 1);
    frontier[old] = vertex;
}

texture<IdxType, 1, cudaReadModeElementType> startIndicesTexture;

static __global__ void kernel_bfs_texture(
    DNeighborGraph graph,
    unsigned int * d_visited,
    unsigned int visitedTag, // must be != 0
    IdxType * cntFrontier,
    IdxType * frontier,
    IdxType * cntNewFrontier,
    IdxType * newFrontier
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    auto processFrontierEntry = [&] (IdxType i) {
        IdxType vertex = frontier[i];
        IdxType incidenceListStart = tex1D(startIndicesTexture, vertex);
        IdxType incidenceListEnd = tex1D(startIndicesTexture, vertex + 1);

        for (IdxType j = incidenceListStart; j < incidenceListEnd; ++j) {
            IdxType destination = graph.d_incidenceAry[j];
            unsigned int destinationVisited = d_visited[destination];
            if(destinationVisited <= 1) {
                d_visited[destination] = visitedTag;
                if (destinationVisited == 0) appendToFrontier(cntNewFrontier, newFrontier, destination);
            }
        }
    };

    IdxType cnt = *cntFrontier;
    IdxType strideBegin = 0;
    if (cnt > stride) for (; strideBegin < cnt - stride; strideBegin += stride) processFrontierEntry(strideBegin + tid);
    if (tid < cnt - strideBegin) processFrontierEntry(strideBegin + tid);
}

template <>
struct FindComponent<graphTexturePolicy> {
    static void findComponent(
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

        for (;;) {
            CUDA_CHECK(cudaMemset(fd->frontiers[!fd->currentFrontier].d_cntFrontier, 0, sizeof(IdxType)))

            kernel_bfs_texture <<<nBlocks, nThreadsPerBlock>>> (
                *graph,
                d_visited,
                visitedTag,
                fd->frontiers[fd->currentFrontier].d_cntFrontier,
                fd->frontiers[fd->currentFrontier].d_frontier,
                fd->frontiers[!fd->currentFrontier].d_cntFrontier,
                fd->frontiers[!fd->currentFrontier].d_frontier
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
// markNonCore: helper function for initializing clusters array by marking non-core elements
// ******************************************************************************************************************************

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

static void markNonCore(IdxType * d_visited, DNeighborGraph const * graph) {
    constexpr int nThreadsPerBlock = 128;
    kernel_markNonCore <<<
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
// FindNextUnvisited:
//   template struct, FindNextUnvisited<FrontierPolicyKey>::findNextUnvisited finds next unvisited (core) node
// ******************************************************************************************************************************

template <int FindNextUnvisitedPolicyKey>
struct FindNextUnvisited {
    struct Result {
        bool wasFound;
        IdxType idx;
    };
    static Result findNextUnvisited(
        IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
    );
};

// ******************************************************************************************************************************
// FindNextUnvisited<findNextUnvisitedNaivePolicy>
//   naive, but simple way of finding next unvisited (core) node
// ******************************************************************************************************************************

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

template <>
auto FindNextUnvisited<findNextUnvisitedNaivePolicy>::findNextUnvisited(
    IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
) -> FindNextUnvisited<findNextUnvisitedNaivePolicy>::Result {
    constexpr int nThreadsPerBlock = 128;
    CUDA_CHECK(cudaMemset(d_resultBuffer, 0, 2 * sizeof(IdxType)))

    kernel_findUnvisitedNaive <<<
        (nVertices + nThreadsPerBlock - 1) / nThreadsPerBlock,
        nThreadsPerBlock
    >>> (d_resultBuffer, d_visited, nVertices, startIdx);

    IdxType localBuffer [2];
    CUDA_CHECK(cudaMemcpy(localBuffer, d_resultBuffer, 2 * sizeof(IdxType), cudaMemcpyDeviceToHost))

    return {!!localBuffer[0], localBuffer[1]};
}

// ******************************************************************************************************************************
// FindNextUnvisited<findNextUnvisitedSuccessivePolicy>
//   start over where you stopped, rather than at the beginning
// ******************************************************************************************************************************

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
auto FindNextUnvisited<findNextUnvisitedSuccessivePolicy>::findNextUnvisited(
    IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
) -> FindNextUnvisited<findNextUnvisitedSuccessivePolicy>::Result {
    constexpr int nThreadsPerBlock = 32;
    constexpr int blocks = 1;
    constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

    IdxType localBuffer;
    kernel_findUnvisitedSuccessive <<<
        blocks, nThreadsPerBlock
    >>> (d_resultBuffer, d_visited, nVertices, startIdx);

    CUDA_CHECK(cudaMemcpy(&localBuffer, d_resultBuffer, sizeof(IdxType), cudaMemcpyDeviceToHost))

    return {localBuffer != maxIdxType, localBuffer};
}

// ******************************************************************************************************************************
// FindNextUnvisited<findNextUnvisitedSuccessiveSimplifiedPolicy>
//   __ballot_sync rather than __shfl_sync
// ******************************************************************************************************************************

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
auto FindNextUnvisited<findNextUnvisitedSuccessiveSimplifiedPolicy>::findNextUnvisited(
    IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
) -> FindNextUnvisited<findNextUnvisitedSuccessiveSimplifiedPolicy>::Result {
    constexpr int nThreadsPerBlock = 32;
    constexpr int blocks = 1;
    constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

    if (startIdx >= nVertices) return Result{false, 0};

    IdxType localBuffer;
    kernel_findUnvisitedSuccessiveSimplified <<<
        blocks, nThreadsPerBlock
    >>> (d_resultBuffer, d_visited, nVertices, startIdx);

    CUDA_CHECK(cudaMemcpy(&localBuffer, d_resultBuffer, sizeof(IdxType), cudaMemcpyDeviceToHost))

    return Result{localBuffer != maxIdxType, localBuffer};
}

// ******************************************************************************************************************************
// FindNextUnvisited<findNextUnvisitedSuccessiveMultWarpPolicy>
//   __ballot_sync rather than __shfl_sync, employ several warps
// ******************************************************************************************************************************

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
auto FindNextUnvisited<findNextUnvisitedSuccessiveMultWarpPolicy>::findNextUnvisited(
    IdxType * d_resultBuffer, IdxType * d_visited, IdxType nVertices, IdxType startIdx
) -> FindNextUnvisited<findNextUnvisitedSuccessiveMultWarpPolicy>::Result {
    constexpr int nThreadsPerBlock = 2 * 32;
    constexpr int blocks = 1;
    constexpr IdxType maxIdxType = (IdxType)0 - (IdxType)1;

    IdxType localBuffer;
    kernel_findUnvisitedSuccessiveMultWarp <<<
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

template <int FindNextUnvisitedPolicyKey, int FrontierPolicyKey>
void findAllComponents(
    int nSm,
    IdxType * d_visited,
    FindComponentsProfilingData * profile,
    DNeighborGraph const * graph
) {
    FrontierData fd{graph->lenIncidenceAry};

    IdxType nextFreeTag = 2;
    ManagedDeviceArray<IdxType> d_resultBuffer {2};

    CUDA_CHECK(cudaMemset(d_visited, 0, graph->nVertices * sizeof(IdxType)))

    cudaArray * texArray = nullptr;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<IdxType> ();
    CUDA_CHECK(cudaMallocArray(&texArray, &channelDesc, graph->nVertices + 1))
    CUDA_CHECK(cudaMemcpy2DToArray(
        texArray, 0, 0, graph->d_startIndices,
        graph->nVertices + 1, graph->nVertices + 1, 1,
        cudaMemcpyDeviceToDevice
    ))
    CUDA_CHECK(cudaBindTextureToArray(startIndicesTexture, texArray))

    profile->timeMarkNonCore = runAndMeasureCuda(markNonCore, d_visited, graph);
    profile->timeFindComponents = runAndMeasureCuda([&]{
        IdxType nIterations = 0;
        IdxType startIdx = 1;
        for (;;) {
            auto nextUnvisited = FindNextUnvisited<FindNextUnvisitedPolicyKey>::findNextUnvisited(
                d_resultBuffer.ptr(), d_visited, graph->nVertices, startIdx
            );
            if (!nextUnvisited.wasFound) break;
            FindComponent<graphTexturePolicy>::findComponent(nSm, d_visited, &fd, graph, nextUnvisited.idx, nextFreeTag);
            startIdx = nextUnvisited.idx + 1;
            ++nextFreeTag;
            ++nIterations;
        }
    });
}

static void forceInstantiation() __attribute__ ((unused));
static void forceInstantiation() {
    findAllComponents<findNextUnvisitedSuccessivePolicy, frontierSharedPolicy> (0, nullptr, nullptr, nullptr);
}
