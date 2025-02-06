#include "cluster_expansion.h"
#include "types.h"
#include "cuda_helpers.h"

#include <iostream>

DPoints copyPointsToDevice(float const * x, float const * y, IdxType n) {
  float * d_x, * d_y;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)))
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)))
  CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice))
  CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice))
  return { n, d_x, d_y };
}

constexpr IdxType maxSeedLength = 1024; // TODO: Could be larger!?
constexpr IdxType nChains = 512; // TODO: ??


struct ThreadData {
  IdxType currentClusterId;
  IdxType pointBeingProcessedIdx;
  bool * collisionMatrix;
  IdxType * clusters;
  unsigned int * pointStates;
  IdxType * seedList;
  IdxType * seedClusterIds;
  IdxType * seedLength;
  IdxType * seedReserved;
  IdxType * neighborBuffer;
  IdxType * neighborCount;
};

// collisionMatrix:
//   in theory: n x n, but only entries (i,j) with j >= i used
//   so entry (i,j) is preceded by how many entries?
//   n for row 0
//   n-1 for row 1
//   ...
//   n-i+1 for row (i-1)
//   j-i for row i
//   -> i * (n + n - i + 1)/2 + j - i = j + i * (2n - i + 1 - 2) / 2 = j + i * (2 * n - (i + 1)) / 2
// correct?
//   entry (0, n-1) -> n-1 + 0
//   entry (1, 1) -> 1 + 1 * (2n - 2) / 2 -> n
//   entry (2, 2) -> 2 + 2 * (2n - 3) / 2 -> 2n - 1
//         (i, i) -> i * n - i * (i - 1) / 2
//         (i, j) -> i * n - i * (i + 1) / 2 + j
static __device__ void markAsCandidate(
  ThreadData const & td,
  IdxType pointIdx
) {
  unsigned int state = td.pointStates[pointIdx];
  if (state == stateFree) {
    IdxType oldReserved = atomicAdd(td.seedReserved, 1);
    if (oldReserved < maxSeedLength) {
      state = atomicCAS(&td.pointStates[pointIdx], stateFree, stateReserved);
      if (state == stateFree) {
        int h = atomicAdd(td.seedLength, 1);
        td.seedList[h] = pointIdx; td.seedClusterIds[h] = td.currentClusterId;
      } else {
        atomicSub(td.seedReserved, 1);
      }
    } else {
      atomicSub(td.seedReserved, 1);
    }
  }
  switch (state) {
    case stateCore: {
      // TODO: handle collision!
    } break;
    case stateNoiseOrBorder: {
      td.clusters[pointIdx] = td.currentClusterId;
    } break;
    case stateUnderInspection: {
      (void)atomicCAS(&td.clusters[pointIdx], 0, td.currentClusterId);
    } break;
    default:
      ; // do nothing
  }
  /*
  if (oldState == cUnprocessed) {
  } else {
    if (oldState != cNoise && oldState != cChain + blockIdx.x) {
      if (oldState < pointIdx) {
        // (oldState, pointIdx)
        collisionMatrix[pointIdx + oldState * (2 * nChains - (oldState + 1)) / 2] = true;
      } else {
        collisionMatrix[oldState + pointIdx * (2 * nChains - (pointIdx + 1)) / 2] = true;
      }
    }
  }
  */
}

static __device__ void processObject(
  ThreadData const & td,
  float px, float py,
  float const * xs, float const * ys, IdxType pointIdx,
  IdxType coreThreshold, float rsq
) {
  int lane = threadIdx.x % 32;
  float dx = xs[pointIdx] - px;
  float dy = ys[pointIdx] - py;
  bool isNeighbor = dx * dx + dy * dy <= rsq;
  unsigned int neighborMask = __ballot_sync(__activemask(), isNeighbor);
  if (isNeighbor) {
    int leaderLane = __ffs(neighborMask) - 1;
    int nNeighbors = __popc(neighborMask);
    int oldNeighborCount = 0;
    if (lane == leaderLane) oldNeighborCount = atomicAdd(td.neighborCount, nNeighbors);
    oldNeighborCount = __shfl_sync(neighborMask, oldNeighborCount, leaderLane);
    if (oldNeighborCount < coreThreshold && oldNeighborCount + nNeighbors >= coreThreshold && lane == leaderLane) {
      td.clusters[td.pointBeingProcessedIdx] = td.currentClusterId;
      td.pointStates[td.pointBeingProcessedIdx] = stateCore;
    }
    int h = oldNeighborCount + __popc(neighborMask & ((1u << lane) - 1));
    if (h >= coreThreshold) {
      markAsCandidate(td, pointIdx);
    } else {
      td.neighborBuffer[h] = pointIdx;
    }
  }
}


// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * cluster, unsigned int * pointState,
  bool * collisionMatrix,
  float const * xs, float const * ys, IdxType n,
  IdxType * seedLists, unsigned int * seedClusterIds, IdxType * seedLengths,
  IdxType coreThreshold, float rsq
) {
  extern __shared__ unsigned char sMem []; // coreThreshold IdxType values in blocks of 128 bytes + 2 IdxType value

  unsigned int threadGroupIdx = blockIdx.x;
  unsigned int stride = blockDim.x;

  ThreadData td;
  td.clusters = cluster;
  td.pointStates = pointState;
  td.collisionMatrix = collisionMatrix;

  td.neighborBuffer = (IdxType *)sMem; // Length: coreThreshold elements
  td.neighborCount = (IdxType *) (sMem + (coreThreshold * sizeof(IdxType) + 127) / 128 * 128);
  td.seedReserved = td.neighborCount + 1;
  static_assert(128 % alignof(IdxType) == 0, "");
  
  if (threadIdx.x == 0) *td.neighborCount = 0;

  td.seedLength = &seedLengths[threadGroupIdx];
  td.seedList = &seedLists[maxSeedLength * threadGroupIdx];
  td.seedClusterIds = &seedClusterIds[maxSeedLength * threadGroupIdx];

  IdxType seedLength = *td.seedLength;

  __syncthreads();

  if (seedLength > 0) {
    --seedLength;

    if (threadIdx.x == 0) { *td.seedLength = seedLength; *td.seedReserved = seedLength; }

    __syncthreads();
    
    td.pointBeingProcessedIdx = td.seedList[seedLength];
    td.currentClusterId = td.seedClusterIds[seedLength];
    if (td.currentClusterId == 0) {
      td.currentClusterId = td.pointBeingProcessedIdx + 1;
    }
    if (threadIdx.x == 0) td.pointStates[td.pointBeingProcessedIdx] = stateUnderInspection;

    __syncthreads();

    float x = xs[td.pointBeingProcessedIdx], y = ys[td.pointBeingProcessedIdx];
    {
      IdxType strideIdx = 0;
      for (; strideIdx < (n - 1) / stride; ++strideIdx) {
        processObject(
          td,
          x, y,
          xs, ys, strideIdx * stride + threadIdx.x,
          coreThreshold, rsq
        );
      }
      if (threadIdx.x < n - strideIdx * stride) {
        processObject(
          td,
          x, y,
          xs, ys, strideIdx * stride + threadIdx.x,
          coreThreshold, rsq
        );
      }
    }

    __syncthreads();

    if (*td.neighborCount >= coreThreshold) {
      //if (threadIdx.x == 0) pointState[seedPointIdx] = stateCore;
      for (int i = threadIdx.x; i < coreThreshold; i += stride) {
        markAsCandidate(td, td.neighborBuffer[i]);
      }
    } else {
      if (threadIdx.x == 0) td.pointStates[td.pointBeingProcessedIdx] = stateNoiseOrBorder;
    }

  }

}

void allocateDeviceMemory(
  unsigned int ** d_pointStates, IdxType ** d_clusters,
  IdxType ** d_seedLists, IdxType ** d_seedClusterIds, IdxType ** d_seedLengths,
  bool ** d_collisionMatrix,
  int nBlocks,
  IdxType n
) {
  CUDA_CHECK(cudaMalloc(d_pointStates, n * sizeof(unsigned int)))
  // TODO: Change later
  CUDA_CHECK(cudaMemset(*d_pointStates, 0, n * sizeof(unsigned int)))
  CUDA_CHECK(cudaMalloc(d_clusters, n * sizeof(IdxType)))
  // TODO: Change later
  CUDA_CHECK(cudaMemset(*d_clusters, 0, n * sizeof(IdxType)))

  CUDA_CHECK(cudaMalloc(d_seedLists, nBlocks * maxSeedLength * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(d_seedClusterIds, nBlocks * maxSeedLength * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(d_seedLengths, nBlocks * sizeof(IdxType)))
  CUDA_CHECK(cudaMemset(*d_seedLengths, 0, nBlocks * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(d_collisionMatrix, nBlocks * (nBlocks + 1) / 2 * sizeof(bool)))
  CUDA_CHECK(cudaMemset(*d_collisionMatrix, 0, nBlocks * (nBlocks + 1) / 2 * sizeof(bool)))
}

static __global__ void kernel_populateSeedLists(
  IdxType * d_seedLists, IdxType * d_seedClusterIds, IdxType * d_seedLengths, IdxType nLists
) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < nLists; i += stride) {
    d_seedLengths[i] = 1;
    d_seedLists[i * maxSeedLength] = i; // TODO: Change later!!!
    d_seedClusterIds[i * maxSeedLength] = i + 1; // TODO: Change later!!!
  }
}

static __global__ void kernel_refillSeed(
    IdxType * d_foundAt,
    IdxType * d_seedLists, IdxType * d_seedClusterIds, IdxType * d_seedLengths, int k,
    unsigned int * d_pointState, IdxType * d_clusters, IdxType n,
    IdxType startPos
) {
    constexpr unsigned int wrp = 32;
    IdxType result = (IdxType)-1;
    for (IdxType strideIdx = startPos / wrp; strideIdx <= ((n - 1) / wrp); ++strideIdx) {
        IdxType idx = strideIdx * wrp + threadIdx.x;
        int unvisitedMask = __ballot_sync(0xffffffff, idx >= startPos && idx < n && !d_pointState[idx]);
        if (unvisitedMask != 0) {
            result = strideIdx * wrp + __ffs(unvisitedMask) - 1;
            break;
        }
    }
    if (threadIdx.x == 0) {
      if (result == (IdxType)-1) {
        *d_foundAt = result;
      } else {
        *d_foundAt = result;
        d_pointState[result] = stateReserved2;
        d_seedLists[k * maxSeedLength] = result;
        d_seedClusterIds[k * maxSeedLength] = d_clusters[result];
        d_seedLengths[k] = 1;
      }
    }
}

void findClusters(
  unsigned int * d_pointStates, IdxType * d_clusters, bool * d_collisionMatrix,
  float * xs, float * ys, IdxType n,
  IdxType * d_seedLists, IdxType * d_seedClusterIds, IdxType * d_seedLengths,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 6;
  constexpr int nThreadsPerBlock = 256;

  IdxType * d_foundAt;
  CUDA_CHECK(cudaMalloc(&d_foundAt, sizeof(IdxType)))

  IdxType seedLengths [nBlocks];
  IdxType startPos = 0;
  for (;;) {
    CUDA_CHECK(cudaMemcpy(seedLengths, d_seedLengths, nBlocks * sizeof(IdxType), cudaMemcpyDeviceToHost))
    bool stillWork = false;

    for (int k = 0; k < nBlocks; ++k) {
      if (seedLengths[k]) {
        stillWork = true;
      } else if (startPos != (IdxType)-1) {
        IdxType foundAt = (IdxType)-1;
        kernel_refillSeed <<<dim3(1), dim3(32)>>> (d_foundAt, d_seedLists, d_seedClusterIds, d_seedLengths, k, d_pointStates, d_clusters, n, startPos);
        CUDA_CHECK(cudaGetLastError())
        CUDA_CHECK(cudaMemcpy(&foundAt, d_foundAt, sizeof(IdxType), cudaMemcpyDeviceToHost))
        //std::cerr << "Refilled " << k << " " << foundAt << "\n";
        startPos = foundAt + (foundAt != (IdxType)-1);
        stillWork = stillWork || foundAt != (IdxType)-1;
      }
    }

    if (!stillWork) break;

    kernel_clusterExpansion <<<dim3(nBlocks), dim3(nThreadsPerBlock), ( (coreThreshold * sizeof(IdxType) + 127) / 128 * 128 + 2 * sizeof(IdxType) ) >>> (
      d_clusters, d_pointStates, d_collisionMatrix, xs, ys, n, d_seedLists, d_seedClusterIds, d_seedLengths, coreThreshold, rsq
    );
    CUDA_CHECK(cudaGetLastError())
  }
}

