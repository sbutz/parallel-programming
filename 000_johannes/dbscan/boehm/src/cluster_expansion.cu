#include "cluster_expansion.h"
#include "types.h"
#include "cuda_helpers.h"

#include <iostream>
#include <cuda.h>

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
  unsigned int threadGroupIdx;
  IdxType pointBeingProcessedIdx;
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
  unsigned int * collisions,
  IdxType pointIdx
) {
  unsigned int state = td.pointStates[pointIdx];
  unsigned int potentiallyFreeMask = __ballot_sync(__activemask(), state == stateFree);
  if (state == stateFree) {
    int lane = threadIdx.x % 32;
    int leader = __ffs(potentiallyFreeMask) - 1;
    int nPotentiallyFree = __popc(potentiallyFreeMask);
    IdxType oldReserved;
    if (lane == leader) oldReserved = atomicAdd(td.seedReserved, nPotentiallyFree);
    oldReserved = __shfl_sync(potentiallyFreeMask, oldReserved, leader);
    int potentialThreadPosition = __popc(potentiallyFreeMask & ((1u << (lane + 1)) - 1));
    if (oldReserved < maxSeedLength && potentialThreadPosition <= maxSeedLength - oldReserved) {
      state = atomicCAS(&td.pointStates[pointIdx], stateFree, stateReserved);
      unsigned int actuallyFreeMask = __ballot_sync(__activemask(), state == stateFree);
      int nActuallyFree = __popc(actuallyFreeMask);
      int actualLeader = __ffs(actuallyFreeMask) - 1;
      if (state == stateFree) {
        int h;
        if (lane == actualLeader) h = atomicAdd(td.seedLength, __popc(actuallyFreeMask));
        h = __shfl_sync(actuallyFreeMask, h, actualLeader);
        int offset = __popc(actuallyFreeMask & ((1u << lane) - 1));
        td.seedList[h + offset] = pointIdx; td.seedClusterIds[h + offset] = td.currentClusterId;
      }
      if (lane == leader) atomicSub(td.seedReserved, nPotentiallyFree - nActuallyFree);
    } else {
      if (lane == leader) atomicSub(td.seedReserved, nPotentiallyFree);
    }
  }
  switch (state & stateStateBitsMask) {
    case stateCore: {
      // TODO: handle collision!
    } break;
    case stateNoiseOrBorder: {
      td.clusters[pointIdx] = td.currentClusterId;
    } break;
    case stateUnderInspection: {
      *collisions |= (1u << (state & stateThreadGroupIdxMask));
      //printf("inspection\n");
      //(void)atomicCAS(&td.clusters[pointIdx], 0, td.currentClusterId);
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
  unsigned int * collisions,
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
      __threadfence();
      td.pointStates[td.pointBeingProcessedIdx] = stateCore;
    }
    int h = oldNeighborCount + __popc(neighborMask & ((1u << lane) - 1));
    if (h >= coreThreshold) {
      markAsCandidate(td, collisions, pointIdx);
    } else {
      td.neighborBuffer[h] = pointIdx;
    }
  }
}


// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * cluster, unsigned int * pointState,
  float const * xs, float const * ys, IdxType n,
  IdxType * seedLists, unsigned int * seedClusterIds, IdxType * seedLengths,
  unsigned int * syncCounter, unsigned int * collisionMatrix, IdxType * processedIdxs,
  IdxType coreThreshold, float rsq
) {
  extern __shared__ unsigned char sMem []; // coreThreshold IdxType values in blocks of 128 bytes + 2 IdxType value
  unsigned int collisions = 0;

  unsigned int stride = blockDim.x;

  ThreadData td;
  td.threadGroupIdx = blockIdx.x;

  td.clusters = cluster;
  td.pointStates = pointState;

  td.neighborBuffer = (IdxType *)sMem; // Length: coreThreshold elements
  td.neighborCount = (IdxType *) (sMem + (coreThreshold * sizeof(IdxType) + 127) / 128 * 128);
  td.seedReserved = td.neighborCount + 1;
  unsigned int * s_collisions = (unsigned int *)td.seedReserved + 1;
  static_assert(128 % alignof(IdxType) == 0, "");
  static_assert(alignof(IdxType) == alignof(unsigned int) && sizeof(IdxType) == sizeof(unsigned int), "");

  if (threadIdx.x == 0) { *td.neighborCount = 0; *s_collisions = 0; }

  td.seedLength = &seedLengths[td.threadGroupIdx];
  td.seedList = &seedLists[maxSeedLength * td.threadGroupIdx];
  td.seedClusterIds = &seedClusterIds[maxSeedLength * td.threadGroupIdx];

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
    if (threadIdx.x == 0) td.pointStates[td.pointBeingProcessedIdx] = stateUnderInspection | td.threadGroupIdx;

    __syncthreads();

    float x = xs[td.pointBeingProcessedIdx], y = ys[td.pointBeingProcessedIdx];
    {
      IdxType strideIdx = 0;
      for (; strideIdx < (n - 1) / stride; ++strideIdx) {
        processObject(
          td, &collisions,
          x, y,
          xs, ys, strideIdx * stride + threadIdx.x,
          coreThreshold, rsq
        );
      }
      if (threadIdx.x < n - strideIdx * stride) {
        processObject(
          td, &collisions,
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
        markAsCandidate(td, &collisions, td.neighborBuffer[i]);
      }
    } else {
      if (threadIdx.x == 0) td.pointStates[td.pointBeingProcessedIdx] = stateNoiseOrBorder;
    }
  }

  int lane = threadIdx.x % 32;
  #pragma unroll
  for (int i = 1; i < 32; i *= 2) collisions |= __shfl_down_sync(0xffffffff, collisions, i);
  if (lane == 0) (void)atomicOr(s_collisions, collisions);

  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned int ourCollisions = *s_collisions;

    collisionMatrix[td.threadGroupIdx] = ourCollisions;
    processedIdxs[td.threadGroupIdx] = td.pointBeingProcessedIdx;
    __threadfence();
    unsigned int finishedBlocks = atomicOr(syncCounter, 1u << td.threadGroupIdx) & ~(1u << td.threadGroupIdx);

    for (int i = 0; i < gridDim.x; ++i) {
      if ((1u << i) & finishedBlocks) {
        if ((collisionMatrix[i] && (1u << td.threadGroupIdx)) || (ourCollisions & (1u << i))) {
          // handle collision
          if (*td.neighborCount >= coreThreshold) {
            IdxType otherIdx = processedIdxs[i];
            // we are core
            if (td.pointStates[i] == stateCore) {
              // mark conflict in union-find datastructure
            } else {
              td.clusters[i] = td.currentClusterId;
            }
          } else {
            // we are noise
            if (td.pointStates[i] == stateCore) {
              td.clusters[td.pointBeingProcessedIdx] = td.clusters[i];
            }
          }
        }
      }
    }
  }
}

void allocateDeviceMemory(
  unsigned int ** d_pointStates, IdxType ** d_clusters,
  IdxType ** d_seedLists, IdxType ** d_seedClusterIds, IdxType ** d_seedLengths,
  unsigned int ** d_syncCounter, unsigned int ** d_collisionMatrix, IdxType ** d_processedIdxs,
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
  CUDA_CHECK(cudaMalloc(d_syncCounter, sizeof(unsigned int)))
  CUDA_CHECK(cudaMalloc(d_collisionMatrix, nBlocks * sizeof(unsigned int)))
  CUDA_CHECK(cudaMalloc(d_processedIdxs, nBlocks * sizeof(IdxType)))
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
  unsigned int * d_pointStates, IdxType * d_clusters,
  float * xs, float * ys, IdxType n,
  IdxType * d_seedLists, IdxType * d_seedClusterIds, IdxType * d_seedLengths,
  unsigned int * d_syncCounter, unsigned int * d_collisionMatrix, IdxType * d_processedIdxs,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 6;
  constexpr int nThreadsPerBlock = 128;

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
        CUDA_CHECK(cudaDeviceSynchronize())
        // std::cerr << "Refilled " << k << " " << foundAt << "\n";
        startPos = foundAt + (foundAt != (IdxType)-1);
        stillWork = stillWork || foundAt != (IdxType)-1;
      }
    }

    if (!stillWork) break;

    CUDA_CHECK(cudaMemset(d_syncCounter, 0, sizeof(unsigned int)))
    kernel_clusterExpansion <<<dim3(nBlocks), dim3(nThreadsPerBlock), ( (coreThreshold * sizeof(IdxType) + 127) / 128 * 128 + 2 * sizeof(IdxType) ) >>> (
      d_clusters, d_pointStates, xs, ys, n, d_seedLists, d_seedClusterIds, d_seedLengths, d_syncCounter, d_collisionMatrix, d_processedIdxs, coreThreshold, rsq
    );
    CUDA_CHECK(cudaGetLastError())
    CUDA_CHECK(cudaDeviceSynchronize())
  }
}

