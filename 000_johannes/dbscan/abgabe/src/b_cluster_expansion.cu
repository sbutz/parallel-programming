#include "b_cluster_expansion.h"
#include "b_types.h"
#include "cuda_helpers.h"

#include <iostream>
#include <cuda.h>

constexpr IdxType maxSeedLength = 1024; // TODO: Could be larger!?

static __device__ __forceinline__ IdxType appendToSeedList(
  IdxType * pointStates,
  IdxType * seedLength, IdxType * seedList, IdxType * seedClusterIds, IdxType * s_seedReserved,
  unsigned int otherState, IdxType pointIdx,
  IdxType ourClusterId,
  unsigned int potentiallyFreeMask
) {
  int lane = threadIdx.x % 32;
  int leader = __ffs(potentiallyFreeMask) - 1;
  int nPotentiallyFree = __popc(potentiallyFreeMask);
  IdxType oldReserved;
  if (lane == leader) oldReserved = atomicAdd(s_seedReserved, nPotentiallyFree);
  oldReserved = __shfl_sync(potentiallyFreeMask, oldReserved, leader);
  int potentialThreadPosition = __popc(potentiallyFreeMask & ((1u << (lane + 1)) - 1));
  if (oldReserved < maxSeedLength && potentialThreadPosition <= maxSeedLength - oldReserved) {
    otherState = atomicCAS(&pointStates[pointIdx], stateFree, stateReserved);
    unsigned int actuallyFreeMask = __ballot_sync(__activemask(), otherState == stateFree);
    int nActuallyFree = __popc(actuallyFreeMask);
    int actualLeader = __ffs(actuallyFreeMask) - 1;
    if (otherState == stateFree) {
      int h;
      if (lane == actualLeader) h = atomicAdd(seedLength, __popc(actuallyFreeMask));
      h = __shfl_sync(actuallyFreeMask, h, actualLeader);
      int offset = __popc(actuallyFreeMask & ((1u << lane) - 1));
      seedList[h + offset] = pointIdx; seedClusterIds[h + offset] = ourClusterId;
    }
    if (lane == leader) atomicSub(s_seedReserved, nPotentiallyFree - nActuallyFree);
  } else {
    if (lane == leader) atomicSub(s_seedReserved, nPotentiallyFree);
  }
  return otherState;
}

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
  IdxType * clusters, unsigned int * pointStates, 
  IdxType * seedList, IdxType * seedClusterIds, IdxType * seedLength, IdxType * s_seedReserved,
  bool * s_collisions,
  IdxType ourClusterId,
  IdxType otherPointIdx
) {
  unsigned int otherState = pointStates[otherPointIdx];
  unsigned int potentiallyFreeMask = __ballot_sync(__activemask(), otherState == stateFree);
  if (otherState == stateFree) {
    otherState = appendToSeedList(
      pointStates,
      seedLength, seedList, seedClusterIds, s_seedReserved,
      otherState, otherPointIdx,
      ourClusterId,
      potentiallyFreeMask
    );
  }
  switch (otherState & stateStateBitsMask) {
    case stateCore: {
      // TODO: handle collision!
    } break;
    case stateNoiseOrBorder:
    case stateFree:
    case stateReserved: {
      clusters[otherPointIdx] = ourClusterId;
    } break;
    case stateUnderInspection: {
      s_collisions[otherState & stateThreadGroupIdxMask] = true;
    } break;
    default:
      ; // do nothing
  }
}

static __device__ void processObject(
  IdxType * clusters, unsigned int * pointStates,
  IdxType * seedList, IdxType * seedClusterIds, IdxType * seedLength, IdxType * s_seedReserved,
  bool * s_collisions, IdxType * s_neighborBuffer, IdxType * s_neighborCount,
  IdxType ourPointIdx, IdxType ourClusterId,
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
    if (lane == leaderLane) oldNeighborCount = atomicAdd(s_neighborCount, nNeighbors);
    oldNeighborCount = __shfl_sync(neighborMask, oldNeighborCount, leaderLane);
    if (oldNeighborCount < coreThreshold && oldNeighborCount + nNeighbors >= coreThreshold && lane == leaderLane) {
      clusters[ourPointIdx] = ourClusterId;
      __threadfence();
      pointStates[ourPointIdx] = stateCore;
    }
    int h = oldNeighborCount + __popc(neighborMask & ((1u << lane) - 1));
    if (h >= coreThreshold) {
      markAsCandidate(
        clusters, pointStates,
        seedList, seedClusterIds, seedLength, s_seedReserved,
        s_collisions, ourClusterId, pointIdx);
    } else {
      s_neighborBuffer[h] = pointIdx;
    }
  }
}

// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * clusters, unsigned int * pointStates,
  float const * xs, float const * ys, IdxType n,
  IdxType * seedLists, unsigned int * seedClusterIds, IdxType * seedLengths,
  unsigned int * syncCounter, CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  // shared memory:
  //   s_neighborBuffer: coreThreshold IdxType values in blocks of 128 bytes
  //   s_neighborCount:  1 IdxType (aligned to a 128 byte block)
  //   s_seedReserved:   1 IdxType
  //   s_collisions:   nBlock bools, aligned to a 128 byte block
  //   s_doneWithIdx:  nBlock IdxType values, aligned to a 128 byte block
  extern __shared__ unsigned char sMem [];

  unsigned int stride = blockDim.x;
  unsigned int nBlocks = gridDim.x;

  unsigned int threadGroupIdx = blockIdx.x;

  IdxType * s_neighborBuffer = (IdxType *)sMem; // Length: coreThreshold elements
  IdxType * s_neighborCount = (IdxType *) (sMem + (coreThreshold * sizeof(IdxType) + 127) / 128 * 128);
  IdxType * s_seedReserved = s_neighborCount + 1;
  bool * s_collisions = (bool *) ((char *)s_neighborCount + 128);
  static_assert(128 % alignof(IdxType) == 0, "");
  static_assert(alignof(IdxType) == alignof(unsigned int) && sizeof(IdxType) == sizeof(unsigned int), "");

  if (threadIdx.x == 0) { *s_neighborCount = 0; }
  for (unsigned int i = threadIdx.x; i < nBlocks; i += stride) s_collisions[i] = false;

  IdxType * ourSeedLength = &seedLengths[threadGroupIdx];
  IdxType * ourSeedList = &seedLists[maxSeedLength * threadGroupIdx];
  IdxType * ourSeedClusterIds = &seedClusterIds[maxSeedLength * threadGroupIdx];

  IdxType seedLengthTemp = *ourSeedLength;

  __syncthreads();

  if (seedLengthTemp > 0) {
    --seedLengthTemp;

    if (threadIdx.x == 0) { *ourSeedLength = seedLengthTemp; *s_seedReserved = seedLengthTemp; }
    IdxType ourPointIdx = ourSeedList[seedLengthTemp];
    IdxType ourClusterId = ourSeedClusterIds[seedLengthTemp];
    if (ourClusterId == 0) ourClusterId = ourPointIdx + 1;
    if (threadIdx.x == 0) pointStates[ourPointIdx] = stateUnderInspection | threadGroupIdx;

    __syncthreads();

    float x = xs[ourPointIdx], y = ys[ourPointIdx];
    {
      IdxType strideIdx = 0;
      for (; strideIdx < (n - 1) / stride; ++strideIdx) {
        processObject(
          clusters, pointStates, 
          ourSeedList, ourSeedClusterIds, ourSeedLength, s_seedReserved,
          s_collisions, s_neighborBuffer, s_neighborCount,
          ourPointIdx, ourClusterId,
          x, y,
          xs, ys, strideIdx * stride + threadIdx.x,
          coreThreshold, rsq
        );
      }
      if (threadIdx.x < n - strideIdx * stride) {
        processObject(
          clusters, pointStates,
          ourSeedList, ourSeedClusterIds, ourSeedLength, s_seedReserved,          
          s_collisions, s_neighborBuffer, s_neighborCount,
          ourPointIdx, ourClusterId,
          x, y,
          xs, ys, strideIdx * stride + threadIdx.x,
          coreThreshold, rsq
        );
      }
    }

    __syncthreads();

    IdxType cnt = *s_neighborCount;
    if (cnt >= coreThreshold) {
      for (int i = threadIdx.x; i < coreThreshold; i += stride) {
        markAsCandidate(
          clusters, pointStates,
          ourSeedList, ourSeedClusterIds, ourSeedLength, s_seedReserved,
          s_collisions, ourClusterId, s_neighborBuffer[i]);
      }
    } else {
      if (threadIdx.x == 0) pointStates[ourPointIdx] = stateNoiseOrBorder;
    }

    __syncthreads();

    // copy our collisions to global memory
    for (unsigned int i = threadIdx.x; i < nBlocks; i += stride) {
      collisionHandlingData.d_collisionMatrix[nBlocks * threadGroupIdx + i] = s_collisions[i];
    }

    __threadfence();

    if (threadIdx.x == 0) collisionHandlingData.d_doneWithIdx[threadGroupIdx] = ourPointIdx + 1;

    __threadfence();

    if (threadIdx.x == 0) (void)atomicAdd(collisionHandlingData.d_mutex, 1);

    __threadfence();

    for (unsigned int i = threadIdx.x; i < nBlocks; i += stride) {
      if (i != threadGroupIdx) {
        IdxType otherIdx = collisionHandlingData.d_doneWithIdx[i] - 1;
        if (otherIdx != (IdxType)-1) {
          bool collision = s_collisions[i] || collisionHandlingData.d_collisionMatrix[i * nBlocks + threadGroupIdx];
          if (collision) {
            if (cnt >= coreThreshold) {
              // we are core
              if (pointStates[otherIdx] == stateCore) {
                // mark conflict in union-find datastructure
              } else {
                clusters[otherIdx] = ourClusterId;
              }
            } else {
              // we are noise
              if (pointStates[otherIdx] == stateCore) {
                clusters[ourPointIdx] = clusters[otherIdx];
              }
            }
          }
        }
      }
    }
  }
}

void allocateDeviceMemory(
  unsigned int ** d_pointStates, IdxType ** d_clusters,
  CollisionHandlingData * collisionHandlingData,
  int nBlocks,
  IdxType n
) {
  CUDA_CHECK(cudaMalloc(d_pointStates, n * sizeof(unsigned int)))
  CUDA_CHECK(cudaMemset(*d_pointStates, 0, n * sizeof(unsigned int)))
  CUDA_CHECK(cudaMalloc(d_clusters, n * sizeof(IdxType)))
  CUDA_CHECK(cudaMemset(*d_clusters, 0, n * sizeof(IdxType)))


  auto chdSizes = CollisionHandlingData::calculateSizes(nBlocks);
  CUDA_CHECK(cudaMalloc(&collisionHandlingData->d_mutex, sizeof(unsigned int)))
  CUDA_CHECK(cudaMalloc(&collisionHandlingData->d_doneWithIdx, chdSizes.szDoneWithIdx))
  CUDA_CHECK(cudaMalloc(&collisionHandlingData->d_collisionMatrix, chdSizes.szCollisionMatrix))

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
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 6;
  constexpr int nThreadsPerBlock = 512;

  IdxType * d_seedLists;
  IdxType * d_seedClusterIds;
  IdxType * d_seedLengths;
  unsigned int * d_syncCounter;
  CUDA_CHECK(cudaMalloc(&d_seedLists, nBlocks * maxSeedLength * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(&d_seedClusterIds, nBlocks * maxSeedLength * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(&d_seedLengths, nBlocks * sizeof(IdxType)))
  CUDA_CHECK(cudaMemset(d_seedLengths, 0, nBlocks * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(&d_syncCounter, sizeof(unsigned int)))

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
        kernel_refillSeed <<<1, 32>>> (
          d_foundAt, d_seedLists, d_seedClusterIds, d_seedLengths, k, d_pointStates, d_clusters, n, startPos
        );
        CUDA_CHECK(cudaGetLastError())
        CUDA_CHECK(cudaMemcpy(&foundAt, d_foundAt, sizeof(IdxType), cudaMemcpyDeviceToHost))
        startPos = foundAt + (foundAt != (IdxType)-1);
        stillWork = stillWork || foundAt != (IdxType)-1;
      }
    }

    if (!stillWork) break;

    CUDA_CHECK(cudaMemset(d_syncCounter, 0, sizeof(unsigned int)))
    CUDA_CHECK(cudaMemset((void *)collisionHandlingData.d_doneWithIdx, 0, nBlocks * sizeof(IdxType)))
    kernel_clusterExpansion <<<
      nBlocks,
      nThreadsPerBlock,
      (
        (coreThreshold * sizeof(IdxType) + 127) / 128 * 128 + 128 
        + (nBlocks * sizeof(bool) + 127) / 128 * 128 + (nBlocks * sizeof(IdxType) + 127) / 128 * 128
      )
    >>> (
      d_clusters, d_pointStates,
      xs, ys, n,
      d_seedLists, d_seedClusterIds, d_seedLengths, d_syncCounter, collisionHandlingData,
      coreThreshold, rsq
    );
    CUDA_CHECK(cudaGetLastError())
    CUDA_CHECK(cudaDeviceSynchronize())
  }
}

