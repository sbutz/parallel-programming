#include "cluster_expansion.h"
#include "types.h"
#include "cuda_helpers.h"

#include <iostream>
#include <cuda.h>
#include <vector>

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
  IdxType beginCurrentlyProcessedIdx;
  IdxType endCurrentlyProcessedIdx;
  IdxType * neighborBuffer;
  IdxType * neighborCount;
  bool * s_collisions;
};

static __device__ void unionizeClusters(
  ThreadData & td,
  IdxType point2Idx
) {
  //printf("%u %u %u %u\n", point1Idx, point2Idx, td.clusters[point1Idx], td.clusters[point2Idx]);

  IdxType grandchild, child, parent, top2, top1;

  child = td.currentClusterId;
  parent = td.clusters[child - 1];
  if (child != parent) {
    child = parent;
    parent = td.clusters[child - 1];
    if (child != parent) {
      for (;;) {
        grandchild = child;
        child = parent;
        parent = td.clusters[child - 1];
        if (child == parent) break;
        (void)atomicCAS(&td.clusters[grandchild - 1], child, parent);
      }
    }
  }
  top1 = child;

  parent = point2Idx + 1;
  for (;;) {
    child = parent;
    parent = td.clusters[child - 1];
    if (child != parent) {
      for (;;) {
        grandchild = child;
        child = parent;
        parent = td.clusters[child - 1];
        if (child == parent) break;
        (void)atomicCAS(&td.clusters[grandchild - 1], child, parent);
      }
    }
    top2 = child;

    if (top1 == top2) break;
    if (top1 > top2) { IdxType tmp = top2; top2 = top1; top1 = tmp; }

    IdxType old = atomicCAS(&td.clusters[top2 - 1], top2, top1);
    if (old == top2) break;
    parent = top2;
  }
  td.currentClusterId = top1;
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
  ThreadData & td,
  IdxType pointIdx
) {
  if (pointIdx < td.beginCurrentlyProcessedIdx) {
    unsigned int state = td.pointStates[pointIdx];
    if (state & stateStateBitsMask == stateCore) {
      unionizeClusters(td, pointIdx);
    } else {
      // TODO: simple write should be sufficient
      (void)atomicCAS(&td.clusters[pointIdx], 0, td.currentClusterId);
    }
  } else if (pointIdx < td.endCurrentlyProcessedIdx) {
    td.s_collisions[pointIdx - td.beginCurrentlyProcessedIdx] = true;
  } else {
      // TODO: simple write should be sufficient
      (void)atomicCAS(&td.clusters[pointIdx], 0, td.currentClusterId);
  }
}

static __device__ void processObject(
  ThreadData & td,
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
      markAsCandidate(td, pointIdx);
    } else {
      td.neighborBuffer[h] = pointIdx;
    }
  }
}

// make VS Code happy
__device__ void __nanosleep(unsigned int);

static __device__ __forceinline__ void lockMutex(unsigned int * d_mutex) {
  unsigned int old;
  int ns = 1;
  do {
    old = atomicCAS(d_mutex, 0, 1);
    
    __nanosleep(ns);
    if (ns < 256) ns *= 2;
  } while (!old);
}

static __device__ __forceinline__ void unlockMutex(unsigned int * d_mutex) {
  *d_mutex = 0;
}



// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * cluster, unsigned int * pointState,
  float const * xs, float const * ys, IdxType n,
  IdxType beginCurrentlyProcessedIdx,
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  // shared memory:
  //   neighborBuffer: coreThreshold IdxType values in blocks of 128 bytes
  //   neighborCount:  1 IdxType (aligned to a 128 byte block)
  //   s_collisions:   nBlock bools, aligned to a 128 byte block
  //   s_doneWithIdx:  nBlock IdxType values, aligned to a 128 byte block
  extern __shared__ unsigned char sMem [];

  unsigned int stride = blockDim.x;
  unsigned int nBlocks = gridDim.x;

  ThreadData td;
  td.threadGroupIdx = blockIdx.x;

  td.clusters = cluster;
  td.pointStates = pointState;
  td.beginCurrentlyProcessedIdx = beginCurrentlyProcessedIdx;
  td.endCurrentlyProcessedIdx = nBlocks < n - beginCurrentlyProcessedIdx ? beginCurrentlyProcessedIdx + nBlocks : n;

  td.neighborBuffer = (IdxType *)sMem; // Length: coreThreshold elements
  td.neighborCount = (IdxType *) (sMem + (coreThreshold * sizeof(IdxType) + 127) / 128 * 128);
  td.s_collisions = (bool *) ((char *)td.neighborCount + 128);
  IdxType * s_doneWithIdx = (IdxType *) ((char *)td.s_collisions + (nBlocks + 127) / 128 * 128);
  static_assert(128 % alignof(IdxType) == 0, "");
  static_assert(alignof(IdxType) == alignof(unsigned int) && sizeof(IdxType) == sizeof(unsigned int), "");

  if (threadIdx.x == 0) { *td.neighborCount = 0; }
  for (unsigned int i = threadIdx.x; i < nBlocks; i += stride) td.s_collisions[i] = false;

  __syncthreads();

  if (blockIdx.x < n - beginCurrentlyProcessedIdx) {
    
    td.pointBeingProcessedIdx = beginCurrentlyProcessedIdx + blockIdx.x;
    td.currentClusterId = td.clusters[td.pointBeingProcessedIdx];
    if (td.currentClusterId == 0) {
      td.currentClusterId = td.pointBeingProcessedIdx + 1;
    }

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
      for (int i = threadIdx.x; i < coreThreshold; i += stride) {
        markAsCandidate(td, td.neighborBuffer[i]);
      }
    } else {
      if (threadIdx.x == 0) td.pointStates[td.pointBeingProcessedIdx] = stateNoiseOrBorder;
    }
  }

  __syncthreads();

  // copy our collisions to global memory
  for (unsigned int i = threadIdx.x; i < nBlocks; i += stride) collisionHandlingData.d_collisionMatrix[nBlocks * td.threadGroupIdx + i] = td.s_collisions[i];
  //if (threadIdx.x == 0) { lockMutex(collisionHandlingData.d_mutex); }

  __threadfence();

  if (threadIdx.x == 0) collisionHandlingData.d_doneWithIdx[td.threadGroupIdx] = td.pointBeingProcessedIdx;

  __threadfence();

  if (threadIdx.x == 0) (void)atomicAdd(collisionHandlingData.d_mutex, 1);
  //if (threadIdx.x == 0) unlockMutex(collisionHandlingData.d_mutex);

  __threadfence();

  for (unsigned int i = threadIdx.x; i < nBlocks; i += stride) {
    if (i != td.threadGroupIdx) {
      IdxType otherIdx = collisionHandlingData.d_doneWithIdx[i];
      if (otherIdx) {
        bool collision = td.s_collisions[i] || collisionHandlingData.d_collisionMatrix[i * nBlocks + td.threadGroupIdx];
        if (collision) {
          //printf("%u\n", otherIdx);
          if (*td.neighborCount >= coreThreshold) {
            // we are core
            if (td.pointStates[otherIdx] == stateCore) {
              // mark conflict in union-find datastructure
              unionizeClusters(td, otherIdx);
            } else {
              td.clusters[otherIdx] = td.currentClusterId;
            }
          } else {
            // we are noise
            if (td.pointStates[otherIdx] == stateCore) {
              td.clusters[td.pointBeingProcessedIdx] = td.clusters[otherIdx];
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
  // TODO: Change later
  CUDA_CHECK(cudaMemset(*d_pointStates, 0, n * sizeof(unsigned int)))
  CUDA_CHECK(cudaMalloc(d_clusters, n * sizeof(IdxType)))
  // TODO: Change later
  CUDA_CHECK(cudaMemset(*d_clusters, 0, n * sizeof(IdxType)))

  auto chdSizes = CollisionHandlingData::calculateSizes(nBlocks);
  unsigned int chdTotalSize = chdSizes.szMutex + chdSizes.szDoneWithIdx + chdSizes.szCollisionMatrix;
  char * d_memCollisionData;
  CUDA_CHECK(cudaMalloc(&d_memCollisionData, chdTotalSize))
  collisionHandlingData->d_mutex = (unsigned int *)d_memCollisionData;
  CUDA_CHECK(cudaMemset(collisionHandlingData->d_mutex, 0, sizeof(unsigned int)))
  collisionHandlingData->d_doneWithIdx = (IdxType *)(d_memCollisionData + chdSizes.szMutex);
  collisionHandlingData->d_collisionMatrix = (bool *)(d_memCollisionData + chdSizes.szMutex + chdSizes.szDoneWithIdx);
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
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 6;
  constexpr int nThreadsPerBlock = 512;

  IdxType startPos = 0;
  for (;;) {

    CUDA_CHECK(cudaMemset((void *)collisionHandlingData.d_doneWithIdx, 0, nBlocks * sizeof(IdxType)))
    kernel_clusterExpansion <<<dim3(nBlocks), dim3(nThreadsPerBlock), ( (coreThreshold * sizeof(IdxType) + 127) / 128 * 128 + 128 
    + (nBlocks * sizeof(bool) + 127) / 128 * 128 + (nBlocks * sizeof(IdxType) + 127) / 128 * 128) >>> (
      d_clusters, d_pointStates, xs, ys, n, startPos, collisionHandlingData, coreThreshold, rsq
    );
    CUDA_CHECK(cudaGetLastError())
    CUDA_CHECK(cudaDeviceSynchronize())

    if (n - startPos <= nBlocks) break;
    startPos += nBlocks;
  }
}

void unionizeCpu(std::vector<IdxType> & clusters) {
  std::vector<IdxType> stack;
  for (IdxType i = 0; i < clusters.size(); ++i) {
    IdxType child = i + 1;
    IdxType parent = clusters[child - 1];
    if (parent == 0) continue; // noise
    for (;;) {
      stack.push_back(child);
      child = parent;
      parent = clusters[child - 1];
      if (child == parent) break;
    }
    IdxType top = stack.back(); stack.pop_back();
    while (stack.size() > 0) {
      IdxType current = stack.back(); stack.pop_back();
      clusters[current - 1] = top;
    }
  }
}