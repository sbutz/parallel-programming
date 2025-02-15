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

static __device__ IdxType unionizeClusters(
  IdxType * clusters,
  IdxType currentClusterId,
  IdxType point2Idx
) {
  IdxType grandchild, child, parent, top2, top1;

  child = currentClusterId;
  parent = clusters[child - 1];
  if (child != parent) {
    child = parent;
    parent = clusters[child - 1];
    if (child != parent) {
      for (;;) {
        grandchild = child;
        child = parent;
        parent = clusters[child - 1];
        if (child == parent) break;
        (void)atomicCAS(&clusters[grandchild - 1], child, parent);
      }
    }
  }
  top1 = child;

  parent = point2Idx + 1;
  for (;;) {
    child = parent;
    parent = clusters[child - 1];
    if (child != parent) {
      for (;;) {
        grandchild = child;
        child = parent;
        parent = clusters[child - 1];
        if (child == parent) break;
        (void)atomicCAS(&clusters[grandchild - 1], child, parent);
      }
    }
    top2 = child;

    if (top1 == top2) break;
    if (top1 > top2) { IdxType tmp = top2; top2 = top1; top1 = tmp; }

    IdxType old = atomicCAS(&clusters[top2 - 1], top2, top1);
    if (old == top2) break;
    parent = top2;
  }
  return top1;
}

static __device__ __forceinline__ unsigned int laneId() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

struct LargeStridePolicy {
  // returns:
  //   1 if append was possible and item was not last element
  //   -1 if append was possible and item was last element
  //   0 if append was not possible
  static __device__ auto tryAppendToNeighborBuffer(
    IdxType * s_neighborBuffer,
    IdxType * s_neighborCount,
    IdxType maxLength,
    IdxType pointIdx
  ) {
    int lane = laneId(); // or threadIdx.x & 0x1f
    unsigned int neighborMask = __ballot_sync(__activemask(), 1);
    int leader = __ffs(neighborMask) - 1;
    int nNeighbors = __popc(neighborMask);
    int oldNeighborCount;
    if (laneId() == leader) oldNeighborCount = atomicAdd(s_neighborCount, nNeighbors);
    oldNeighborCount = __shfl_sync(neighborMask, oldNeighborCount, leader);
    int h = oldNeighborCount + __popc(neighborMask & ((1u << lane) - 1));

    struct Result {
      bool wasAppended;
      bool maxLengthReached;
    };
    bool shouldAppend = h < maxLength;
    if (shouldAppend) s_neighborBuffer[h] = pointIdx;
    return Result { shouldAppend, oldNeighborCount + nNeighbors >= maxLength };
  }
};

/*
      } else {
        unsigned int strideMask = ((1u << stride) - 1u) << (lane & ~(stride - 1));
        unsigned int neighborMask = __ballot_sync(__activemask(), isNeighbor);
        int leaderLane = __ffs(neighborMask & strideMask) - 1;
        int nNeighbors = __popc(neighborMask & strideMask);
        int oldNeighborCount = *s_neighborCount;
        __syncwarp();
        if (lane == leaderLane) *s_neighborCount += nNeighbors;
        if (oldNeighborCount < coreThreshold && oldNeighborCount + nNeighbors >= coreThreshold && lane == leaderLane) {
          clusters[currentPointIdx] = currentClusterId;
          __threadfence();
          pointStates[currentPointIdx] = stateCore;
        }
        int h = oldNeighborCount + __popc(neighborMask & strideMask & ((1u << lane) - 1));
        if (h >= coreThreshold) {
          markAsCandidate(pointIdx);
        } else {
          s_neighborBuffer[h] = pointIdx;
        }
      }

*/

// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * clusters, unsigned int * pointStates,
  float const * xs, float const * ys, IdxType n,
  IdxType beginCurrentlyProcessedIdx,
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  unsigned int stride = blockDim.x;
  unsigned int nThreadGroupsPerBlock = blockDim.y;
  unsigned int nBlocks = gridDim.y;
  unsigned int nThreadGroupsTotal = nBlocks * nThreadGroupsPerBlock;
  unsigned int threadGroupIdx = blockDim.y * blockIdx.y + threadIdx.y;

  IdxType endCurrentlyProcessedIdx = nThreadGroupsTotal < n - beginCurrentlyProcessedIdx ? beginCurrentlyProcessedIdx + nThreadGroupsTotal : n;

  // Shared memory:
  // s_collisions: nThreadGroupsTotal bools
  //   size in bytes: (nThreadGroupsTotal + 3) / 4 * 4
  // s_neighborBuffer: IdxType[coreThreshold]
  //   size in bytes: coreThreshold * 4       TODO: Why not (coreThreshold - 1) * 4?
  // s_neighborCount: IdxType
  //   size in bytes: 4
  // s_groupClusterId: IdxType
  //   size in bytes: 4
  // -> Total size per thread group:
  //   [(nThreadGroupsTotal + 3) / 4 + coreThreshold + 2] * 4 bytes
  extern __shared__ unsigned char sMem [];
  unsigned int sMemBytesPerThreadGroup = 4 * ((nThreadGroupsTotal + 3) / 4 + coreThreshold + 2);

  static_assert(sizeof(IdxType) == 4, "");
  static_assert(alignof(IdxType) == 4, "");
  bool * s_collisions        = (bool *)    (sMem                              + sMemBytesPerThreadGroup * threadIdx.y);
  IdxType * s_neighborBuffer = (IdxType *) ((unsigned char *)s_collisions     + (nThreadGroupsTotal + 3) / 4 * 4);
  IdxType * s_neighborCount  = (IdxType *) ((unsigned char *)s_neighborBuffer + coreThreshold * 4);
  IdxType * s_groupClusterId = (IdxType *) ((unsigned char *)s_neighborCount  + 4);

  // clear s_collisions
  {
    IdxType strideStart = 0;
    if (nThreadGroupsTotal > stride) for (; strideStart < nThreadGroupsTotal - stride; strideStart += stride) {
      s_collisions[strideStart + threadIdx.x] = false;
    }
    if (threadIdx.x < nThreadGroupsTotal - strideStart) s_collisions[strideStart + threadIdx.x] = false;
  }

  // clear s_neighborCount
  if (threadIdx.x == 0) *s_neighborCount = 0;

  __syncthreads();

  if (threadGroupIdx >= n - beginCurrentlyProcessedIdx) return;

  IdxType currentPointIdx = beginCurrentlyProcessedIdx + threadGroupIdx;
  IdxType currentClusterId = clusters[currentPointIdx];
  if (currentClusterId == 0) currentClusterId = currentPointIdx + 1;

  if (threadIdx.x == 0) *s_groupClusterId = currentClusterId;

  __syncthreads();

  auto markAsCandidate = [&] (IdxType pointIdx) {
    if (pointIdx < beginCurrentlyProcessedIdx) {
      unsigned int state = pointStates[pointIdx];
      if (state == stateCore) {
        currentClusterId = unionizeClusters(clusters, currentClusterId, pointIdx);
        *s_groupClusterId = currentClusterId;
      } else {
        // TODO: simple write should be sufficient
        (void)atomicCAS(&clusters[pointIdx], 0, currentClusterId);
      }
    } else if (pointIdx < endCurrentlyProcessedIdx) {
      s_collisions[pointIdx - beginCurrentlyProcessedIdx] = true;
    } else {
      // TODO: simple write should be sufficient
      (void)atomicCAS(&clusters[pointIdx], 0, currentClusterId);
    }
  };  

  {
    float x = xs[currentPointIdx], y = ys[currentPointIdx];
    bool isDefinitelyCore = false;

    auto processObject = [&] (IdxType pointIdx) {
      int lane = threadIdx.x % 32;
      float dx = xs[pointIdx] - x;
      float dy = ys[pointIdx] - y;
      bool isNeighbor = dx * dx + dy * dy <= rsq;
      if (isNeighbor) {
        bool handleImmediately = isDefinitelyCore;
        if (!handleImmediately) {
          auto r = LargeStridePolicy::tryAppendToNeighborBuffer(s_neighborBuffer, s_neighborCount, coreThreshold, pointIdx);
          handleImmediately = !r.wasAppended;
          isDefinitelyCore = r.maxLengthReached;
        }
        if (handleImmediately) markAsCandidate(pointIdx);
      }
    };

    {
      IdxType strideIdx = 0;
      for (; strideIdx < (n - 1) / stride; ++strideIdx) { processObject(strideIdx * stride + threadIdx.x); }

      if (threadIdx.x < n - strideIdx * stride) processObject(strideIdx * stride + threadIdx.x);
    }

    __syncthreads();

    if (*s_neighborCount >= coreThreshold) {
      for (int i = threadIdx.x; i < coreThreshold; i += stride) {
        markAsCandidate(s_neighborBuffer[i]);
      }
      if (threadIdx.x == 0) {
        clusters[currentPointIdx] = currentClusterId;
        pointStates[currentPointIdx] = stateCore;
      }
    } else {
      if (threadIdx.x == 0) pointStates[currentPointIdx] = stateNoiseOrBorder;
    }
  }

  __syncthreads();

  // copy our collisions to global memory
  for (unsigned int i = threadIdx.x; i < nThreadGroupsTotal; i += stride) collisionHandlingData.d_collisionMatrix[nThreadGroupsTotal * threadGroupIdx + i] = s_collisions[i];

  __threadfence();

  if (threadIdx.x == 0) collisionHandlingData.d_doneWithIdx[threadGroupIdx] = currentPointIdx;

  __threadfence();

  if (threadIdx.x == 0) (void)atomicAdd(collisionHandlingData.d_mutex, 1);

  __threadfence();

  for (unsigned int i = threadIdx.x; i < nThreadGroupsTotal; i += stride) {
    if (i != blockIdx.x) {
      IdxType otherIdx = collisionHandlingData.d_doneWithIdx[i];
      if (otherIdx) {
        bool collision = s_collisions[i] || collisionHandlingData.d_collisionMatrix[i * nThreadGroupsTotal + threadGroupIdx];
        if (collision) {
          if (*s_neighborCount >= coreThreshold) {
            // we are core
            if (pointStates[otherIdx] == stateCore) {
              unionizeClusters(clusters, *s_groupClusterId, otherIdx);
            } else {
              clusters[otherIdx] = *s_groupClusterId;
            }
          } else {
            // we are noise
            if (pointStates[otherIdx] == stateCore) {
              clusters[currentPointIdx] = clusters[otherIdx];
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

void findClusters(
  unsigned int ** d_pointStates, IdxType ** d_clusters,
  float * xs, float * ys, IdxType n,
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 6;
  constexpr int nThreadGroupsPerBlock = 32;
  constexpr int nThreadsPerBlock = 1024;

  //int nThreadGroupsPerBlock = 46000 / 4 / ((nThreadGroupsTotal + 3) / 4 + coreThreshold + 2);
  //int dimy = (nThreadsPerBlock / nThreadGroupsPerBlock + 31) / 32 * 32;
  //nThreadGroupsPerBlock = nThreadsPerBlock / dimy;
  int nThreadGroupsTotal = nBlocks * nThreadGroupsPerBlock;

  unsigned int sharedBytesPerBlock = nThreadGroupsPerBlock * ((nThreadGroupsTotal + 3) / 4 + coreThreshold + 2) * 4;
  std::cerr << sharedBytesPerBlock << "\n";

  allocateDeviceMemory(d_pointStates, d_clusters, &collisionHandlingData, nThreadGroupsTotal, n);

  IdxType startPos = 0;
  for (;;) {
    CUDA_CHECK(cudaMemset((void *)collisionHandlingData.d_doneWithIdx, 0, nThreadGroupsTotal * sizeof(IdxType)))
    kernel_clusterExpansion <<<dim3(1, nBlocks), dim3(nThreadsPerBlock / nThreadGroupsPerBlock, nThreadGroupsPerBlock), sharedBytesPerBlock >>> (
      *d_clusters, *d_pointStates, xs, ys, n, startPos, collisionHandlingData, coreThreshold, rsq
    );
    CUDA_CHECK(cudaGetLastError())
    CUDA_CHECK(cudaDeviceSynchronize())

    if (n - startPos <= nThreadGroupsTotal) break;
    startPos += nThreadGroupsTotal;
  }
}

void unionizeCpu(std::vector<IdxType> & clusters) {
  std::vector<IdxType> stack;
  for (IdxType i = 0; i < clusters.size(); ++i) {
    IdxType child = i + 1;
    IdxType parent = clusters[child - 1];
    if (parent == 0 || parent == child) continue; // noise
    do {
      stack.push_back(child);
      child = parent;
      parent = clusters[child - 1];
    } while (parent != child);
    IdxType top = child;
    while (stack.size() > 0) {
      IdxType current = stack.back(); stack.pop_back();
      clusters[current - 1] = top;
    }
  }
}

static __global__ void kernel_unionize(IdxType * clusters, IdxType n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  IdxType strideBegin = 0;

  auto doWork = [&] {
    IdxType idx = strideBegin + tid;
    IdxType cl = idx + 1;
    IdxType p = clusters[idx];
    if (p != 0 && p != cl) {
      for (;;) {
        cl = p;
        p = clusters[cl - 1];
        if (p == cl) break;
      }
      IdxType top = cl;
      cl = idx + 1;
      for (;;) {
        //p = atomicExch(&clusters[cl - 1], top);
        p = clusters[cl - 1];
        if (p == top) break;
        clusters[cl - 1] = top;
      }
    }
  };

  if (n > stride) for (; strideBegin < n - stride; strideBegin += stride) doWork();
  if (tid < n - strideBegin) doWork();
}

void unionizeGpu(IdxType * d_clusters, IdxType n) {
  constexpr unsigned int nBlocks = 6;
  constexpr unsigned int nThreadsPerBlock = 1024;

  kernel_unionize <<<dim3(nBlocks), dim3(nThreadsPerBlock)>>> (d_clusters, n);
  CUDA_CHECK(cudaGetLastError())
  CUDA_CHECK(cudaDeviceSynchronize())
}