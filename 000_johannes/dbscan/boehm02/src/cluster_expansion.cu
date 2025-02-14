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

// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * clusters, unsigned int * pointStates,
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

  IdxType endCurrentlyProcessedIdx = nBlocks < n - beginCurrentlyProcessedIdx ? beginCurrentlyProcessedIdx + nBlocks : n;

  IdxType * neighborBuffer = (IdxType *)sMem; // Length: coreThreshold elements
  IdxType * neighborCount = (IdxType *) (sMem + (coreThreshold * sizeof(IdxType) + 127) / 128 * 128);
  IdxType * s_blockClusterId = neighborCount + 1;
  bool * s_collisions = (bool *) ((char *)neighborCount + 128);
  static_assert(128 % alignof(IdxType) == 0, "");
  static_assert(alignof(IdxType) == alignof(unsigned int) && sizeof(IdxType) == sizeof(unsigned int), "");

  if (threadIdx.x == 0) { *neighborCount = 0; }
  for (unsigned int i = threadIdx.x; i < nBlocks; i += blockDim.x) s_collisions[i] = false;

  __syncthreads();

  if (blockIdx.x >= n - beginCurrentlyProcessedIdx) return;
    
  IdxType currentPointIdx = beginCurrentlyProcessedIdx + blockIdx.x;
  IdxType currentClusterId = clusters[currentPointIdx];
  if (currentClusterId == 0) {
    currentClusterId = currentPointIdx + 1;
  }
  if (threadIdx.x == 0) *s_blockClusterId = currentClusterId;

  auto markAsCandidate = [&] (IdxType pointIdx) {
    if (pointIdx < beginCurrentlyProcessedIdx) {
      unsigned int state = pointStates[pointIdx];
      if (state == stateCore) {
        currentClusterId = unionizeClusters(clusters, currentClusterId, pointIdx);
        *s_blockClusterId = currentClusterId;
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

  __syncthreads();

  {
    float x = xs[currentPointIdx], y = ys[currentPointIdx];

    auto processObject = [&] (IdxType pointIdx) {
      int lane = threadIdx.x % 32;
      float dx = xs[pointIdx] - x;
      float dy = ys[pointIdx] - y;
      bool isNeighbor = dx * dx + dy * dy <= rsq;
      unsigned int neighborMask = __ballot_sync(__activemask(), isNeighbor);
      if (isNeighbor) {
        int leaderLane = __ffs(neighborMask) - 1;
        int nNeighbors = __popc(neighborMask);
        int oldNeighborCount = 0;
        if (lane == leaderLane) oldNeighborCount = atomicAdd(neighborCount, nNeighbors);
        oldNeighborCount = __shfl_sync(neighborMask, oldNeighborCount, leaderLane);
        if (oldNeighborCount < coreThreshold && oldNeighborCount + nNeighbors >= coreThreshold && lane == leaderLane) {
          clusters[currentPointIdx] = currentClusterId;
          __threadfence();
          pointStates[currentPointIdx] = stateCore;
        }
        int h = oldNeighborCount + __popc(neighborMask & ((1u << lane) - 1));
        if (h >= coreThreshold) {
          markAsCandidate(pointIdx);
        } else {
          neighborBuffer[h] = pointIdx;
        }
      }  
    };

    IdxType strideIdx = 0;
    for (; strideIdx < (n - 1) / stride; ++strideIdx) processObject(strideIdx * stride + threadIdx.x);
    if (threadIdx.x < n - strideIdx * stride) processObject(strideIdx * stride + threadIdx.x);

    __syncthreads();

    if (*neighborCount >= coreThreshold) {
      for (int i = threadIdx.x; i < coreThreshold; i += stride) {
        markAsCandidate(neighborBuffer[i]);
      }
    } else {
      if (threadIdx.x == 0) pointStates[currentPointIdx] = stateNoiseOrBorder;
    }
  }

  __syncthreads();

  // copy our collisions to global memory
  for (unsigned int i = threadIdx.x; i < nBlocks; i += blockDim.x) collisionHandlingData.d_collisionMatrix[nBlocks * blockIdx.x + i] = s_collisions[i];

  __threadfence();

  if (threadIdx.x == 0) collisionHandlingData.d_doneWithIdx[blockIdx.x] = currentPointIdx;

  __threadfence();

  if (threadIdx.x == 0) (void)atomicAdd(collisionHandlingData.d_mutex, 1);

  __threadfence();

  for (unsigned int i = threadIdx.x; i < nBlocks; i += stride) {
    if (i != blockIdx.x) {
      IdxType otherIdx = collisionHandlingData.d_doneWithIdx[i];
      if (otherIdx) {
        bool collision = s_collisions[i] || collisionHandlingData.d_collisionMatrix[i * nBlocks + blockIdx.x];
        if (collision) {
          if (*neighborCount >= coreThreshold) {
            // we are core
            if (pointStates[otherIdx] == stateCore) {
              unionizeClusters(clusters, *s_blockClusterId, otherIdx);
            } else {
              clusters[otherIdx] = *s_blockClusterId;
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
  unsigned int * d_pointStates, IdxType * d_clusters,
  float * xs, float * ys, IdxType n,
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 32;
  constexpr int nThreadsPerBlock = 256;

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