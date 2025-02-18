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

static __device__ IdxType unionizeClusters2(
  IdxType * clusters,
  IdxType cluster1,
  IdxType cluster2
) {
  IdxType grandchild, child, parentOffset, top2, top1;

  // the following seems to save some time
  if (cluster1 == cluster2) return cluster1;

  top1 = cluster1; // we initially assume cluster1 is the top node
  child = cluster2;
  for (;;) {
    parentOffset = clusters[child];
    if (parentOffset) {
      grandchild = child;
      child += parentOffset;
      parentOffset = clusters[child];
      while (parentOffset) {
        (void)atomicCAS(&clusters[grandchild], child - grandchild, child + parentOffset - grandchild);
        grandchild = child;
        child += parentOffset;
        parentOffset = clusters[child];
      }
    }
    top2 = child;

    if (top1 == top2) break; // necessary?
    if (top1 > top2) { IdxType tmp = top2; top2 = top1; top1 = tmp; }

    IdxType old = atomicCAS(&clusters[top2], 0, top1 - top2);
    if (!old) break;
    child = top2;
  }
  return top1;
}


static __device__ __forceinline__ IdxType runToTop(IdxType * clusters, IdxType clusterId) {
  IdxType child = clusterId;
  IdxType parentOffset = clusters[child];
  while (parentOffset) { child += parentOffset; parentOffset = clusters[child]; }
  return child;
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
    if (lane == leader) oldNeighborCount = atomicAdd(s_neighborCount, nNeighbors);
    oldNeighborCount = __shfl_sync(neighborMask, oldNeighborCount, leader);
    int h = oldNeighborCount + __popc(neighborMask & ((1u << lane) - 1));

    bool shouldAppend = h < maxLength;
    if (shouldAppend) s_neighborBuffer[h] = pointIdx;

    struct Result {
      bool wasAppended;
      bool maxLengthReached;
    }; 
    return Result { shouldAppend, oldNeighborCount + nNeighbors >= maxLength };
  }
};

static constexpr __host__ __device__ __forceinline__ IdxType dhi_max(IdxType a, IdxType b) {
  return a > b ? a : b;
}

static __global__ void kernel_clusterExpansion(
  IdxType * clusters, bool * coreMarkers,
  float const * xs, float const * ys, IdxType n,
  IdxType beginCurrentlyProcessedIdx,
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  unsigned int stride = blockDim.x; // blockDim.x must be a multiple of 32
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
  // -> simultaneously used for s_interWarpUnionize: IdxType[blockDim.x / 32]
  //   size in bytes: (blockDim.x + 31) / 32 * 4
  // => size in bytes: max(coreThreshold, (blockDim.x + 31) / 32) * 4
  // s_neighborCount: IdxType
  //   size in bytes: 4
  // -> Total size per thread group in bytes:
  //   [(nThreadGroupsTotal + 3) / 4 + max(coreThreshold, (blockDim.x + 31) / 32) + 1] * 4 bytes
  extern __shared__ char sMem [];
  unsigned int sMemBytesPerThreadGroup = 4 * (
    (nThreadGroupsTotal + 3) / 4 +
    dhi_max(coreThreshold, (blockDim.x + 31) / 32) +
    1
  );

  static_assert(
    sizeof(IdxType)  == 4 &&
    alignof(IdxType) == 4 &&
    sizeof(bool) == 1, ""
  );
  bool * s_collisions        = (bool *)    (sMem                     + sMemBytesPerThreadGroup * threadIdx.y);
  IdxType * s_neighborBuffer = (IdxType *) ((char *)s_collisions     + (nThreadGroupsTotal + 3) / 4 * 4);
  volatile IdxType * s_interWarpUnionize = s_neighborBuffer;
  IdxType * s_neighborCount  = (IdxType *) ((char *)s_neighborBuffer + dhi_max(coreThreshold, (blockDim.x + 31) / 32) * 4);

  // clear all shared memory
  {
    // zeroing one byte per thread is faster than one unsigned int per thread
    //   -- reason unclear, but may be related to memory bank conflicts
    IdxType strideStart = 0;
    IdxType myOffset = blockDim.x * threadIdx.y + threadIdx.x;
    IdxType total = sMemBytesPerThreadGroup * blockDim.y;
    while (total - strideStart > blockDim.x * blockDim.y) {
      sMem [strideStart + myOffset] = 0;
      strideStart += blockDim.x * blockDim.y;
    }
    if (myOffset < total - strideStart) sMem [strideStart + myOffset] = 0;
  }

  if (threadGroupIdx >= n - beginCurrentlyProcessedIdx) return;

  IdxType currentPointIdx = beginCurrentlyProcessedIdx + threadGroupIdx;
  IdxType currentClusterId = currentPointIdx + clusters[currentPointIdx];

  __syncthreads();

  auto markAsCandidate = [&] (IdxType pointIdx) {
    if (pointIdx < beginCurrentlyProcessedIdx) {
      if (coreMarkers[pointIdx]) {
        currentClusterId = unionizeClusters2(clusters, pointIdx, currentClusterId);
      } else {
        clusters[pointIdx] = currentClusterId - pointIdx;
      }
    } else if (pointIdx < endCurrentlyProcessedIdx) {
      s_collisions[pointIdx - beginCurrentlyProcessedIdx] = true;
    } else {
      clusters[pointIdx] = currentClusterId - pointIdx;
    }
  };

  {
    float x = xs[currentPointIdx], y = ys[currentPointIdx];
    bool isDefinitelyCore = false;

    auto processObject = [&] (IdxType pointIdx, unsigned int mask) {
      int lane = threadIdx.x % 32;
      float dx = xs[pointIdx] - x;
      float dy = ys[pointIdx] - y;
      unsigned int oldClusterId = currentClusterId;
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
      /*
      //unsigned int clusterIdChangedMask = __ballot_sync(mask, currentClusterId != oldClusterId);
      //if (clusterIdChangedMask) {
      //  int leader = __ffs(clusterIdChangedMask) - 1;
      //  currentClusterId = __shfl_sync(mask, currentClusterId, leader);
      //}
      
      unsigned int clusterIdChanged = __any_sync(mask, currentClusterId != oldClusterId);
      if (clusterIdChanged) {
        for (unsigned int i = 1; i != 32; i <<= 1) {
          unsigned int otherId = currentClusterId;
          otherId = __shfl_down_sync(mask, otherId, i);
          if (otherId < currentClusterId) currentClusterId = otherId;
        }
        currentClusterId = __shfl_sync(mask, currentClusterId, 0);
      }
      */
    };

    {
      IdxType strideIdx = 0;
      for (; strideIdx < (n - 1) / stride; ++strideIdx) { processObject(strideIdx * stride + threadIdx.x, 0xffffffff); }
      unsigned int remaining = n - strideIdx * stride;
      if (threadIdx.x < remaining) {
        unsigned int remainingFromWarpStart = remaining - (threadIdx.x & ~0x1fu);
        unsigned int mask = remainingFromWarpStart >= 32 ? 0xffffffff : (1u << remainingFromWarpStart) - 1u;
        processObject(strideIdx * stride + threadIdx.x, mask);
      }
    }
    
    __syncthreads();

    if (*s_neighborCount >= coreThreshold) {
      for (int i = threadIdx.x; i < coreThreshold; i += stride) {
        markAsCandidate(s_neighborBuffer[i]);
      }
    }

    //seems to make no difference
    //currentClusterId = runToTop(clusters, currentClusterId);
    for (unsigned int i = 1; i < 32; i <<= 1) {
      IdxType otherClusterId = __shfl_down_sync(0xffffffff, currentClusterId, i);
      if (!(laneId() & ((i << 1) - 1))) currentClusterId = unionizeClusters2(clusters, currentClusterId, otherClusterId);
    }
    currentClusterId = __shfl_sync(0xffffffff, currentClusterId, 0);

    __syncthreads();

    // unionize warps within every thread group
    unsigned int nValuesToUnionize = (blockDim.x + 31) / 32;
    if (nValuesToUnionize > 1) {
      int lane = laneId();
      int wid = threadIdx.x / 32;
      if (lane == 0) s_interWarpUnionize[threadIdx.x / 32] = currentClusterId;
      while ((wid + 1) * 32 <= nValuesToUnionize) {
        __syncthreads();
        IdxType myValue = threadIdx.x < nValuesToUnionize ? s_interWarpUnionize[threadIdx.x] : (IdxType)-1;
        unsigned int limit = nValuesToUnionize - wid * 32;
        if (limit > 32) limit = 32;
        for (unsigned int i = 1; i < limit; i <<= 1) {
          IdxType otherValue = __shfl_down_sync(0xffffffff, myValue, i);
          if (
            (otherValue + 1) &&
            !(laneId() & ((i << 1) - 1))
          ) myValue = unionizeClusters2(clusters, myValue, otherValue);
        }
        nValuesToUnionize = (nValuesToUnionize + 31) / 32;
        if (laneId() == 0) s_interWarpUnionize[threadIdx.x / 32] == myValue;
        if (nValuesToUnionize == 1) break;
      }
      __syncthreads();
      currentClusterId = s_interWarpUnionize[0];
    }

    __syncthreads();

    if (*s_neighborCount >= coreThreshold) {
      if (threadIdx.x == 0) {
        clusters[currentPointIdx] = currentClusterId - currentPointIdx;
        coreMarkers[currentPointIdx] = true;
      }
    } else {
      int cnt = *s_neighborCount;
      for (int i = threadIdx.x; i < cnt; i += stride) {
        IdxType neighbor = s_neighborBuffer[i];
        if (coreMarkers[neighbor]) {
          if (neighbor < beginCurrentlyProcessedIdx) {
            clusters[currentPointIdx] = neighbor + clusters[neighbor] - currentPointIdx;
          } else if (neighbor < endCurrentlyProcessedIdx) {
            s_collisions[neighbor - beginCurrentlyProcessedIdx] = true;
          }
          break;
        }
      }
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
            if (coreMarkers[otherIdx]) {
              currentClusterId = unionizeClusters2(clusters, currentClusterId, otherIdx);
            } else {
              clusters[otherIdx] = currentClusterId - otherIdx;
            }
          } else {
            // we are noise
            if (coreMarkers[otherIdx]) {
              clusters[currentPointIdx] = otherIdx + clusters[otherIdx] - currentPointIdx;
            }
          }
        }
      }
    }
  }
}

void allocateDeviceMemory(
  bool ** d_coreMarkers, IdxType ** d_clusters,
  CollisionHandlingData * collisionHandlingData,
  int nBlocks,
  IdxType n
) {
  CUDA_CHECK(cudaMalloc(d_coreMarkers, n * sizeof(bool)))
  // TODO: Change later
  CUDA_CHECK(cudaMemset(*d_coreMarkers, 0, n * sizeof(bool)))
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
  bool ** d_coreMarkers, IdxType ** d_clusters,
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

  unsigned int sharedBytesPerBlock = nThreadGroupsPerBlock * 4 * (
    (nThreadGroupsTotal + 3) / 4 +
    dhi_max(coreThreshold, (nThreadGroupsPerBlock + 31) / 32) +
    1
  );
  std::cerr << sharedBytesPerBlock << "\n";

  allocateDeviceMemory(d_coreMarkers, d_clusters, &collisionHandlingData, nThreadGroupsTotal, n);

  IdxType startPos = 0;
  for (;;) {
    CUDA_CHECK(cudaMemset((void *)collisionHandlingData.d_doneWithIdx, 0, nThreadGroupsTotal * sizeof(IdxType)))
    kernel_clusterExpansion <<<dim3(1, nBlocks), dim3(nThreadsPerBlock / nThreadGroupsPerBlock, nThreadGroupsPerBlock), sharedBytesPerBlock >>> (
      *d_clusters, *d_coreMarkers, xs, ys, n, startPos, collisionHandlingData, coreThreshold, rsq
    );
    //CUDA_CHECK(cudaDeviceSynchronize())

    if (n - startPos <= nThreadGroupsTotal) break;
    startPos += nThreadGroupsTotal;
  }
  CUDA_CHECK(cudaGetLastError())
}

void unionizeCpu(std::vector<IdxType> & clusters) {
  std::vector<IdxType> stack;
  for (IdxType i = 0; i < clusters.size(); ++i) {
    IdxType child = i + 1;
    IdxType parent = child + clusters[child - 1] + 1;
    if (parent == 0 || parent == child) continue; // noise
    do {
      stack.push_back(child);
      child = parent;
      parent = child + clusters[child - 1] + 1;
    } while (parent != child);
    IdxType top = child;
    while (stack.size() > 0) {
      IdxType current = stack.back(); stack.pop_back();
      clusters[current - 1] = top - current;
    }
  }
}

static __global__ void kernel_unionize(IdxType * clusters, IdxType n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  IdxType strideBegin = 0;

  auto doWork = [&] {
    IdxType idx = strideBegin + tid;
    IdxType cl = idx;
    IdxType pOffset = clusters[idx];
    if (!pOffset) {
      for (;;) {
        cl += pOffset;
        pOffset = clusters[cl];
        if (!pOffset) break;
      }
      IdxType top = cl;
      cl = idx;
      for (;;) {
        pOffset = clusters[cl];
        if (!pOffset) break;
        clusters[cl] = top - cl;
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