#include "c_cluster_expansion.h"
#include "c_types.h"
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

static __device__ IdxType unionizeWithinThreadGroup(
  volatile IdxType * s_interWarpUnionize,
  IdxType * clusters,
  IdxType myClusterId
) {
  // unionize within every warp
  for (unsigned int i = 1; i < 32; i <<= 1) {
    IdxType otherClusterId = __shfl_down_sync(0xffffffff, myClusterId, i);
    if (!(laneId() & ((i << 1) - 1))) myClusterId = unionizeClusters(clusters, myClusterId, otherClusterId);
  }
  myClusterId = __shfl_sync(0xffffffff, myClusterId, 0);

  __syncthreads();

  // unionize among warps within thread group
  unsigned int nValuesToUnionize = (blockDim.x + 31) / 32;
  if (nValuesToUnionize > 1) {
    int lane = laneId();
    int wid = threadIdx.x / 32;
    if (lane == 0) s_interWarpUnionize[threadIdx.x / 32] = myClusterId;
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
        ) myValue = unionizeClusters(clusters, myValue, otherValue);
      }
      nValuesToUnionize = (nValuesToUnionize + 31) / 32;
      if (laneId() == 0) s_interWarpUnionize[threadIdx.x / 32] == myValue;
      if (nValuesToUnionize == 1) break;
    }
    __syncthreads();
    myClusterId = s_interWarpUnionize[0];
  }

  return myClusterId;
}

static __global__ void kernel_handleCollisions(
  unsigned int * collisionMatrix,
  IdxType * clusters,
  bool * coreMarkers,
  IdxType n,
  IdxType beginStep,
  IdxType nThreadGroupsTotal
) {
  unsigned int collisionPitch = (nThreadGroupsTotal + 31) / 32;
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int nRequiredThreads = nThreadGroupsTotal * nThreadGroupsTotal;
  if (tid < nRequiredThreads) {
    unsigned int r = tid / nThreadGroupsTotal;
    unsigned int c = tid % nThreadGroupsTotal;

    bool collisionRToC = collisionMatrix[r * collisionPitch + c / 32] & (1u << (c % 32));
    bool collisionCToR = collisionMatrix[c * collisionPitch + r / 32] & (1u << (r % 32));
    if (c < r && collisionRToC && collisionCToR) {
      // both points are core
      (void)unionizeClusters(clusters, beginStep + r, beginStep + c);
    } else if (c < r && collisionRToC) {
      // r is core
      clusters[beginStep + c] = r - c;
    } else if (c < r && collisionCToR) {
      // c is core
      clusters[beginStep + r] = c - r;
    }
  }
}

static __device__ void sharedMemZero(
  char * s_mem, IdxType nBytes
) {
  // zeroing one byte per thread is faster than one unsigned int per thread
  //   -- reason unclear, but may be related to memory bank conflicts
  IdxType strideStart = 0;
  IdxType myOffset = blockDim.x * threadIdx.y + threadIdx.x;
  while (nBytes - strideStart > blockDim.x * blockDim.y) {
    s_mem [strideStart + myOffset] = 0;
    strideStart += blockDim.x * blockDim.y;
  }
  if (myOffset < nBytes - strideStart) s_mem [strideStart + myOffset] = 0;

}

static __device__ IdxType processPoints(
  bool * coreMarkers, IdxType * clusters, bool * s_collisions,
  IdxType * s_neighborBuffer, IdxType * s_neighborCount,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float rsq,
  IdxType beginStep, IdxType endStep,
  IdxType ourPointIdx, IdxType myClusterId
) {
  unsigned int stride = blockDim.x;

  auto markAsCandidate = [&] (IdxType pointIdx) {
    if (pointIdx < beginStep) {
      if (coreMarkers[pointIdx]) {
        myClusterId = unionizeClusters(clusters, pointIdx, myClusterId);
      } else {
        clusters[pointIdx] = myClusterId - pointIdx;
      }
    } else if (pointIdx < endStep) {
      s_collisions[pointIdx - beginStep] = true;
    } else {
      clusters[pointIdx] = myClusterId - pointIdx;
    }
  };

  float x = xs[ourPointIdx], y = ys[ourPointIdx];
  bool isDefinitelyCore = false;

  auto processObject = [&] (IdxType pointIdx) {
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

  IdxType strideIdx = 0;
  for (; strideIdx < (n - 1) / stride; ++strideIdx) processObject(strideIdx * stride + threadIdx.x);
  if (threadIdx.x < n - strideIdx * stride) processObject(strideIdx * stride + threadIdx.x);
    
  __syncthreads();

  if (*s_neighborCount >= coreThreshold) {
    for (int i = threadIdx.x; i < coreThreshold; i += stride) {
      markAsCandidate(s_neighborBuffer[i]);
    }
  }

  return myClusterId;
}

static __device__ void writeCollisions(
  unsigned int * collisionMatrix,
  bool * s_collisions,
  IdxType nThreadGroupsTotal
) {
  unsigned int stride = blockDim.x;
  unsigned int threadGroupIdx = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int collisionPitch = (nThreadGroupsTotal + 31) / 32;

  unsigned int strideBegin = 0;

  auto assembleMask = [&] (unsigned int m) {
    #pragma unroll
    for (int j = 1; j < 32; j <<= 1) {
      m |= __shfl_down_sync(0xffffffff, m, j) << j;
    }
    return m;
  };

  if (stride < nThreadGroupsTotal) {
    for (; strideBegin < nThreadGroupsTotal - stride; strideBegin += stride) {
      unsigned int collisionMask = assembleMask(s_collisions[strideBegin + threadIdx.x]);
      if (laneId() == 0) {
        collisionMatrix[collisionPitch * threadGroupIdx + (strideBegin + threadIdx.x) / 32] = collisionMask;
      }
    }
  }
  unsigned int collisionMask = assembleMask(strideBegin + threadIdx.x < nThreadGroupsTotal ? s_collisions[strideBegin + threadIdx.x] : 0);
  if (laneId() == 0 && strideBegin + threadIdx.x < nThreadGroupsTotal) collisionMatrix[collisionPitch * threadGroupIdx + (strideBegin + threadIdx.x) / 32] = collisionMask;
}

static __global__ void kernel_clusterExpansion(
  IdxType * clusters, bool * coreMarkers,
  float const * xs, float const * ys, IdxType n,
  IdxType beginStep,
  unsigned int * collisionMatrix,
  IdxType coreThreshold, float rsq
) {
  unsigned int stride = blockDim.x; // blockDim.x must be a multiple of 32
  unsigned int nThreadGroupsPerBlock = blockDim.y;
  unsigned int nBlocks = gridDim.y;
  unsigned int nThreadGroupsTotal = nBlocks * nThreadGroupsPerBlock;
  unsigned int threadGroupIdx = blockDim.y * blockIdx.y + threadIdx.y;

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
  sharedMemZero(sMem, sMemBytesPerThreadGroup * blockDim.y);

  if (threadGroupIdx < n - beginStep) {
    // all threads in a thread group always examine the same point, but the
    //   cluster ids may diverge during the process
    IdxType ourPointIdx = beginStep + threadGroupIdx;
    IdxType myClusterId = ourPointIdx + clusters[ourPointIdx];

    __syncthreads();

    myClusterId = processPoints(
      coreMarkers, clusters, s_collisions,
      s_neighborBuffer, s_neighborCount,
      xs, ys, n,
      coreThreshold, rsq,
      beginStep, nThreadGroupsTotal < n - beginStep ? beginStep + nThreadGroupsTotal : n,
      ourPointIdx, myClusterId
    );

    __syncthreads();

    IdxType neighborCount = *s_neighborCount;

    if (neighborCount >= coreThreshold) {
      myClusterId = unionizeWithinThreadGroup(s_interWarpUnionize, clusters, myClusterId);

      __syncthreads();
  
      if (threadIdx.x == 0) {
        clusters[ourPointIdx] = myClusterId - ourPointIdx;
        coreMarkers[ourPointIdx] = true;
      }
    } else {
      for (int i = threadIdx.x; i < neighborCount; i += stride) {
        IdxType neighbor = s_neighborBuffer[i];
        if (neighbor < beginStep && coreMarkers[neighbor]) { clusters[ourPointIdx] = neighbor - ourPointIdx; break; }
      }
    }
  }

  __syncthreads();

  writeCollisions(collisionMatrix, s_collisions, nThreadGroupsTotal);
}

void allocateDeviceMemory(
  bool ** d_coreMarkers, IdxType ** d_clusters,
  unsigned int ** d_collisionMatrix,
  int nThreadGroups,
  IdxType n
) {
  CUDA_CHECK(cudaMalloc(d_coreMarkers, n * sizeof(bool)))
  // TODO: Change later
  CUDA_CHECK(cudaMemset(*d_coreMarkers, 0, n * sizeof(bool)))
  CUDA_CHECK(cudaMalloc(d_clusters, n * sizeof(IdxType)))
  // TODO: Change later
  CUDA_CHECK(cudaMemset(*d_clusters, 0, n * sizeof(IdxType)))

  unsigned int collisionPitch = (nThreadGroups + 31) / 32;
  CUDA_CHECK(cudaMalloc(d_collisionMatrix, collisionPitch * nThreadGroups * sizeof(unsigned int)))
}


void unionizeCpu(std::vector<IdxType> & clusters) {
  std::vector<IdxType> stack;
  for (IdxType i = 0; i < clusters.size(); ++i) {
    IdxType child = i;
    IdxType parentOffset = clusters[child];
    if (parentOffset == 0) continue; // noise
    do {
      stack.push_back(child);
      child = child + parentOffset;
      parentOffset = clusters[child];
    } while (parentOffset);
    IdxType top = child;
    while (stack.size() > 0) {
      IdxType current = stack.back(); stack.pop_back();
      clusters[current] = top - current;
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
    if (pOffset) {
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
        cl += pOffset;
      }
    }
  };

  if (n > stride) for (; strideBegin < n - stride; strideBegin += stride) doWork();
  if (tid < n - strideBegin) doWork();
}

void findClusters(
  bool ** d_coreMarkers, IdxType ** d_clusters,
  float * xs, float * ys, IdxType n,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 6;
  constexpr int nThreadGroupsPerBlock = 32;
  constexpr int nThreadsPerBlock = 1024;

  int nThreadGroupsTotal = nBlocks * nThreadGroupsPerBlock;

  unsigned int sharedBytesPerBlock = nThreadGroupsPerBlock * 4 * (
    (nThreadGroupsTotal + 3) / 4 +
    dhi_max(coreThreshold, (nThreadGroupsPerBlock + 31) / 32) +
    1
  );

  unsigned int * d_collisionMatrix;
  allocateDeviceMemory(d_coreMarkers, d_clusters, &d_collisionMatrix, nThreadGroupsTotal, n);

  IdxType startPos = 0;
  for (;;) {
    kernel_clusterExpansion <<<dim3(1, nBlocks), dim3(nThreadsPerBlock / nThreadGroupsPerBlock, nThreadGroupsPerBlock), sharedBytesPerBlock >>> (
      *d_clusters, *d_coreMarkers, xs, ys, n, startPos, d_collisionMatrix, coreThreshold, rsq
    );
    unsigned int nCHThreads = nThreadGroupsTotal * nThreadGroupsTotal;
    unsigned int nCHThreadsPerBlock = 128;
    unsigned int nCHBlocks = (nCHThreads + nCHThreadsPerBlock - 1) / nCHThreadsPerBlock;
    kernel_handleCollisions <<<nCHBlocks, nCHThreadsPerBlock>>> (
      d_collisionMatrix, *d_clusters, *d_coreMarkers, n, startPos, nThreadGroupsTotal
    );

    //CUDA_CHECK(cudaDeviceSynchronize())

    if (n - startPos <= nThreadGroupsTotal) break;
    startPos += nThreadGroupsTotal;
  }
  kernel_unionize <<<nBlocks, nThreadsPerBlock>>> (*d_clusters, n);
  CUDA_CHECK(cudaGetLastError())
}

void unionizeGpu(IdxType * d_clusters, IdxType n) {
  constexpr unsigned int nBlocks = 6;
  constexpr unsigned int nThreadsPerBlock = 1024;

  kernel_unionize <<<dim3(nBlocks), dim3(nThreadsPerBlock)>>> (d_clusters, n);
  CUDA_CHECK(cudaGetLastError())
  CUDA_CHECK(cudaDeviceSynchronize())
}