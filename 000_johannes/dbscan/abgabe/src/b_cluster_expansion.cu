#include "b_cluster_expansion.h"
#include "b_types.h"
#include "cuda_helpers.h"

#include <iostream>
#include <cuda.h>

constexpr IdxType maxSeedLength = 1024;

struct CollisionHandlingData {
  unsigned int * d_synchronizer;
  IdxType * d_doneWithIdx;
  bool * d_collisionMatrix;

  CollisionHandlingData(int nThreadGroupsTotal, IdxType n) {
    CUDA_CHECK(cudaMalloc(&this->d_synchronizer, sizeof(unsigned int)))
    CUDA_CHECK(cudaMalloc(&this->d_doneWithIdx, nThreadGroupsTotal * sizeof(IdxType)))
    CUDA_CHECK(cudaMalloc(&this->d_collisionMatrix, nThreadGroupsTotal * nThreadGroupsTotal))
  }

  CollisionHandlingData(CollisionHandlingData const &) = delete;

  ~CollisionHandlingData() {
    (void)cudaFree(this->d_collisionMatrix);
    (void)cudaFree(this->d_doneWithIdx);
    (void)cudaFree(this->d_synchronizer);
  }
};

struct CollisionHandlingDataView {
  unsigned int * d_synchronizer;
  IdxType * d_doneWithIdx;
  bool * d_collisionMatrix;

  explicit CollisionHandlingDataView(CollisionHandlingData const & chd)
  : d_synchronizer(chd.d_synchronizer), d_doneWithIdx(chd.d_doneWithIdx), d_collisionMatrix(chd.d_collisionMatrix) {
  }
};

static __device__ __forceinline__ IdxType tryAppendToSeedListIfStateFree(
  IdxType * pointStates,
  IdxType * seedList, IdxType * seedClusterIds, IdxType * seedLength, IdxType * s_seedReserved,
  unsigned int state, IdxType pointIdx,
  IdxType ourClusterId
) {
unsigned int threadGroupIdx = blockDim.y * blockIdx.y + threadIdx.y;

  bool stateLikelyFree = state == stateFree;

  if (stateLikelyFree) {
    IdxType oldReserved = atomicAdd(s_seedReserved, 1);
    bool iCanWrite = oldReserved < maxSeedLength;
    if (iCanWrite) {
      state = atomicCAS(&pointStates[pointIdx], stateFree, stateReserved | threadGroupIdx);
      if (state == stateFree) {
        IdxType oldSeedLength = atomicAdd(seedLength, 1);
        seedList[oldSeedLength] = pointIdx; seedClusterIds[oldSeedLength] = ourClusterId;
      } else {
        (void)atomicSub(s_seedReserved, 1);
      }
    } else {
      (void)atomicSub(s_seedReserved, 1);
    }
  }

  return state;
}

static __device__ void markAsCandidate(
  IdxType * clusters, unsigned int * pointStates, 
  IdxType * seedList, IdxType * seedClusterIds, IdxType * seedLength, IdxType * s_seedReserved,
  bool * s_collisions,
  IdxType ourClusterId,
  IdxType otherPointIdx
) {
  unsigned int otherState = pointStates[otherPointIdx];
  
  otherState = tryAppendToSeedListIfStateFree(
    pointStates,
    seedList, seedClusterIds, seedLength, s_seedReserved,
    otherState, otherPointIdx,
    ourClusterId
  );

  switch (otherState & stateStateBitsMask) {
    case stateCore: {
      // MISSING: Unionize
    } break;
    case stateNoiseOrBorder: {
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
        s_collisions, ourClusterId, pointIdx
      );
    } else {
      s_neighborBuffer[h] = pointIdx;
    }
  }
}

static __device__ void sharedMemZero(
  unsigned char * s_mem, IdxType nBytes
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

// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * clusters, unsigned int * pointStates,
  float const * xs, float const * ys, IdxType n,
  IdxType * seedLists, unsigned int * seedClusterIds, IdxType * seedLengths,
  unsigned int * syncCounter, CollisionHandlingDataView collisionHandlingData,
  IdxType coreThreshold, float rsq
) {
  unsigned int stride = blockDim.x; // blockDim.x must be a multiple of 32
  unsigned int nThreadGroupsPerBlock = blockDim.y;
  unsigned int nBlocks = gridDim.y;
  unsigned int nThreadGroupsTotal = nBlocks * nThreadGroupsPerBlock;
  unsigned int threadGroupIdx = blockDim.y * blockIdx.y + threadIdx.y;

  // shared memory:
  // s_collisions: nThreadGroupsTotal bools
  //   size in bytes: (nThreadGroupsTotal + 3) / 4 * 4
  // s_neighborBuffer: IdxType[coreThreshold]
  //   size in bytes: coreThreshold * 4
  // s_neighborCount:  1 IdxType
  //   size in bytes: 4
  // s_seedReserved:   1 IdxType
  //   size in bytes: 4
  extern __shared__ unsigned char sMem [];
  unsigned int sMemBytesPerThreadGroup = 4 * (
    (nThreadGroupsTotal + 3) / 4 +
    coreThreshold +
    1 +
    1
  );

  static_assert(
    sizeof(IdxType)  == 4 &&
    alignof(IdxType) == 4 &&
    sizeof(bool) == 1, ""
  );
  bool * s_collisions        = (bool *)    (sMem                              + sMemBytesPerThreadGroup * threadIdx.y);
  IdxType * s_neighborBuffer = (IdxType *) ((unsigned char *)s_collisions     + (nThreadGroupsTotal + 3) / 4 * 4);
  IdxType * s_neighborCount  = (IdxType *) ((unsigned char *)s_neighborBuffer + coreThreshold * 4);
  IdxType * s_seedReserved   = (IdxType *) ((unsigned char *)s_neighborCount  + 4);

  sharedMemZero(sMem, sMemBytesPerThreadGroup * blockDim.y);

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

    // TODO: This is possibly not correct. We cannot guarantee that all threads in the grid
    //   will observe this change!
    if (threadIdx.x == 0) pointStates[ourPointIdx] = stateUnderInspection | threadGroupIdx;

    __syncthreads();

    float x = xs[ourPointIdx], y = ys[ourPointIdx];
    {
      IdxType strideBegin = 0;
      if (n > stride) for (; strideBegin < n - stride; strideBegin += stride) {
        processObject(
          clusters, pointStates, 
          ourSeedList, ourSeedClusterIds, ourSeedLength, s_seedReserved,
          s_collisions, s_neighborBuffer, s_neighborCount,
          ourPointIdx, ourClusterId,
          x, y,
          xs, ys, strideBegin + threadIdx.x,
          coreThreshold, rsq
        );
      }
      if (threadIdx.x < n - strideBegin) {
        processObject(
          clusters, pointStates,
          ourSeedList, ourSeedClusterIds, ourSeedLength, s_seedReserved,          
          s_collisions, s_neighborBuffer, s_neighborCount,
          ourPointIdx, ourClusterId,
          x, y,
          xs, ys, strideBegin + threadIdx.x,
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
          s_collisions, ourClusterId, s_neighborBuffer[i]
        );
      }
    } else {
      if (threadIdx.x == 0) pointStates[ourPointIdx] = stateNoiseOrBorder;
    }

    __syncthreads();

    // copy our collisions to global memory
    for (unsigned int i = threadIdx.x; i < nThreadGroupsPerBlock; i += stride) {
      collisionHandlingData.d_collisionMatrix[nThreadGroupsTotal * threadGroupIdx + i] = s_collisions[i];
    }

    __threadfence();

    if (threadIdx.x == 0) collisionHandlingData.d_doneWithIdx[threadGroupIdx] = ourPointIdx + 1;

    __threadfence();

    if (threadIdx.x == 0) (void)atomicAdd(collisionHandlingData.d_synchronizer, 1);

    __threadfence();

    for (unsigned int i = threadIdx.x; i < nThreadGroupsTotal; i += stride) {
      if (i != threadGroupIdx) {
        IdxType otherIdx = collisionHandlingData.d_doneWithIdx[i] - 1;
        if (otherIdx != (IdxType)-1) {
          bool collision = s_collisions[i] || collisionHandlingData.d_collisionMatrix[i * nThreadGroupsTotal+ threadGroupIdx];
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
        d_pointState[result] = stateReserved;
        d_seedLists[k * maxSeedLength] = result;
        d_seedClusterIds[k * maxSeedLength] = d_clusters[result];
        d_seedLengths[k] = 1;
      }
    }
}

static inline auto determineCEBlockGeometry(
  unsigned int nCEBlocks, unsigned int nCEThreadsPerBlock, IdxType coreThreshold
) {
  constexpr int maxSharedBytesPerBlock = 47000;
  unsigned int nCEThreadGroupsPerBlock = 32;
  unsigned int requiredSharedBytes, nCEThreadGroupsTotal;
  for (;;) {
    nCEThreadGroupsTotal = nCEBlocks * nCEThreadGroupsPerBlock;
    requiredSharedBytes = nCEThreadGroupsPerBlock * 4 * (
      (nCEThreadGroupsTotal + 3) / 4 +
      coreThreshold +
      1 +
      1
    );
    if (requiredSharedBytes <= maxSharedBytesPerBlock) break;
    nCEThreadGroupsPerBlock >>= 1;
    if (!nCEThreadGroupsPerBlock) {
      fprintf(
        stderr, "coreThreshold too large.\n"
      );
      exit (1);
    }
  }
  struct Result {
    unsigned int nCEThreadGroupsPerBlock;
    unsigned int nCEThreadGroupsTotal;
    unsigned int requiredSharedBytes;
  };
  return Result { nCEThreadGroupsPerBlock, nCEThreadGroupsTotal, requiredSharedBytes };
}

void findClusters(
  int nSm,
  unsigned int * d_pointStates, IdxType * d_clusters,
  float * xs, float * ys, IdxType n,
  IdxType coreThreshold, float rsq
) {
  int nCEBlocks = nSm;
  constexpr int nCEThreadsPerBlock = 1024;
  auto blockGeom = determineCEBlockGeometry(nCEBlocks, nCEThreadsPerBlock, coreThreshold);
  unsigned int nCEThreadGroupsPerBlock = blockGeom.nCEThreadGroupsPerBlock;
  unsigned int nCEThreadGroupsTotal = blockGeom.nCEThreadGroupsTotal;
  unsigned int sharedBytesPerBlock = blockGeom.requiredSharedBytes;


  auto && d_seedLists = ManagedDeviceArray<IdxType> (nCEThreadGroupsTotal * maxSeedLength);
  auto && d_seedClusterIds = ManagedDeviceArray<IdxType> (nCEThreadGroupsTotal * maxSeedLength);
  auto && d_seedLengths = ManagedDeviceArray<IdxType> (nCEThreadGroupsTotal);
  auto && d_synchronizer = ManagedDeviceArray<unsigned int> (1);
  auto && d_foundAt = ManagedDeviceArray<IdxType> (1);

  auto && collisionHandlingData = CollisionHandlingData(nCEThreadGroupsTotal, n);

  CUDA_CHECK(cudaMemset(d_pointStates, 0, n * sizeof(unsigned int)))
  CUDA_CHECK(cudaMemset(d_clusters, 0, n * sizeof(IdxType)))
  CUDA_CHECK(cudaMemset(d_seedLengths.ptr(), 0, nCEThreadGroupsTotal * sizeof(IdxType)))

  IdxType seedLengths [nCEThreadGroupsTotal];
  IdxType startPos = 0;
  for (;;) {
    CUDA_CHECK(cudaMemcpy(seedLengths, d_seedLengths.ptr(), nCEThreadGroupsTotal * sizeof(IdxType), cudaMemcpyDeviceToHost))
    bool stillWork = false;

    for (int k = 0; k < nCEThreadGroupsTotal; ++k) {
      if (seedLengths[k]) {
        stillWork = true;
      } else if (startPos != (IdxType)-1) {
        IdxType foundAt = (IdxType)-1;
        kernel_refillSeed <<<1, 32>>> (
          d_foundAt.ptr(), d_seedLists.ptr(), d_seedClusterIds.ptr(), d_seedLengths.ptr(), k, d_pointStates, d_clusters, n, startPos
        );
        CUDA_CHECK(cudaGetLastError())
        CUDA_CHECK(cudaMemcpy(&foundAt, d_foundAt.ptr(), sizeof(IdxType), cudaMemcpyDeviceToHost))
        startPos = foundAt + (foundAt != (IdxType)-1);
        stillWork = stillWork || foundAt != (IdxType)-1;
      }
    }

    if (!stillWork) break;

    CUDA_CHECK(cudaMemset(d_synchronizer.ptr(), 0, sizeof(unsigned int)))
    CUDA_CHECK(cudaMemset((void *)collisionHandlingData.d_doneWithIdx, 0, nCEThreadGroupsTotal * sizeof(IdxType)))
    kernel_clusterExpansion <<<
      dim3(1, nCEBlocks),
      dim3(nCEThreadsPerBlock / nCEThreadGroupsPerBlock, nCEThreadGroupsPerBlock),
      sharedBytesPerBlock
    >>> (
      d_clusters, d_pointStates,
      xs, ys, n,
      d_seedLists.ptr(), d_seedClusterIds.ptr(), d_seedLengths.ptr(), d_synchronizer.ptr(),
      CollisionHandlingDataView{ collisionHandlingData },
      coreThreshold, rsq
    );
  }

  // Missing: Unionize
  
  CUDA_CHECK(cudaGetLastError())
}

