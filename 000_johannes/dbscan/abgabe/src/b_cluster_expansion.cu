#include "b_cluster_expansion.h"
#include "b_types.h"
#include "cuda_helpers.h"

#include <iostream>
#include <cuda.h>
#include <vector>

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
  IdxType * seedList, IdxType * seedLength, IdxType * s_seedReserved,
  unsigned int state, IdxType pointIdx,
  IdxType ourClusterId
) {
  unsigned int threadGroupIdx = blockDim.y * blockIdx.y + threadIdx.y;

  bool stateLikelyFree = state == stateFree;

  if (stateLikelyFree) {
    IdxType oldReserved = atomicAdd(s_seedReserved, 1);
    bool iCanWrite = oldReserved < maxSeedLength;
    if (iCanWrite) {
      state = atomicCAS(&pointStates[pointIdx], stateFree, stateReserved);
      if (state == stateFree) {
        IdxType oldSeedLength = atomicAdd(seedLength, 1);
        seedList[oldSeedLength] = pointIdx;
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
  IdxType * seedList, IdxType * seedLength, IdxType * s_seedReserved,
  bool * s_collisions,
  IdxType ourClusterId,
  IdxType otherPointIdx
) {
  unsigned int otherState = pointStates[otherPointIdx];
  
  otherState = tryAppendToSeedListIfStateFree(
    pointStates,
    seedList, seedLength, s_seedReserved,
    otherState, otherPointIdx,
    ourClusterId
  );

  switch (otherState & stateStateBitsMask) {
    case stateCore: {
      // MISSING: Unionize
    } break;
    case stateNoiseOrBorder:
    case stateReserved:
    case stateFree: {
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
  IdxType * seedList,IdxType * seedLength, IdxType * s_seedReserved,
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
  if (isNeighbor) {
    int oldNeighborCount = atomicAdd(s_neighborCount, 1);
    if (oldNeighborCount >= coreThreshold) {
      markAsCandidate(
        clusters, pointStates,
        seedList, seedLength, s_seedReserved,
        s_collisions, ourClusterId, pointIdx
      );
    } else {
      s_neighborBuffer[oldNeighborCount] = pointIdx;
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
  IdxType * seedLists, IdxType * seedLengths,
  IdxType * pointsUnderInspection, IdxType * threadGroupClusterIds,
  unsigned int * synchronizer, CollisionHandlingDataView collisionHandlingData,
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

  IdxType ourPointIdx = pointsUnderInspection[threadGroupIdx];

  if (threadIdx.x == 0) { *s_seedReserved = *ourSeedLength; }
  IdxType ourClusterId = threadGroupClusterIds[threadGroupIdx];
  if (ourClusterId == 0) ourClusterId = ourPointIdx + 1;

  __syncthreads();

  if (ourPointIdx + 1 != 0) {
    //printf("%u %u\n", threadGroupIdx, ourPointIdx);
    float x = xs[ourPointIdx], y = ys[ourPointIdx];
    {
      IdxType strideBegin = 0;
      if (n > stride) for (; strideBegin < n - stride; strideBegin += stride) {
        processObject(
          clusters, pointStates, 
          ourSeedList, ourSeedLength, s_seedReserved,
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
          ourSeedList, ourSeedLength, s_seedReserved,          
          s_collisions, s_neighborBuffer, s_neighborCount,
          ourPointIdx, ourClusterId,
          x, y,
          xs, ys, strideBegin + threadIdx.x,
          coreThreshold, rsq
        );
      }
    }
 
    __syncthreads();

    if (ourPointIdx + 1 != 0) {
      IdxType cnt = *s_neighborCount;
      if (cnt >= coreThreshold) {
        for (int i = threadIdx.x; i < coreThreshold; i += stride) {
          markAsCandidate(
            clusters, pointStates,
            ourSeedList, ourSeedLength, s_seedReserved,
            s_collisions, ourClusterId, s_neighborBuffer[i]
          );
        }
        if (threadIdx.x == 0) {
          pointStates[ourPointIdx] = stateCore;
          clusters[ourPointIdx] = ourClusterId;
        }
      } else {
        if (threadIdx.x == 0) pointStates[ourPointIdx] = stateNoiseOrBorder;
      }
    }
return;
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

    if (ourPointIdx + 1 != 0) {
      IdxType cnt = *s_neighborCount;
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
}

static __global__ void kernel_refillSeeds(
  IdxType * d_continueAt,
  IdxType * d_seedLists, IdxType * d_seedLengths,
  IdxType * d_pointsUnderInspection, IdxType * d_threadGroupClusterIds,
  unsigned int * d_pointState, IdxType * d_clusters, IdxType n,
  IdxType nThreadGroupsTotal
) {
  constexpr unsigned int wrp = 32;
  __shared__ IdxType foundIndices[wrp];
  int nAvailable = 0;

  IdxType strideIdx = *d_continueAt / wrp;

  bool stillWork = false;
  int tg = 0;
  for (; tg < nThreadGroupsTotal; ++tg) {
    IdxType seedLength = d_seedLengths[tg];
    if (seedLength == 0) {
      if (nAvailable == 0) {
        for (; strideIdx <= (n - 1) / wrp; ++strideIdx) {
          IdxType idx = strideIdx * wrp + threadIdx.x;
          bool haveOne = idx < n && !d_pointState[idx];
          int foundMask = __ballot_sync(0xffffffff, haveOne);
          if (haveOne) foundIndices[__popc(foundMask & ((1u << threadIdx.x) - 1))] = idx;
          nAvailable = __popc(foundMask);
          if (nAvailable) break;
        }
      }
      __syncthreads();
      if (nAvailable == 0) {
        d_pointsUnderInspection[tg] = (IdxType)-1;
      } else {
        stillWork = true;
        --nAvailable;
        if (threadIdx.x == 0) {
          IdxType pointIdx = foundIndices[nAvailable];
          //if (threadIdx.x == 0) printf("Refill %u from array with %u\n", tg, pointIdx);
          d_pointState[pointIdx] = stateUnderInspection | (unsigned int)tg;
          d_pointsUnderInspection[tg] = pointIdx;
          d_threadGroupClusterIds[tg] = d_clusters[pointIdx];
        }
      }
    } else {
      stillWork = true;
      if (threadIdx.x == 0) {
        --seedLength;
        d_seedLengths[tg] = seedLength;
        IdxType pointIdx = d_seedLists[tg * maxSeedLength + seedLength];
        //printf("Refill %u from seed with %u\n", tg, pointIdx);
        d_pointState[pointIdx] = stateUnderInspection | (unsigned int)tg;
        d_pointsUnderInspection[tg] = pointIdx;
      }
      //d_threadGroupClusterIds[tg] = d_clusters[pointIdx];
    }
  }
  if (threadIdx.x == 0) *d_continueAt = !stillWork ? (IdxType)-1 : strideIdx * wrp;
}

static inline auto determineCEBlockGeometry(
  unsigned int nCEBlocks, unsigned int nCEThreadsPerBlock, IdxType coreThreshold
) {
  constexpr int maxSharedBytesPerBlock = 40000;
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
  int nCEBlocks = 1; //nSm;
  constexpr int nCEThreadsPerBlock = 512;
  auto blockGeom = determineCEBlockGeometry(nCEBlocks, nCEThreadsPerBlock, coreThreshold);
  unsigned int nCEThreadGroupsPerBlock = blockGeom.nCEThreadGroupsPerBlock;
  unsigned int nCEThreadGroupsTotal = blockGeom.nCEThreadGroupsTotal;
  unsigned int sharedBytesPerBlock = blockGeom.requiredSharedBytes;


  auto && d_seedLists = ManagedDeviceArray<IdxType> (nCEThreadGroupsTotal * maxSeedLength);
  auto && d_seedLengths = ManagedDeviceArray<IdxType> (nCEThreadGroupsTotal);
  auto && d_synchronizer = ManagedDeviceArray<unsigned int> (1);
  auto && d_continueAt = ManagedDeviceArray<IdxType> (1);
  auto && d_pointsUnderInspection = ManagedDeviceArray<IdxType> (nCEThreadGroupsTotal);
  auto && d_threadGroupClusterIds = ManagedDeviceArray<IdxType> (nCEThreadGroupsTotal);

  auto && collisionHandlingData = CollisionHandlingData(nCEThreadGroupsTotal, n);

  CUDA_CHECK(cudaMemset(d_pointStates, 0, n * sizeof(unsigned int)))
  CUDA_CHECK(cudaMemset(d_clusters, 0, n * sizeof(IdxType)))
  CUDA_CHECK(cudaMemset(d_seedLengths.ptr(), 0, nCEThreadGroupsTotal * sizeof(IdxType)))
  CUDA_CHECK(cudaMemset(d_continueAt.ptr(), 0, sizeof(IdxType)))

  std::vector<IdxType> h_seedLengths (nCEThreadGroupsTotal);

  int nIterations = 0;
  for (;;) {
    kernel_refillSeeds<<<1,32>>>(
      d_continueAt.ptr(), d_seedLists.ptr(), d_seedLengths.ptr(),
      d_pointsUnderInspection.ptr(), d_threadGroupClusterIds.ptr(),
      d_pointStates, d_clusters, n, nCEThreadGroupsTotal
    );
    IdxType continueAt;
    CUDA_CHECK(cudaMemcpy(&continueAt, d_continueAt.ptr(), sizeof(IdxType), cudaMemcpyDeviceToHost))
    if (continueAt + 1 == 0) break;

    CUDA_CHECK(cudaMemset(d_synchronizer.ptr(), 0, sizeof(unsigned int)))
    CUDA_CHECK(cudaMemset((void *)collisionHandlingData.d_doneWithIdx, 0, nCEThreadGroupsTotal * sizeof(IdxType)))
    kernel_clusterExpansion <<<
      dim3(1, nCEBlocks),
      dim3(nCEThreadsPerBlock / nCEThreadGroupsPerBlock, nCEThreadGroupsPerBlock),
      sharedBytesPerBlock
    >>> (
      d_clusters, d_pointStates,
      xs, ys, n,
      d_seedLists.ptr(), d_seedLengths.ptr(),
      d_pointsUnderInspection.ptr(), d_threadGroupClusterIds.ptr(),
      d_synchronizer.ptr(),
      CollisionHandlingDataView{ collisionHandlingData },
      coreThreshold, rsq
    );
    cudaMemcpy(h_seedLengths.data(), d_seedLengths.ptr(), nCEThreadGroupsTotal * sizeof(IdxType), cudaMemcpyDeviceToHost);
//    std::cerr << "Lengths:\n";
//    for (int i = 0; i < nCEThreadGroupsTotal; ++i) std::cerr << h_seedLengths[i] << std::endl;
//    std::cerr << std::endl;
    CUDA_CHECK(cudaGetLastError())
    ++nIterations;
    //if (nIterations == 1000) break;
  }

  // MISSING: Unionize

  CUDA_CHECK(cudaGetLastError())
}

