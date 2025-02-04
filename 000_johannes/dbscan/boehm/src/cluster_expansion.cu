#include "cluster_expansion.h"
#include "types.h"
#include "cuda_helpers.h"

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
  IdxType currentClusterId,
  IdxType pointIdx,
  bool * collisionMatrix,
  IdxType * cluster, unsigned int * pointState,
  IdxType * seedList, IdxType * seedClusterIds, IdxType * seedLength
) {
  unsigned int oldState = atomicCAS(&pointState[pointIdx], stateFree, stateReserved); // TODO: check chain index
  switch (oldState) {
    case stateFree: { // now state is stateReserved
      int h = atomicAdd(seedLength, 1); // TODO: check overflow
      if (h < maxSeedLength) { seedList[h] = pointIdx; seedClusterIds[h] = currentClusterId; }
    } break;
    case stateCore: {
      // TODO: handle collision!
    } break;
    case stateNoiseOrBorder: {
      cluster[pointIdx] = currentClusterId;
    } break;
    case stateUnderInspection: {
      (void)atomicCAS(&cluster[pointIdx], 0, currentClusterId);
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
  float px, float py, IdxType myselfIdx,
  float * xs, float * ys, IdxType pointIdx,
  IdxType currentClusterId,
  IdxType * cluster, unsigned int * pointState,
  IdxType * s_neighborBuffer, IdxType * s_neighborCount,
  bool * collisionMatrix, IdxType * seedList, IdxType * seedClusterIds, IdxType * seedLength,
  IdxType coreThreshold, float rsq
) {
  float dx = xs[pointIdx] - px;
  float dy = ys[pointIdx] - py;
  if (dx * dx + dy * dy <= rsq) {
    int h = atomicAdd(s_neighborCount, 1) + 1;
    if (h == coreThreshold) {
      cluster[myselfIdx] = currentClusterId;
      pointState[myselfIdx] = stateCore;
      markAsCandidate(currentClusterId, pointIdx, collisionMatrix, cluster, pointState, seedList, seedClusterIds, seedLength);
    } else if (h > coreThreshold) {
      markAsCandidate(currentClusterId, pointIdx, collisionMatrix, cluster, pointState, seedList, seedClusterIds, seedLength);
    } else {
      s_neighborBuffer[h] = pointIdx;
    }
  }
}


// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * cluster, unsigned int * pointState,
  bool * collisionMatrix,
  float * xs, float * ys, IdxType n,
  IdxType * seedLists, unsigned int * seedClusterIds, IdxType * seedLengths,
  IdxType coreThreshold, float rsq
) {
  extern __shared__ unsigned char sMem []; // coreThreshold IdxType values in blocks of 128 bytes + 1 IdxType value

  unsigned int threadGroupIdx = blockIdx.x;
  unsigned int stride = blockDim.x;

  IdxType * neighborBuffer = (IdxType *)sMem; // Length: coreThreshold elements
  IdxType * neighborCount = (IdxType *) (sMem + (coreThreshold * sizeof(IdxType) + 127) / 128 * 128);
  static_assert(128 % alignof(IdxType) == 0, "");
  
  if (threadIdx.x == 0) *neighborCount = 0;

  IdxType seedLength = seedLengths[threadGroupIdx];

  __syncthreads();

  if (seedLength > 0) {
    --seedLength;

    if (threadIdx.x == 0) seedLengths[threadGroupIdx] = seedLength;

    IdxType seedPointIdx = seedLists[maxSeedLength * threadGroupIdx + seedLength];
    unsigned int currentClusterId = seedClusterIds[maxSeedLength * threadGroupIdx + seedLength];

    pointState[seedPointIdx] = stateUnderInspection;
    float x = xs[seedPointIdx], y = ys[seedPointIdx];
    {
      IdxType strideIdx = 0;
      for (; strideIdx < (n - 1) / stride; ++strideIdx) {
        processObject(
          x, y, seedPointIdx, xs, ys, strideIdx * stride + threadIdx.x, currentClusterId, cluster, 
          pointState, neighborBuffer, neighborCount, collisionMatrix, &seedLists[maxSeedLength * threadGroupIdx], &seedClusterIds[maxSeedLength * threadGroupIdx], &seedLengths[threadGroupIdx], coreThreshold, rsq
        );
      }
      if (threadIdx.x < n - strideIdx * stride) {
        processObject(
          x, y, seedPointIdx, xs, ys, strideIdx * stride + threadIdx.x, currentClusterId, cluster,
          pointState, neighborBuffer, neighborCount, collisionMatrix, &seedLists[maxSeedLength * threadGroupIdx], &seedLengths[threadGroupIdx], &seedClusterIds[maxSeedLength * threadGroupIdx], coreThreshold, rsq
        );
      }
    }

    __syncthreads();

    if (*neighborCount >= coreThreshold) {
      //if (threadIdx.x == 0) pointState[seedPointIdx] = cCore;
      for (int i = threadIdx.x; i < coreThreshold; i += stride) {
        markAsCandidate(
          currentClusterId, neighborBuffer[i], collisionMatrix, cluster, pointState,
          &seedLists[maxSeedLength * threadGroupIdx], &seedClusterIds[maxSeedLength * threadGroupIdx], &seedLengths[threadGroupIdx]
        );
      }
    } else {
      if (threadIdx.x == 0) pointState[seedPointIdx] = stateNoiseOrBorder;
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

void findClusters(
  unsigned int * d_pointStates, IdxType * d_clusters, bool * d_collisionMatrix,
  float * xs, float * ys, IdxType n,
  IdxType * d_seedLists, IdxType * d_seedClusterIds, IdxType * d_seedLengths,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 6;
  constexpr int nThreadsPerBlock = 256;

  kernel_populateSeedLists <<<dim3(1), dim3(nThreadsPerBlock)>>> (d_seedLists, d_seedClusterIds, d_seedLengths, nBlocks);
  kernel_clusterExpansion <<<dim3(nBlocks), dim3(nThreadsPerBlock), ( (coreThreshold * sizeof(IdxType) + 127) / 128 * 128 + sizeof(IdxType) ) >>> (
    d_clusters, d_pointStates, d_collisionMatrix, xs, ys, n, d_seedLists, d_seedClusterIds, d_seedLengths, coreThreshold, rsq
  );
}

