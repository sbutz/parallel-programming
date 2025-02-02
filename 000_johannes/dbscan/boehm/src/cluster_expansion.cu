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

constexpr IdxType cUnprocessed = 0;
constexpr IdxType cNoise = 1;
constexpr IdxType cChain = 2;

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
  IdxType pointIdx,
  bool * collisionMatrix,
  IdxType * pointState,
  IdxType * seedList, IdxType * seedLength
) {
  int oldState = atomicCAS(&pointState[pointIdx], cUnprocessed, cChain + blockIdx.x); // TODO: check chain index
  if (oldState == cUnprocessed) {
    int h = atomicAdd(seedLength, 1); // TODO: check overflow
    if (h < maxSeedLength) seedList[h] = pointIdx;
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
}

static __device__ void processObject(
  float px, float py,
  float * xs, float * ys, IdxType pointIdx,
  IdxType * pointState,
  IdxType * s_neighborBuffer, IdxType * s_neighborCount,
  bool * collisionMatrix, IdxType * seedList, IdxType * seedLength,
  IdxType coreThreshold, float rsq
) {
  float dx = xs[pointIdx] - px;
  float dy = ys[pointIdx] - py;
  if (dx * dx + dy * dy <= rsq) {
    int h = atomicAdd(s_neighborCount, 1) + 1;
    if (h >= coreThreshold) {
      markAsCandidate(pointIdx, collisionMatrix, pointState, seedList, seedLength);
    } else {
      s_neighborBuffer[h] = pointIdx;
    }
  }
}

// len(seedLists) must equal maxSeedLength * maxNumberOfThreadGroups
// shared memory required: ( (coreThreshold + 127) / 128 * 128 + 1 ) * sizeof(IdxType)
static __global__ void kernel_clusterExpansion(
  IdxType * pointState, bool * collisionMatrix,
  float * xs, float * ys, IdxType n,
  IdxType * seedLists, IdxType * seedLengths,
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

    float x = xs[seedPointIdx], y = ys[seedPointIdx];    
    {
      IdxType strideIdx = 0;
      for (; strideIdx < (n - 1) / stride; ++strideIdx) {
        processObject(
          x, y, xs, ys, strideIdx * stride + threadIdx.x, pointState, neighborBuffer, neighborCount, collisionMatrix, &seedLists[maxSeedLength * threadGroupIdx], &seedLengths[threadGroupIdx], coreThreshold, rsq
        );
      }
      if (threadIdx.x < n - strideIdx * stride) {
        processObject(
          x, y, xs, ys, strideIdx * stride + threadIdx.x, pointState, neighborBuffer, neighborCount, collisionMatrix, &seedLists[maxSeedLength * threadGroupIdx], &seedLengths[threadGroupIdx], coreThreshold, rsq
        );
      }
    }

    __syncthreads();

    if (*neighborCount >= coreThreshold) {
      if (threadIdx.x == 0) pointState[seedPointIdx] = cChain + threadGroupIdx;
      for (int i = threadIdx.x; i < coreThreshold; i += stride) {
        markAsCandidate(neighborBuffer[i], collisionMatrix, pointState, &seedLists[maxSeedLength * threadGroupIdx], &seedLengths[threadGroupIdx]);
      }
    } else {
      if (threadIdx.x == 0) pointState[seedPointIdx] = cNoise;
    }

  }

}

void allocateDeviceMemory(
  IdxType ** d_pointStates,
  IdxType ** d_seedLists, IdxType ** d_seedLengths,
  bool ** d_collisionMatrix,
  int nBlocks,
  IdxType n
) {
  CUDA_CHECK(cudaMalloc(d_pointStates, n * sizeof(IdxType)))
  // TODO: Change later
  CUDA_CHECK(cudaMemset(*d_pointStates, 0, n * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(d_seedLists, nBlocks * maxSeedLength * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(d_seedLengths, nBlocks * sizeof(IdxType)))
  CUDA_CHECK(cudaMalloc(d_collisionMatrix, nBlocks * (nBlocks + 1) / 2 * sizeof(bool)))
  CUDA_CHECK(cudaMemset(*d_collisionMatrix, 0, nBlocks * (nBlocks + 1) / 2 * sizeof(bool)))
}

static __global__ void kernel_populateSeedLists(
  IdxType * d_seedLists, IdxType * d_seedLengths, IdxType nLists
) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < nLists; i += stride) {
    d_seedLengths[i] = 1;
    d_seedLists[i * maxSeedLength] = i; // TODO: Change later!!!
  }
}

void findClusters(
  IdxType * d_pointStates, bool * d_collisionMatrix,
  float * xs, float * ys, IdxType n,
  IdxType * d_seedLists, IdxType * d_seedLengths,
  IdxType coreThreshold, float rsq
) {
  constexpr int nBlocks = 6;
  constexpr int nThreadsPerBlock = 256;

  kernel_populateSeedLists <<<dim3(1), dim3(nThreadsPerBlock)>>> (d_seedLists, d_seedLengths, nBlocks);
  kernel_clusterExpansion <<<dim3(nBlocks), dim3(nThreadsPerBlock), ( (coreThreshold * sizeof(IdxType) + 127) / 128 * 128 + sizeof(IdxType) ) >>> (
    d_pointStates, d_collisionMatrix, xs, ys, n, d_seedLists, d_seedLengths, coreThreshold, rsq
  );
}

