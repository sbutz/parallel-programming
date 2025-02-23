#include "build_graph.h"
#include "types.h"
#include "cuda_helpers.h"
#include <cuda.h>

// ********************************************************+*********************************************************************
// Neighbor counting
// ********************************************************+*********************************************************************

// Kernel counts the number of neighboring points including the point itself.
// If this number is >= coreThresholds, the number MINUS 1 is stored the array given by parameter counts.
// If the number is < coreThresholds, 0 is stored.
static __global__ void kernel_countNeighbors (
  IdxType * counts,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  auto countForPoint = [&] (IdxType ourPointIdx) {
    float x = xs[ourPointIdx];
    float y = ys[ourPointIdx];
    float rsq = r * r;
    IdxType cnt = 0;
    for (IdxType i = 0; i < n; ++i) {
      cnt += ( (xs[i] - x) * (xs[i] - x) + (ys[i] - y) * (ys[i] - y) <= rsq );
    }
    counts[ourPointIdx] = cnt >= coreThreshold ? cnt - 1 : 0;  
  };

  IdxType strideBegin = 0;
  if (n > stride) for (; strideBegin < n - stride; strideBegin += stride) countForPoint(strideBegin + tid);
  if (tid < n - strideBegin) countForPoint(strideBegin + tid);
}

void countNeighbors(
  IdxType * d_counts,
  float const * d_xs, float const * d_ys, IdxType n,
  IdxType coreThreshold, float r
) {
  constexpr unsigned int nThreadsPerBlock = 256;
  unsigned int nBlocks = (n + nThreadsPerBlock - 1) / nThreadsPerBlock;
  kernel_countNeighbors <<<nBlocks, nThreadsPerBlock>>> (d_counts, d_xs, d_ys, n, coreThreshold, r);
}

// ********************************************************+*********************************************************************
// Prefix Scan
// ********************************************************+*********************************************************************

static __global__ void kernel_prefixScanStep(
  IdxType * __restrict__ dest, IdxType * __restrict__ src, IdxType n, IdxType delta
) {
  // assert: delta <= n
  // assert: delta is a power of 2

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  IdxType s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      auto idx = s + tid;
      IdxType res = src[idx];
      if (idx & delta) res += src[( idx & ~(delta - 1u) ) - 1u];
      dest[idx] = res;
    }
  }
  if (tid < n - s) {
    auto idx = s + tid;
    IdxType res = src[idx];
    if (idx & delta) res += src[( idx & ~(delta - 1u) ) - 1u];
    dest[idx] = res;
  }
}

static void prefixScanOnDevice(IdxType ** res, IdxType * dest1, IdxType * dest2, IdxType * src, IdxType n) {
  using std::swap;

  if (n == 0) { *res = dest1; return; }

  constexpr unsigned int nThreadsPerBlock = 256;
  unsigned int nBlocks = (n + nThreadsPerBlock - 1) / nThreadsPerBlock;

  kernel_prefixScanStep <<<nBlocks, nThreadsPerBlock>>> (dest1, src, n, 1);
  // subtract 1 on both sides of the inequality to avoid overflow issues
  //   if leftmost bit of n is 1
  for (IdxType delta = 2; delta - 1 < n - 1; delta <<= 1) {
    swap(dest1, dest2);
    kernel_prefixScanStep <<<nBlocks, nThreadsPerBlock>>> (dest1, dest2, n, delta);
  }
  *res = dest1;
}

// ********************************************************+*********************************************************************
// Building the incidence lists
// ********************************************************+*********************************************************************

static __global__ void kernel_buildIncidenceLists (
  IdxType * __restrict__ listArray,
  float const * xs, float const * ys, IdxType const * __restrict__ cumulative, IdxType n,
  float r
) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  auto buildIncidenceListForPoint = [&] (IdxType listStartIdx, IdxType listEndIdx, IdxType ourPointIdx) {
    if (listEndIdx - listStartIdx > 0) {
      float x = xs[ourPointIdx];
      float y = ys[ourPointIdx];
      float rsq = r * r;

      IdxType currentListIdx = listStartIdx;
      for (IdxType i = 0; i < n; ++i) {
        float xd = xs[i] - x;
        float yd = ys[i] - y;
        if (xd * xd + yd * yd <= rsq && i != ourPointIdx) {
          listArray[currentListIdx] = i;
          ++currentListIdx;
        }
      }
    }
  };

  IdxType strideBegin = 0;
  if (n > stride) for (; strideBegin < n - stride; strideBegin += stride) {
    buildIncidenceListForPoint(cumulative[strideBegin + tid], cumulative[strideBegin + tid + 1], strideBegin + tid);
  }
  if (tid < n - strideBegin) 
    buildIncidenceListForPoint(cumulative[strideBegin + tid], cumulative[strideBegin + tid + 1], strideBegin + tid);
}

static void buildIncidenceListsOnDevice(
  IdxType * d_listArray,
  float const * d_xs, float const * d_ys, IdxType const * d_cumulative, IdxType n,
  float r
) {
  constexpr unsigned int nThreadsPerBlock = 256;
  unsigned int nBlocks = (n + nThreadsPerBlock - 1) / nThreadsPerBlock;
  kernel_buildIncidenceLists <<<nBlocks, nThreadsPerBlock>>> (d_listArray, d_xs, d_ys, d_cumulative, n, r);
}

// ********************************************************+*********************************************************************
// Graph construction and graph memory deallocation
// ********************************************************+*********************************************************************

DNeighborGraph buildNeighborGraph(
  BuildNeighborGraphProfile * profile,
  float const * d_xs, float const * d_ys, IdxType n,
  IdxType coreThreshold, float r
) {
  IdxType * d_counts;
  CUDA_CHECK(cudaMalloc(&d_counts, n * sizeof(IdxType)))

  profile->timeNeighborCount = runAndMeasureCuda(
    countNeighbors,
    d_counts, d_xs, d_ys, n, coreThreshold, r
  );

  IdxType * d_temp;
  CUDA_CHECK(cudaMalloc(&d_temp, 2 * n * sizeof(IdxType)))
  IdxType * d_dest1 = d_temp, * d_dest2 = d_dest1 + n;
  IdxType * s;
  profile->timePrefixScan = runAndMeasureCuda(
    prefixScanOnDevice,
    &s,
    d_dest1,
    d_dest2,
    d_counts,
    n
  );

  IdxType * d_startIndices;
  CUDA_CHECK(cudaMalloc(&d_startIndices, (n + 1) * sizeof(IdxType)))
  CUDA_CHECK(cudaMemset(d_startIndices, 0, sizeof(IdxType)))
  CUDA_CHECK(cudaMemcpy(d_startIndices + 1, s, n * (sizeof(IdxType)), cudaMemcpyDeviceToDevice))
  (void)cudaFree(d_temp);

  IdxType lenIncidenceAry;
  CUDA_CHECK(cudaMemcpy(&lenIncidenceAry, d_startIndices + n, sizeof(IdxType), cudaMemcpyDeviceToHost));

  IdxType * d_incidenceAry;
  CUDA_CHECK(cudaMalloc(&d_incidenceAry, lenIncidenceAry * sizeof(IdxType)))
  profile->timeBuildIncidenceList = runAndMeasureCuda(
    buildIncidenceListsOnDevice,
    d_incidenceAry, d_xs, d_ys, d_startIndices, n, r
  );

  return { n, lenIncidenceAry, d_counts, d_startIndices, d_incidenceAry };
}

void freeDNeighborGraph(DNeighborGraph & g) {
  (void)cudaFree(g.d_neighborCounts);
  (void)cudaFree(g.d_startIndices);
  (void)cudaFree(g.d_incidenceAry);
  g = DNeighborGraph{}; // avoid dangling pointers
}
