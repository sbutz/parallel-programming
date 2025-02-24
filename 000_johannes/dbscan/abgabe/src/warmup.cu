#include "warmup.h"
#include "cuda_helpers.h"
#include <cuda.h>

using IdxType = unsigned int;

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

void warmup() {
  constexpr size_t sz64Mi = (size_t)1 << 26;
  IdxType * tempAry, * temp;
  CUDA_CHECK(cudaMalloc(&tempAry, 2 * sz64Mi * sizeof(IdxType) ))
  for (int i = 0; i < 5; ++i) {
    prefixScanOnDevice(
      &temp, tempAry, tempAry + sz64Mi, tempAry + sz64Mi, sz64Mi
    );
  }
  (void)cudaFree(tempAry);
}