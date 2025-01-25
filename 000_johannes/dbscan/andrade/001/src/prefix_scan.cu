#include "prefix_scan.h"
#include "types.h"
#include "cuda_helpers.h"
#include <cuda.h>
#include <utility>

static __global__ void prefixScanStep(
  IdxType * dest, IdxType * src, IdxType n, IdxType delta
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

void prefixScanOnDevice(IdxType ** res, IdxType * dest1, IdxType * dest2, IdxType * src, IdxType n) {
  using std::swap;

  if (n == 0) { *res = dest1; return; }

  constexpr unsigned int nThreadsPerBlock = 256;
  dim3 dimBlock(nThreadsPerBlock, 1, 1);
  dim3 dimGrid((n + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);

  prefixScanStep <<<dimGrid, dimBlock>>> (dest1, src, n, 1);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  // subtract 1 on both sides of the inequality to avoid overflow issues
  //   if leftmost bit of n is 1
  for (IdxType delta = 2; delta - 1 < n - 1; delta <<= 1) {
    swap(dest1, dest2);
    prefixScanStep <<<dimGrid, dimBlock>>> (dest1, dest2, n, delta);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
  }
  *res = dest1;
  return;
}

void prefixScan(IdxType * dest, IdxType * src, IdxType n) {
  IdxType * d_ary = nullptr;

  CUDA_CHECK(cudaMalloc(&d_ary, 2 * n * sizeof(IdxType)))
  CUDA_CHECK(cudaMemcpy(d_ary + n, src, n * sizeof(IdxType), cudaMemcpyHostToDevice))
  IdxType * s;
  prefixScanOnDevice(&s, d_ary, d_ary + n, d_ary + n, n);
  CUDA_CHECK(cudaMemcpy(dest + 1, s, n * sizeof(IdxType), cudaMemcpyDeviceToHost))
  CUDA_CHECK(cudaMemset(dest, 0, sizeof(IdxType)))
  CUDA_CHECK(cudaFree(d_ary))
}

void prefixScanCpu(IdxType * dest, IdxType * src, IdxType n) {
  dest[0] = 0;
  if (n == 0) return;
  for (IdxType i = 1; i <= n; ++i) {
    dest[i] = src[i - 1] + dest[i - 1];
  }
}