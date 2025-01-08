#include "cuda_helpers.h"
#include <cuda.h>

using Count = std::size_t;

static __global__ void accumulationStep(
  Count * dest, Count * src, std::size_t n, std::size_t delta
) {
  // assert: delta <= n
  // assert: delta is a power of 2

  using size_t = std::size_t;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  size_t s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      auto idx = s + tid;
      Count res = src[idx];
      if (idx & delta) res += src[( idx & ~(delta - 1u) ) - 1u];
      dest[idx] = res;
    }
  }
  if (tid < n - s) {
    auto idx = s + tid;
    Count res = src[idx];
    if (idx & delta) res += src[( idx & ~(delta - 1u) ) - 1u];
    dest[idx] = res;
  }
}

std::size_t accumulateOnDevice(Count * ary, std::size_t n) {
  // dest must have space for 2*n elements
  constexpr unsigned int nThreadsPerBlock = 256;

  dim3 dimBlock(nThreadsPerBlock, 1, 1);
  dim3 dimGrid((n + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);

  std::size_t delta = 1, s = 0, t = s ^ n;
  for (; delta <= n; delta <<= 1, s ^= n, t ^= n) {
    accumulationStep <<<dimGrid, dimBlock>>> (&ary[t], &ary[s], n, delta);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
  }
  return s;
}

void accumulate(Count * dest, Count * src, std::size_t n) {
  Count * d_ary = nullptr;

  CUDA_CHECK(cudaMalloc(&d_ary, 2 * n * sizeof(Count)));
  CUDA_CHECK(cudaMemcpy(d_ary, src, n * sizeof(Count), cudaMemcpyHostToDevice));
  auto s = accumulateOnDevice(d_ary, n);
  CUDA_CHECK(cudaMemcpy(dest, &d_ary[s], n * sizeof(Count), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_ary));
}

void accumulateCpu(Count * dest, Count * src, std::size_t n) {
  if (n == 0) return;
  dest[0] = src[0];
  for (std::size_t i = 1; i < n; ++i) {
    dest[i] = src[i] + dest[i - 1];
  }
}