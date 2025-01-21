#include "accumulate.h"
#include "types.h"
#include "cuda_helpers.h"
#include <cuda.h>

static __global__ void accumulationStep(
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

IdxType accumulateOnDevice(IdxType * ary, IdxType n) {
  // dest must have space for 2*n elements
  constexpr unsigned int nThreadsPerBlock = 256;

  dim3 dimBlock(nThreadsPerBlock, 1, 1);
  dim3 dimGrid((n + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);

  IdxType delta = 1, s = 0, t = s ^ n;
  for (; delta <= n; delta <<= 1, s ^= n, t ^= n) {
    accumulationStep <<<dimGrid, dimBlock>>> (&ary[t], &ary[s], n, delta);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
  }
  return s;
}

void accumulate(IdxType * dest, IdxType * src, IdxType n) {
  IdxType * d_ary = nullptr;

  CUDA_CHECK(cudaMalloc(&d_ary, 2 * (n + 1) * sizeof(IdxType)));
  CUDA_CHECK(cudaMemset(&d_ary[0], 0, sizeof(IdxType)));
  CUDA_CHECK(cudaMemcpy(&d_ary[1], src, n * sizeof(IdxType), cudaMemcpyHostToDevice));
  auto s = accumulateOnDevice(d_ary, n + 1);
  CUDA_CHECK(cudaMemcpy(dest, &d_ary[s], (n + 1) * sizeof(IdxType), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_ary));
}

void accumulateCpu(IdxType * dest, IdxType * src, IdxType n) {
  dest[0] = 0;
  if (n == 0) return;
  for (IdxType i = 1; i <= n; ++i) {
    dest[i] = src[i - 1] + dest[i - 1];
  }
}