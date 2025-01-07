#include <cuda.h>
#include <cuda_runtime.h>

using Count = size_t;

__device__ void countForPoint (
  Count * dcounts,
  std::size_t idx,
  float * xs, float * ys, Count n,
  float r
) {
  using size_t = std::size_t;
  float x = xs[idx];
  float y = ys[idx];
  float rsq = r * r;
  Count cnt = 0;
  for (size_t i = 0; i < n; ++i) {
    cnt += ( (xs[i] - x) * (xs[i] - x) + (ys[i] - y) * (ys[i] - y) <= rsq );
  }
  dcounts[idx] = cnt;
}

__global__ void countNeighborsKernel (
  Count * dcounts,
  float * xs, float * ys, Count n,
  float r
) {
  using size_t = std::size_t;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  size_t s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      countForPoint(dcounts, s + tid, xs, ys, n, r);
    }
  }
  if (tid < n - s) {
    countForPoint(dcounts, s + tid, xs, ys, n, r);
  }
}