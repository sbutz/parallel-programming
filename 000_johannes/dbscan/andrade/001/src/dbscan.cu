#include "dbscan.h"

#include "cuda_helpers.h"

#include <cuda.h>
#include <cuda_runtime.h>

using Count = size_t;

static __device__ void countForPoint (
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
  dcounts[idx] = cnt - 1; // nobody is oneself's neighbour, so subtract 1
}

static __global__ void countNeighborsKernel (
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

void countNeighbors(
  Count * dcounts,
  float * xs, float * ys, Count n,
  float r
) {
  constexpr unsigned int nThreadsPerBlock = 256;

  float * d_xs = nullptr, * d_ys = nullptr;
  Count * d_dcounts = nullptr;

  CUDA_CHECK(cudaMalloc (&d_xs, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc (&d_ys, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc (&d_dcounts, n * sizeof(Count)));
  CUDA_CHECK(cudaMemcpy (d_xs, xs, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy (d_ys, ys, n * sizeof(float), cudaMemcpyHostToDevice));

  dim3 dimBlock(nThreadsPerBlock, 1, 1);
  dim3 dimGrid((n + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
  countNeighborsKernel <<<dimGrid, dimBlock>>> (d_dcounts, d_xs, d_ys, n, r);
	CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy (dcounts, d_dcounts, n * sizeof(Count), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree (d_dcounts));
  CUDA_CHECK(cudaFree (d_ys));
  CUDA_CHECK(cudaFree (d_xs));
}

void countNeighborsCpu(
  Count * dcounts,
  float * xs, float * ys, Count n,
  float r
) {
  float rsq = r * r;
  for (size_t i = 0; i < n; ++i) {
    Count cnt = 0;
    for (size_t j = 0; j < n; ++j) {
      float xd = xs[i] - xs[j];
      float yd = ys[i] - ys[j];
      cnt += (xd * xd + yd * yd <= rsq);
    }
    dcounts[i] = cnt - 1;
  }
}
