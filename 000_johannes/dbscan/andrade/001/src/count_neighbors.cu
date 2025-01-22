#include "count_neighbors.h"

#include "types.h"
#include "cuda_helpers.h"

#include <cuda.h>
#include <cuda_runtime.h>

static __device__ void countForPoint (
  IdxType * dcounts,
  IdxType idx,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  float x = xs[idx];
  float y = ys[idx];
  float rsq = r * r;
  IdxType cnt = 0;
  for (IdxType i = 0; i < n; ++i) {
    cnt += ( (xs[i] - x) * (xs[i] - x) + (ys[i] - y) * (ys[i] - y) <= rsq );
  }
  --cnt;  // nobody is oneself's neighbour, so subtract 1
  dcounts[idx] = cnt >= coreThreshold ? cnt : 0;
}

static __global__ void countNeighborsKernel (
  IdxType * dcounts,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  IdxType s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      countForPoint(dcounts, s + tid, xs, ys, n, coreThreshold, r);
    }
  }
  if (tid < n - s) {
    countForPoint(dcounts, s + tid, xs, ys, n, coreThreshold, r);
  }
}

void countNeighborsOnDevice(
  IdxType * d_dcounts,
  float const * d_xs, float const * d_ys, IdxType n,
  IdxType coreThreshold, float r
) {
  constexpr unsigned int nThreadsPerBlock = 256;

  dim3 dimBlock(nThreadsPerBlock, 1, 1);
  dim3 dimGrid((n + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
  countNeighborsKernel <<<dimGrid, dimBlock>>> (d_dcounts, d_xs, d_ys, n, coreThreshold, r);
	CUDA_CHECK(cudaGetLastError());
}

void countNeighbors(
  IdxType * dcounts,
  float * xs, float * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  float * d_xs = nullptr, * d_ys = nullptr;
  IdxType * d_dcounts = nullptr;

  CUDA_CHECK(cudaMalloc (&d_xs, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc (&d_ys, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc (&d_dcounts, n * sizeof(IdxType)));
  CUDA_CHECK(cudaMemcpy (d_xs, xs, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy (d_ys, ys, n * sizeof(float), cudaMemcpyHostToDevice));

  countNeighborsOnDevice(d_dcounts, d_xs, d_ys, n, coreThreshold, r);

  CUDA_CHECK(cudaMemcpy (dcounts, d_dcounts, n * sizeof(IdxType), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree (d_dcounts));
  CUDA_CHECK(cudaFree (d_ys));
  CUDA_CHECK(cudaFree (d_xs));
}

void countNeighborsCpu(
  IdxType * dcounts,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  float rsq = r * r;
  for (IdxType i = 0; i < n; ++i) {
    IdxType cnt = 0;
    for (IdxType j = 0; j < n; ++j) {
      float xd = xs[i] - xs[j];
      float yd = ys[i] - ys[j];
      cnt += (xd * xd + yd * yd <= rsq);
    }
    --cnt;
    dcounts[i] = (cnt >= coreThreshold) ? cnt : 0;
  }
}
