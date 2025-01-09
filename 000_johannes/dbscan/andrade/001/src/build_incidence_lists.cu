#include "device_vector.h"
#include "cuda_helpers.h"

using Count = std::size_t;

static __device__ void buildIncidenceListForPoint (
  Count * listArray, std::size_t listStartIdx,
  std::size_t pointIdx,
  float const * xs, float const * ys, Count n,
  float r
) {
  using size_t = std::size_t;
  float x = xs[pointIdx];
  float y = ys[pointIdx];
  float rsq = r * r;
  Count currentListIdx = listStartIdx;
  for (size_t i = 0; i < n; ++i) {
    float xd = xs[i] - x;
    float yd = ys[i] - y;
    if (xd * xd + yd * yd <= rsq && i != pointIdx) {
      listArray[currentListIdx] = i;
      ++currentListIdx;
    }
  }
}

static __global__ void buildIncidenceListsKernel (
  Count * listArray,
  float const * xs, float const * ys, Count const * cumulative, Count n,
  float r
) {
  using size_t = std::size_t;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  size_t s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      buildIncidenceListForPoint(listArray, cumulative[s + tid], s + tid, xs, ys, n, r);
    }
  }
  if (tid < n - s) {
    buildIncidenceListForPoint(listArray, cumulative[s + tid], s + tid, xs, ys, n, r);
  }
}

void buildIncidenceListsOnDevice(
  Count * d_listArray,
  float const * d_xs, float const * d_ys, Count const * d_cumulative, Count n,
  float r
) {
  constexpr unsigned int nThreadsPerBlock = 256;

  dim3 dimBlock(nThreadsPerBlock, 1, 1);
  dim3 dimGrid((n + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
  buildIncidenceListsKernel <<<dimGrid, dimBlock>>> (d_listArray, d_xs, d_ys, d_cumulative, n, r);
	CUDA_CHECK(cudaGetLastError());
}

void buildIncidenceLists(
  Count * listArray,
  float const * xs, float const * ys, Count const * cumulative, Count n,
  float r
) {
  DeviceVector<float> d_xs(xs, n);
  DeviceVector<float> d_ys(ys, n);
  DeviceVector<Count> d_cumulative(cumulative, n + 1);
  DeviceVector<Count> d_listArray(UninitializedDeviceVectorTag {}, cumulative[n]);

  buildIncidenceListsOnDevice(d_listArray.data(), d_xs.data(), d_ys.data(), d_cumulative.data(), n, r);

  d_listArray.memcpyToHost(listArray);
}

void buildIncidenceListsCpu(
  Count * listArray,
  float const * xs, float const * ys, Count const * cumulative, Count n,
  float r
) {
  float rsq = r * r;
  for (Count i = 0; i < n; ++i) {
    Count listArrayIdx = cumulative[i]; // TODO: not necessary!!!!
    for (Count j = 0; j < n; ++j) {
      float xd = xs[i] - xs[j];
      float yd = ys[i] - ys[j];
      if (xd * xd + yd * yd <= rsq && i != j) {
        listArray[listArrayIdx++] = j;
      }
    }
  }
}

