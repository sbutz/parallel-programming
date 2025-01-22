#include "types.h"
#include "device_vector.h"
#include "cuda_helpers.h"

static __device__ void buildIncidenceListForPoint (
  IdxType * listArray, IdxType listStartIdx, IdxType listEndIdx,
  IdxType pointIdx,
  float const * xs, float const * ys, IdxType n,
  float r
) {
  if (listEndIdx - listStartIdx > 0) {
    float x = xs[pointIdx];
    float y = ys[pointIdx];
    float rsq = r * r;

    IdxType currentListIdx = listStartIdx;
    for (IdxType i = 0; i < n; ++i) {
      float xd = xs[i] - x;
      float yd = ys[i] - y;
      if (xd * xd + yd * yd <= rsq && i != pointIdx) {
        listArray[currentListIdx] = i;
        ++currentListIdx;
      }
    }
  }
}

static __global__ void buildIncidenceListsKernel (
  IdxType * listArray,
  float const * xs, float const * ys, IdxType const * cumulative, IdxType n,
  float r
) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  IdxType s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      buildIncidenceListForPoint(listArray, cumulative[s + tid], cumulative[s + tid + 1], s + tid, xs, ys, n, r);
    }
  }
  if (tid < n - s) {
    buildIncidenceListForPoint(listArray, cumulative[s + tid], cumulative[s + tid + 1], s + tid, xs, ys, n, r);
  }
}

void buildIncidenceListsOnDevice(
  IdxType * d_listArray,
  float const * d_xs, float const * d_ys, IdxType const * d_cumulative, IdxType n,
  float r
) {
  constexpr unsigned int nThreadsPerBlock = 256;

  dim3 dimBlock(nThreadsPerBlock, 1, 1);
  dim3 dimGrid((n + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
  buildIncidenceListsKernel <<<dimGrid, dimBlock>>> (d_listArray, d_xs, d_ys, d_cumulative, n, r);
	CUDA_CHECK(cudaGetLastError());
}

void buildIncidenceLists(
  IdxType * listArray,
  float const * xs, float const * ys, IdxType const * cumulative, IdxType n,
  float r
) {
  DeviceVector<float> d_xs(xs, n);
  DeviceVector<float> d_ys(ys, n);
  DeviceVector<IdxType> d_cumulative(cumulative, n + 1);
  DeviceVector<IdxType> d_listArray(UninitializedDeviceVectorTag {}, cumulative[n]);

  buildIncidenceListsOnDevice(d_listArray.data(), d_xs.data(), d_ys.data(), d_cumulative.data(), n, r);

  d_listArray.memcpyToHost(listArray);
}

void buildIncidenceListsCpu(
  IdxType * listArray,
  float const * xs, float const * ys, IdxType const * cumulative, IdxType n,
  float r
) {
  float rsq = r * r;
  for (IdxType i = 0; i < n; ++i) {
    if (cumulative[i+1] - cumulative[i] == 0) continue; // non-core point
    auto listArrayIdx = cumulative[i];
    for (IdxType j = 0; j < n; ++j) {
      float xd = xs[i] - xs[j];
      float yd = ys[i] - ys[j];
      if (xd * xd + yd * yd <= rsq && i != j) {
        listArray[listArrayIdx++] = j;
      }
    }
  }
}

