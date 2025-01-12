#include "device_vector.h"
#include "cuda_helpers.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cooperative_groups.h>

template <std::size_t logStep = 0, int cWarpSize = 32, int cWarpsPerBlock = 256 / 32>
__device__ void scanSingleStrideStep(
  float * dest, std::size_t n, 
  float * sTemp,
  float * values
) {
  // assert: warpSize == cWarpSize
  // assert: blockDim.x = cWarpSize * cWarpsPerBlock
  static_assert(cWarpsPerBlock <= cWarpSize, "");

  constexpr unsigned int cBlockSize = cWarpSize * cWarpsPerBlock;

  unsigned int const tid = cBlockSize * blockIdx.x + threadIdx.x;
  unsigned int const stride = gridDim.x * blockDim.x;
  unsigned int const wid = threadIdx.x / cWarpSize;
  unsigned int const lane = threadIdx.x % cWarpSize;
  unsigned int const warpInGrid = tid / cWarpSize;

  using MaskType = unsigned int;
  static_assert(8 * sizeof(MaskType) == cWarpSize, "");

  if (warpInGrid <= (n - 1) / cWarpSize) {
    auto v = (tid << logStep) < n ? values[tid << logStep] : 0;

    __syncwarp();

    auto const mask = MaskType{} - 1u;
    for (int w = 1; w != cWarpSize; w <<= 1) {
      v += __shfl_up_sync(mask, v, w);
    }

    if (lane == cWarpSize - 1) sTemp[wid] = v;

    __syncthreads();

    if (wid == 0) {
      v = threadIdx.x < cWarpsPerBlock ? sTemp[threadIdx.x] : 0;
      
      for (int w = 1; w < cWarpsPerBlock; w <<= 1) {
        v += __shfl_up_sync(mask, v, w);
      }

      __syncwarp();

      sTemp[threadIdx.x] = v;

      if (threadIdx.x == cWarpsPerBlock - 1) {
        dest[(cBlockSize * (blockIdx.x + 1) - 1) << logStep] = v;
      }
    }

    __syncthreads();
  }
}

template <int cWarpSize = 32, int cWarpsPerBlock = 256 / 32, std::size_t... logSteps>
__forceinline__ __device__ void scanSingleStrideSteps(
  float * dest, std::size_t n, 
  float * sTemp,
  float * values,
  std::index_sequence<logSteps...>
) {
  auto grid = cooperative_groups::this_grid();

  scanSingleStrideStep<0>(dest, n, sTemp, values);

  grid.sync();
  
  (void) std::initializer_list<int>{ 
    ((void)scanSingleStrideStep<logSteps>(dest, n, sTemp, dest), 0) ...
  };
}

// assumption: n > 0
template <int cWarpSize = 32, int cWarpsPerBlock = 256 / 32>
__global__ void kernel_scan(float * dest, std::size_t n, float * values) {
  // assert: warpSize == cWarpSize
  // assert: blockDim.x = cWarpSize * cWarpsPerBlock

  // assert: warpSize == cWarpSize
  // assert: blockDim.x = cWarpSize * cWarpsPerBlock
  static_assert(cWarpsPerBlock <= cWarpSize, "");

  constexpr unsigned int cBlockSize = cWarpSize * cWarpsPerBlock;

  unsigned int const tid = cBlockSize * blockIdx.x + threadIdx.x;
  unsigned int const stride = gridDim.x * blockDim.x;
  unsigned int const wid = threadIdx.x / cWarpSize;
  unsigned int const lane = threadIdx.x % cWarpSize;

  // adjust dimension
  __shared__ float temp[cWarpsPerBlock]; 

  using MaskType = unsigned int;

  std::size_t s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      scanSingleStrideSteps(
        dest + s, stride, temp, values + s,
        std::index_sequence<7> {}
      );
    }
  }
  if (n - s > 0)  {
    scanSingleStrideSteps(
      dest + s, n - s, temp, values + s,
      std::index_sequence<7> {}
    );
  }
}

constexpr int cWarpSize = 32;
constexpr int cWarpsPerBlock = 4;
constexpr int cBlocksPerGrid = 32;
constexpr int cStrideSize = cWarpSize * cWarpsPerBlock * cBlocksPerGrid;
constexpr std::size_t nSampleData = cStrideSize;

std::array<float, nSampleData> sampleData;
std::array<float, nSampleData> result;
std::array<float, nSampleData> cpuResult;

int main() {
  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void *)kernel_scan<cWarpSize, cWarpsPerBlock>, 
  cWarpSize * cWarpsPerBlock, 0);
  std::cerr << numBlocksPerSm << "\n";


  for (std::size_t i = 0; i < nSampleData; ++i) sampleData[i] = i;

  DeviceVector<float> d_data(sampleData);
  DeviceVector<float> d_result(nSampleData);

  auto a1 = d_result.data();
  auto a2 = nSampleData;
  auto a3 = d_data.data();
  void * kernelArgs [] = { (void *)&a1, (void *)&a2, (void *)&a3 };
  cudaLaunchCooperativeKernel(
    (void *)kernel_scan<cWarpSize, cWarpsPerBlock>,
    dim3{cBlocksPerGrid}, dim3{cWarpSize * cWarpsPerBlock},
    kernelArgs
  );
  CUDA_CHECK(cudaGetLastError());

  d_result.memcpyToHost(result.data());

  constexpr auto cBlockSize = cWarpSize * cWarpsPerBlock;
  float s = 0;
  for (auto i = 0; i < nSampleData; ++i) {
    //if (!(i % cBlockSize)) s = 0; 
    s += sampleData[i]; cpuResult[i] = s;
  }
  for (auto i : { 1, 2, 3, 4 }) std::cerr << i * cBlockSize - 1 << " " << result[i * cBlockSize - 1] << " " <<
    cpuResult[i * cBlockSize - 1] << '\n';
  return 0;
}