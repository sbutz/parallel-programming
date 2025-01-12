#include "device_vector.h"
#include "cuda_helpers.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cooperative_groups.h>

// return ceil(a / b) without overflow issues
constexpr __host__ __device__ __forceinline__ std::size_t ceilDiv(std::size_t a, std::size_t b) {
  return (a / b) + ((a / b) * b > 0);
}

// return result of ceil(a / b) < c without overflow issues
constexpr __host__ __device__ __forceinline__ bool ceilDivLt(std::size_t a, std::size_t b, std::size_t c) {
  return (a - 1) / b + 1 < c;
}

// return result of a < ceil(b / c) without overflow issues
constexpr __host__ __device__ __forceinline__ bool ltCeilDiv(std::size_t a, std::size_t b, std::size_t c) {
  return (b != 0) && (b - 1) / c >= a;
}

template <int cWarpSize = 32, int cWarpsPerBlock = 256 / 32>
__device__ void scanSingleStrideStep(
  float * dest, std::size_t n, 
  std::size_t step,
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

  if (ltCeilDiv(warpInGrid, n, cWarpSize)) {
    auto v = tid < n ? values[tid * step + (step - 1)] : 0;

    __syncwarp();

    auto const mask = MaskType{} - 1u;
    for (int w = 1; w != cWarpSize; w <<= 1) {
      float x = __shfl_up_sync(mask, v, w);
      if (lane >= w) v += x; 
    }

    if (lane == cWarpSize - 1) sTemp[wid] = v;

    __syncthreads();

    if (blockIdx.x < n / cBlockSize && wid == 0) {
      v = threadIdx.x < cWarpsPerBlock ? sTemp[threadIdx.x] : 0;
      __syncwarp();

      for (int w = 1; w < cWarpsPerBlock; w <<= 1) {
        float x = __shfl_up_sync(mask, v, w);
        if (lane >= w) v += x;
      }

      if (threadIdx.x < cWarpsPerBlock) sTemp[threadIdx.x] = v;

      if (threadIdx.x == cWarpsPerBlock - 1) {
        dest[cBlockSize * (blockIdx.x + 1) * step - 1] = v;
      }
    }

    __syncthreads();
  }
}

template <int cWarpSize = 32, int cWarpsPerBlock = 256 / 32>
__device__ void scanSingleStrideFillinStep(
  float * dest, std::size_t n, 
  std::size_t step,
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

  if (ltCeilDiv(warpInGrid, n, cWarpSize)) {
    auto basePrevious = blockIdx.x != 0 ? dest[blockIdx.x * cBlockSize * step - 1] : 0;
    auto baseFromShared = threadIdx.x >= cWarpSize ? sTemp[threadIdx.x / cWarpSize - 1] : 0;
    auto v = tid < n ? values[tid * step] : 0;
    __syncwarp();

    auto const mask = MaskType{} - 1u;
    for (int w = 1; w < cWarpSize; w <<= 1) {
      float x = __shfl_up_sync(mask, v, w);
      if (lane >= w) v += x;
    }
    __syncwarp();

    if (tid < n) dest[tid * step] = basePrevious + baseFromShared + v;
    __syncthreads();
  }
}


template <int cWarpSize = 32, int cWarpsPerBlock = 256 / 32>
__forceinline__ __device__ void scanSingleStrideSteps(
  float * dest, std::size_t n, 
  float * sTemp,
  float * values
) {
  constexpr unsigned int cBlockSize = cWarpSize * cWarpsPerBlock;

  std::size_t step = 1;
  std::size_t nn = n;
  float * currentSTemp = sTemp;
  float * vs = values;
  for (;;) {
    scanSingleStrideStep(dest, nn, step, currentSTemp, vs);
    auto grid = cooperative_groups::this_grid();
    grid.sync();

    if (nn / step == 0) break;
    step *= cBlockSize;
    nn /= step;
    currentSTemp += cWarpsPerBlock;
    vs = dest;
    //if (blockIdx.x == 0 && threadIdx.x == 0) printf("Step is %lu\n", step);
  }
  for (;;) {
    nn = n / step;
    if (blockIdx.x == 0 && threadIdx.x == 0) printf("Step is %lu\n", step);
    scanSingleStrideFillinStep(dest, nn, step, currentSTemp, values);
    auto grid = cooperative_groups::this_grid();
    grid.sync();

    if (step == 1) break;
    step /= cBlockSize;
    currentSTemp -= cWarpsPerBlock;
  }
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
  extern __shared__ float temp[]; 

  using MaskType = unsigned int;

  std::size_t s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      scanSingleStrideSteps(
        dest + s, stride, temp, values + s
      );
    }
  }
  if (n - s > 0)  {
    scanSingleStrideSteps(
      dest + s, n - s, temp, values + s
    );
  }
}

constexpr int cWarpSize = 32;
constexpr int cWarpsPerBlock = 8;
constexpr int cBlockSize = cWarpSize * cWarpsPerBlock;
constexpr std::size_t nSampleData = 5 * cBlockSize * cBlockSize;

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

  std::size_t stride = numBlocksPerSm * cWarpSize * cWarpsPerBlock;
  std::cerr << "Stride: " << stride << "\n";

  // calculate the amount of shared memory we will need
  std::size_t nSharedFloats = 0;
  for (std::size_t n = stride, step = 1; n != 0; n /= (step *= cBlockSize)) {
    nSharedFloats += cWarpsPerBlock;
  }

  std::cerr << "nSharedFloats: " << nSharedFloats << " (= cWarpsPerBlock * " << nSharedFloats / cWarpsPerBlock << ")\n";
  cudaLaunchCooperativeKernel(
    (void *)kernel_scan<cWarpSize, cWarpsPerBlock>,
    dim3{numBlocksPerSm}, dim3{cWarpSize * cWarpsPerBlock},
    kernelArgs, nSharedFloats * sizeof(float)
  );
  CUDA_CHECK(cudaGetLastError());

  d_result.memcpyToHost(result.data());

  constexpr auto cBlockSize = cWarpSize * cWarpsPerBlock;
  float s = 0;
  for (auto i = 0; i < nSampleData; ++i) {
    //if (!(i % cBlockSize)) s = 0; 
    s += sampleData[i]; cpuResult[i] = s;
  }
  for (int i = 0; i < 512; ++i)
    std::cerr << i << " " << result[i] << " " <<
    cpuResult[i] << '\n';
  return 0;
}