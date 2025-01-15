#include "./device_vector.h"
#include <array>
#include <vector>


// according to:
//   https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/, Listing 9
__device__ std::size_t warpAggregatedAtomicAdd(std::size_t * d_cnt) {
  static_assert(sizeof(std::size_t) == sizeof(unsigned long long int), "");

  unsigned int lane = threadIdx.x % warpSize;
  unsigned int activeMask = __ballot_sync(__activemask(), 1);
  int leader = __ffs(activeMask) - 1;
  std::size_t res;
  if (lane == leader) res = atomicAdd((unsigned long long int *)d_cnt, __popc(activeMask));
  res = __shfl_sync(activeMask, res, leader);
  return res + __popc(activeMask & ((1 << lane) - 1));
}

__global__ void kernel_filter(
  float * d_positiveValues, std::size_t * d_cntPositive,
  float * d_negativeValues, std::size_t * d_cntNegative,
  float * values,
  std::size_t n
) {
  unsigned int const tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < n) {
    float v = values[tid];
    if (v > 0) {
      d_positiveValues[warpAggregatedAtomicAdd(d_cntPositive)] = v;
    };
  }
  if (tid < n) {
    float v = values[tid];
    if (v <= 0) {
      d_positiveValues[warpAggregatedAtomicAdd(d_cntPositive)] = v;
//      d_negativeValues[warpAggregatedAtomicAdd(d_cntNegative)] = v;
    }
  }
}

constexpr std::size_t nSampleData = 500000u;
//std::array<float, nSampleData> sampleData;

int main() {
  std::vector<float> sampleData(nSampleData);

  for (int i = 0; i < nSampleData; ++i) sampleData[i] = (i % 1779) - 1779/2;
  DeviceVector<float> d_data(sampleData);
  DeviceVector<float> d_positiveValues(nSampleData);
  DeviceVector<float> d_negativeValues(nSampleData);
  DeviceVector<std::size_t> d_cntPositive(1);
  DeviceVector<std::size_t> d_cntNegative(1);

  constexpr int nThreadsPerBlock = 8 * 32;
  dim3 dimGrid((nSampleData + nThreadsPerBlock - 1) / nThreadsPerBlock);
  dim3 dimBlock(nThreadsPerBlock);
  kernel_filter <<<dimGrid, dimBlock>>> (
    d_positiveValues.data(), d_cntPositive.data(),
    d_negativeValues.data(), d_cntNegative.data(),
    d_data.data(), nSampleData
  );
  cudaDeviceSynchronize();

  std::vector<float> positiveValues(nSampleData);
  d_positiveValues.memcpyToHost(positiveValues.data());
  std::size_t cntPositive;
  CUDA_CHECK(cudaMemcpy(&cntPositive, d_cntPositive.data(), sizeof(std::size_t), cudaMemcpyDeviceToHost));

  std::vector<float> negativeValues(nSampleData);
  d_negativeValues.memcpyToHost(negativeValues.data());
  std::size_t cntNegative;
  CUDA_CHECK(cudaMemcpy(&cntNegative, d_cntNegative.data(), sizeof(std::size_t), cudaMemcpyDeviceToHost));

  std::cerr << "Positive values: " << cntPositive << "\n";
  std::cerr << "Negative values: " << cntNegative << "\n";
  for (std::size_t i = 0; i < cntPositive; ++i) {
    if (!(positiveValues[i] > 0)) {
      std::cerr << "Non-positive value detected.\n";
      return 1;
    }
  }
  for (std::size_t i = 0; i < cntNegative; ++i) {
    if (negativeValues[i] > 0) {
      std::cerr << "Positive value detected.\n";
      return 1;
    }
  }

  return 0;
  
}
