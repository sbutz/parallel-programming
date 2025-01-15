#include "./device_vector.h"
#include <array>
#include <vector>


// according to:
//   https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/, Listing 9
__device__ std::size_t warpAggregatedAtomicAdd(std::size_t * d_cnt) {
  static_assert(sizeof(std::size_t) == sizeof(unsigned long long int), "");

  unsigned int lane = threadIdx.x % warpSize;
  unsigned int activeMask = __match_any_sync(__activemask(), (unsigned long int)d_cnt);
  int leader = __ffs(activeMask) - 1;
  std::size_t res;
  if (lane == leader) res = atomicAdd((unsigned long long int *)d_cnt, __popc(activeMask));
  res = __shfl_sync(activeMask, res, leader);
  return res + __popc(activeMask & ((1 << lane) - 1));
}

__global__ void kernel_pivotPartition(
  float * d_partitioned, float * d_equal,
  std::size_t * d_cntLower, std::size_t * d_cntUpper, std::size_t * d_cntEqual,
  float * values,
  std::size_t n,
  float pivot
) {
  unsigned int const tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < n) {
    float v = values[tid];
    auto cnt = v < pivot ? d_cntLower : v > pivot ? d_cntUpper : d_cntEqual;
    auto idx = warpAggregatedAtomicAdd(cnt);
    if (v < pivot) {
      d_partitioned[idx] = v;
    } else if (v > pivot) {
      d_partitioned[n - 1 - idx] = v;
    } else {
      d_equal[idx] = v;
    }
  }
}


constexpr std::size_t nSampleData = 500000u;
//std::array<float, nSampleData> sampleData;

int main() {
  std::vector<float> sampleData(nSampleData);

  for (int i = 0; i < nSampleData; ++i) sampleData[i] = (i % 1779) - 1779/2;
  DeviceVector<float> d_data(sampleData);
  DeviceVector<float> d_partitioned(nSampleData);
  DeviceVector<float> d_equal(nSampleData);
  DeviceVector<float> d_negativeValues(nSampleData);
  DeviceVector<std::size_t> d_cntLower(1);
  DeviceVector<std::size_t> d_cntUpper(1);
  DeviceVector<std::size_t> d_cntEqual(1);

  float pivot = 0.1f;

  constexpr int nThreadsPerBlock = 8 * 32;
  dim3 dimGrid((nSampleData + nThreadsPerBlock - 1) / nThreadsPerBlock);
  dim3 dimBlock(nThreadsPerBlock);
  kernel_pivotPartition <<<dimGrid, dimBlock>>> (
    d_partitioned.data(), d_equal.data(),
    d_cntLower.data(), d_cntUpper.data(), d_cntEqual.data(),
    d_data.data(), nSampleData, pivot
  );
  cudaDeviceSynchronize();

  std::vector<float> partitioned(nSampleData);
  std::size_t cntLower;
  std::size_t cntUpper;
  std::size_t cntEqual;
  CUDA_CHECK(cudaMemcpy(&cntLower, d_cntLower.data(), sizeof(std::size_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&cntUpper, d_cntUpper.data(), sizeof(std::size_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&cntEqual, d_cntEqual.data(), sizeof(std::size_t), cudaMemcpyDeviceToHost));
  d_partitioned.memcpyToHost(partitioned.data());
  CUDA_CHECK(cudaMemcpy(partitioned.data() + cntLower, d_equal.data(), cntEqual * sizeof(float), cudaMemcpyDeviceToHost));

  std::cerr << cntLower << "\n";
  std::cerr << cntUpper << "\n";

  int status = -1;
  for (std::size_t i = 0; i < nSampleData; ++i) {
    int newStatus = (partitioned[i] != pivot) * (1 - 2 * (partitioned[i] <= pivot));
    if (newStatus < status) {
      std::cerr << "Partitioning not correct " << i << " " << partitioned[i-1] << " " << partitioned[i] << "\n";
    }
    status = newStatus;
  }

  return 0;
  
}
