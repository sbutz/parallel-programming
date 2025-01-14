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

template <int cWarpSize, int nLanes = cWarpSize>
constexpr __host__ __device__ unsigned int getMask() {
  using MaskType = unsigned int;
  // slightly complicated to avoid "shift count is too large" warning
  return (nLanes == cWarpSize ? MaskType{0} : MaskType{1u << nLanes}) - MaskType{1};
}

// The function will perform the necessary synchronization itself.
template <int cWarpSize, int nLanes = cWarpSize>
__device__ float scanPerWarpSync(float v) {
  using MaskType = unsigned int;

  static_assert(nLanes <= cWarpSize, "");
  static_assert(8 * sizeof(MaskType) == cWarpSize, "");

  MaskType constexpr mask = getMask<cWarpSize, nLanes> ();

  unsigned int const lane = threadIdx.x % cWarpSize;
  for (int w = 1; w < nLanes; w <<= 1) {
    float x = __shfl_up_sync(mask, v, w);
    if (lane >= w) v += x;
  }
  return v;
}

// The function will perform the necessary synchronization itself.
template <int cWarpSize>
__device__ float sumToLastOfWarpSync(float v) {
  using MaskType = unsigned int;

  static_assert(8 * sizeof(MaskType) == cWarpSize, "");

  MaskType constexpr mask = MaskType{0} - MaskType{1};

  for (int w = 1; w < cWarpSize; w <<= 1) v += __shfl_up_sync(mask, v, w);
  return v;
}

// Thread cWarpsPerBlock - 1 returns the block sum, the other
//   threads return an undefined value
// block with index idx will handle the values at indices
//   {
//     step * blockDim.x * idx,
//     step * blockDim.x * idx + step,
//     ...,
//     step * blockDim.x * idx + step * (blockDim.x - 1)
//   }
// where step = 2 ** logStep
template <int cWarpSize, int cWarpsPerBlock>
__device__ float blockwiseScan (
  float * sTemp,
  float * values,
  std::size_t n,
  int logStep
) {
  // assert: warpSize == cWarpSize
  // assert: blockDim.x = cWarpSize * cWarpsPerBlock
  static_assert(cWarpsPerBlock <= cWarpSize, "");

  constexpr unsigned int cBlockSize = cWarpSize * cWarpsPerBlock;
  unsigned int const tid = cBlockSize * blockIdx.x + threadIdx.x;
  unsigned int const wid = threadIdx.x / cWarpSize;
  unsigned int const lane = threadIdx.x % cWarpSize;
  unsigned int const warpInGrid = tid / cWarpSize;

  float returnValue = 0.0f;

  // We always employ entire warps.
  if (warpInGrid <= n / cWarpSize) {
    // We calculate the sum for each warp and store these sums in shared memory.
    {
      auto v = tid < n ? values[(std::size_t)tid << logStep] : 0;

      v = sumToLastOfWarpSync<cWarpSize>(v);
      if (lane == cWarpSize - 1) sTemp[wid] = v;
    }
  }

  __syncthreads();

  if (warpInGrid <= n / cWarpSize) {
    // Now we have the sums per warp in shared memory.
    // We perform a scan over the shared memory, in order to get
    //   accumulated warpwise sums in shared memory.
    // The scan is performed by the first warp.
    if (threadIdx.x < cWarpsPerBlock) {
      static_assert(cWarpsPerBlock <= cWarpSize,
        "This code is only correct if the per-warp sums within a block "
        "can be scanned by a single warp."
      );

      float v = sTemp[threadIdx.x];
      
      v = scanPerWarpSync<cWarpSize, cWarpsPerBlock> (v);

      sTemp[threadIdx.x] = v;

      returnValue = v;
    }
  }
  return returnValue;
}


template <int cWarpSize, int cWarpsPerBlock>
__device__ void scanSingleStrideFillinStep(
  float * dest, std::size_t n, 
  int logStep,
  float * sTemp,
  float blockBaseValue,
  float * values
) {
  // assert: warpSize == cWarpSize
  // assert: blockDim.x = cWarpSize * cWarpsPerBlock
  static_assert(cWarpsPerBlock <= cWarpSize, "");

  constexpr unsigned int cBlockSize = cWarpSize * cWarpsPerBlock;

  unsigned int const tid = cBlockSize * blockIdx.x + threadIdx.x;
  unsigned int const warpInGrid = tid / cWarpSize;

  using MaskType = unsigned int;
  static_assert(8 * sizeof(MaskType) == cWarpSize, "");

  if (warpInGrid <= n / cWarpSize) {
    auto warpBaseValue = threadIdx.x >= cWarpSize ? sTemp[threadIdx.x / cWarpSize - 1] : 0;
    auto v = tid < n ? values[(std::size_t)tid << logStep] : 0;

    v = scanPerWarpSync<cWarpSize>(v);

    if (tid < n && threadIdx.x != cBlockSize - 1) {
      printf("Index: %lu\n", (std::size_t)tid << logStep);
      dest[(std::size_t)tid << logStep] = blockBaseValue + warpBaseValue + v;
    }
  }
}

constexpr __host__ __device__ std::size_t uexp2(int n) {
  return (std::size_t)1 << n;
}

constexpr __host__ __device__ int ulog2(std::size_t n) {
  int res = 0;
  std::size_t mask = 1;
  while (n & ~mask) { mask = (mask << 1) + 1; ++res; }
  return res;
}

static_assert(ulog2(1) == 0, "");
static_assert(ulog2(2) == 1, "");
static_assert(ulog2(0x2a000) == 17, "");

template <int cWarpSize, int cWarpsPerBlock>
__forceinline__ __device__ void scanSingleStrideSteps(
  float * dest, std::size_t n, std::size_t nWriteable,
  bool haveMilestoneAtMinusOne,
  float * sTemp,
  float * values
) {
  constexpr std::size_t cBlockSize = cWarpSize * cWarpsPerBlock;
  unsigned int const tid = cBlockSize * blockIdx.x + threadIdx.x;

  auto grid = cooperative_groups::this_grid();

  float blockSum = blockwiseScan<cWarpSize, cWarpsPerBlock>(sTemp, values, n, 0);

  if (threadIdx.x == cWarpsPerBlock - 1 && blockIdx.x < n / cBlockSize)
    dest[cBlockSize * (blockIdx.x + 1) - 1] = blockSum;

  if (n > cBlockSize) {
    int logStep = 0;
    float * currentSTemp = sTemp;

    for (;;) {
      grid.sync();

      logStep += ulog2(cBlockSize);
      currentSTemp += cWarpsPerBlock;

      // nPoints is defined by the number of whole steps of size "step"
      //   we can make in the interval [step - 1, n - 1].
      // This is the same as asking how many numbers in the interval
      // [step, n] are divisible by step.
      // We make use of the fact that step is a power of 2.
      // Further, step <= n.
      //
      // E.g., in binary,
      //    n = 10110100
      //    step = 1000 -> logStep = 3
      //    nPoints = n >> logStep = 10110
      std::size_t nPoints = n >> logStep;

      std::size_t step = uexp2(logStep);
      float blockSum = blockwiseScan<cWarpSize, cWarpsPerBlock>(
        currentSTemp, dest + step - 1, nPoints, logStep
      );

      if (nPoints <= cBlockSize) break;

      if (threadIdx.x == cWarpsPerBlock - 1 && blockIdx.x < nPoints / cBlockSize)
        dest[cBlockSize * step * (blockIdx.x + 1) - 1] = blockSum;
    }

    grid.sync();

    do {
      std::size_t nPoints = n >> logStep;
      if (nPoints < cBlockSize) printf("%lu %lu %d %d %u\n", n, nPoints, logStep, ulog2(cBlockSize), cBlockSize);

      float blockBaseValue = (haveMilestoneAtMinusOne || blockIdx.x > 0) && (blockIdx.x * cBlockSize <= nPoints)?
        dest[((blockIdx.x * cBlockSize) << logStep) - 1]
      : 0;
      scanSingleStrideFillinStep<cWarpSize, cWarpsPerBlock> (
        dest + uexp2(logStep) - 1, nPoints, logStep, currentSTemp, blockBaseValue, dest + uexp2(logStep) - 1
      );
      printf("%lu\n", uexp2(logStep));

      auto grid = cooperative_groups::this_grid();
      grid.sync();

      logStep -= ulog2(cBlockSize);
      currentSTemp -= cWarpsPerBlock;
    } while (logStep);
  }

  float blockBaseValue = haveMilestoneAtMinusOne || blockIdx.x > 0 ? dest[blockIdx.x * cBlockSize - 1] : 0;
  auto grid6 = cooperative_groups::this_grid();
  grid6.sync();
  scanSingleStrideFillinStep<cWarpSize, cWarpsPerBlock> (dest, n, 0, sTemp, blockBaseValue, values);
  auto grid7 = cooperative_groups::this_grid();
  grid7.sync();
}

// assumption: n > 0
template <int cWarpSize = 32, int cWarpsPerBlock = 256 / 32>
__global__ void kernel_scan(float * dest, std::size_t n, float * values) {
  // assert: warpSize == cWarpSize
  // assert: blockDim.x = cWarpSize * cWarpsPerBlock

  // assert: warpSize == cWarpSize
  // assert: blockDim.x = cWarpSize * cWarpsPerBlock
  static_assert(cWarpsPerBlock <= cWarpSize, "");

  unsigned int const stride = gridDim.x * blockDim.x;

  // adjust dimension
  extern __shared__ float temp[]; 

  using MaskType = unsigned int;


  bool firstStride = true;
  std::size_t s = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      scanSingleStrideSteps<cWarpSize, cWarpsPerBlock>(
        dest + s, stride, n, !firstStride, temp, values + s
      );
      firstStride = false;
      auto grid = cooperative_groups::this_grid();
      grid.sync();
    }
  }
  if (n - s > 0)  {
    scanSingleStrideSteps<cWarpSize, cWarpsPerBlock>(
      dest + s, n - s, n, !firstStride, temp, values + s
    );
  }
  auto grid = cooperative_groups::this_grid();
  grid.sync();

}

constexpr int cWarpSize = 32;
constexpr int cWarpsPerBlock = 4;
constexpr int cBlockSize = cWarpSize * cWarpsPerBlock;
constexpr std::size_t nSampleData = 100 * cBlockSize * cBlockSize;

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


  for (std::size_t i = 0; i < nSampleData; ++i) sampleData[i] = 2.0f * ((int)(i & 0xff) - 129) + 1;

  DeviceVector<float> d_data(sampleData);
  float * d_result;
  CUDA_CHECK(cudaMalloc(&d_result, nSampleData * sizeof(float)));


  auto a1 = d_result;
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
    dim3{(unsigned int)numBlocksPerSm}, dim3{cWarpSize * cWarpsPerBlock},
    kernelArgs, nSharedFloats * sizeof(float)
  );
  CUDA_CHECK(cudaGetLastError());

  float w;
//  CUDA_CHECK(cudaMemcpy(&w, &(d_result.data()[6208]), sizeof(float), cudaMemcpyDeviceToHost));

//  std::cerr << nSampleData << " " << w << "\n";

  CUDA_CHECK(cudaMemcpy(&result, d_result, nSampleData * sizeof(float), cudaMemcpyDeviceToHost))
  //d_result.memcpyToHost(result.data());
  

  float s = 0;
  for (auto i = 0; i < nSampleData; ++i) {
    //if (!(i % cBlockSize)) s = 0; 
    s += sampleData[i]; cpuResult[i] = s;
  }

  bool ok = true;
  for (int i = 0; i < nSampleData; ++i) {
    if (result[i] != cpuResult[i]) {
      ok = false;
      std::cerr << i-1 << " " << sampleData[i-1] << " " << result[i-1] << " " << cpuResult[i-1] << '\n';
      std::cerr << i << " " << sampleData[i] << " " << result[i] << " " << cpuResult[i] << '\n';
      return 1;
    }
  }
  if (!ok) std::cerr << "There were errors.\n";
  return !ok;
}