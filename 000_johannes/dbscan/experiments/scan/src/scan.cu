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


constexpr __host__ __device__ std::size_t uexp2(int n) {
  return (std::size_t)1 << n;
}



template <unsigned int WarpSize, unsigned int WarpsPerBlock>
struct Scan {

  static constexpr int ulog2(std::size_t n) {
    int res = 0;
    std::size_t mask = 1;
    while (n & ~mask) { mask = (mask << 1) + 1; ++res; }
    return res;
  }
  static_assert(ulog2(1) == 0, "");
  static_assert(ulog2(2) == 1, "");
  static_assert(ulog2(0x2a000) == 17, "");
  static_assert(ulog2((std::size_t)0 - (std::size_t)1) == 8 * sizeof(std::size_t) - 1, "");

  static constexpr unsigned int cWarpSize = WarpSize;
  static constexpr unsigned int cWarpsPerBlock = WarpsPerBlock;
  static constexpr unsigned int cBlockSize = cWarpSize * cWarpsPerBlock;
  static constexpr int cLogBlockSize = ulog2(cBlockSize);

  static_assert(cWarpsPerBlock <= cWarpSize,
    "This code requires that the per-warp sums within a block "
    "can be scanned by a single warp."
  );
  // assert: warpSize == cWarpSize
  // assert: blockDim.x = cWarpSize * cWarpsPerBlock


  // In phase 1a, we calculate per-warp sums and store them into shared memory.
  // Requirements:
  //   0 < n <= number of threads in grid
  //   len(sTemp) >= cWarpsPerBlock
  //   ( (n + 1) << logStep ) <= len(values)
  //
  // Precisely:
  // Let step := 2 ** logStep.
  // Let a[i] := values[(step - 1) + i * step)] = values[( (i + 1) << logStep ) - 1].
  //   This requires that (step - 1) + n * step < len(values),
  //   which is equivalent to ( (n + 1) << logStep ) <= len(values).
  // Let w(i) := ent(i / cWarpSize) be the number of the warp which i belongs to.
  // Let b(w) := ent(w / cBlockSize) be the number of the block which warp w belongs to.
  // Then, for w := 0, ..., floor( (n - 1) / cWarpSize )
  //   sTemp[[b(w)]][w % cWarpsPerBlock] := sum_{k = cWarpSize * w}^{cWarpSize * (w + 1) - 1}
  //                 (k < n) ? a[k] : 0.
  static __device__ void phase1a (
    float * sTemp,
    float * values,
    std::size_t n,
    int logStep
  ) {
    unsigned int const tid = cBlockSize * blockIdx.x + threadIdx.x; // index of thread in grid
    unsigned int const wid = threadIdx.x / cWarpSize; // index of warp in block
    unsigned int const lane = threadIdx.x % cWarpSize; // index of thread in warp

    // We always employ entire warps.
    // -> The divisions are *necessary*!
    if (tid / cWarpSize <= (n - 1) / cWarpSize) {
      // We calculate the sum for each warp and store these sums in shared memory.
      auto v = tid < n ? values[(((std::size_t)tid + 1) << logStep) - 1] : 0;
      v = sumToLastOfWarpSync<cWarpSize>(v);
      if (lane == cWarpSize - 1) sTemp[wid] = v;
    }
  }

  // In phase 1b, we perform a scan over the per-warp sums in shared memory,
  //   calculating per-block sums.
  static __device__ float phase1b (float * sTemp) {
    unsigned int const wid = threadIdx.x / cWarpSize; // index of warp in block

    float returnValue = 0.0f;

    static_assert(cWarpsPerBlock <= cWarpSize, "Only the very first warp will do the job.");
    if (threadIdx.x < cWarpsPerBlock) {
      float v = sTemp[threadIdx.x];
      v = scanPerWarpSync<cWarpSize, cWarpsPerBlock> (v);
      sTemp[threadIdx.x] = v;
      returnValue = v;
    }

    return returnValue;
  }

  static __device__ float phase1 (
    float * sTemp,
    float * values,
    std::size_t n,
    int logStep
  ) {
    phase1a(sTemp, values, n, logStep);
    __syncthreads();
    return phase1b(sTemp);
  }

  // In phase 2, we scan over milestones in global memory.
  static __device__ void phase2 (
    float * dest,
    std::size_t n,
    int logStep,
    float * sTemp,
    float blockBaseValue,
    float * values
  ) {
    unsigned int const tid = cBlockSize * blockIdx.x + threadIdx.x; // index of thread in grid

    // We always employ entire warps.
    // -> The divisions are *necessary*!
    if (tid / cWarpSize <= (n - 1) / cWarpSize) {
      auto warpBaseValue = threadIdx.x >= cWarpSize ? sTemp[threadIdx.x / cWarpSize - 1] : 0;
      auto v = tid < n ? values[(((std::size_t)tid + 1) << logStep) - 1] : 0;

      v = scanPerWarpSync<cWarpSize>(v);

      if (tid < n && threadIdx.x != cBlockSize - 1) {
        dest[(((std::size_t)tid + 1) << logStep) - 1] = blockBaseValue + warpBaseValue + v;
      }
    }
  }

  static __device__ void scanSingleStride(
    float * dest, std::size_t n, std::size_t nWriteable,
    bool haveMilestoneAtMinusOne,
    float * sTemp,
    float * values
  ) {
    unsigned int const tid = cBlockSize * blockIdx.x + threadIdx.x;
    unsigned int const tidOfFirstInBlock = (tid / cBlockSize) * cBlockSize;
    unsigned int const tidOfLastInBlock = tidOfFirstInBlock + cBlockSize - 1;

    auto grid = cooperative_groups::this_grid();

    float blockSum = phase1(sTemp, values, n, 0);

    if (tidOfLastInBlock < n) {
      if (threadIdx.x == cWarpsPerBlock - 1) dest[tidOfLastInBlock] = blockSum;
    }

    if (n > cBlockSize) {
      int logStep = 0;
      float * currentSTemp = sTemp;

      for (;;) {
        grid.sync();

        logStep += cLogBlockSize;
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

        float blockSum = phase1(
          currentSTemp, dest, nPoints, logStep
        );

        if (nPoints <= cBlockSize) break;

        if (tidOfLastInBlock < nPoints) {
          if (threadIdx.x == cWarpsPerBlock - 1) dest[( (tidOfLastInBlock + 1) << logStep ) - 1] = blockSum;
        }
      }

      grid.sync();

      do {
        std::size_t nPoints = n >> logStep;

        if (tidOfFirstInBlock < nPoints) {

          // We *cannot* use array notation here, since the index might be -1.
          // If the index type is unsigned, we get a wrong result!
          float blockBaseValue = (haveMilestoneAtMinusOne || blockIdx.x > 0) ? *( dest + (tidOfFirstInBlock << logStep) - 1 ) : 0;
          phase2(
            dest, nPoints, logStep, currentSTemp, blockBaseValue, dest
          );

        }

        grid.sync();

        logStep -= cLogBlockSize;
        currentSTemp -= cWarpsPerBlock;
      } while (logStep);

    }

    if (tidOfFirstInBlock < n) {

      // Again, we cannot use array notation, since the index might be -1.
      float blockBaseValue = (haveMilestoneAtMinusOne || blockIdx.x > 0) ? *(dest + tidOfFirstInBlock - 1) : 0;
      phase2(dest, n, 0, sTemp, blockBaseValue, values);

    }
  }
};

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
  auto grid = cooperative_groups::this_grid();

  if (n > stride) {
    for (; s < n - stride; s += stride) {
      Scan<cWarpSize, cWarpsPerBlock>::scanSingleStride(
        dest + s, stride, n, !firstStride, temp, values + s
      );
      firstStride = false;

      grid.sync();
    }
  }
  if (n - s > 0)  {
    Scan<cWarpSize, cWarpsPerBlock>::scanSingleStride(
      dest + s, n - s, n, !firstStride, temp, values + s
    );
  }

  grid.sync();
}

constexpr int cWarpSize = 32;
constexpr int cWarpsPerBlock = 8;
constexpr int cBlockSize = cWarpSize * cWarpsPerBlock;
constexpr std::size_t nSampleData = 500 * cBlockSize * cBlockSize;

std::array<float, nSampleData> sampleData;
std::array<float, nSampleData> result;
std::array<float, nSampleData> cpuResult;

int main() {
  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void *)kernel_scan<cWarpSize, cWarpsPerBlock>, 
  cWarpSize * cWarpsPerBlock, 4 * cWarpsPerBlock * sizeof(float));
  std::cerr << numBlocksPerSm << "\n";

  for (std::size_t i = 0; i < nSampleData; ++i) sampleData[i] = (int)(i & 0x7fffffff) % 1779 - 1779/2;

  DeviceVector<float> d_data(sampleData);
  float * d_result;
  CUDA_CHECK(cudaMalloc(&d_result, nSampleData * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(&result, d_result, nSampleData * sizeof(float), cudaMemcpyDeviceToHost))

  auto a1 = d_result;
  auto a2 = nSampleData;
  auto a3 = d_data.data();
  void * kernelArgs [] = { (void *)&a1, (void *)&a2, (void *)&a3 };

  std::size_t stride = numBlocksPerSm * cWarpSize * cWarpsPerBlock;
  std::cerr << "Stride: " << stride << "\n";

  // calculate the amount of shared memory we will need
  //std::size_t nSharedFloats = 0;
  //for (std::size_t n = stride, step = 1; n != 0; n /= (step *= cBlockSize)) {
  //  nSharedFloats += cWarpsPerBlock;
  //}
  std::size_t nSharedFloats = 4 * cWarpsPerBlock;
  std::cerr << "nSharedFloats: " << nSharedFloats << " (= cWarpsPerBlock * " << nSharedFloats / cWarpsPerBlock << ")\n";
  cudaLaunchCooperativeKernel(
    (void *)kernel_scan<cWarpSize, cWarpsPerBlock>,
    dim3{(unsigned int)numBlocksPerSm}, dim3{cWarpSize * cWarpsPerBlock},
    kernelArgs, nSharedFloats * sizeof(float), 0
  );
  CUDA_CHECK(cudaGetLastError());

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