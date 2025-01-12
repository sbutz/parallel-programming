#include <cuda.h>

// assumption: n > 0
template <unsigned int WarpsPerBlock>
__global__ void kernel_scan(float * dest, std::size_t n, float * values) {
  unsigned int const tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int const stride = gridDim.x * blockDim.x;
  unsigned int const bDim = blockDim.x;
  unsigned int const wid = bDim / warpSize;
  unsigned int const lane = bDim % warpSize;
  unsigned int const warpsPerBlock = bDim / warpSize;

  // assert: bDim is integer multiple of warpSize
  // assert: bDim <= warpSize * warpSize

  // adjust dimension
  __shared__ float temp[WarpsPerBlock]; 

  using MaskType = unsigned int;

  // currently only works if warpSize divides n
  std::size_t const sFirst = tid;
  std::size_t const sLast = n > stride ? n - stride : 0;
  for (auto s = sFirst; s <= sLast; s += stride) {
    auto v = values[s];

    auto const mask = MaskType{1 << warpSize} - 1u;
    for (int w = 1; w != warpSize; w <<= 1) {
      v += __shfl_up_sync(mask, v, w);
    }

    if (lane + 1 == warpSize) temp[wid] = v;

    __syncthreads();

    if (threadIndex.x < WarpsPerBlock) {
      v = temp[threadIndex.x];
      for (int w = 1; w < warpsPerBlock; w <<= 1) {
        v += __shfl_up_sync(mask, v, w);
      }

      __syncwarp();

      temp[threadIndex.x] = v;

      if (threadIndex.x == WarpsPerBlock - 1) {
        dest[blockDim.x * (blockIdx.x + 1) - 1] = v;
      }
    }

    __syncthreads();


  }


}