#include "util.h"
#include <cstdlib>
#include <cuda.h>
#include <iostream>

static constexpr std::size_t N_ITERATIONS = 1024;
static constexpr std::size_t N_THREADS = 1024;
static constexpr std::size_t N_CASCADE = 32;

__global__ void Init(float *in, std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = gridDim.x * blockDim.x;

    for (; i < size; i += stride) {
        in[i] = 1.0;
    }
}

inline __device__ float WarpReduceSum(float val) {
    for (std::size_t offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

inline __device__ float BlockReduceSum(float val) {

    static __shared__ float shared[32]; // Storage each warp's local sum
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warpwise local sum
    val = WarpReduceSum(val);

    // Store the result of each warp
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) {
        val = WarpReduceSum(val); // Final reduce within first warp
    }

    return val;
}

__global__ void Sum(float *in, float *out, int size) {
    std::size_t tid = threadIdx.x;

    // Load data in shared memory
    float sum = 0;
    for (std::size_t i = (blockIdx.x * blockDim.x) + tid; i < size; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    sum = WarpReduceSum(sum);
    if ((threadIdx.x % warpSize) == 0) {
        atomicAdd(out, sum);
    }
}

void Reduce(std::size_t input_size) {
    float *input;
    CUDA_ASSERT(cudaMalloc((void **)&input, input_size * sizeof(float)));

    std::size_t output_size = 1;
    float *output;
    CUDA_ASSERT(cudaMalloc((void **)&output, output_size * sizeof(float)));

    dim3 blockSize(N_THREADS, 1, 1);
    dim3 gridSize((input_size / N_CASCADE + blockSize.x - 1) / blockSize.x, 1, 1);

    // Initialize Vector
    Init<<<gridSize, blockSize>>>(input, input_size);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    // Reduction
    // CUDA_TRACE_START(TIME_REDUCTION);
    Sum<<<gridSize, blockSize>>>(input, output, input_size);
    // CUDA_TRACE_END(TIME_REDUCTION);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    float result = 0;
    CUDA_ASSERT(cudaMemcpy(&result, output, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "Result: " << result << std::endl;
    ASSERT(result == input_size, "Wrong result");

    CUDA_ASSERT(cudaFree(input));
    CUDA_ASSERT(cudaFree(output));
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " input_size" << std::endl;
        return 1;
    }

    cudaInit();

    std::size_t input_size = std::atoi(argv[1]);
    for (std::size_t i = 0; i < N_ITERATIONS; i++) {
        Reduce(input_size);
    }

    return 0;
}
