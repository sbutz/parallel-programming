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

__device__ void WarpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void Sum(float *in, float *out, std::size_t size) {
    __shared__ float sdata[N_THREADS];
    std::size_t tid = threadIdx.x;
    std::size_t i = (blockIdx.x * blockDim.x) + tid;

    // Load data in shared memory
    sdata[tid] = 0;
    for (auto i = (blockIdx.x * blockDim.x) + tid; i < size; i += blockDim.x * gridDim.x) {
        sdata[tid] += in[i];
    }
    __syncthreads();

    // Local Sum
    for (std::size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        WarpReduce(sdata, tid);
    }

    // Global Sum
    if (threadIdx.x == 0)
        atomicAdd(out, sdata[0]);
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
    for (auto i = 0; i < N_ITERATIONS; i++) {
        Reduce(input_size);
    }

    return 0;
}
