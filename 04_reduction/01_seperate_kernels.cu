#include "util.h"
#include <cstdlib>
#include <cuda.h>
#include <iostream>

static constexpr std::size_t N_ITERATIONS = 1024;
static constexpr std::size_t N_THREADS = 1024;

__global__ void Init(float *in, std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = gridDim.x * blockDim.x;

    for (; i < size; i += stride) {
        in[i] = 1.0;
    }
}

__global__ void Sum(float *in, float *out, std::size_t size) {
    __shared__ float sdata[N_THREADS];
    std::size_t tid = threadIdx.x;
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data in shared memory
    if (i < size) {
        sdata[tid] = in[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Local Sum
    for (std::size_t s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Global Sum
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

void Reduce(std::size_t input_size) {
    float *input;
    CUDA_ASSERT(cudaMalloc((void **)&input, input_size * sizeof(float)));

    std::size_t output_size = input_size / N_THREADS;
    float *output;
    CUDA_ASSERT(cudaMalloc((void **)&output, output_size * sizeof(float)));

    dim3 blockSize(N_THREADS, 1, 1);
    dim3 gridSize((input_size + blockSize.x - 1) / blockSize.x, 1, 1);

    // Initialize Vector
    Init<<<gridSize, blockSize>>>(input, input_size);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    std::size_t remaining_size = input_size;
    while (remaining_size != 1) {
        gridSize = dim3{(remaining_size + blockSize.x - 1) / blockSize.x, 1, 1};
        Sum<<<gridSize, blockSize>>>(input, output, remaining_size);
        CUDA_ASSERT(cudaGetLastError());
        CUDA_ASSERT(cudaDeviceSynchronize());

        remaining_size = gridSize.x;
        std::swap(input, output);
    }

    float result = 0;
    CUDA_ASSERT(cudaMemcpy(&result, input, 1 * sizeof(float), cudaMemcpyDeviceToHost));
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