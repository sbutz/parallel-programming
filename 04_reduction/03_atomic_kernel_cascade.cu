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
    std::size_t i = (blockIdx.x * blockDim.x) + tid;

    // Load data in shared memory
    sdata[tid] = 0;
    while (i < size) {
        sdata[tid] += in[i];
        i += blockDim.x * gridDim.x;
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
    if (threadIdx.x == 0)
        atomicAdd(out, sdata[0]);
}

void Reduce(std::size_t input_size, std::size_t cascade_size) {
    float *input;
    CUDA_ASSERT(cudaMalloc((void **)&input, input_size * sizeof(float)));

    std::size_t output_size = 1;
    float *output;
    CUDA_ASSERT(cudaMalloc((void **)&output, output_size * sizeof(float)));

    dim3 blockSize(N_THREADS, 1, 1);
    dim3 gridSize((input_size / cascade_size + blockSize.x - 1) / blockSize.x, 1, 1);

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
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " input_size cascade_size" << std::endl;
        return 1;
    }

    cudaInit();

    std::size_t input_size = std::atoi(argv[1]);

    std::size_t cascade_size = 32;
    if (argc == 3) {
        cascade_size = std::atoi(argv[2]);
    }

    for (auto i = 0; i < N_ITERATIONS; i++) {
        Reduce(input_size, cascade_size);
    }

    return 0;
}
