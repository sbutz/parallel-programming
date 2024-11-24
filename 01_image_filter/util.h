#pragma once

#include <cuda.h>
#include <iostream>

#define ASSERT(ans, msg) Assert((ans), msg, __FILE__, __LINE__);
inline void Assert(bool v, const char *message, const char *file, int line) {
    if (!v) {
        std::cerr << "Error: " << message << " " << file << ":" << line << std::endl;
        std::abort();
    }
}

#define CUDA_ASSERT(ans) CudaAssert((ans), __FILE__, __LINE__);
inline void CudaAssert(cudaError_t code, const char *file, int line) {
    Assert(code == cudaSuccess, cudaGetErrorString(code), file, line);
}

void cudaInit() {
    // The first access of the gpu is initializing the driver etc.
    // To untaint the performance reports to capture all startup overhead in cudaFree().
    // See ../REAMDE.md.
    CUDA_ASSERT(cudaFree(0));
}

// Prefer using nvprof (or nsys profile)
#define CUDA_TRACE_START(key)                                                                      \
    float time_##key;                                                                              \
    cudaEvent_t start_##key, end_##key;                                                            \
    CUDA_ASSERT(cudaEventCreate(&start_##key));                                                    \
    CUDA_ASSERT(cudaEventCreate(&end_##key));                                                      \
    CUDA_ASSERT(cudaEventRecord(start_##key, 0))

#define CUDA_TRACE_END(key)                                                                        \
    CUDA_ASSERT(cudaEventRecord(end_##key, 0));                                                    \
    CUDA_ASSERT(cudaEventSynchronize(end_##key));                                                  \
    CUDA_ASSERT(cudaEventElapsedTime(&time_##key, start_##key, end_##key));                        \
    std::cout << "TRACE: " << #key << " " << time_##key << "ms" << std::endl

#define CUDA_TRACE(key, ans)                                                                       \
    {                                                                                              \
        CUDA_TRACE_START(key);                                                                     \
        ans;                                                                                       \
        CUDA_TRACE_END(key);                                                                       \
    }
