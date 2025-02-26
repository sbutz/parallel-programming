#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#endif // __CUDACC__
#include <chrono>
#include <iostream>

#define ASSERT(ans, msg) Assert((ans), msg, __FILE__, __LINE__);
inline void Assert(bool v, const char *message, const char *file, int line) {
    if (!v) {
        std::cerr << "Error: " << message << " " << file << ":" << line << std::endl;
        std::abort();
    }
}

#ifdef __CUDACC__
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
#endif // __CUDACC__

#define TRACE_MSG(key, msg) std::cerr << "TRACE: " << #key << " " << msg << std::endl;

#define TRACE_START(key) auto time_start_##key = std::chrono::high_resolution_clock::now();

#define TRACE_END(key)                                                                             \
    auto time_end_##key = std::chrono::high_resolution_clock::now();                               \
    auto duration_##key =                                                                          \
        std::chrono::duration_cast<std::chrono::nanoseconds>(time_end_##key - time_start_##key);   \
    TRACE_MSG(key, duration_##key.count() << " ns");

#define TRACE(key, ans)                                                                            \
    TRACE_START(key);                                                                              \
    ans;                                                                                           \
    TRACE_END(key);

template <class size_type> size_type next_smaller_power_of_two(size_type n) {
    size_type p = 1;
    while (p < n) {
        p *= 2;
    }
    return p / 2;
}