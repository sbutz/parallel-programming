#pragma once

#include <iostream>

#define ASSERT(ans, msg) Assert((ans), msg, __FILE__, __LINE__);
inline void Assert(bool v, const char* message, const char* file, int line)
{
    if (!v)
    {
        std::cerr << "Error: " << message << " " << file << ":" << line << std::endl;
        std::abort();
    }
}

#define CUDA_ASSERT(ans) CudaAssert((ans), __FILE__, __LINE__);
inline void CudaAssert(cudaError_t code, const char * file, int line)
{
    Assert(code == cudaSuccess, cudaGetErrorString(code), file, line);
}
