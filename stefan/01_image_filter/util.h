#pragma once

#include <iostream>

inline void Assert(bool v, const char* message)
{
    if (!v)
    {
        std::cerr << "Error: " << message << std::endl;
        std::abort();
    }
}

#define GpuAssert(ans) { GpuAssert_((ans), __FILE__, __LINE__); }
inline void GpuAssert_(cudaError_t code, const char * file, int line)
{
    if (code != cudaSuccess) 
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        std::abort();
    }
}