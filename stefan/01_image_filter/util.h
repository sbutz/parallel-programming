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

inline void GpuAssert(cudaError_t code)
{
    if (code != cudaSuccess) 
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(code) << std::endl;
        std::abort();
    }
}