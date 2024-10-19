# Profiling

## Tools
### CUDA Events
```cpp
float time;
cudaEvent_t start, end;
CUDA_ASSERT(cudaEventCreate(&start));
CUDA_ASSERT(cudaEventCreate(&end));
CUDA_ASSERT(cudaEventRecord(start, 0));
// CUDA Operaion
CUDA_ASSERT(cudaEventRecord(end, 0));
CUDA_ASSERT(cudaEventSynchronize(end));
CUDA_ASSERT(cudaEventElapsedTime(&time, start, end));
std::cout << "Time elapsed: " << time << "ms" << std::endl
```
There are handy macros available in `stefan/01_image_filter/util.h`.

### nvprof
```bash
nvprof program program_args
```
See [nvprof documentation](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview)

*Hint*: nvprof is a legacy tool not supported on newer compute capability versions.
To use nvprof on newer chips, run `nsys nvprof ARGS` or `nsys profile --stats=true ARGS`.

### Nsight
New nvidia profiling tools.
`nsys` -> overall system stats
`ncu` -> stats on a single kernel execution

Generated traces can be inspected using `nsys-ui` and `ncu-ui`.
It can store traces information in sqlite databases as well.
