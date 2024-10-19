# Image Filters

## Performance Analysis

### CUDA Init
The first access to the gpu is expensive since it initalizes the driver etc.

First API Call is `cudaMalloc`:
```
Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)           Name
--------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------
    97.4        123535495          2  61767747.5  61767747.5     92276  123443219   87222288.3  cudaMalloc
    2.1          2601594          2   1300797.0   1300797.0    570777    2030817    1032404.2  cudaMemcpy
    0.3           382490          2    191245.0    191245.0    131151     251339      84985.7  cudaFree
```
First API Call is `cudaFree`:
```
Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)  Min (ns)  Max (ns)   StdDev (ns)           Name
--------  ---------------  ---------  ----------  --------  --------  ---------  -----------  ----------------------
    98.8        121656948          3  40552316.0  247833.0    123799  121285316   69916856.4  cudaFree
    0.8          1011191          2    505595.5  505595.5    454578     556613      72149.6  cudaMemcpy
    0.2           195261          2     97630.5   97630.5     73809     121452      33688.7  cudaMalloc
```
To untaint the performance measurements `cudaFree` is called in the beginning of
`main()` to capture call startup overhead.
