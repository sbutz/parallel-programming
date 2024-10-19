# Image Filters - Performance Analysis

Possible Optimizations:
- One thread per color channel (3 for rgb, instead of one)
- Change row/coloumn iteration (memory alignment)
- Adjust block/grid size to image size
- Pad image to avoid margins and reduce conditions 


## Test Parameters
- Execution Environment:
    - GPU: NVIDIA AD104, CPU: Intel Xeon, OS: Ubuntu 24.04 LTS
- Image Size
    - `./images/small.jpg`  (100x100px)
    - `./images/medium.jpg` (1200×675px)
    - `./images/large.jpg`  (5616×3744px)
- Margin (blur factor): 1, 2, 3
- Implementation versions (different optimizations)
    - `./01_image_filter/stefan/blur.cu`

Test command:
```bash
make -C ./stefan all
nsys profile --status=true ./stefan/build/blur MARGIN INPUT_IMAGE OUTPUT_IMAGE
```

To reduce test cases and only one parameter is changed in each run.

## CUDA Init
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

## Image Size
Margin: 3

| Metric/Image Size | small | medium | big |
| --- | --- | --- | --- |
| Kernel Execution | 5us | 114us | 3.279us |
| Transfer (one-way) | 4us | 313us | 30.080us |

Result: With increasing image size the filter execution time increases less
than the time needed for the data transfer.

## Margin (blur factor)
`(2n+1)^2` pixels are included for calculating each pixel's new value.

Image: medium
| Metric/n | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Kernel Execution | 37us | 64us| 113us |
| Data Transfer | 302us | 298us | 305us |

Result: Same effect as for increasing image size.
The number of pixels included in the calcuation increases quadratic, the
execution time does not.
