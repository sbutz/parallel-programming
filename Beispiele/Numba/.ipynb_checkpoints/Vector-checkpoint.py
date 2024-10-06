import numpy as np
from numba import cuda 

@cuda.jit('void(float64 [:], float64[:], float64[:])') 
def f(a, b, c):
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    tid = cuda.grid(1)
    size = len(c)
    if tid < size:
        c[tid] = a[tid] + b[tid]

start = cuda.event(timing=True)
stop = cuda.event(timing=True)

N = 1000000
h_a  = np.random.random(N)
h_b = np.random.random(N)
start.record()
d_a = cuda.to_device(h_a)
d_b = cuda.to_device(h_b)
stop.record()
stop.synchronize()
time = start.elapsed_time(stop)

print("It took me {} mseconds to get data to GPU".format(time))
d_c = cuda.device_array_like(d_a)
start.record()
f[N//256, 256](d_a, d_b, d_c)
stop.record()
stop.synchronize()
time = start.elapsed_time(stop)
print("It took me {} mseconds to add vectors on the GPU".format(time))
start.record()
c = d_c.copy_to_host()
stop.record()
stop.synchronize()
time = start.elapsed_time(stop)
print("It took me {} mseconds to copy result back".format(time))