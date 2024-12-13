#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stddef.h>
#include <math.h>
#include <sys/time.h>
#define FULL_MASK 0xffffffff


__global__ void deviceReduceKernel(float *in, float* out, size_t N) {
	size_t tid = threadIdx.x;
	size_t bid = blockIdx.x;

	extern __shared__ float s_sums[];
	size_t i = bid * blockDim.x * 2 + tid;
	size_t j = i + blockDim.x;
	s_sums[tid] = (i < N ? in[i] : 0.0f) + (j < N ? in[j] : 0.0f);
	__syncthreads();

	for (size_t k = blockDim.x >> 1; k != 0; k >>= 1) {
		if (tid < k) s_sums[tid] += s_sums[tid + k];
		__syncthreads();
	}

	if (tid == 0) out[bid] = s_sums[0];
}



__global__ void InitKernel(float *in, size_t size)
{

	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += gridDim.x * blockDim.x)
		in[id]=1.0;

}

void init(float *in, size_t size){
	size_t threads = 1024;
	size_t blocks = min(size_t(size + threads - 1) / threads, (size_t)1024);
	InitKernel<<<blocks,threads>>>(in, size);
	cudaDeviceSynchronize();
}

float reduction(float *in, size_t N)
{
	size_t threads = 1024;
	size_t blocks = size_t(N + threads - 1) / threads;
	float* out;
	float ret;

	cudaMalloc((void **) &out, blocks * sizeof(float));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	deviceReduceKernel<<<blocks, threads, threads * sizeof(float)>>> (in, out, N);
	size_t n = blocks;
	for (; n > 1; n = (n + threads - 1) / threads) {
		deviceReduceKernel<<<n, threads, threads * sizeof(float)>>> (out, out, n);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float reduceTime = 0;
	cudaEventElapsedTime(&reduceTime, start, stop);

	// kopiere die Daten zurück auf den HOST
	cudaMemcpy(&ret, out, sizeof(float), cudaMemcpyDeviceToHost);

	printf("%.2f ", ret);
	printf("\n");

	return ret;
}


#define ITER 10
int main (int argc, const char** argv)
{
	size_t size = 8;
	float * input;
	int i;

	//Größe kann als Parameter übergeben werden
	if (argc > 1) {
		size = atoi(argv[1]);
	}

	printf("Size: %lu * 2^20 floating-point values\n", size);

	float res;
	size = size * (1024 * 1024);
	cudaMalloc(&input, size * sizeof(float));
	init(input, size);

	//Wir machen hier mehrere Interationen, um ein gutes Ergebnis zu bekommen
	for (i = 0; i < ITER; i++) {


		res = reduction(input, size);
		if (res != size) printf("Wrong result.\n");
	}

	return 0;

}
