#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

__global__ void VecAddKernel(const float *a, const float *b,  float *c, const int n) {
	int i = threadIdx.x+blockDim.x*blockIdx.x;
	if(i<n) c[i] = a[i] + b[i];
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{

	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_C, size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float memory_time1 = 0;
	cudaEventElapsedTime(&memory_time1, start, stop);

	dim3 DimGrid((n-1)/256 + 1, 1, 1);
	dim3 DimBlock(256, 1, 1);

	cudaEventRecord(start);
	VecAddKernel<<<DimGrid,DimBlock>>>(d_A, d_B, d_C, n);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float kernel_time = 0;
	cudaEventElapsedTime(&kernel_time, start, stop);


	cudaEventRecord(start);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float memory_time2 = 0;
	cudaEventElapsedTime(&memory_time2, start, stop);
	printf("memory time 1 %.2f,  memory time 1 %.2f, kernel time %.2f\n", memory_time1, memory_time2, kernel_time);

}

int main() {
	int n = 1<<20;

	float *h_A, *h_B, *h_C;
	h_A = (float*)malloc(sizeof(float)*n);
	h_B = (float*)malloc(sizeof(float)*n);

	h_C = (float*)malloc(sizeof(float)*n);

	for(int i = 0; i<n; i++) {
		h_A[i] = (float)(rand()%100);
		h_B[i]= (float)(rand()%200);
	}

	vecAdd(h_A, h_B, h_C,n);

	free(h_A);
	free(h_B);
	free(h_C);

}
