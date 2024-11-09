#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "read_file.h"
#include <sys/time.h>
#define NUM_BINS 128
#include <cctype> 

#define CUDA_CHECK(ans)                                                   \
{ gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(
	cudaError_t code, const char *file, int line, bool abort = true
) {
	if (code == cudaSuccess) return;
	fprintf(
		stderr, "GPUassert: %s %s %d\n",
		cudaGetErrorString(code), file, line
	);
	if (abort) exit(code);
}


__global__ void histogram_kernel(
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numElements) return;
	unsigned char c = input[idx];
	atomicAdd(&bins[c % numBins], 1);
}

__global__ void histogram_kernel_atomic_private(
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ unsigned int sBins[];
	for (unsigned int t = threadIdx.x; t < numBins; t += blockDim.x) {
		sBins[t] = 0;
	}

	if (idx >= numElements) return;
	unsigned char c = input[idx];
	atomicAdd(&sBins[c % numBins], 1);
	__syncthreads();

	for (unsigned int t = threadIdx.x; t < numBins; t += blockDim.x) {
		atomicAdd(&bins[t], sBins[t]);
	}
}

__global__ void histogram_kernel_atomic_private_stride(
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ unsigned int sBins[];
	for (unsigned int t = threadIdx.x; t < numBins; t += blockDim.x) {
		sBins[t] = 0;
	}
	__syncthreads();

	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < numElements; i += stride) {
		unsigned char c = input[idx];
		atomicAdd(&sBins[c], c < numBins);
	}
	__syncthreads();

	for (unsigned int t = threadIdx.x; t < numBins; t += blockDim.x) {
		atomicAdd(&bins[t], sBins[t]);
	}
}

void histogram(
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
) {
	CUDA_CHECK(cudaMemset(bins, 0, numBins * sizeof(unsigned int)));

	{ 
		// starte den Kernel
		dim3 dimGrid(2048, 1, 1);
		dim3 dimBlock(256, 1, 1);
		//histogram_kernel<<<dimGrid, dimBlock>>> (input, bins, numElements, numBins);
		//	histogram_kernel_atomic_private<<<dimGrid, dimBlock, numBins * sizeof(unsigned int)>>> (
		//		input, bins, numElements, numBins
		//	);

		histogram_kernel_atomic_private_stride<<<dimGrid, dimBlock, numBins * sizeof(unsigned int)>>> (
			input, bins, numElements, numBins
		);

		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

	}
}


int main(int argc, char *argv[]) {
	// allokiert den Speicher und gibt die Anzahl der gelesenen Werte zurück
	unsigned char * hostInput = (unsigned char *)malloc(sizeof(unsigned char));
	int inputLength = read_file("input_data/test.txt", &hostInput); 

	unsigned int * hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

	printf("The input length is %d\n ", inputLength);
	printf("The number of bins is %d\n", NUM_BINS);

	// allokiere GPU-Speicher
	cudaError_t err = cudaSuccess;
	int sizeBins = NUM_BINS * sizeof(unsigned int);
	unsigned int sizeInput = inputLength * sizeof(unsigned char);
	unsigned char * deviceInput;
	unsigned int * deviceBins;

	cudaMalloc((void **) &deviceInput, sizeInput);
	cudaMalloc((void **) &deviceBins, sizeBins);

	// kopiere die Daten auf die GPU
	cudaMemcpy(deviceInput, hostInput, sizeInput, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// starte den Kernel
	histogram(deviceInput, deviceBins, inputLength, NUM_BINS);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float histogramTime = 0;
	cudaEventElapsedTime(&histogramTime, start, stop);

	// kopiere die Daten zurück auf den HOST
	cudaMemcpy(hostBins, deviceBins, sizeBins, cudaMemcpyDeviceToHost);

	for (int i = 0; i < NUM_BINS; i++) {
		printf("i: %d ,num: %d \n ",i,  hostBins[i]);
	}

	// gib den GPU-Speicher frei
	cudaFree(deviceInput);
	cudaFree(deviceBins);

	// gib den Host-Speicher frei
	free(hostBins);
	free(hostInput);

	printf("Histogram time %.4f\n", histogramTime);

	return 0;
}
