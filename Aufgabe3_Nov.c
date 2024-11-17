#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include<cuda_runtime.h>
//#include "read_file.h"
//#include <sys/time.h>
#define NUM_BINS 128
#include <cctype> 

#define BLOCK_SIZE 1024
#define MAXCHAR 1024
#define LOAD_SIZE 16
#define ITER 1

#define CUDA_CHECK(ans)                                                   \
{ gpuAssert((ans), __FILE__, __LINE__); }

int read_file(const char* filename, unsigned char** data) {
	FILE* fp;
	char str[MAXCHAR];
	size_t read;
	size_t total_size = 0;
	*data = NULL;
	fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("Could not open file %s\n", filename);
		return 1;
	}
	while (fgets(str, MAXCHAR, fp) != NULL) {
		read = strlen(str);
		*data = (unsigned char*)realloc(*data, (total_size + read) * sizeof(char));
		memcpy(*data + total_size, str, read * sizeof(char));
		total_size += read;
	}
	fclose(fp);
	return total_size;
}



inline void gpuAssert(cudaError_t code, const char* file, int line,
	bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
			file, line);
		if (abort)
			exit(code);
	}
}

__global__ void delBins(unsigned int* bins) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < NUM_BINS)bins[tid] = 0;
}

__global__ void histogram_kernel_atomic_private(unsigned char* input, unsigned int* bins,
	unsigned int num_elements,
	unsigned int num_bins) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int elem = (int)input[tid];

	if (tid < num_elements && elem < NUM_BINS) atomicAdd(&bins[elem], 1);

}

__global__ void histogram_kernel_atomM(unsigned char* input, unsigned int* bins,
	unsigned int num_elements,
	unsigned int num_bins) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int elem[LOAD_SIZE];
	int j;

	for (int i = 0; i < LOAD_SIZE; ++i) {
		j = tid + i * blockDim.x;
		elem[i] = (int)input[j];
		if (elem[i] < NUM_BINS && j < num_elements)atomicAdd(&bins[elem[i]], 1);
	}
}

// kopiert von Johannes
__global__ void histogram_kernel_stride(unsigned char* input, unsigned int* bins,
	unsigned int num_elements,
	unsigned int num_bins) {

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ unsigned int sBins[];
	for (unsigned int t = threadIdx.x; t < num_bins; t += blockDim.x)sBins[t] = 0;
	__syncthreads();

	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < num_elements; i += stride) {
		unsigned char c = input[idx];
		atomicAdd(&sBins[c], c < num_bins);
	}
	__syncthreads();

	for (unsigned int t = threadIdx.x; t < num_bins; t += blockDim.x) {
		atomicAdd(&bins[t], sBins[t]);
	}
}

void del_Bins(unsigned int* bins) {
	delBins << <1, NUM_BINS >> > (bins);
}


void histogram(unsigned char* input, unsigned int* bins,
	unsigned int num_elements, unsigned int num_bins) {


	CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));

	{

		//TODO starte den Kernel hier
		int numBlocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		for (int i = 0; i < ITER; ++i) {

			del_Bins(bins);

			//histogram_kernel_atomM << < numBlocks / LOAD_SIZE, BLOCK_SIZE >> > (input, bins, num_elements, num_bins);
			//histogram_kernel_atomic_private << < numBlocks, BLOCK_SIZE >> > (input, bins, num_elements, num_bins);
			histogram_kernel_stride << < numBlocks / 256, BLOCK_SIZE >> > (input, bins, num_elements, num_bins);
		}

		cudaEventRecord(stop, 0);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("\Zeit: %f ms\n", time);



		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

	}
}


int main(int argc, char* argv[]) {

	int inputLength;
	unsigned char* hostInput;
	unsigned int* hostBins;
	unsigned char* deviceInput;
	unsigned int* deviceBins;

	char* datei = "C:/Users/nbaum/Documents/fu_hagen/parallel/Aufgabe3/Histogram/C_CUDA/test.txt";

	/*allokiert den Speicher und gibt die Anzahl der Gelesenen Werte zurrück*/

	hostInput = (unsigned char*)malloc(sizeof(unsigned char));
	inputLength = read_file(datei, &hostInput);

	hostBins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

	printf("The input length is %d\n ", inputLength);
	printf("The number of bins is %d\n", NUM_BINS);


	//TODO Allokiere GPU Speicher Hier
	int size_bins = NUM_BINS;
	int size_input = inputLength;

	cudaMalloc(&deviceInput, inputLength * sizeof(char));
	cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

	cudaError_t err = cudaSuccess;

	//TODO Copiere Daten hier auf die GPU
	cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

	// starte dem kernel
	// ----------------------------------------------------------

	histogram(deviceInput, deviceBins, inputLength, NUM_BINS);

	//TODO Kopiere die Daten zurrück auf den HOST

	cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < NUM_BINS; i++) {

		printf("i: %d ,num: %d \n ", i, hostBins[i]);

	}//*/

	//TODO Gebe den GPU speicher frei
	cudaFree(deviceBins);
	cudaFree(deviceInput);
	free(hostBins);
	free(hostInput);

	return 0;
}