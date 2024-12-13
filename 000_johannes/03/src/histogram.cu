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

// Template-Funktion für die Messung der Ausführungsdauer via Cuda-Events
template <typename Fct, typename ... Args>
float RunAndMeasureCuda(Fct f, Args ... args) {
	float timeInMilliseconds = 0;

	// Zeitmessung: Zeitpunkte definieren
	cudaEvent_t start; CUDA_CHECK(cudaEventCreate(&start));
	cudaEvent_t stop; CUDA_CHECK(cudaEventCreate(&stop));

	// Zeitmessung: Start-Zeitpunkt
	CUDA_CHECK(cudaEventRecord(start));

	(*f)(args ...);

	// Zeitmessung: Stop-Zeitpunkt
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	
	// Berechne das Messergebnis.
	CUDA_CHECK(cudaEventElapsedTime(&timeInMilliseconds, start, stop));

	return timeInMilliseconds;
}

// Einfacher Histogramm-Kernel
// Vorgehen analog Bildfilter in Aufgabe 1:
// Die Anzahl Threads entspricht der Anzahl Zeichen im Input-String;
//   jeder Thread erhöht atomar den Bin für "sein" Zeichen um 1.
__global__ void histogram_kernel_one_thread_per_character(
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numElements) return;
	unsigned char c = input[idx];
	atomicAdd(&bins[c % numBins], 1);
}

// Histogramm-Funktion, die den Kernel histogram_kernel_one_thread_per_character verwendet.
void histogram_one_thread_per_character(
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
) {
	constexpr size_t nThreadsPerBlock = 128;

	CUDA_CHECK(cudaMemset(bins, 0, numBins * sizeof(unsigned int)));

	dim3 dimGrid((numElements + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);

	histogram_kernel_one_thread_per_character<<<dimGrid, dimBlock>>> (input, bins, numElements, numBins);
	
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
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

// Histogramm-Funktion, die den Kernel histogram_kernel_atomic_private_stride verwendet.
void histogram_atomic_private_stride(
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
) {
	constexpr size_t nThreadsPerBlock = 256;

	CUDA_CHECK(cudaMemset(bins, 0, numBins * sizeof(unsigned int)));

	dim3 dimGrid(2048, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);
	histogram_kernel_atomic_private_stride<<<dimGrid, dimBlock, numBins * sizeof(unsigned int)>>> (
		input, bins, numElements, numBins
	);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void jsonPrintFloatAry(float * ary, size_t n) {
	printf("[ ");
	if (n > 0) {
		size_t i = 0;
		for (;;) {
			printf("%.4f", ary[i]);
			++i;
			if (i == n) break;
			printf(", ");
		}
	}
	printf(" ]\n");
}

int main(int argc, char *argv[]) {
	// allokiert den Speicher und gibt die Anzahl der gelesenen Werte zurück
	unsigned char * hostInput = (unsigned char *)malloc(sizeof(unsigned char));
	int inputLength = read_file("input_data/test.txt", &hostInput); 

	unsigned int * hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

	printf("The input length is %d\n ", inputLength);
	printf("The number of bins is %d\n", NUM_BINS);

	// allokiere GPU-Speicher
	//cudaError_t err = cudaSuccess;
	int sizeBins = NUM_BINS * sizeof(unsigned int);
	unsigned int sizeInput = inputLength * sizeof(unsigned char);
	unsigned char * deviceInput;
	unsigned int * deviceBins;

	CUDA_CHECK(cudaMalloc((void **) &deviceInput, sizeInput));
	CUDA_CHECK(cudaMalloc((void **) &deviceBins, sizeBins));

	// kopiere die Daten auf die GPU
	CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, sizeInput, cudaMemcpyHostToDevice));

	constexpr size_t nWarmupRuns = 10;
	constexpr size_t nRuns = 10;
	float measuredTimes[nRuns] = {};

	// mach Warmup-Runs und ignoriere die Ergebnisse
	for (size_t i = 0; i < nWarmupRuns; ++i) {
		histogram_one_thread_per_character(deviceInput, deviceBins, inputLength, NUM_BINS);
	}

	// führe die zu testende Histogramm-Funktion nRuns mal aus
	//   und schreibe die gemessenen Zeiten ins Array measuredTimes
	for (size_t i = 0; i < nRuns; ++i) {
		measuredTimes[i] = RunAndMeasureCuda(
			&histogram_atomic_private_stride, //&histogram_one_thread_per_character,
			deviceInput, deviceBins, inputLength, NUM_BINS
		);
	}
	
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

	jsonPrintFloatAry(measuredTimes, nRuns);

	return 0;
}
