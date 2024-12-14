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

	f(args ...);

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

	// initialisiere Array im shared memory mit 0
	extern __shared__ unsigned int sBins[];
	for (unsigned int t = threadIdx.x; t < numBins; t += blockDim.x) {
		sBins[t] = 0;
	}
	__syncthreads();

	{
		// baseLimit ist die kleinste Zeichenposition, ab der ein ab baseLimit beginnender
		//   Stride genau am letzten Zeichen des Inputs endet oder über den Input hinausragt.
		//   Im Prinizip wäre das die letzte Iteration der Schlefe. Da allerdings hier durch
		//   eine if-Abfrage geprüft werden müsste, ob idx noch innerhalb des Inputs liegt,
		//   spart es etwas Zeit, den letzten Stride separat zu behandeln.
		int stride = blockDim.x * gridDim.x;
		unsigned int baseLimit = numElements >= stride ? numElements - stride : 0;
		unsigned int base = 0;
		for (; base < baseLimit; base += stride) {
			unsigned char c = input[base + idx];
			atomicAdd(&sBins[c % numBins], 1);
		}
		if (base + idx < numElements) {
			unsigned char c = input[base + idx];
			atomicAdd(&sBins[c % numBins], 1);
		}
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

	dim3 dimGrid(1024, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);
	histogram_kernel_atomic_private_stride<<<dimGrid, dimBlock, numBins * sizeof(unsigned int)>>> (
		input, bins, numElements, numBins
	);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

// Histogramm auf der CPU (zum Test, ob das Ergebnis der Kernels korrekt ist)
void histogramCpu(
	unsigned char * input, unsigned int * bins,
	unsigned int numElements, unsigned int numBins
) {
	for (size_t i = 0; i < numBins; ++i) bins[i] = 0;
	for (size_t i = 0; i < numElements; ++i) ++bins[input[i] % numBins];
}

void checkGpuResults(
	unsigned char * input, unsigned int * bins,
	unsigned int numElements, unsigned int numBins
) {
	unsigned int * binsCpu = (unsigned int *)malloc(numBins * sizeof(unsigned int));
	histogramCpu(input, binsCpu, numElements, numBins);
	for (size_t i = 0; i < numBins; ++i)
		if (binsCpu[i] != bins[i]) {
			for (size_t j = 0; j < numBins; ++j) {
				printf("Character %lu: CPU: %u, GPU: %u\n", j, binsCpu[j], bins[j]);
			}
			printf("Gpu result seems to be wrong.\n");
			exit(1);
		}
	free(binsCpu);
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
	printf(" ]");
}

template <typename Fct>
void runForKernel(Fct f,
	char const * inputFileName,
	unsigned char * hostInput, unsigned char * deviceInput, unsigned int sizeInput, unsigned int inputLength,
	unsigned int * hostBins, unsigned int * deviceBins, unsigned int sizeBins
) {
	constexpr size_t nRuns = 50;
	float timesTransferToDevice[nRuns] = {};
	float timesExecution[nRuns] = {};
	float timesTransferFromDevice[nRuns] = {};

	// führe die zu testende Histogramm-Funktion (inkl. Transfers) nRuns mal aus
	//   und schreibe die gemessenen Zeiten ins Array timesExecution
	for (size_t i = 0; i < nRuns; ++i) {
		timesTransferToDevice[i] = RunAndMeasureCuda(
			[&] {
				CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, sizeInput, cudaMemcpyHostToDevice))
			}
		);
		timesExecution[i] = RunAndMeasureCuda(
			histogram_atomic_private_stride,
			//histogram_one_thread_per_character,
			deviceInput, deviceBins, inputLength, NUM_BINS
		);
		timesTransferFromDevice[i] = RunAndMeasureCuda(
			[&] {
				CUDA_CHECK(cudaMemcpy(hostBins, deviceBins, sizeBins, cudaMemcpyDeviceToHost))
			}
		);
	}

	// prüfe das Ergebnis des letzten Durchlaufs auf Korrektheit
	checkGpuResults(hostInput, hostBins, inputLength, NUM_BINS);

	// schreibe JSON-Output auf stdout
	//   ziemlich naiver Code,
	//   z.B. keine korrekte Behandlung von Double Quotes und Backslashes im Dateinamen
	printf("{\n");
		printf("\"fileName\": \"%s\",\n", inputFileName);
		printf("\"inputLengthInCharacters\": %d,\n", inputLength);

		printf("\"timesTransferToDevice\": "); jsonPrintFloatAry(timesTransferToDevice, nRuns); printf(",\n");
		printf("\"timesExecution\": "); jsonPrintFloatAry(timesExecution, nRuns); printf(",\n");
		printf("\"timesTransferFromDevice\": "); jsonPrintFloatAry(timesTransferFromDevice, nRuns); printf(",\n");
	printf("}\n");
}

int main(int argc, char *argv[]) {
	// allokiert den Speicher und gibt die Anzahl der gelesenen Werte zurück
	unsigned char * hostInput = (unsigned char *)malloc(sizeof(unsigned char));
	const char * inputFileName = "input_data/test.txt";
	int inputLength = read_file(inputFileName, &hostInput); 

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


	constexpr size_t nWarmupRuns = 100;

	// mache Warmup-Runs und ignoriere die Ergebnisse
	CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, sizeInput, cudaMemcpyHostToDevice));
	for (size_t i = 0; i < nWarmupRuns; ++i) {
		histogram_one_thread_per_character(deviceInput, deviceBins, inputLength, NUM_BINS);
	}

	printf("[\n\n");

	runForKernel(
		histogram_kernel_atomic_private,
		inputFileName,
		hostInput, deviceInput, sizeInput, inputLength,
		hostBins, deviceBins, sizeBins
	);

	printf(",\n");

	runForKernel(
		histogram_atomic_private_stride,
		inputFileName,
		hostInput, deviceInput, sizeInput, inputLength,
		hostBins, deviceBins, sizeBins
	);

	printf("]\n");

	// gib den GPU-Speicher frei
	cudaFree(deviceInput);
	cudaFree(deviceBins);

	// gib den Host-Speicher frei
	free(hostBins);
	free(hostInput);

	return 0;
}
