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

	if (idx < numElements) {
		unsigned char c = input[idx];
		atomicAdd(&sBins[c % numBins], 1);
	}
	__syncthreads();

	for (unsigned int t = threadIdx.x; t < numBins; t += blockDim.x) {
		atomicAdd(&bins[t], sBins[t]);
	}
}

// Histogramm-Funktion, die den Kernel histogram_kernel_atomic_private verwendet.
void histogram_atomic_private(
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
) {
	constexpr size_t nThreadsPerBlock = 256;

	CUDA_CHECK(cudaMemset(bins, 0, numBins * sizeof(unsigned int)));

	dim3 dimGrid((numElements + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);
	histogram_kernel_atomic_private<<<dimGrid, dimBlock, numBins * sizeof(unsigned int)>>> (
		input, bins, numElements, numBins
	);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
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

bool checkGpuResults(
	char const * descr,
	unsigned char * input, unsigned int * bins,
	unsigned int numElements, unsigned int numBins
) {
	unsigned int * binsCpu = (unsigned int *)malloc(numBins * sizeof(unsigned int));
	histogramCpu(input, binsCpu, numElements, numBins);
	for (size_t i = 0; i < numBins; ++i)
		if (binsCpu[i] != bins[i]) {
			printf("Gpu result seems to be wrong for %s.\n", descr);
			for (size_t j = 0; j < numBins; ++j) {
				printf("Character %lu: CPU: %u, GPU: %u\n", j, binsCpu[j], bins[j]);
			}
			return false;
		}
	free(binsCpu);
	return true;
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

using HistogramFunction = void (
	unsigned char *input, unsigned int *bins,
	unsigned int numElements, unsigned int numBins
);

void performWarmupRuns(
	HistogramFunction histogramFn,
	unsigned char * deviceWarmupInput, unsigned int * deviceBins, size_t warmupInputLength
) {
	constexpr size_t nWarmupRuns = 200;
	for (size_t i = 0; i < nWarmupRuns; ++i) {
		histogramFn(deviceWarmupInput, deviceBins, warmupInputLength, NUM_BINS);
	}
}

void runForHistogramFunction(
	HistogramFunction histogramFn,
	unsigned char * deviceWarmupInput, unsigned int lengthWarmupInput,
	unsigned char * hostInput, unsigned char * deviceInput, unsigned int sizeInput, unsigned int inputLength,
	unsigned int * hostBins, unsigned int * deviceBins, unsigned int sizeBins
) {
	constexpr size_t nRuns = 100;
	float timesTransferToDevice[nRuns] = {};
	float timesExecution[nRuns] = {};
	float timesTransferFromDevice[nRuns] = {};

	// mache Warmup-Runs und ignoriere die Ergebnisse
	//   Wir machen die Warmup-Runs mit allokiertem und uninitialisiertem Speicher,
	//   aber das ist ok, denn uns sind die Ergebnisse sowieso egal.
	performWarmupRuns(histogramFn, deviceWarmupInput, deviceBins, lengthWarmupInput);

	// führe die zu testende Histogramm-Funktion (inkl. Transfers) nRuns mal aus
	//   und schreibe die gemessenen Zeiten ins Array timesExecution
	for (size_t i = 0; i < nRuns; ++i) {
		timesTransferToDevice[i] = RunAndMeasureCuda(
			[&] {
				CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, sizeInput, cudaMemcpyHostToDevice))
			}
		);
		timesExecution[i] = RunAndMeasureCuda(
			histogramFn,
			deviceInput, deviceBins, inputLength, NUM_BINS
		);
		timesTransferFromDevice[i] = RunAndMeasureCuda(
			[&] {
				CUDA_CHECK(cudaMemcpy(hostBins, deviceBins, sizeBins, cudaMemcpyDeviceToHost))
			}
		);
	}

	// schreibe JSON-Output auf stdout
	printf("{\n");
		printf("\"timesTransferToDevice\": "); jsonPrintFloatAry(timesTransferToDevice, nRuns); printf(",\n");
		printf("\"timesExecution\": "); jsonPrintFloatAry(timesExecution, nRuns); printf(",\n");
		printf("\"timesTransferFromDevice\": "); jsonPrintFloatAry(timesTransferFromDevice, nRuns); printf("\n");
	printf("}\n");
}

void randomFill(unsigned char * ary, size_t nChars) {
	// simple pseudo-random number generator copy/pasted from here:
	//   https://en.wikipedia.org/wiki/Lehmer_random_number_generator#Sample_C99_code
	// Das sind zugegeben sehr schlechte Zufallszahlen, aber wir müssen nur offensichtliche
	//   sehr kurze Patterns vermeiden.
	constexpr uint32_t seed = 2236631296; // eine beliegbige zufällige Zahl < 0x7fffffff
	auto lcg_parkmiller = [state = (uint32_t)seed] () mutable {
		return state = (uint64_t)state * 48271 % 0x7fffffff;
	};
	for (size_t i = 0; i < nChars; ++i) ary[i] = (lcg_parkmiller() >> 16) & 0xff;
}

int main(int argc, char *argv[]) {
	unsigned char * hostInput = nullptr;
	size_t inputLength = 0;
	char const * inputFileName = nullptr;

	if (argc <= 2) {
		if (argc <= 1) {
			inputFileName = "input_data/test.txt";
		} else {
			inputFileName = argv[1];
		}
		hostInput = (unsigned char *)malloc(sizeof(unsigned char));
		inputLength = read_file(inputFileName, &hostInput);
	} else if (
		argc == 3 && !strcmp(argv[1], "--") &&
		(inputLength = strtol(argv[2], nullptr, 10)) > 0
	) {
		hostInput = (unsigned char *)malloc(inputLength * sizeof(unsigned char));
		randomFill(hostInput, inputLength);
	} else {
		printf("Usage:\nhistogram\nor\nhistogram filename\nor\nhistogram -- number_of_characters\n");
		exit(1);
	}

	unsigned int * hostBins_one_thread_per_character = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	unsigned int * hostBins_atomic_private = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	unsigned int * hostBins_atomic_private_stride = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

	// GPU-Speicher für Warmup-Runs
	constexpr size_t lengthWarmupInput = 1u << 22; // 100 MiB
	constexpr size_t sizeWarmupInput = lengthWarmupInput * sizeof (unsigned char);
	unsigned char * deviceWarmupInput;
	CUDA_CHECK(cudaMalloc((void **) &deviceWarmupInput, sizeWarmupInput))

	// allokiere GPU-Speicher
	int sizeBins = NUM_BINS * sizeof(unsigned int);
	unsigned int sizeInput = inputLength * sizeof(unsigned char);
	unsigned char * deviceInput;
	unsigned int * deviceBins;
	CUDA_CHECK(cudaMalloc((void **) &deviceInput, sizeInput));
	CUDA_CHECK(cudaMalloc((void **) &deviceBins, sizeBins));

	// mache die Messungen und schreibe JSON-Output
	// ziemlich naiv, z.B. keine korrekte Behandlung von Double Quotes,
	//   aber hier wohl ausreichend
	printf("{\n");

		if (inputFileName) printf("\"fileName\": \"%s\",\n", inputFileName);
		printf("\"inputLengthInCharacters\": %lu,\n", inputLength);

		printf("\"measurements\": {\n");

			printf("\"%s\": ", "histogram_one_thread_per_character");
			runForHistogramFunction(
				histogram_one_thread_per_character,
				deviceWarmupInput, lengthWarmupInput,
				hostInput, deviceInput, sizeInput, inputLength,
				hostBins_one_thread_per_character, deviceBins, sizeBins
			);

			printf(",\n");

			printf("\"%s\": ", "histogram_atomic_private");
			runForHistogramFunction(
				histogram_atomic_private,
				deviceWarmupInput, lengthWarmupInput,
				hostInput, deviceInput, sizeInput, inputLength,
				hostBins_atomic_private, deviceBins, sizeBins
			);

			printf(",\n");

			printf("\"%s\": ", "histogram_atomic_private_stride");
			runForHistogramFunction(
				histogram_atomic_private_stride,
				deviceWarmupInput, lengthWarmupInput,
				hostInput, deviceInput, sizeInput, inputLength,
				hostBins_atomic_private_stride, deviceBins, sizeBins
			);

		printf("}\n");

	printf("}\n");


	// gib den GPU-Speicher frei
	cudaFree(deviceBins);
	cudaFree(deviceInput);
	cudaFree(deviceWarmupInput);

	// prüfe das Ergebnis des jeweils letzten Durchlaufs auf Korrektheit
	if (!!(
		checkGpuResults("one_thread_per_character", hostInput, hostBins_one_thread_per_character, inputLength, NUM_BINS) &
		checkGpuResults("atomic_private", hostInput, hostBins_atomic_private, inputLength, NUM_BINS) &
		checkGpuResults("atomic_private_stride", hostInput, hostBins_atomic_private_stride, inputLength, NUM_BINS)
	)) {
		exit(1);
	}

	// gib den Host-Speicher frei
	free(hostBins_atomic_private_stride);
	free(hostBins_atomic_private);
	free(hostBins_one_thread_per_character);
	free(hostInput);

	return 0;
}
