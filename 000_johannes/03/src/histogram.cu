#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "read_file.h"
#include <sys/time.h>
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

using MappingFn = unsigned char (unsigned char);

// Einfacher Histogramm-Kernel
// Vorgehen analog Bildfilter in Aufgabe 1:
// Die Anzahl Threads entspricht der Anzahl Zeichen im Input-String;
//   jeder Thread erhöht atomar den Bin für "sein" Zeichen um 1.
template<typename Mapping>
__global__ void histogram_kernel_one_thread_per_character(
	unsigned char *input, unsigned int *bins,
	size_t numElements
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numElements) return;
	unsigned char c = input[idx];
	atomicAdd(&bins[Mapping::map(c)], 1);
}

// Histogramm-Funktion, die den Kernel histogram_kernel_one_thread_per_character verwendet.
template <typename Mapping>
void histogram_one_thread_per_character(
	unsigned char *input, unsigned int *bins,
	size_t numElements	
) {
	constexpr size_t nThreadsPerBlock = 128;

	CUDA_CHECK(cudaMemset(bins, 0, Mapping::nBins * sizeof(unsigned int)));

	dim3 dimGrid((numElements + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);

	histogram_kernel_one_thread_per_character<Mapping> <<<dimGrid, dimBlock>>> (
		input, bins, numElements
	);
	
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Mapping>
__global__ void histogram_kernel_atomic_private(
	unsigned char *input, unsigned int *bins,
	size_t numElements
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ unsigned int sBins[];
	for (unsigned int t = threadIdx.x; t < Mapping::nBins; t += blockDim.x) {
		sBins[t] = 0;
	}
	__syncthreads();

	if (idx < numElements) {
		unsigned char c = input[idx];
		atomicAdd(&sBins[Mapping::map(c)], 1);
	}
	__syncthreads();

	for (unsigned int t = threadIdx.x; t < Mapping::nBins; t += blockDim.x) {
		atomicAdd(&bins[t], sBins[t]);
	}
}

// Histogramm-Funktion, die den Kernel histogram_kernel_atomic_private verwendet.
template <typename Mapping>
void histogram_atomic_private(
	unsigned char *input, unsigned int *bins,
	size_t numElements
) {
	constexpr size_t nThreadsPerBlock = 256;

	CUDA_CHECK(cudaMemset(bins, 0, Mapping::nBins * sizeof(unsigned int)));
	dim3 dimGrid((numElements + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);
	histogram_kernel_atomic_private<Mapping> <<<dimGrid, dimBlock, Mapping::nBins * sizeof(unsigned int)>>> (
		input, bins, numElements
	);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Mapping>
__global__ void histogram_kernel_atomic_private_stride(
	unsigned char *input, unsigned int *bins,
	size_t numElements
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// initialisiere Array im shared memory mit 0
	extern __shared__ unsigned int sBins[];
	for (unsigned int t = threadIdx.x; t < Mapping::nBins; t += blockDim.x) {
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
		size_t baseLimit = numElements >= stride ? numElements - stride : 0;
		size_t base = 0;
		for (; base < baseLimit; base += stride) {
			unsigned char c = input[base + idx];
			atomicAdd(&sBins[Mapping::map(c)], 1);
		}
		if (base + idx < numElements) {
			unsigned char c = input[base + idx];
			atomicAdd(&sBins[Mapping::map(c)], 1);
		}
	}
	__syncthreads();

	for (unsigned int t = threadIdx.x; t < Mapping::nBins; t += blockDim.x) {
		atomicAdd(&bins[t], sBins[t]);
	}
}

// Histogramm-Funktion, die den Kernel histogram_kernel_atomic_private_stride verwendet.
template <typename Mapping>
void histogram_atomic_private_stride(
	unsigned char *input, unsigned int *bins,
	size_t numElements
) {
	constexpr size_t nThreadsPerBlock = 256;

	CUDA_CHECK(cudaMemset(bins, 0, Mapping::nBins * sizeof(unsigned int)));

	dim3 dimGrid(1024, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);
	histogram_kernel_atomic_private_stride<Mapping> <<<dimGrid, dimBlock, Mapping::nBins * sizeof(unsigned int)>>> (
		input, bins, numElements
	);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

// Histogramm auf der CPU (zum Test, ob das Ergebnis der Kernels korrekt ist)
template <typename Mapping>
void histogramCpu(
	unsigned char * input, unsigned int * bins,
	size_t numElements, size_t numBins
) {
	for (size_t i = 0; i < Mapping::nBins; ++i) bins[i] = 0;
	for (size_t i = 0; i < numElements; ++i) ++bins[Mapping::map(input[i])];
}

template <typename Mapping>
bool checkGpuResults(
	char const * descr,
	unsigned char * input, unsigned int * bins,
	size_t numElements
) {
	unsigned int * binsCpu = (unsigned int *)malloc(Mapping::nBins * sizeof(unsigned int));
	histogramCpu<Mapping>(input, binsCpu, numElements, Mapping::nBins);
	for (size_t i = 0; i < Mapping::nBins; ++i)
		if (binsCpu[i] != bins[i]) {
			printf("Gpu result seems to be wrong for %s.\n", descr);
			for (size_t j = 0; j < Mapping::nBins; ++j) {
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
			printf("%.7f", ary[i]);
			++i;
			if (i == n) break;
			printf(", ");
		}
	}
	printf(" ]");
}

void jsonPrintUnsignedIntAry(unsigned int * ary, size_t n) {
	printf("[ ");
	if (n > 0) {
		size_t i = 0;
		for (;;) {
			printf("%u", ary[i]);
			++i;
			if (i == n) break;
			printf(", ");
		}
	}
	printf(" ]");
}


using HistogramFunction = void (
	unsigned char *input, unsigned int *bins,
	size_t numElements
);

void performWarmupRuns(
	HistogramFunction histogramFn,
	unsigned char * deviceWarmupInput, unsigned int * deviceBins, size_t warmupInputLength
) {
	constexpr size_t nWarmupRuns = 200;
	for (size_t i = 0; i < nWarmupRuns; ++i) {
		histogramFn(deviceWarmupInput, deviceBins, warmupInputLength);
	}
}

void runForHistogramFunction(
	HistogramFunction histogramFn, int nRuns, size_t nBins,
	unsigned char * deviceWarmupInput, size_t lengthWarmupInput,
	unsigned char * hostInput, unsigned char * deviceInput, size_t sizeInput, size_t inputLength,
	unsigned int * hostBins, unsigned int * deviceBins, size_t sizeBins
) {
	float * timesTransferToDevice = (float *)malloc(nRuns * sizeof(float));
	float * timesExecution = (float *)malloc(nRuns * sizeof(float));
	float * timesTransferFromDevice = (float *)malloc(nRuns * sizeof(float));

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
			deviceInput, deviceBins, inputLength
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
		printf("\"timesTransferFromDevice\": "); jsonPrintFloatAry(timesTransferFromDevice, nRuns); printf(",\n");
		printf("\"bins\": "); jsonPrintUnsignedIntAry(hostBins, nBins); printf("\n");
	printf("}\n");

	free(timesTransferFromDevice);
	free(timesExecution);
	free(timesTransferToDevice);
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

constexpr unsigned int ONE_THREAD_PER_CHARACTER = 1;
constexpr unsigned int ATOMIC_PRIVATE = 2;
constexpr unsigned int ATOMIC_PRIVATE_STRIDE = 4;

struct CommandLineArguments {
	unsigned int kernelsToRun = 0;
	char const * inputFileName = nullptr;
	size_t inputLength = 0;
	int nRuns = 1;
	bool uniformInput = false;
	bool letterMode = false;
};

CommandLineArguments parseCommandLineArguments(int argc, char * argv []) {
	CommandLineArguments cla = {
		.kernelsToRun = ONE_THREAD_PER_CHARACTER |
			ATOMIC_PRIVATE |
			ATOMIC_PRIVATE_STRIDE,
		.inputFileName = "input_data/test.txt"
	};

	do {
		if (argc <= 1) {
			// use defaults
			break;
		}

		int idx = 1;
		if (strcmp(argv[1], "--")) {
			cla.inputFileName = argv[1];
			++idx;
		} else {
			cla.inputFileName = nullptr;
			if (argc < 3) goto exitFailUsage;
			// read input length
			cla.inputLength = strtol(argv[2], nullptr, 10);
			idx += 2;
		}

		if (argc == idx) break;

		char const * argstr = argv[idx];
		cla.kernelsToRun = 0;
		for (size_t j = 0; argstr[j]; ++j) {
			switch (argstr[j]) {
				case 'o': {
					cla.kernelsToRun |= ONE_THREAD_PER_CHARACTER;
				} break;
				case 'a': {
					cla.kernelsToRun |= ATOMIC_PRIVATE;
				} break;
				case 's': {
					cla.kernelsToRun |= ATOMIC_PRIVATE_STRIDE;
				} break;
				case 'u': {
					cla.uniformInput = true;
				} break;
				case 'l': {
					cla.letterMode = true;
				}
			}
		}

		++idx;
		if (idx == argc) break;

		cla.nRuns = strtol(argv[idx], nullptr, 10);

	} while (false);

	return cla;

exitFailUsage:
		printf("Usage:\nhistogram\nor\nhistogram filename\nor\nhistogram -- number_of_characters\n");
		exit(1);
}

struct Mapping128 {
	constexpr static size_t nBins = 128;
	static __host__ __device__ unsigned char map(unsigned char c) {
		return (c - 1) & 0x7f;
	}
};

struct MappingLetter {
	constexpr static size_t nBins = 27;
	static __host__ __device__ unsigned char map(unsigned char c) {
		c = (c & 0xbf) - 64;
		return c & -(c <= 26);
	}
};

template <typename Mapping>
void performEverything(
	unsigned int kernelsToRun, bool uniformInput, int nRuns,
	char const * inputFileName,
	unsigned char * hostInput, size_t inputLength
) {
	unsigned int * hostBins_one_thread_per_character = (unsigned int *)malloc(Mapping::nBins * sizeof(unsigned int));
	unsigned int * hostBins_atomic_private = (unsigned int *)malloc(Mapping::nBins * sizeof(unsigned int));
	unsigned int * hostBins_atomic_private_stride = (unsigned int *)malloc(Mapping::nBins * sizeof(unsigned int));

	// GPU-Speicher für Warmup-Runs
	constexpr size_t lengthWarmupInput = 1u << 22; // 100 MiB
	constexpr size_t sizeWarmupInput = lengthWarmupInput * sizeof (unsigned char);
	unsigned char * deviceWarmupInput;
	CUDA_CHECK(cudaMalloc((void **) &deviceWarmupInput, sizeWarmupInput))

	// allokiere GPU-Speicher
	size_t sizeBins = Mapping::nBins * sizeof(unsigned int);
	size_t sizeInput = inputLength * sizeof(unsigned char);
	unsigned char * deviceInput;
	unsigned int * deviceBins;
	CUDA_CHECK(cudaMalloc((void **) &deviceInput, sizeInput));
	CUDA_CHECK(cudaMalloc((void **) &deviceBins, sizeBins));

  if (!deviceInput || !deviceBins) {
		fprintf(stderr, "Device memory allocation failed");
		exit(1);
	}

	// mache die Messungen und schreibe JSON-Output
	// ziemlich naiv, z.B. keine korrekte Behandlung von Double Quotes,
	//   aber hier wohl ausreichend
	printf("{\n");

		if (inputFileName) printf("\"fileName\": \"%s\",\n", inputFileName);
		printf("\"inputLengthInCharacters\": %lu,\n", inputLength);
		printf("\"uniformInput\": %s,\n", uniformInput ? "true" : "false");

		printf("\"measurements\": {\n");
		bool first = true;

			if (kernelsToRun & ONE_THREAD_PER_CHARACTER) {
				if (!first) printf(",\n");
				printf("\"%s\": ", "histogram_one_thread_per_character");
				runForHistogramFunction(
					histogram_one_thread_per_character<Mapping>, nRuns, Mapping::nBins,
					deviceWarmupInput, lengthWarmupInput,
					hostInput, deviceInput, sizeInput, inputLength,
					hostBins_one_thread_per_character, deviceBins, sizeBins
				);
				first = false;
			}

			if (kernelsToRun & ATOMIC_PRIVATE) {
				if (!first) printf(",\n");
				printf("\"%s\": ", "histogram_atomic_private");
				runForHistogramFunction(
					histogram_atomic_private<Mapping>, nRuns, Mapping::nBins,
					deviceWarmupInput, lengthWarmupInput,
					hostInput, deviceInput, sizeInput, inputLength,
					hostBins_atomic_private, deviceBins, sizeBins
				);
				first = false;
			}

			if (kernelsToRun & ATOMIC_PRIVATE_STRIDE) {
				if (!first) printf(",\n");
				printf("\"%s\": ", "histogram_atomic_private_stride");
				runForHistogramFunction(
					histogram_atomic_private_stride<Mapping>, nRuns, Mapping::nBins,
					deviceWarmupInput, lengthWarmupInput,
					hostInput, deviceInput, sizeInput, inputLength,
					hostBins_atomic_private_stride, deviceBins, sizeBins
				);
				first = false;
			}

		printf("}\n");

	printf("}\n");


	// gib den GPU-Speicher frei
	cudaFree(deviceBins);
	cudaFree(deviceInput);
	cudaFree(deviceWarmupInput);

	// prüfe das Ergebnis des jeweils letzten Durchlaufs auf Korrektheit
	if (!(
		(
			(kernelsToRun & ONE_THREAD_PER_CHARACTER) == 0 ||
			checkGpuResults<Mapping>("one_thread_per_character", hostInput, hostBins_one_thread_per_character, inputLength)
		) & (
			(kernelsToRun & ATOMIC_PRIVATE) == 0 ||
			checkGpuResults<Mapping>("atomic_private", hostInput, hostBins_atomic_private, inputLength)
		) & (
			(kernelsToRun & ATOMIC_PRIVATE_STRIDE) == 0 ||
			checkGpuResults<Mapping>("atomic_private_stride", hostInput, hostBins_atomic_private_stride, inputLength)
		)
	)) {
		exit(1);
	}

	// gib den Host-Speicher frei
	free(hostBins_atomic_private_stride);
	free(hostBins_atomic_private);
	free(hostBins_one_thread_per_character);
}

int main(int argc, char *argv[]) {
	unsigned char * hostInput = nullptr;
	size_t inputLength = 0;
	char const * inputFileName = nullptr;
	bool uniformInput = false;
	bool letterMode = false;
	unsigned int kernelsToRun = 0;
	int nRuns = 1;

	CommandLineArguments cla = parseCommandLineArguments(argc, argv);

	if (!!cla.inputFileName) {
		inputFileName = cla.inputFileName;
		hostInput = (unsigned char *)malloc(sizeof(unsigned char));
		inputLength = read_file(inputFileName, &hostInput);
	} else {
		inputLength = cla.inputLength;
		uniformInput = cla.uniformInput;
		hostInput = (unsigned char *)malloc(inputLength * sizeof(unsigned char));
		if (!hostInput) {
			fprintf(stderr, "Host memory allocation failed");
			exit(1);
		}
		if (uniformInput) {
			for (size_t j = 0; j < inputLength; ++j) hostInput[j] = 'a';
		} else {
			randomFill(hostInput, inputLength);
		}
	}
	kernelsToRun = cla.kernelsToRun;
	nRuns = cla.nRuns;
	letterMode = cla.letterMode;

	if (letterMode) {
		performEverything<MappingLetter> (
			kernelsToRun, uniformInput, nRuns, inputFileName, 
			hostInput, inputLength
		);
	} else {
		performEverything<Mapping128> (
			kernelsToRun, uniformInput, nRuns, inputFileName, 
			hostInput, inputLength
		);
	}

	free(hostInput);

	return 0;
}
