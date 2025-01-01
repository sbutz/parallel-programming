#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "read_file.h"
#include <sys/time.h>
#include <cctype> 

constexpr char const * usageText =
	"Usage:\n"
	"\n"
	"histogram file_name [commands] [number_of_runs]\n"
	"histogram -- number_of_characters [commands] [number_of_runs]\n"
	"\n"
	"Measures the runtimes of four different histogram kernels for\n"
	"a given text file or the specified number of pseudo-random characters.\n"
	"\n"
	"Parameters:\n"
	"\n"
	"file_name: Input text file.\n"
	"number_of_characters: Number of pseudo-random characters to generate.\n"
	"commands: String of characters from the set {o,a,s,u,l}.\n"
	"  o: Run histogram_kernel_atomic_global.\n"
	"  a: Run histogram_kernel_atomic_private.\n"
	"  s: Run histogram_kernel_atomic_private_stride.\n"
	"  g: Run histogram_kernel_atomic_global_stride.\n"
	"  u: Generate uniform input data consisting only of the character 'a'.\n"
	"       Has no effect for text file input.\n"
	"  l: Use only 27 bins,\n"
	"       bins 1 to 26 for characters 'A'/'a' to 'Z'/'z',\n"
	"       bin 0 for everything else.\n"
	"number_of_runs: Number of measurement runs to perform per kernel.\n"
	"  Does not affect the number of warmup runs.\n";

// Datentyp für die einzelnen Bins
//   unsigned int hat nur 32 Bit, d.h. wollen wir Inputdaten >= 4 GiB korrekt
//   behandeln, brauchen wir size_t mit 64 Bit
using BinType = unsigned int;

// *****************************************************************************
// Utilities für Error-Checking
// *****************************************************************************

#define CUDA_CHECK(ans)                                                   \
{ gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(
	hipError_t code, const char *file, int line, bool abort = true
) {
	if (code == hipSuccess) return;
	fprintf(
		stderr, "GPUassert: %s %s %d\n",
		hipGetErrorString(code), file, line
	);
	if (abort) exit(code);
}

// *****************************************************************************
// Template-Funktion zur Messung der Ausführungsdauer via Cuda-Events
// *****************************************************************************

template <typename Fct, typename ... Args>
float runAndMeasureCuda(Fct f, Args ... args) {
	float timeInMilliseconds = 0;

	// Zeitmessung: Zeitpunkte definieren
	hipEvent_t start; CUDA_CHECK(hipEventCreate(&start));
	hipEvent_t stop; CUDA_CHECK(hipEventCreate(&stop));

	// Zeitmessung: Start-Zeitpunkt
	CUDA_CHECK(hipEventRecord(start));

	f(args ...);

	// Zeitmessung: Stop-Zeitpunkt
	CUDA_CHECK(hipEventRecord(stop));
	CUDA_CHECK(hipEventSynchronize(stop));
	
	// Berechne das Messergebnis.
	CUDA_CHECK(hipEventElapsedTime(&timeInMilliseconds, start, stop));

	return timeInMilliseconds;
}

// *****************************************************************************
// Mappings Zeichen -> Bins
// *****************************************************************************

// Mapping für Teil a
//   Zeichen-Codes 1 bis 128 -> Bins 0 bis 127
//   Mapping der übrigen Zeichen "undefiniert", da sie nach Annahme
//     nicht vorkommen sollen. Wir machen das so, dass es möglichst
//     einfach ist.
//   Die Implementierung mappt
//     1 -> 0
//     2 -> 1
//     ...
//     128 -> 127
//     129 -> 0
//     ...
//     255 -> 126
//     0 -> 127
struct Mapping128 {
	constexpr static size_t numBins = 128;
	constexpr static __host__ __device__ unsigned char map(unsigned char c) {
		return (c - 1u) & 0x7f;
	}
};

static_assert(Mapping128::map(1) == 0, "");
static_assert(Mapping128::map(2) == 1, "");
static_assert(Mapping128::map(128) == 127, "");

// Mapping für Teil b
//   Zeichen 'A' bis 'Z' -> Bins 1 bis 26
//   Zeichen 'a' bis 'z' -> Bins 1 bis 26
//   alle anderen Zeichen -> Bin 0
struct MappingLetter {
	constexpr static size_t numBins = 27;
	constexpr static __host__ __device__ unsigned char map(unsigned char c) {
		c = (c & 0xdf) - 64u;
		return c & (0u - (c <= 26u));
	}
};

static_assert(MappingLetter::map(0) == 0, "");
static_assert(MappingLetter::map('@') == 0, "");
static_assert(MappingLetter::map('A') == 1, "");
static_assert(MappingLetter::map('C') == 3, "");
static_assert(MappingLetter::map('Z') == 26, "");
static_assert(MappingLetter::map('[') == 0, "");
static_assert(MappingLetter::map('`') == 0, "");
static_assert(MappingLetter::map('a') == 1, "");
static_assert(MappingLetter::map('c') == 3, "");
static_assert(MappingLetter::map('z') == 26, "");
static_assert(MappingLetter::map('{') == 0, "");

// *****************************************************************************
// Histogramm-Kernels und Funktionen
// *****************************************************************************

// Funktionstyp für Histogramm-Funktionen
using HistogramFunction = void (
	unsigned char * input, BinType * bins,	size_t numElements
);

// Einfacher Histogramm-Kernel.
// Vorgehen analog Bildfilter in Aufgabe 1:
// Die Anzahl Threads entspricht der Anzahl Zeichen im Input-String;
//   jeder Thread erhöht atomar den Bin für "sein" Zeichen um 1.
template<typename Mapping>
__global__ void histogram_kernel_atomic_global(
	unsigned char * input, BinType * bins, size_t numElements
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numElements) return;
	unsigned char c = input[idx];
	atomicAdd(&bins[Mapping::map(c)], 1);
}

// Histogramm-Funktion, die den Kernel histogram_kernel_atomic_global verwendet.
template <typename Mapping>
void histogram_atomic_global(
	unsigned char * input, BinType * bins,	size_t numElements	
) {
	constexpr size_t nThreadsPerBlock = 256;

	CUDA_CHECK(hipMemset(bins, 0, Mapping::numBins * sizeof(BinType)));

	dim3 dimGrid((numElements + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);

	histogram_kernel_atomic_global<Mapping> <<<dimGrid, dimBlock>>> (
		input, bins, numElements
	);
	
	CUDA_CHECK(hipGetLastError());
	CUDA_CHECK(hipDeviceSynchronize());
}

// Histogramm-Kernel, der Shared Memory benutzt, aber ohne Stride.
// Jeder Block zählt zunächst im Shared Memory. Abschliessend werden die Ergebnisse
//   der Blöcke addiert.
// Vorgehen weiterhin ähnlich dem Bildfilter in Aufgabe 1:
// Die Anzahl Threads entspricht der Anzahl Zeichen im Input-String;
//   jeder Thread erhöht atomar den Bin für "sein" Zeichen um 1, aber eben
//   zunächst im Shared Memory.
template <typename Mapping>
__global__ void histogram_kernel_atomic_private(
	unsigned char * input, BinType * bins, size_t numElements
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ BinType sBins[Mapping::numBins * sizeof(BinType)];
	for (unsigned int t = threadIdx.x; t < Mapping::numBins; t += blockDim.x) {
		sBins[t] = 0;
	}
	__syncthreads();

	if (idx < numElements) {
		unsigned char c = input[idx];
		atomicAdd(&sBins[Mapping::map(c)], 1);
	}
	__syncthreads();

	for (unsigned int t = threadIdx.x; t < Mapping::numBins; t += blockDim.x) {
		atomicAdd(&bins[t], sBins[t]);
	}
}

// Histogramm-Funktion, die den Kernel histogram_kernel_atomic_private verwendet.
template <typename Mapping>
void histogram_atomic_private(
	unsigned char * input, BinType * bins, size_t numElements
) {
	constexpr size_t nThreadsPerBlock = 256;

	CUDA_CHECK(hipMemset(bins, 0, Mapping::numBins * sizeof(BinType)));
	dim3 dimGrid((numElements + nThreadsPerBlock - 1) / nThreadsPerBlock, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);
	histogram_kernel_atomic_private<Mapping> <<<dimGrid, dimBlock>>> (
		input, bins, numElements
	);
	CUDA_CHECK(hipGetLastError());
	CUDA_CHECK(hipDeviceSynchronize());
}

// Histogramm-Kernel, der Shared Memory benutzt, aber nun mit Stride.
// Jeder Block zählt zunächst im Shared Memory. Abschliessend werden die Ergebnisse
//   der Blöcke addiert.
// Die Anzahl der Threads kann nun unabhängig von der Grösse des Inputs gewählt
//   werden. Die Threads arbeiten einen Abschnitt (Stride) der Eingabe nach dem
//   anderen ab.
template <typename Mapping>
__global__ void histogram_kernel_atomic_private_stride(
	unsigned char * input, BinType * bins,	size_t numElements
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// initialisiere Array im shared memory mit 0
	__shared__ BinType sBins[Mapping::numBins * sizeof(BinType)];
	for (unsigned int t = threadIdx.x; t < Mapping::numBins; t += blockDim.x) {
		sBins[t] = 0;
	}
	__syncthreads();

	{
		// baseLimit ist die kleinste Zeichenposition, ab der ein ab baseLimit beginnender
		//   Stride genau am letzten Zeichen des Inputs endet oder über den Input hinausragt.
		//   Im Prinzip wäre das die letzte Iteration der Schleife. Da allerdings hier durch
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

	for (unsigned int t = threadIdx.x; t < Mapping::numBins; t += blockDim.x) {
		atomicAdd(&bins[t], sBins[t]);
	}
}

// Histogramm-Funktion, die den Kernel histogram_kernel_atomic_private_stride verwendet.
template <typename Mapping>
void histogram_atomic_private_stride(
	unsigned char *input, BinType * bins, size_t numElements
) {
	constexpr size_t nThreadsPerBlock = 256;

	CUDA_CHECK(hipMemset(bins, 0, Mapping::numBins * sizeof(BinType)));

	dim3 dimGrid(1024, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);
	histogram_kernel_atomic_private_stride<Mapping> <<<dimGrid, dimBlock>>> (
		input, bins, numElements
	);
	CUDA_CHECK(hipGetLastError());
	CUDA_CHECK(hipDeviceSynchronize());
}

// Histogramm-Kernel mit Stride, aber ohne Shared Memory.
// Die Anzahl der Threads kann wieder unabhängig von der Grösse des Inputs gewählt
//   werden. Die Threads arbeiten einen Abschnitt (Stride) der Eingabe nach dem
//   anderen ab.
template <typename Mapping>
__global__ void histogram_kernel_atomic_global_stride(
	unsigned char * input, BinType * bins,	size_t numElements
) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
			atomicAdd(&bins[Mapping::map(c)], 1);
		}
		if (base + idx < numElements) {
			unsigned char c = input[base + idx];
			atomicAdd(&bins[Mapping::map(c)], 1);
		}
	}
}

// Histogramm-Funktion, die den Kernel histogram_kernel_atomic_global_stride verwendet.
template <typename Mapping>
void histogram_atomic_global_stride(
	unsigned char *input, BinType * bins, size_t numElements
) {
	constexpr size_t nThreadsPerBlock = 256;

	CUDA_CHECK(hipMemset(bins, 0, Mapping::numBins * sizeof(BinType)));

	dim3 dimGrid(1024, 1, 1);
	dim3 dimBlock(nThreadsPerBlock, 1, 1);
	histogram_kernel_atomic_global_stride<Mapping> <<<dimGrid, dimBlock>>> (
		input, bins, numElements
	);
	CUDA_CHECK(hipGetLastError());
	CUDA_CHECK(hipDeviceSynchronize());
}

// Histogramm-Funktion, die das Histogramm auf der CPU berechnet.
//   Wird nachher zur Prüfung der GPU-Resultate benötigt.
template <typename Mapping>
void histogram_cpu(
	unsigned char * input, BinType * bins,	size_t numElements
) {
	for (size_t i = 0; i < Mapping::numBins; ++i) bins[i] = 0;
	for (size_t i = 0; i < numElements; ++i) ++bins[Mapping::map(input[i])];
}

// *****************************************************************************
// Hilfsfunktionen für die Ausgabe von Arrays im JSON-Format
// *****************************************************************************

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

void jsonPrintBinTypeAry(BinType * ary, size_t n) {
	constexpr char const * formatStr = sizeof(BinType) == 4? "%u" : "%lu";
	printf("[ ");
	if (n > 0) {
		size_t i = 0;
		for (;;) {
			printf(formatStr, ary[i]);
			++i;
			if (i == n) break;
			printf(", ");
		}
	}
	printf(" ]");
}

// *****************************************************************************
// Hilfsfunktion zur Generierung von pseudo-zufälligem Input
// *****************************************************************************

void randomFill(unsigned char * ary, size_t numChars) {
	// Einfacher Pseudo-Zufallszahlengenerator von hier:
	//   https://en.wikipedia.org/wiki/Lehmer_random_number_generator#Sample_C99_code
	// Das sind wohl sehr schlechte Zufallszahlen, aber wir wollen nur offensichtliche,
	//   sehr kurze Muster vermeiden.
	// Es werden Zahlen zwischen 1 und 128 generiert.
	constexpr uint32_t seed = 2236631296; // eine beliegbige zufällige Zahl < 0x7fffffff
	auto lcg_parkmiller = [state = (uint32_t)seed] () mutable {
		return state = (uint64_t)state * 48271 % 0x7fffffff;
	};
	for (size_t i = 0; i < numChars; ++i) ary[i] = ((lcg_parkmiller() >> 16) & 0x7f) + 1;
}

// *****************************************************************************
// Config: struct, die beschreibt, was aufgrund der Kommandozeile zu tun ist
// *****************************************************************************

struct Config {
	static constexpr unsigned int ATOMIC_GLOBAL = 1;
	static constexpr unsigned int ATOMIC_PRIVATE = 2;
	static constexpr unsigned int ATOMIC_PRIVATE_STRIDE = 4;
	static constexpr unsigned int ATOMIC_GLOBAL_STRIDE = 8;

	unsigned int kernelsToRun = 0;
	char const * inputFileName = nullptr;
	size_t inputLength = 0;
	int nRuns = 1;
	bool uniformInput = false;
	bool useMappingLetter = false;
};

// *****************************************************************************
// Hilfsfunktionen zum Parsen und Interpretieren der Kommandozeile
// *****************************************************************************

void abortWithUsageMessage() {
	fprintf(stderr, usageText);
	exit(1);
}

Config parseCommandLineArguments(int argc, char * argv []) {
	// defaults
	Config config {};
	config.kernelsToRun = Config::ATOMIC_GLOBAL |
			Config::ATOMIC_PRIVATE |
			Config::ATOMIC_PRIVATE_STRIDE |
			Config::ATOMIC_GLOBAL_STRIDE;

	if (argc <= 1) abortWithUsageMessage();

	int idx = 1;
	if (strcmp(argv[1], "--")) {
		config.inputFileName = argv[1];
		++idx;
	} else {
		config.inputFileName = nullptr;
		if (argc < 3) abortWithUsageMessage();
		// read input length
		config.inputLength = strtol(argv[2], nullptr, 10);
		idx += 2;
	}

	if (argc == idx) return config;

	char const * argstr = argv[idx];
	config.kernelsToRun = 0;
	for (size_t j = 0; argstr[j]; ++j) {
		switch (argstr[j]) {
			case 'o': {
				config.kernelsToRun |= Config::ATOMIC_GLOBAL;
			} break;
			case 'a': {
				config.kernelsToRun |= Config::ATOMIC_PRIVATE;
			} break;
			case 's': {
				config.kernelsToRun |= Config::ATOMIC_PRIVATE_STRIDE;
			} break;
			case 'g': {
				config.kernelsToRun |= Config::ATOMIC_GLOBAL_STRIDE;
			} break;
			case 'u': {
				config.uniformInput = true;
			} break;
			case 'l': {
				config.useMappingLetter = true;
			} break;
			default:
				abortWithUsageMessage();
		}
	}

	++idx;
	if (idx == argc) return config;

	config.nRuns = strtol(argv[idx], nullptr, 10);

	return config;
}

// *****************************************************************************
// measureHistogramFunction
// *****************************************************************************

// Miss einen Histogram-Algorithmus (nach einigen Warmup-Runs).
// Ausgabe der Messwerte als JSON auf stdout.
void measureHistogramFunction(
	HistogramFunction histogramFn, int nRuns, size_t numBins,
	unsigned char * deviceWarmupInput, size_t lengthWarmupInput,
	unsigned char * hostInput, unsigned char * deviceInput, size_t sizeInput, size_t inputLength,
	BinType * hostBins, BinType * deviceBins, size_t sizeBins
) {
	float * timesTransferToDevice = (float *)malloc(nRuns * sizeof(float));
	float * timesExecution = (float *)malloc(nRuns * sizeof(float));
	float * timesTransferFromDevice = (float *)malloc(nRuns * sizeof(float));

	// mache Warmup-Runs und ignoriere die Ergebnisse
	//   Wir machen die Warmup-Runs mit allokiertem, aber uninitialisiertem Speicher.
	//   Das ist ok, denn uns sind die Ergebnisse der Warmup-Runs sowieso egal.
	{
		constexpr size_t nWarmupRuns = 200;
		for (size_t i = 0; i < nWarmupRuns; ++i) {
			histogramFn(deviceWarmupInput, deviceBins, lengthWarmupInput);
		}
	}

	// führe die zu testende Histogramm-Funktion (inkl. Transfers) nRuns mal aus
	//   und schreibe die gemessenen Zeiten ins Array timesExecution
	for (size_t i = 0; i < nRuns; ++i) {
		timesTransferToDevice[i] = runAndMeasureCuda(
			[&] {
				CUDA_CHECK(hipMemcpy(deviceInput, hostInput, sizeInput, hipMemcpyHostToDevice))
			}
		);
		timesExecution[i] = runAndMeasureCuda(
			histogramFn,
			deviceInput, deviceBins, inputLength
		);
		timesTransferFromDevice[i] = runAndMeasureCuda(
			[&] {
				CUDA_CHECK(hipMemcpy(hostBins, deviceBins, sizeBins, hipMemcpyDeviceToHost))
			}
		);
	}

	// schreibe JSON-Output auf stdout
	printf("{\n");
		printf("\"timesTransferToDevice\": "); jsonPrintFloatAry(timesTransferToDevice, nRuns); printf(",\n");
		printf("\"timesExecution\": "); jsonPrintFloatAry(timesExecution, nRuns); printf(",\n");
		printf("\"timesTransferFromDevice\": "); jsonPrintFloatAry(timesTransferFromDevice, nRuns); printf(",\n");
		printf("\"bins\": "); jsonPrintBinTypeAry(hostBins, numBins); printf("\n");
	printf("}\n");

	free(timesTransferFromDevice);
	free(timesExecution);
	free(timesTransferToDevice);
}

// *****************************************************************************
// main() im weiteren Sinne
// *****************************************************************************

// Hilfsfunktion zum Vergleich der GPU-Resultate mit dem Histogramm, das auf der CPU generiert wurde
template <typename Mapping>
bool checkGpuResults(
	char const * descr,
	unsigned char * input, BinType * binsGpu, BinType * binsCpu, size_t numElements
) {
	for (size_t i = 0; i < Mapping::numBins; ++i)
		if (binsCpu[i] != binsGpu[i]) {
			fprintf(stderr, "Gpu result seems to be wrong for %s.\n", descr);
			for (size_t j = 0; j < Mapping::numBins; ++j) {
				fprintf(stderr, "Character %lu: CPU: %u, GPU: %u\n", j, binsCpu[j], binsGpu[j]);
			}
			return false;
		}
	return true;
}

// Hauptfunktion, die nach dem Parsen der Kommandozeile von main() aufgerufen wird und
// die wesentliche Arbeit macht:
// - Speicher allokieren (Host und Device)
// - JSON-Output auf stdout beginnen und abschliessen
// - für jede der zu untersuchenden Histogram-Funktionen measureHistogramFunction aufrufen
// - Vergleichsresultate auf CPU generieren
// - GPU-Resultate mithilfe der Vergleichsresultaten prüfen
// - Speicher deallokieren
template <typename Mapping>
void run(
	Config config,
	unsigned char * hostInput
) {
	BinType * hostBins_atomic_global = (BinType *)malloc(Mapping::numBins * sizeof(BinType));
	BinType * hostBins_atomic_private = (BinType *)malloc(Mapping::numBins * sizeof(BinType));
	BinType * hostBins_atomic_private_stride = (BinType *)malloc(Mapping::numBins * sizeof(BinType));
	BinType * hostBins_atomic_global_stride = (BinType *)malloc(Mapping::numBins * sizeof(BinType));

	// GPU-Speicher für Warmup-Runs
	constexpr size_t lengthWarmupInput = 1u << 22; // 100 MiB
	constexpr size_t sizeWarmupInput = lengthWarmupInput * sizeof (unsigned char);
	unsigned char * deviceWarmupInput;
	CUDA_CHECK(hipMalloc((void **) &deviceWarmupInput, sizeWarmupInput))

	// allokiere GPU-Speicher
	size_t sizeBins = Mapping::numBins * sizeof(BinType);
	size_t sizeInput = config.inputLength * sizeof(unsigned char);
	unsigned char * deviceInput;
	BinType * deviceBins;
	CUDA_CHECK(hipMalloc((void **) &deviceInput, sizeInput));
	CUDA_CHECK(hipMalloc((void **) &deviceBins, sizeBins));

	if (!deviceInput || !deviceBins) {
		fprintf(stderr, "Device memory allocation failed.\n");
		exit(1);
	}

	// mache die Messungen und schreibe JSON-Output
	// ziemlich naiv, z.B. keine korrekte Behandlung von Double Quotes,
	//   aber hier wohl ausreichend
	printf("{\n");

		if (config.inputFileName) printf("\"fileName\": \"%s\",\n", config.inputFileName);
		printf("\"inputLengthInCharacters\": %lu,\n", config.inputLength);
		printf("\"uniformInput\": %s,\n", config.uniformInput ? "true" : "false");
		printf("\"useMappingLetter\": %s,\n", config.useMappingLetter ? "true" : "false");

		printf("\"measurements\": {\n");
		bool first = true;

			if (config.kernelsToRun & Config::ATOMIC_GLOBAL) {
				if (!first) printf(",\n");
				printf("\"%s\": ", "histogram_atomic_global");
				measureHistogramFunction(
					histogram_atomic_global<Mapping>, config.nRuns, Mapping::numBins,
					deviceWarmupInput, lengthWarmupInput,
					hostInput, deviceInput, sizeInput, config.inputLength,
					hostBins_atomic_global, deviceBins, sizeBins
				);
				first = false;
			}

			if (config.kernelsToRun & Config::ATOMIC_PRIVATE) {
				if (!first) printf(",\n");
				printf("\"%s\": ", "histogram_atomic_private");
				measureHistogramFunction(
					histogram_atomic_private<Mapping>, config.nRuns, Mapping::numBins,
					deviceWarmupInput, lengthWarmupInput,
					hostInput, deviceInput, sizeInput, config.inputLength,
					hostBins_atomic_private, deviceBins, sizeBins
				);
				first = false;
			}

			if (config.kernelsToRun & Config::ATOMIC_PRIVATE_STRIDE) {
				if (!first) printf(",\n");
				printf("\"%s\": ", "histogram_atomic_private_stride");
				measureHistogramFunction(
					histogram_atomic_private_stride<Mapping>, config.nRuns, Mapping::numBins,
					deviceWarmupInput, lengthWarmupInput,
					hostInput, deviceInput, sizeInput, config.inputLength,
					hostBins_atomic_private_stride, deviceBins, sizeBins
				);
				first = false;
			}

			if (config.kernelsToRun & Config::ATOMIC_GLOBAL_STRIDE) {
				if (!first) printf(",\n");
				printf("\"%s\": ", "histogram_atomic_global_stride");
				measureHistogramFunction(
					histogram_atomic_global_stride<Mapping>, config.nRuns, Mapping::numBins,
					deviceWarmupInput, lengthWarmupInput,
					hostInput, deviceInput, sizeInput, config.inputLength,
					hostBins_atomic_global_stride, deviceBins, sizeBins
				);
				first = false;
			}

		printf("}\n");

	printf("}\n");


	// gib den GPU-Speicher frei
	(void)hipFree(deviceBins);
	(void)hipFree(deviceInput);
	(void)hipFree(deviceWarmupInput);

	// prüfe das Ergebnis des jeweils letzten Durchlaufs auf Korrektheit ...
    // ... generiere Histogramm auf der CPU, zum Vergleich
	BinType * binsCpu = (BinType *)malloc(Mapping::numBins * sizeof(BinType));
	histogram_cpu<Mapping>(hostInput, binsCpu, config.inputLength);

    // ... und vergleiche
	if (!(
		(
			(config.kernelsToRun & Config::ATOMIC_GLOBAL) == 0 ||
			checkGpuResults<Mapping>(
				"atomic_global",
				hostInput, hostBins_atomic_global, binsCpu, config.inputLength
			)
		) & (
			(config.kernelsToRun & Config::ATOMIC_PRIVATE) == 0 ||
			checkGpuResults<Mapping>(
				"atomic_private",
				hostInput, hostBins_atomic_private, binsCpu, config.inputLength
			)
		) & (
			(config.kernelsToRun & Config::ATOMIC_PRIVATE_STRIDE) == 0 ||
			checkGpuResults<Mapping>(
				"atomic_private_stride",
				hostInput, hostBins_atomic_private_stride, binsCpu, config.inputLength
			)
		) & (
			(config.kernelsToRun & Config::ATOMIC_GLOBAL_STRIDE) == 0 ||
			checkGpuResults<Mapping>(
				"atomic_global_stride",
				hostInput, hostBins_atomic_global_stride, binsCpu, config.inputLength
			)
		)
	)) {
		exit(1);
	}

	// gib den Host-Speicher frei
	free(binsCpu);
	free(hostBins_atomic_global_stride);
	free(hostBins_atomic_private_stride);
	free(hostBins_atomic_private);
	free(hostBins_atomic_global);
}

int main(int argc, char *argv[]) {
	unsigned char * hostInput = nullptr;

	Config config = parseCommandLineArguments(argc, argv);

	if (config.inputFileName) {
		hostInput = (unsigned char *)malloc(sizeof(unsigned char));
		config.inputLength = read_file(config.inputFileName, &hostInput);
		config.uniformInput = false;
	} else {
		hostInput = (unsigned char *)malloc(config.inputLength * sizeof(unsigned char));
		if (!hostInput) {
			fprintf(stderr, "Host memory allocation failed");
			exit(1);
		}
		if (config.uniformInput) {
			for (size_t j = 0; j < config.inputLength; ++j) hostInput[j] = 'a';
		} else {
			randomFill(hostInput, config.inputLength);
		}
	}

	if (config.useMappingLetter) {
		run<MappingLetter> (config, hostInput);
	} else {
		run<Mapping128> (config, hostInput);
	}

	free(hostInput);

	return 0;
}
