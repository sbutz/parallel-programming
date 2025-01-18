#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

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

#endif