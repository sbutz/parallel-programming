#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>

// *****************************************************************************
// Utilities für Error-Checking
// *****************************************************************************

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

// *****************************************************************************
// Device-Speicher, der bei Verlassen des Scopes freigegeben wird
// *****************************************************************************

template <typename T>
struct ManagedDeviceArray {
  explicit ManagedDeviceArray (std::size_t n): n(n) {
    CUDA_CHECK(cudaMalloc(&this->p, n * sizeof(T)))
  }
  ManagedDeviceArray (ManagedDeviceArray<T> const &) = delete;
  ~ManagedDeviceArray () { if (this->p) (void)cudaFree(this->p); }
  inline T * ptr () const { return this->p; }
  inline std::size_t size() const { return this->n; }
private:
  T * p = nullptr;
  std::size_t n = 0;
};

// *****************************************************************************
// Template-Funktion zur Messung der Ausführungsdauer via Cuda-Events
// *****************************************************************************

template <typename Fct, typename ... Args>
inline float runAndMeasureCuda(Fct && f, Args && ... args) {
	float timeInMilliseconds = 0;

	// Zeitmessung: Zeitpunkte definieren
	cudaEvent_t start; CUDA_CHECK(cudaEventCreate(&start));
	cudaEvent_t stop; CUDA_CHECK(cudaEventCreate(&stop));

	// Zeitmessung: Start-Zeitpunkt
	CUDA_CHECK(cudaEventRecord(start));

	std::forward<Fct>(f) (std::forward<Args>(args) ...);

	// Zeitmessung: Stop-Zeitpunkt
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	
	// Berechne das Messergebnis.
	CUDA_CHECK(cudaEventElapsedTime(&timeInMilliseconds, start, stop));

	return timeInMilliseconds;
}

#endif