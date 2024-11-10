#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using std::size_t;

constexpr size_t M = 1000;
constexpr size_t K = 2000;
constexpr size_t N = 1500;

__global__ void mmult_simple_kernel(
	float * C, float * A, float * B, size_t m, size_t k, size_t n
) {
	size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  size_t c = blockIdx.y * blockDim.y + threadIdx.y;

  if (r >= m || c >= n) return;
  float s = 0.0f;
  for (size_t kk = 0; kk < k; ++kk) {
    s += A[k * r + kk] * B[c + n * kk];
  }
  C[n * r + c] = s;
}

int main () {
  float * A = (float *) malloc (M * K * sizeof(float));
  float * B = (float *) malloc (K * N * sizeof(float));
  float * C = (float *) malloc (M * N * sizeof(float));

  for (size_t i = 0; i < M * K; ++i) A[i] = (float) i;
  for (size_t i = 0; i < K * N; ++i) B[i] = (float) i;

  float * deviceA, * deviceB, * deviceC;
	cudaMalloc((void **) &deviceA, M * K * sizeof(float));
	cudaMalloc((void **) &deviceB, K * N * sizeof(float));
  cudaMalloc((void **) &deviceC, M * N * sizeof(float));

	// kopiere die Daten auf die GPU
	cudaMemcpy(deviceA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  dim3 dimGrid((M - 1)/32 + 1, (N - 1) / 32 + 1, 1);
  dim3 dimBlock(32, 32, 1);
  cudaEventRecord(start);
  mmult_simple_kernel<<<dimGrid, dimBlock>>> (deviceC, deviceA, deviceB, M, K, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaMemcpy(C, deviceC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	float calcTime = 0;
	cudaEventElapsedTime(&calcTime, start, stop);
	printf("Calc time %.4f ms\n", calcTime);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  // check
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j) {
      float s = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        s += ((float)i * K + k) * ((float) j + (float)k * N);
      }
      if (abs(C[N * i + j]/s - 1) > 0.0001) {
        printf("Wrong result at %lu %lu -- expected: %f, actual: %f", i, j, s, C[N*i +j]); exit(0);    
      }
    }
  return 0;
}