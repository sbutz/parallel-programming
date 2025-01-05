#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using std::size_t;

constexpr size_t M = 1000;
constexpr size_t K = 2000;
constexpr size_t N = 1500;

template <size_t BM, size_t BN, size_t BK, size_t TM>
__global__ void mmult_1dblocktile_kernel(
	float * C, float * A, float * B, size_t m, size_t k, size_t n
) {
	size_t r = blockIdx.y;
  size_t c = blockIdx.x;
  size_t threadR = threadIdx.x / BN;
  size_t threadC = threadIdx.x % BN;

  __shared__ float As[BM * BK], Bs[BK * BN];

  size_t baseIdxA = r * BM * k;
  size_t baseIdxB = c * BN;

  size_t innerRowA = threadIdx.x / BK;
  size_t innerColA = threadIdx.x % BK;
  size_t innerRowB = threadIdx.x / BN;
  size_t innerColB = threadIdx.x % BN;

  float threadResults[TM] = { 0.0f };
  for (size_t blkIdx = 0; blkIdx < k; blkIdx += BK) {
    As[innerRowA * BK + innerColA] =
      r * BM + innerRowA < m &&
      blkIdx + innerColA < k ?
        A[baseIdxA + innerRowA * k + innerColA] 
      :
        0.0f;
    Bs[innerRowB * BN + innerColB] = 
      blkIdx + innerRowB < k &&
      c * BN + innerColB < n ?
        B[baseIdxB + innerRowB * n + innerColB]
      :
        0.0f;
    __syncthreads();

    baseIdxA += BK;
    baseIdxB += BK * n;

    for (size_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadC];
      for (size_t resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] += As[(threadR * TM + resIdx) * BK + dotIdx] * tmpB;
      } 
    }
    __syncthreads();
  }

  for (size_t resIdx = 0; resIdx < TM; ++resIdx) {
    if (
      r * BM + threadR * TM + resIdx < m &&
      c * BN + threadC < n
    ) {
      C [(r * BM + threadR * TM + resIdx) * n + c * BN + threadC] = threadResults[resIdx];
    }
  }
}

int main () {
  float * A = (float *) malloc (M * K * sizeof(float));
  float * B = (float *) malloc (K * N * sizeof(float));
  float * C = (float *) malloc (M * N * sizeof(float));

  for (size_t i = 0; i < M * K; ++i) A[i] = (float) i;
  for (size_t i = 0; i < K * N; ++i) B[i] = (float) 1;

  float * deviceA, * deviceB, * deviceC;
	cudaMalloc((void **) &deviceA, M * K * sizeof(float));
	cudaMalloc((void **) &deviceB, K * N * sizeof(float));
  cudaMalloc((void **) &deviceC, M * N * sizeof(float));

	// kopiere die Daten auf die GPU
	cudaMemcpy(deviceA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(deviceC, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  constexpr size_t BM = 64;
  constexpr size_t BN = 64;
  constexpr size_t BK = 8;
  constexpr size_t TM = 8;

  dim3 dimGrid((N - 1) / BN + 1, (M - 1) / BM + 1, 1);
  dim3 dimBlock(BM * BN / TM);
  cudaEventRecord(start);
  mmult_1dblocktile_kernel<BM, BN, BK, TM> <<<dimGrid, dimBlock>>> (deviceC, deviceA, deviceB, M, K, N);
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
        s += A[i * K + k] * B[j + k * N];
      }
      if (abs(C[N * i + j]/s - 1) > 0.0001) {
        printf("Wrong result at %lu %lu -- expected: %f, actual: %f", i, j, s, C[N*i +j]); exit(0);    
      }
    }
  return 0;
}