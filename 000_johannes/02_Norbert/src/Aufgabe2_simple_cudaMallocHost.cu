#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

// Thread block size: BLOCK_SIZE * BLOCK_SIZE
#define BLOCK_SIZE 16 

__global__ void dgemm_gpu_simple(const float* a, const float* b, float* c, const int n) {

	//Todo: implement Kernel Here
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = x + y * blockDim.x * gridDim.x;

	int vx = id % n; // Spalte
	int vy = id - vx; // Zeile
	int i;

	float s = 0.0;
	for (i = 0; i < n; ++i)
		s += a[vy + i] * b[vx + i * n];
	c[id] = s;
}

// get compute performance
float getGflops(int width, float time) {

	float gf = (1.0e-6 * width * width * width / time);

	return gf;
}

int main(int argc, const char** argv) {

	int n = 8192; // dimension of square matrices
	float* h_a = 0, * h_b = 0, * h_c = 0;
	float* d_a, * d_b, * d_c;
	int row, col;
	float absError, maxAbsError = 0.0, sumAbsError = 0.0;
	size_t size;
	float time;
	cudaEvent_t start, stop;
	//MÃ¶glichkeit mit GPUs Zeit zu messen 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// GrÃ¶ÃŸe der Matrix kann als Argument  Ã¼bergeben werden
	if (argc > 1) {
		n = atoi(argv[1]);
	}

	size = n * n * sizeof(float);
	//TODO  Allokiere Speicher auf dem host und GPU
	h_a = (float*)malloc(size);
	h_b = (float*)malloc(size);
	cudaMallocHost((void**)&h_c, size);
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	//Initialisierung, parallelisiert mit OpenMP
	if (h_a && h_b) {

		//#pragma omp parallel for
		for (row = 0; row < n; row++) {
			for (col = 0; col < n; col++) {
				h_a[row * n + col] = (row == col) ? 1.0 : 0.0;
				h_b[row * n + col] = row * n + col;
			}
		}
	}
	else return 1;

	// FÃ¼hre die Matrix-Matrix Multiplikation aus
	//16x16 ist vorgegeben

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((n + BLOCK_SIZE - 1) / blockDim.x, (n + BLOCK_SIZE - 1) / blockDim.y);

	cudaEventRecord(start, 0);
	//TODO kopiere daten auf die GPU und Rufe den Kernel auf  und Kopiere das Ergebniss zurrÃ¼ck

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	dgemm_gpu_simple << < gridDim, blockDim >> > (d_a, d_b, d_c, n);
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);


	// Teste das Ergebnis  
	for (row = 0; row < n; ++row) {
		for (col = 0; col < n; ++col) {

			absError = fabs(h_c[row * n + col] - h_b[row * n + col]);
			sumAbsError += absError;

			if (absError > maxAbsError)
				maxAbsError = absError;
		}
	}
	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//TODO Gebe den Speicher auf GPU und Host frei
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	cudaFreeHost(h_c);
	free(h_b);
	free(h_a);

	printf("\nmaxAbsError: %4.4f, sumAbsError: %4.4f\n", maxAbsError, sumAbsError);
	if (maxAbsError < 2.0e-5) {
		printf("\nProgram terminated SUCCESSFULLY.\n");
		printf("\nKernel Execution Time: %f ms (dim C: %d * %d)", time, n, n);
		printf("\nThis corresponds to: %4.4f GFLOPS\n\n", getGflops(n, time));
	}
	else {
		printf("\n--> Result not correct:  check your code\n\n");
	}

	return 0;
}
