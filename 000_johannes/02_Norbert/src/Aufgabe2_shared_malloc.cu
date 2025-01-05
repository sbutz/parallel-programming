#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>


// Thread block size: BLOCK_SIZE * BLOCK_SIZE
#define BLOCK_SIZE  16
#define __N__ 1026

// In Anlehnung an http://www.techdarting.com/2014/03/matrix-multiplication-in-cuda-using.html
__global__ void dgemm_gpu_shared(const float* a, const float* b, float* c, const int n) {

	//Todo: implement Kernel Here
	__shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];   // Tile size of 32x32
	__shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;
	float Cvalue = 0.0;
	sA[threadIdx.y][threadIdx.x] = 0.0;
	sB[threadIdx.y][threadIdx.x] = 0.0;

	int Row_n = Row * n;
	int anz_BL = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // Number of Blocks in Matrix
	int ph, j;

	for (ph = 0; ph < anz_BL; ph++) {

		// Load
		int ph_BLOCK_SIZE = ph * BLOCK_SIZE;

		sA[threadIdx.y][threadIdx.x] = a[Row_n + threadIdx.x + ph_BLOCK_SIZE];
		sB[threadIdx.y][threadIdx.x] = b[(threadIdx.y + ph_BLOCK_SIZE) * n + Col];
		__syncthreads();

		// Calc
		for (j = 0; j < BLOCK_SIZE; ++j) {
			Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
		}
	}

	//Store 
	c[Row_n + Col] = Cvalue;
}



// get compute performance
float getGflops(int width, float time) {

	float gf = (1.0e-6 * width * width * width / time);

	return gf;
}

int main(int argc, const char** argv) {

	int n = __N__; // dimension of square matrices
	float* h_a = 0, * h_b = 0, * h_c = 0;
	float* d_a, * d_b, * d_c;
	int row, col;
	float absError, maxAbsError = 0.0, sumAbsError = 0.0;
	size_t size;
	float time;
	cudaEvent_t start, stop;
	//Möglichkeit mit GPUs Zeit zu messen 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Größe der Matrix kann als Argument  übergeben werden
	if (argc > 1) {
		n = atoi(argv[1]);
	}

	size = n * n * sizeof(float);
	//TODO  Allokiere Speicher auf dem host und GPU
	h_a = (float*)malloc(size);
	h_b = (float*)malloc(size);
	h_c = (float*)malloc(size);
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

	// Führe die Matrix-Matrix Multiplikation aus
	//16x16 ist vorgegeben

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((n + BLOCK_SIZE - 1) / blockDim.x, (n + BLOCK_SIZE - 1) / blockDim.y);

	cudaEventRecord(start, 0);
	//TODO kopiere daten auf die GPU und Rufe den Kernel auf  und Kopiere das Ergebniss zurrück

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	dgemm_gpu_shared << < gridDim, blockDim >> > (d_a, d_b, d_c, n);
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

	free(h_c);
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
