#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#define FULL_MASK 0xffffffff

#define N_THREADS 1024
#define SHARED_SIZE N_THREADS * 2
#define N_SIZE 1024 * 1024
#define S_SIZE 8
#define N_TOTAL N_SIZE * S_SIZE
#define ITER 1000 //1000 VEREINFACHUNG

__global__ void deviceReduceKernel(const float* in, float* out, const size_t N) {

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sdata[SHARED_SIZE];

	sdata[tid] = in[i];
	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch	
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (0 == tid)out[blockIdx.x] = sdata[tid];
}

__global__ void deviceReduceKernel_atom(const float* in, float* out, const size_t N) {

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sdata[SHARED_SIZE];

	sdata[tid] = in[i];
	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (0 == tid)atomicAdd(out, sdata[tid]);
}

__global__ void deviceReduceKernel_atom_strided(const float* in, float* out, const size_t N) {

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sdata[SHARED_SIZE];

	sdata[tid] = in[i];
	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if (0 == tid)atomicAdd(out, sdata[tid]);
}

__global__ void deviceReduceKernel_atom_reverse(const float* in, float* out, const size_t N) {

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sdata[SHARED_SIZE];

	sdata[tid] = in[i];
	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch
	for (unsigned int s = blockDim.x /2 ; s > 0; s >>=1) {
		if(tid<s)sdata[tid] += sdata[tid + s];
		
		__syncthreads();
	}

	if (0 == tid)atomicAdd(out, sdata[tid]);
}

__inline__ __device__ void warpReduce(volatile int* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__inline__ __device__ int warpReduceSum(int val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

__inline__ __device__ int blockReduceSum(int val) {

	static __shared__ int shared[32]; // Shared memory für 32 Teil-Threads
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum(val);     // Jeder Warp macht eine Teil-Reduction

	if (lane == 0) shared[wid] = val; // Schreibe den Teil-Wert ins shared memory
	__syncthreads();              // Warte auf alle Teil-Reduktionen 

	//Nur vom Shared memory lesen, wenn der Wert existiert
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) val = warpReduceSum(val); //Jetzt wird der Wert für den letzten Warp Reduziert.

	return val;
}

__global__ void deviceReduceKernelShuffle(const float* in, float* out, const int N) {
	float sum = 0.0;
	/*/reduce multiple elements per thread
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < N;i += blockDim.x * gridDim.x) {
		sum += in[i];
	}//*/

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	sum = in[i] + in[i+blockDim.x]+ in[i+2* blockDim.x] + in[i + 3*blockDim.x]
		+in[i+4*blockDim.x] + in[i + 5*blockDim.x] + in[i + 6 * blockDim.x] + in[i + 7 * blockDim.x];


	sum = blockReduceSum(sum);
	if (threadIdx.x == 0)
		out[blockIdx.x] = sum;
}


__global__ void deviceReduceKernelShuffle2(const float* in, float* out, const int N) {
	float sum = 0.0;
	/*/reduce multiple elements per thread
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < N;i += blockDim.x * gridDim.x) {
		sum += in[i];
	}//*/
		
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sum = in[i]; // +in[i + blockDim.x] + in[i + 2 * blockDim.x] + in[i + 3 * blockDim.x]
		//+ in[i + 4 * blockDim.x] + in[i + 5 * blockDim.x] + in[i + 6 * blockDim.x] + in[i + 7 * blockDim.x];

	int val = (int)sum;

	//blockReduceSum
		static __shared__ int shared_[32]; // Shared memory für 32 Teil-Threads
		int lane = threadIdx.x % warpSize;
		int wid = threadIdx.x / warpSize;
		// Jeder Warp macht eine Teil-Reduction
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
			val += __shfl_down(val, offset);

		if (lane == 0) shared_[wid] = val; // Schreibe den Teil-Wert ins shared memory
		__syncthreads();              // Warte auf alle Teil-Reduktionen 

		//Nur vom Shared memory lesen, wenn der Wert existiert
		val = (threadIdx.x < blockDim.x / warpSize) ? shared_[lane] : 0;

		if (wid == 0) { // pro Block
			//Jetzt wird der Wert für den letzten Warp Reduziert.
			for (int offset = warpSize / 2; offset > 0; offset /= 2)
				val+= __shfl_down(val, offset);			
		}


	//	
	if (threadIdx.x == 0)
		atomicAdd(out, (float)val);
}


float reduction_shuffle(float* in, size_t N)
{
	size_t threads = N_THREADS;
	size_t blocks = size_t(N + threads - 1) / threads;

	float* out, ret = 0;

	cudaMalloc(&out, blocks / 8*sizeof(float));

	deviceReduceKernelShuffle2 << <blocks, N_THREADS >> > (in, out, N);

	cudaMemcpy(&ret, out, sizeof(float), cudaMemcpyDeviceToHost);

	return ret;
}





__global__ void deviceReduceKernel_atom_reverse_unroll(const float* in, float* out, const size_t N) {

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sdata[SHARED_SIZE];

	sdata[tid] = in[i];
	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s)sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// (tid < 32)warpReduce(sdata, tid);
	__syncthreads();
	if (0 == tid)atomicAdd(out, sdata[tid]);
}

//blocks durch 2 bei Aufruf
__global__ void deviceReduceKernel_atom_twoLoads(const float* in, float* out, const size_t N) {

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	extern __shared__ int sdata[SHARED_SIZE];

	sdata[tid] = in[i]+in[i + blockDim.x];
	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if (0 == tid)atomicAdd(out, sdata[tid]);
}

__global__ void deviceReduceKernel_atom_fourLoads(const float* in, float* out, const size_t N) {

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;

	extern __shared__ int sdata[SHARED_SIZE];

	sdata[tid] = in[i] + in[i + blockDim.x]+in[i + blockDim.x*2]+ in[i + blockDim.x*3];
	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if (0 == tid)atomicAdd(out, sdata[tid]);
}

__global__ void deviceReduceKernel_atom_16Loads(const float* in, float* out, const size_t N) {

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 16 + threadIdx.x;

	extern __shared__ int sdata[SHARED_SIZE];

	sdata[tid] = in[i] + in[i + blockDim.x] + in[i + blockDim.x * 2] + in[i + blockDim.x * 3]
		+ in[i+ blockDim.x*5] + in[i + blockDim.x*5] + in[i + blockDim.x * 6] + in[i + blockDim.x * 7]
		+ in[i + blockDim.x * 8] + in[i + blockDim.x * 9] + in[i + blockDim.x * 10] + in[i + blockDim.x * 11]
		+ in[i + blockDim.x * 12] + in[i + blockDim.x * 13] + in[i + blockDim.x * 14] + in[i + blockDim.x * 15];;
	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if (0 == tid)atomicAdd(out, sdata[tid]);
}


__global__ void getSharedKernel(float* sout) {
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[SHARED_SIZE];
	sout[tid] = sdata[tid];
}

void getShared(float* gs, int N) {

	size_t threads = N_THREADS;
	size_t blocks = size_t(N + threads - 1) / threads;

	float* gsd;
	cudaMalloc(&gsd, N_TOTAL * sizeof(float));
	getSharedKernel << <1, N_THREADS >> > (gsd);
	cudaMemcpy(gs, gsd, N_TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
}

void showShared(int blocks, int N, int max) {
	float* get_in = (float*)malloc(sizeof(float) * N_TOTAL);
	getShared(get_in, N);
	printf("\nAusgabe von shared\n");
	for (int i = 0; i < N_THREADS; ++i) {
		printf("%d ", (int)get_in[i]);
		if (0 == ((i + 1) % 10))printf("\n");
		if (max == i) {
			printf("\n\n"); i = blocks;
		}
	}
}

__global__ void InitKernel(float* in, size_t size)
{

	for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += gridDim.x * blockDim.x)
		in[id] = 1.0;

}

void init(float* in, size_t size) {
	size_t threads = N_THREADS;	
	size_t blocks = size_t(size + threads - 1) / threads;
	InitKernel << <blocks, threads >> > (in, size);
	cudaDeviceSynchronize();
}

float reduction(float* in, size_t N)
{
	size_t threads = N_THREADS;
	size_t blocks = size_t(N + threads - 1) / threads; 
		
	float* out;
	cudaMalloc(&out, blocks* sizeof(float));
	float ret=0;

	deviceReduceKernel << <blocks, threads >> > (in, out, N);	
		
	int max_loop = (blocks + 1023) / 1024;
	float* l_ret = (float*)malloc(sizeof(float));
	float* in_out = out;
	
	float* get_in_out = (float*)malloc(blocks * sizeof(float));
	cudaMemcpy(get_in_out, in_out, blocks*sizeof(float), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < max_loop; ++i) {
		deviceReduceKernel << <1, 1024 >> > (in_out, out, blocks);
		cudaMemcpy(l_ret, out, sizeof(float), cudaMemcpyDeviceToHost);
		ret += *l_ret;
		in_out += 1024;
	}

	return ret;
}

__global__ void maxLoad( float* in, const size_t N) {
	extern __shared__ int sdata[SHARED_SIZE];

	sdata[threadIdx.x] = 0.0;
	for (int i = threadIdx.x; i < N; i += gridDim.x * blockDim.x)
		sdata[threadIdx.x] += in[i];

	__syncthreads();
}

__global__ void deviceReduceKernel_maxLoad(const float* in, float* out, const size_t N) {

	extern __shared__ int sdata[SHARED_SIZE];

	unsigned int tid = threadIdx.x;

	// Alle Daten in einen Block aufsummieren
	sdata[threadIdx.x] = 0.0;
	for (int i = threadIdx.x; i < N; i += gridDim.x * blockDim.x)
		sdata[threadIdx.x] += in[i];

	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch	
	for (unsigned int s = 1; s < N_THREADS; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (0 == tid)out[0] = sdata[tid];
}

float reduction_maxLoad(float* in, size_t N)
{
	size_t threads = N_THREADS;
	size_t blocks = size_t(N + threads - 1) / threads;

	float* out, ret = 0;
		
	cudaMalloc(&out, sizeof(float));
		
	deviceReduceKernel_maxLoad << <1, N_THREADS >> > (in, out, N);	
	
	cudaMemcpy(&ret, out, sizeof(float), cudaMemcpyDeviceToHost);

	return ret;
}

float reduction_atom(float* in, size_t N)
{
	size_t threads = N_THREADS;
	size_t blocks = size_t(N + threads - 1) / threads;
	float* out;
	cudaMalloc(&out,sizeof(float));
	float ret;

	deviceReduceKernel_atom_reverse_unroll << <blocks, threads >> > (in, out, N);

	cudaMemcpy(&ret, out, sizeof(float), cudaMemcpyDeviceToHost);

	return ret;
}

int main(int argc, const char** argv)
{
	size_t size = S_SIZE;
	float* input;

	int i;
		
	float res;
	size = size * N_SIZE;

	//Größe kann als Parameter Übergeben werden
	if (argc > 1) {
		size = atoi(argv[1]);
	}

	cudaMalloc(&input, size * sizeof(float));
	init(input, size);

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Wir machen hier mehrere Interationen, um ein gutes Ergebnis zu bekommen
	for (i = 0; i < ITER; i++) {

		res = reduction_shuffle(input, size);
	}
	cudaEventRecord(stop, 0);
	cudaEventElapsedTime(&time, start, stop);

	if (size == res)printf("SUCCESS!"); else printf("Fehler! erwartet: %d, ist: %f\n", size, res);
	printf("\nZeit: %f ms\n", time);

	return 0;

}