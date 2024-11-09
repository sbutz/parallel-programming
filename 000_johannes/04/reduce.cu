#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stddef.h>
#include <math.h>
#include <sys/time.h>
#define FULL_MASK 0xffffffff


__global__ void deviceReduceKernel(float *in, float* out, size_t N) {
}



__global__ void InitKernel(float *in, size_t size)
{

	for( size_t id = blockIdx.x*blockDim.x+threadIdx.x; id<size; id+=gridDim.x*blockDim.x)
		in[id]=1.0;

}

void init(float *in, size_t size){
	size_t threads = 1024;
	size_t blocks = min(size_t(size + threads - 1) / threads, (size_t)1024);
	InitKernel<<<blocks,threads>>>(in,size);
	cudaDeviceSynchronize();
}

float reduction(float *in, size_t N)
{
	size_t threads = 512;
	size_t blocks = min(size_t(N + threads - 1) / threads, (size_t)1024);
	float* out;
	float ret;



	return ret;
}


#define ITER 1000
int main (int argc, const char** argv)
{
	size_t size = 8;
	float *input;


	int i;

	//Größe kann als Parameter Übergeben werden
	if (argc > 1) {
		size = atoi(argv[1]);
	}
	float res;
	size=size*(1024 *1024 );
	cudaMalloc(&input, size*sizeof(float));
	init(input,size);

//Wir machen hier mehrere Interationen, um ein gutes Ergebnis zu bekommen
	for(i=0; i<ITER; i++){


		res = reduction(input, size);
	}

	return 0;

}
