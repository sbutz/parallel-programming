#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>

#define DEBUG true
#define SHARED false
#define SHARED_EXT false
#define SHARED_ALL false
#define CUDAMALLOCHOST true
#define KERNEL_DEBUG_T false
#define SAVE_TO_FILE true
#define SAVE_TO_FILE_STRIDE 1
#define READ_FROM_FILE false
#define CONSOLE_AUSGABE false
#define TRENNZEICHEN " "

#define NUM_POINTS (1024 * 16 + 1111) // Maximum NUM_POINTS * NUM_K : 1024 * 64 
#define NUM_K 11
#define MAX_LOOPS 100
#define NUM_THREADS 1024
#define MAX_X 700
#define MAX_Y 500
#define VAR_LOADS 4

#define POINTSFILE "C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Punktdaten/birch2.txt"

#define TYP int
#define LTYP long

typedef struct {
	TYP x;
	TYP y;
	unsigned int myCentroid;
	LTYP distance; //ubound

	//Hambry
	LTYP lbound;
}Point;

typedef struct {
	TYP x;
	TYP y;
	unsigned int total;
	TYP xsum;
	TYP ysum;
	TYP xdist;
	TYP ydist;

	//Hambry
	TYP sxdist;
	TYP sydist;
}Centroid;

void saveK(int*, int*, int);
void saveK(Centroid*, int);
void savePoints(int*, int*);
void savePoints(Point*);
void savePoints(Point*, int);
int getPoints(Point*, std::string, int);
int getPoints(int*, int*, std::string, int);
bool cudaFail(cudaError_t, char*);
cudaError_t reset_device();


//debugtechnische Hilfsroutine
__global__ void delCentroidsSums(Centroid* c, bool compl) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_K) return;
	c[id].xsum = 0;
	c[id].ysum = 0;
	c[id].total = 0;
	if (!compl)return;
	c[id].x = c[id].y = 0;
}

//debugtechnische Hilfsroutine
__global__ void init_Points(Point* p) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id >= NUM_POINTS)return;

	p[id].distance = -1;
	p[id].myCentroid = -1;
}

__device__ LTYP distance_dev(TYP x1, TYP y1, TYP x2, TYP y2) {
	return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

__global__ void kMeansClusterAssign(Point* p, Centroid* c)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS) return;

	//Punkt zum nÃƒÂ¤chsten Zentroid	
	int cluster = -1;
	LTYP pdist = p[id].distance;

#pragma unroll
	for (int j = 0; j < NUM_K; ++j)
	{
		LTYP dist = distance_dev(p[id].x, p[id].y, c[j].x, c[j].y);

		if ((dist < pdist) || (-1 == cluster))
		{
			pdist = dist;
			cluster = j;
		}
	}

	p[id].myCentroid = cluster;
}//*/




/////////// - reduktionsloesungen

__global__ void delete_3arr__TYP(TYP* d1, TYP* d2, TYP* d3) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS * NUM_K)return;
	d1[id] = d2[id] = d3[id] = 0;
}

__global__ void kMeansCentroidUpdate_reduction_Part1(Point* p, TYP* cXSum_arr, TYP* cYSum_arr, TYP* cTotal_arr)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS) return;

	//Thread 0 summiert pro Block
	if (threadIdx.x == 0)
	{
		float cXSum[NUM_K] = { 0 };
		float cYSum[NUM_K] = { 0 };
		int cTotal[NUM_K] = { 0 };

		//im letzten Block darf die Aufsummierung nicht ÃƒÂ¼ber NUM_POINTS hinausgehen
		int loops = blockDim.x;
		if (((blockIdx.x + 1) * blockDim.x) > NUM_POINTS)loops = NUM_POINTS % blockDim.x;

		for (int j = 0; j < loops; ++j)
		{
			int cid = p[j].myCentroid;
			cXSum[cid] += p[j].x;
			cYSum[cid] += p[j].y;
			cTotal[cid]++;
		}

		//atomare Summierung in den globalen Clustervariablen
#pragma unroll
		for (int z = 0; z < NUM_K; ++z)
		{
			cXSum_arr[id * NUM_K + z] = cXSum[z];
			cYSum_arr[id * NUM_K + z] = cYSum[z];
			cTotal_arr[id * NUM_K + z] = cTotal[z];
		}
	}
}


__global__ void kMeansCentroidUpdate_reduction_Part1_Var(Point* p, TYP* cXSum_arr, TYP* cYSum_arr, TYP* cTotal_arr)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS) return;

	int z = p[id].myCentroid;
	cXSum_arr[id * NUM_K + z] = p[id].x;
	cYSum_arr[id * NUM_K + z] = p[id].y;
	cTotal_arr[id * NUM_K + z] = 1;
}


__global__ void reduction_arr_index(TYP* in, TYP* out) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id > NUM_POINTS * NUM_K - 2)return;

	int index = 0;

	for (unsigned int s = 1; s < NUM_POINTS; s *= 2) {
		for (unsigned int t = 0; t < NUM_K; ++t) {

			index = 2 * s * id * NUM_K + t;

			if (index + s * NUM_K < NUM_K * NUM_POINTS)in[index] += in[index + s * NUM_K];

			__syncthreads();
		}
	}

	
}

__global__ void reduction_arr_index_var(TYP* in, TYP* out) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id > NUM_POINTS * NUM_K - 2)return;

	int index = 0;

	for (unsigned int s = 1; s < NUM_POINTS; s *= 2) {

		index = 2 * s * id * NUM_K;

		for (unsigned int t = 0; t < NUM_K; ++t) {

			if (index + s * NUM_K < NUM_K * NUM_POINTS)in[index] += in[index + s * NUM_K];
			index++;

			__syncthreads();
		}
	}

	if (id < NUM_K)out[id] = in[id];
}


__global__ void reduction_arr_modulo(TYP* in, TYP* out) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id > NUM_POINTS * NUM_K - 2)return;

	int index = 0;

	for (unsigned int s = 1; s < NUM_POINTS; s *= 2) {
		for (unsigned int t = 0; t < NUM_K; ++t) {
			if ((id - t) % (2 * s * NUM_K) == 0) {

				in[id] += in[id + s * NUM_K];
			}
			__syncthreads();
		}
	}

	if (id < NUM_K)out[id] = in[id];
}



__global__ void copyOutToIn3(const TYP* in1, const TYP* in2, const TYP* in3, TYP* out1, TYP* out2, TYP* out3, const int max) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= max)return;

	out1[id] = in1[id];
	out2[id] = in2[id];
	out3[id] = in3[id];
}

__global__ void reduction_arr_modulo_shared_var(TYP* in, TYP* out, const int np) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id > np * NUM_K - 2)return;
	unsigned int tid = threadIdx.x;

	__shared__ TYP sh_mem[NUM_THREADS];
	sh_mem[tid] = in[id];

	int index = 0;

	for (unsigned int s = 1; s < np, s < NUM_THREADS; s *= 2) {
		for (unsigned int t = 0; t < NUM_K; ++t) {
			if ((tid - t) % (2 * s * NUM_K) == 0) {

				if (tid + s * NUM_K < NUM_THREADS)sh_mem[tid] += sh_mem[tid + s * NUM_K];
			}
			__syncthreads();
		}
	}

	if (tid < NUM_K)out[blockIdx.x * NUM_K + tid] = sh_mem[tid];
}

__global__ void reduction_arr_modulo_shared(TYP* in, TYP* out) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id > NUM_POINTS * NUM_K - 2)return;
	unsigned int tid = threadIdx.x;

	__shared__ TYP sh_mem[NUM_THREADS];
	sh_mem[tid] = in[id];

	int index = 0;

	for (unsigned int s = 1; s < NUM_POINTS, s < NUM_THREADS; s *= 2) {
		for (unsigned int t = 0; t < NUM_K; ++t) {
			if ((tid - t) % (2 * s * NUM_K) == 0) {

				if (tid + s * NUM_K < NUM_THREADS)sh_mem[tid] += sh_mem[tid + s * NUM_K];
			}
			__syncthreads();
		}
	}

	if (tid < NUM_K)atomicAdd(&out[tid], sh_mem[tid]);
}

__global__ void reduction_arr_varLoads(TYP* in, TYP* out, int varLoads) {

	unsigned int id = blockIdx.x * blockDim.x * varLoads + threadIdx.x;

	LTYP N_TOTAL = NUM_POINTS * NUM_K;

	if (id > N_TOTAL - 2)return;

	TYP s = 0;

	for (int v = 0; v < varLoads; ++v)
		if (id + v * blockDim.x < N_TOTAL)s += in[id + v * blockDim.x];

	in[id] = s;

	for (unsigned int s = 1; s < NUM_POINTS; s *= 2 * varLoads) {
		for (unsigned int t = 0; t < NUM_K; ++t) {
			if ((id - t) % (2 * s * NUM_K * varLoads) == 0) {

				if (id + s * NUM_K < NUM_THREADS)in[id] += in[id + s * NUM_K];
			}
			__syncthreads();
		}
	}
	if (id < NUM_K)out[id] = in[id];
}

__global__ void reduction_arr_shared_varLoads(TYP* in, TYP* out, const int varLoads) {

	unsigned int id = blockIdx.x * blockDim.x * varLoads + threadIdx.x;

	LTYP N_TOTAL = NUM_POINTS * NUM_K;
	
	unsigned int tid = threadIdx.x;

	extern __shared__ TYP sh_val[];

	if (id > N_TOTAL - 2) {
		sh_val[tid] = 0;
		return;
	}

	sh_val[tid] = in[id];

	TYP s = 0;
	for (int v = 0; v < varLoads; ++v)
		s += sh_val[tid + v * blockDim.x];
	sh_val[tid] = s;
	
	for (unsigned int s = 1; s < NUM_POINTS, s < NUM_THREADS; s *= 2 * varLoads) {
		for (unsigned int t = 0; t < NUM_K; ++t) {
			if ((tid - t) % (2 * s * NUM_K) == 0) {

				if (tid + s * NUM_K < NUM_THREADS)sh_val[tid] += sh_val[tid + s * NUM_K];				
			}
			__syncthreads();
		}
	}

	if (tid < NUM_K)atomicAdd(&out[tid], sh_val[tid]);
}

__global__ void reduction_arr_shared_varLoads2(TYP* in, TYP* out, const int varLoads) {

	unsigned int id = blockIdx.x * blockDim.x * varLoads + threadIdx.x;

	LTYP N_TOTAL = NUM_POINTS * NUM_K;

	unsigned int tid = threadIdx.x;

	__shared__ TYP sh_val2[NUM_THREADS];

	if (id > N_TOTAL - 2) {
		sh_val2[tid] = 0;
		return;
	}

	TYP s = 0;
	for (int v = 0; v < varLoads; ++v)
		s += in[id + v * blockDim.x];
	in[id] = s;

	sh_val2[tid] = in[id];

	for (unsigned int s = 1; s < NUM_POINTS, s < NUM_THREADS; s *= 2 * varLoads) {
		for (unsigned int t = 0; t < NUM_K; ++t) {
			if ((tid - t) % (2 * s * NUM_K) == 0) {

				if (tid + s * NUM_K < NUM_THREADS)sh_val2[tid] += sh_val2[tid + s * NUM_K];
			}
			__syncthreads();
		}
	}

	if (tid < NUM_K)atomicAdd(&out[tid], sh_val2[tid]);
}

__global__ void kMeansCentroidUpdate_reduction_Part2(Centroid* c, bool delC) {

	//Neuberechnung der Clusterzentren
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_K)return;

	TYP ocx = c[id].x;
	TYP ocy = c[id].y;

	c[id].x = c[id].xsum / c[id].total;
	c[id].y = c[id].ysum / c[id].total;

	c[id].xdist = ocx - c[id].x;
	c[id].ydist = ocy - c[id].y;

	if (delC) {
		c[id].xsum = 0;
		c[id].ysum = 0;
		c[id].total = 0;
	}
}

__global__ void copyOutToCentroid(Centroid* c, TYP* cxsum, TYP* cysum, TYP* ctotal) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_K)return;

	c[id].total = ctotal[id];
	c[id].xsum = cxsum[id];
	c[id].ysum = cysum[id];
}

/////////// - reduktionsloesungen --- ENDE

int main()
{
	// Zeitmessung
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;

	cudaEventRecord(start, 0);

	// grundsÃƒÂ¤tzliche Variablen
	cudaError_t cudaStatus;
	int psize = NUM_POINTS * sizeof(Point);
	int csize = NUM_K * sizeof(Centroid);
	int blocks = (NUM_POINTS + NUM_THREADS - 1) / NUM_THREADS;


	//Speicherallokation - Datapoints
	Point* hp = (Point*)malloc(psize), * dp = 0;

	// Centroids
	Centroid* c;  cudaStatus = cudaMallocManaged(&c, csize); if (cudaFail(cudaStatus, "Fehler bei cudaMallocMamaged fÃƒÂ¼r c"))return -1;


	//Initialisierung der Datenpunkte		
	if (READ_FROM_FILE) {
		long anz = getPoints(hp, POINTSFILE, NUM_POINTS);
		if (-1 == anz) {
			printf("Datei konnte nicht gelesen werden\n");
			return 1;
		}
	}
	else {
		for (unsigned long i = 0; i < NUM_POINTS; ++i) {
			hp[i].x = rand() % MAX_X; // NotlÃƒÆ’Ã‚Â¶sung
			hp[i].y = rand() % MAX_Y; // NotlÃƒÆ’Ã‚Â¶sung
		}
	}

	if (SAVE_TO_FILE)savePoints(hp);

	//Initialisierung der ersten Zentroide
#pragma unroll	
	for (int i = 0; i < NUM_K; ++i) {
		c[i].x = hp[i].x;
		c[i].y = hp[i].y;
		c[i].xsum = c[i].ysum = c[i].total = 0;
	}

	//Allokation Datenpunkte werden auf Device kopiert    
	if (CUDAMALLOCHOST) {
		cudaStatus = cudaMallocHost(&dp, psize); if (cudaFail(cudaStatus, "cudaMalloc(dp)"))return -1;
	}
	else {
		cudaStatus = cudaMalloc(&dp, psize); if (cudaFail(cudaStatus, "cudaMalloc(dp)"))return -1;
	}
	cudaStatus = cudaMemcpy(dp, hp, psize, cudaMemcpyHostToDevice); if (cudaFail(cudaStatus, "memcpy(hp)"))return -1;


	// Variablen fÃ¼r die Reduktionsloesung
	int arr_size_typ = NUM_K * NUM_POINTS * sizeof(TYP);
	int out_size_typ = NUM_K * sizeof(TYP);
	int blocks_arr = (NUM_POINTS * NUM_K + NUM_THREADS - 1) / NUM_THREADS;
	TYP* dcxsum_arr, * dcysum_arr, * dctotal_arr, * dcxsum, * dcysum, * dctotal;
	cudaStatus = cudaMalloc(&dcxsum_arr, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dcxsum_arr)"))return -1;
	cudaStatus = cudaMalloc(&dcysum_arr, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dcysum_arr)"))return -1;
	cudaStatus = cudaMalloc(&dctotal_arr, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dctotal_arr)"))return -1;
	cudaStatus = cudaMalloc(&dcxsum, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dcxsum)"))return -1;
	cudaStatus = cudaMalloc(&dcysum, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dcysum)"))return -1;
	cudaStatus = cudaMalloc(&dctotal, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dctotal)"))return -1;

	//Loop, um die Zentroide zu finden
	bool changed = true, maxlooped = false;
	unsigned int loops = 0;
	while (changed && !maxlooped)
	{
		if (CONSOLE_AUSGABE) { //Zeige die aktuellen Centroide
			for (int i = 0; i < NUM_K; ++i)printf("Iteration %d: centroid %d x: %d, y: %d, total: %d, \n", loops, i, c[i].x, c[i].y, c[i].total);
		}

		//*//ordner die Punkte den Centroiden zu
		kMeansClusterAssign << <blocks, NUM_THREADS >> > (dp, c);
		cudaDeviceSynchronize();

		if (SAVE_TO_FILE && (loops % SAVE_TO_FILE_STRIDE == 0)) {
			cudaStatus = cudaMemcpy(hp, dp, psize, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy dp - save to file"))return -1;
			saveK(c, loops);
			savePoints(hp, loops);
		}

		//Bestimme die Zentroide neu

		delete_3arr__TYP << <blocks_arr, NUM_THREADS >> > (dcxsum_arr, dcysum_arr, dctotal_arr);
		cudaDeviceSynchronize();

		kMeansCentroidUpdate_reduction_Part1 << <blocks, NUM_THREADS >> > (dp, dcxsum_arr, dcysum_arr, dctotal_arr);
		cudaDeviceSynchronize();

		//reduction_arr_index << <blocks_arr, NUM_THREADS >> > (dcxsum_arr, dcxsum);
		//reduction_arr_index << <blocks_arr, NUM_THREADS >> > (dcysum_arr, dcysum);
		//reduction_arr_index << <blocks_arr, NUM_THREADS >> > (dctotal_arr, dctotal);


		//reduction_arr_varLoads << <blocks_arr/VAR_LOADS, NUM_THREADS >> > (dcxsum_arr, dcxsum, VAR_LOADS);
		//reduction_arr_varLoads << <blocks_arr/VAR_LOADS, NUM_THREADS >> > (dcysum_arr, dcysum, VAR_LOADS);
		//reduction_arr_varLoads << <blocks_arr/VAR_LOADS, NUM_THREADS >> > (dctotal_arr, dctotal, VAR_LOADS);
		
		reduction_arr_shared_varLoads << <blocks_arr/ VAR_LOADS, NUM_THREADS, NUM_THREADS * VAR_LOADS * sizeof(TYP) >> > (dcxsum_arr, dcxsum, VAR_LOADS);
		reduction_arr_shared_varLoads << <blocks_arr/ VAR_LOADS, NUM_THREADS, NUM_THREADS * VAR_LOADS * sizeof(TYP) >> > (dcysum_arr, dcysum, VAR_LOADS);
		reduction_arr_shared_varLoads << <blocks_arr/ VAR_LOADS, NUM_THREADS, NUM_THREADS * VAR_LOADS * sizeof(TYP) >> > (dctotal_arr, dctotal, VAR_LOADS);
		cudaDeviceSynchronize();

		/*
		if (NUM_POINTS <= NUM_THREADS) {
			reduction_arr_modulo_shared_var << <blocks_arr, NUM_THREADS >> > (dcxsum_arr, dcxsum, NUM_POINTS);
			reduction_arr_modulo_shared_var << <blocks_arr, NUM_THREADS >> > (dcysum_arr, dcysum, NUM_POINTS);
			reduction_arr_modulo_shared_var << <blocks_arr, NUM_THREADS >> > (dctotal_arr, dctotal, NUM_POINTS);
			cudaDeviceSynchronize();
		}
		else {

			int blocks_arr_loop = blocks_arr;
			int point_size = NUM_POINTS;

			do {
				reduction_arr_modulo_shared_var << <blocks_arr_loop, NUM_THREADS >> > (dcxsum_arr, dcxsum, point_size);
				reduction_arr_modulo_shared_var << <blocks_arr_loop, NUM_THREADS >> > (dcysum_arr, dcysum, point_size);
				reduction_arr_modulo_shared_var << < blocks_arr_loop, NUM_THREADS >> > (dctotal_arr, dctotal, point_size);
				cudaDeviceSynchronize();

				copyOutToIn3 << < blocks, NUM_THREADS >> > (dcxsum, dcysum, dctotal, dcxsum_arr, dcysum_arr, dctotal_arr, blocks_arr_loop * NUM_K);
				cudaDeviceSynchronize();

				point_size = blocks_arr_loop;
				blocks_arr_loop = (point_size * NUM_K + NUM_THREADS - 1) / NUM_THREADS;
			} while (point_size > NUM_THREADS);
		}*/



		copyOutToCentroid << < blocks, NUM_THREADS >> > (c, dcxsum, dcysum, dctotal);
		cudaDeviceSynchronize();

		kMeansCentroidUpdate_reduction_Part2 << <blocks, NUM_THREADS >> > (c, true);
		cudaDeviceSynchronize();

		changed = false;
		for (int i = 0; i < NUM_K; ++i) {

			//Blieben die Centroide stehen?
			if ((0 != c[i].xdist) && (0 != c[i].ydist))changed = true;
		}

		// Begrenzung auf maximale Anzahl von DurchlÃƒÆ’Ã‚Â¤ufen
		loops++;
		if (!changed)printf("\nkeine weitere Bewegung der Zentroide, Beendigung des Algorithmus\n");
		if ((-1 != MAX_LOOPS) && (loops > MAX_LOOPS)) {
			maxlooped = true;
			printf("\nmaximale Anzahl der Schleifendurchlauefe erreicht, Abbruch des Algorithmus\n");
			loops--;
		}
	}

	printf("Anzahl Loops %d\n", loops);

	cudaFree(c);
	if (CUDAMALLOCHOST) {
		cudaFreeHost(dp);
	}
	else {
		cudaFree(dp);
	}

	//Reduktionsvariablen
	cudaFree(dcxsum);
	cudaFree(dcysum);
	cudaFree(dctotal);
	cudaFree(dcxsum_arr);
	cudaFree(dcysum_arr);
	cudaFree(dctotal_arr);


	// Zeitmessung Ende
	cudaEventRecord(stop, 0);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("\nAusfÃƒÂ¼hrungszeit: %f ms (Anzahl Datenpunkte: %d, Anzahl Centroide: %d)", time, NUM_POINTS, NUM_K);

	return reset_device();
}



// ----------------------------------------------------------------------------------------------------------------------------------

// Hilfsroutine fÃƒÂ¼r Cuda

cudaError_t reset_device() {
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceReset failed!");
	return cudaStatus;
}

bool cudaFail(cudaError_t cudaStatus, char* dev) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Fehler bei ");
		fprintf(stderr, dev);
		fprintf(stderr, "\n");
		return true;
	}
	return false;
}

// Hilfsroutinen zum Laden & Speichern

int getPoints(Point* p, std::string filename, int maxLines) {
	std::fstream new_file;

	// open a file to perform read operation using file object.
	new_file.open(filename, std::ios::in);
	if (new_file.fail()) { printf("\Schreibfehler Datei (getPoints)\n"); return -1; }

	// Checking whether the file is open.
	if (new_file.is_open()) {
		std::string sa, sz1, sz2;
		std::size_t komma;
		int zaehler = 0;
		// Read data from the file object and put it into a string.
		while (getline(new_file, sa) && zaehler < maxLines) {

			komma = sa.find(TRENNZEICHEN);
			if ((komma == std::string::npos) || (komma < 1) || (komma == sa.length())) {
				printf("Falsches Zahlenformat");
				return -1;
			}

			sz1 = sa.substr(0, komma);
			sz2 = sa.substr(komma + 1, sa.length());

			p[zaehler].x = stoi(sz1);
			p[zaehler].y = stoi(sz2);

			zaehler++;
		}

		// Close the file object.
		new_file.close();
		return zaehler;
	}
}

int getPoints(int* x, int* y, std::string filename, int maxLines) {

	std::fstream new_file;

	// open a file to perform read operation using file object.
	new_file.open(filename, std::ios::in);
	if (new_file.fail()) { printf("\Schreibfehler Datei (getPoints)\n"); return -1; }

	// Checking whether the file is open.
	if (new_file.is_open()) {
		std::string sa, sz1, sz2;
		std::size_t komma;
		int zaehler = 0;
		// Read data from the file object and put it into a string.
		while (getline(new_file, sa) && zaehler < maxLines) {

			komma = sa.find(TRENNZEICHEN);
			if ((komma == std::string::npos) || (komma < 1) || (komma == sa.length())) {
				printf("Falsches Zahlenformat");
				return -1;
			}

			sz1 = sa.substr(0, komma);
			sz2 = sa.substr(komma + 1, sa.length());

			x[zaehler] = stoi(sz1);
			y[zaehler] = stoi(sz2);

			zaehler++;
		}

		// Close the file object.
		new_file.close();
		return zaehler;
	}
}

void savePoints(int* x, int* y) {

	std::ofstream file("C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/points0.dat", std::ios::out);
	if (file.fail()) { printf("\Schreibfehler Datei (Punkte)\n"); return; }

	for (int i = 0; i < NUM_POINTS; ++i)
		file << x[i] << TRENNZEICHEN << y[i] << std::endl;

	file.close();
}

void savePoints(Point* p) {

	std::ofstream file("C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/points0.dat", std::ios::out);
	if (file.fail()) { printf("\Schreibfehler Datei (Punkte)\n"); return; }

	for (int i = 0; i < NUM_POINTS; ++i)
		file << p[i].x << TRENNZEICHEN << p[i].y << std::endl;

	file.close();
}

void savePoints(Point* p, int num) {
	char numstr[33];
	itoa(num, numstr, 10);

	std::ofstream file("C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/points" + std::string(numstr) + ".dat", std::ios::out);
	if (file.fail()) { printf("\Schreibfehler Datei (Punkte)\n"); return; }

	for (int i = 0; i < NUM_POINTS; ++i)
		file << p[i].x << TRENNZEICHEN << p[i].y << TRENNZEICHEN << p[i].myCentroid << std::endl;

	file.close();
}

void saveK(int* x, int* y, int num) {
	char numstr[33];
	itoa(num, numstr, 10);

	std::string s = "C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/pointsK" + std::string(numstr) + ".dat";
	std::ofstream file(s, std::ios::out);

	if (file.fail()) { printf("\Schreibfehler Datei (K)\n"); return; }

	for (int i = 0; i < NUM_K; ++i)
		file << x[i] << TRENNZEICHEN << y[i] << std::endl;

	file.close();
}

void saveK(Centroid* c, int num) {
	char numstr[33];
	itoa(num, numstr, 10);

	std::string s = "C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/pointsK" + std::string(numstr) + ".dat";
	std::ofstream file(s, std::ios::out);

	if (file.fail()) { printf("\Schreibfehler Datei (K)\n"); return; }

	for (int i = 0; i < NUM_K; ++i)
		file << c[i].x << TRENNZEICHEN << c[i].y << std::endl;

	file.close();
}
