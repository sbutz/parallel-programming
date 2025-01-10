#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>

#define DEBUG true
#define SHARED true
#define SHARED_EXT true
#define CUDAMALLOCHOST true
#define KERNEL_DEBUG_T false
#define SAVE_TO_FILE true
#define READ_FROM_FILE false
#define CONSOLE_AUSGABE false

#define NUM_POINTS 1024 * 16
#define NUM_K 13
#define MAX_LOOPS 100
#define NUM_THREADS 1024
#define MAX_X 700;
#define MAX_Y 500;

#define POINTSFILE "C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/points_3012.txt"

#define TYP int
#define LTYP long

typedef struct {
	TYP x;
	TYP y;
	unsigned int myCentroid;
	LTYP distance;
}Point;

typedef struct {
	TYP x;
	TYP y;
	unsigned int total;
	TYP xsum;
	TYP ysum;
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

	//Punkt zum nächsten Zentroid	
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


__global__ void kMeansCentroidUpdate(Point* p, Centroid* c, bool delC)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS) return;	
	
	//Thread 0 summiert pro Block
	if (threadIdx.x == 0)
	{
		float cXSum[NUM_K] = { 0 };
		float cYSum[NUM_K] = { 0 };
		int cTotal[NUM_K] = { 0 };

		//im letzten Block darf die Aufsummierung nicht über NUM_POINTS hinausgehen
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
			atomicAdd(&c[z].xsum, cXSum[z]);
			atomicAdd(&c[z].ysum, cYSum[z]);
			atomicAdd(&c[z].total, cTotal[z]);
		}
	}

	__syncthreads();

	//Neuberechnung der Clusterzentren
	if (id >= NUM_K)return;

	c[id].x = c[id].xsum / c[id].total;
	c[id].y = c[id].ysum / c[id].total;

	if (delC) {
		c[id].xsum = 0;
		c[id].ysum = 0;
		c[id].total = 0;
	}
}


__global__ void kMeansClusterAssignSharedMemoryExt(Point* p, Centroid* c)
{	
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS) return;
	const int tid = threadIdx.x;

	__shared__ TYP scX[NUM_K];
	__shared__ TYP scY[NUM_K];

	if (tid < NUM_K) {
		scX[tid] = c[tid].x;
		scY[tid] = c[tid].y;
	}

	//__syncthreads(); // nicht erforderlich

	__shared__ TYP spX[NUM_THREADS];
	__shared__ TYP spY[NUM_THREADS];

	spX[tid] = p[id].x;
	spY[tid] = p[id].y;	

	__syncthreads();

	//Punkt zum nächsten Zentroid	
	int cluster = -1;
	LTYP pdist = p[id].distance;

	#pragma unroll
	for (int j = 0; j < NUM_K; ++j)
	{		
		LTYP dist = distance_dev(spX[tid], spY[tid], scX[j], scY[j]);

		if ((dist < pdist)||(-1==cluster))
		{
			pdist = dist;
			cluster = j;
		}
	}
		
	p[id].myCentroid = cluster;
}//*/

__global__ void kMeansClusterAssignSharedMemory(Point* p, Centroid* c)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS) return;
	const int tid = threadIdx.x;
			
	__shared__ TYP spX[NUM_THREADS];
	__shared__ TYP spY[NUM_THREADS];

	spX[tid] = p[id].x;
	spY[tid] = p[id].y;

	__syncthreads();

	//Punkt zum nächsten Zentroid	
	int cluster = -1;
	LTYP pdist = p[id].distance;

#pragma unroll
	for (int j = 0; j < NUM_K; ++j)
	{
		LTYP dist = distance_dev(spX[tid], spY[tid], c[j].x, c[j].y);

		if ((dist < pdist) || (-1 == cluster))
		{
			pdist = dist;
			cluster = j;
		}
	}

	p[id].myCentroid = cluster;
}//*/


__global__ void kMeansCentroidUpdateSharedMemory(Point* p, Centroid* c, bool delC)
{		
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS) return;
	const int tid = threadIdx.x;

	//shared memory für Datenpunkte und Cluster
	__shared__ TYP spX[NUM_THREADS];
	__shared__ TYP spY[NUM_THREADS];
	__shared__ int spC[NUM_THREADS];
	spX[tid] = p[id].x;
	spY[tid] = p[id].y;
	spC[tid] = p[id].myCentroid;

	__syncthreads();

	//Thread 0 summiert pro Block
	if (tid == 0)
	{
		float cXSum[NUM_K] = { 0 };
		float cYSum[NUM_K] = { 0 };
		int cTotal[NUM_K] = { 0 };

		//im letzten Block darf die Aufsummierung nicht über NUM_POINTS hinausgehen
		int loops = blockDim.x;
		if (((blockIdx.x + 1) * blockDim.x) > NUM_POINTS)loops = NUM_POINTS % blockDim.x;

		for (int j = 0; j < loops; ++j)
		{
			int cid = spC[j];
			cXSum[cid] += spX[j];
			cYSum[cid] += spY[j];
			cTotal[cid] ++;
		}

		//atomare Summierung in den globalen Clustervariablen
		#pragma unroll
		for (int z = 0; z < NUM_K; ++z)
		{
			atomicAdd(&c[z].xsum, cXSum[z]);
			atomicAdd(&c[z].ysum, cYSum[z]);
			atomicAdd(&c[z].total, cTotal[z]);
		}
	}

	__syncthreads();

	//Neuberechnung der Clusterzentren
	if (id >= NUM_K)return;
	
	c[id].x = c[id].xsum / c[id].total;
	c[id].y = c[id].ysum / c[id].total;	

	if (delC) {
		c[id].xsum = 0;
		c[id].ysum = 0;
		c[id].total = 0;
	}
}





int main()
{
	// Zeitmessung
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;

	cudaEventRecord(start, 0);

	// grundsätzliche Variablen
	cudaError_t cudaStatus;
	int psize = NUM_POINTS * sizeof(Point);
	int csize = NUM_K * sizeof(Centroid);
	int blocks = (NUM_POINTS + NUM_THREADS - 1) / NUM_THREADS;

	//Speicherallokation
	Point* hp = (Point*)malloc(psize), * dp = 0;          // Datapoints
	Centroid* hc = (Centroid*)malloc(csize), * dc = 0;    // Centroids
	Centroid* oc = (Centroid*)malloc(csize);             // former Centroids for comparison purposes
		
	//Initialisierung der Datenpunkte		
	if (READ_FROM_FILE) {
		int anz = getPoints(hp, POINTSFILE, 65);
		if (-1 == anz) {
			printf("Datei konnte nicht gelesen werden\n");
			return 1;
		}
	}
	else {
		for (unsigned long i = 0; i < NUM_POINTS; ++i) {
			hp[i].x = rand() % MAX_X; // NotlÃ¶sung
			hp[i].y = rand() % MAX_Y; // NotlÃ¶sung
		}
	}

	if (SAVE_TO_FILE)savePoints(hp);
	
	//Initialisierung der ersten Zentroide
	#pragma unroll	
	for (int i = 0; i < NUM_K; ++i) {
		oc[i].x = hc[i].x = hp[i].x;
		oc[i].y = hc[i].y = hp[i].y;
		hc[i].xsum = hc[i].ysum = hc[i].total = 0;
	}

	//Allokation Datenpunkte werden auf Device kopiert    
	if (CUDAMALLOCHOST) {
		cudaStatus = cudaMallocHost(&dp, psize); if (cudaFail(cudaStatus, "cudaMalloc(dp)"))return -1;
	}
	else {
		cudaStatus = cudaMalloc(&dp, psize); if (cudaFail(cudaStatus, "cudaMalloc(dp)"))return -1;
	}
	cudaStatus = cudaMemcpy(dp, hp, psize, cudaMemcpyHostToDevice); if (cudaFail(cudaStatus, "memcpy(hp)"))return -1;
	
	//Allokation dc und Copy on device
	if (CUDAMALLOCHOST) {
		cudaStatus = cudaMallocHost(&dc, csize); if (cudaFail(cudaStatus, "cudaMalloc(dc)"))return -1;
	}
	else{
		cudaStatus = cudaMalloc(&dc, csize); if (cudaFail(cudaStatus, "cudaMalloc(dc)"))return -1;
	}
	cudaStatus = cudaMemcpy(dc, hc, csize, cudaMemcpyHostToDevice); if (cudaFail(cudaStatus, "Memcpy hc"))return -1;
	
	//Loop, um die Zentroide zu finden
	bool changed = true, maxlooped = false;
	unsigned int loops = 0;
	while (changed && !maxlooped)
	{
		if(CONSOLE_AUSGABE){ //Zeige die aktuellen Centroide
			cudaStatus = cudaMemcpy(hc, dc, csize, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy dc"))return -1;

			for (int i = 0; i < NUM_K; ++i)printf("Iteration %d: centroid %d x: %d, y: %d, total: %d, \n", loops, i, hc[i].x, hc[i].y, hc[i].total);
		}

		//*//ordner die Punkte den Centroiden zu
		if (SHARED) {
			if ((NUM_K <= NUM_THREADS)&&SHARED_EXT)				
				kMeansClusterAssignSharedMemoryExt << <blocks, NUM_THREADS >> > (dp, dc);
			else
				kMeansClusterAssignSharedMemory << <blocks, NUM_THREADS >> > (dp, dc);
		}
		else
			kMeansClusterAssign << <blocks, NUM_THREADS >> > (dp, dc);
		cudaDeviceSynchronize();

		if (SAVE_TO_FILE) {
			cudaStatus = cudaMemcpy(hp, dp, psize, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy dp - save to file"))return -1;
			saveK(hc, loops);
			savePoints(hp, loops);
		}
		
		//Bestimme die Zentroide neu
		if(SHARED)
			kMeansCentroidUpdateSharedMemory << <blocks, NUM_THREADS >> > (dp, dc, true);
		else
			kMeansCentroidUpdate << <blocks, NUM_THREADS >> > (dp, dc, true);
		cudaDeviceSynchronize();//*/

		//kMeansClusterAssignSharedMemory_AndCentroidUpdate << <blocks, NUM_THREADS >> > (dp, dc, true); cudaDeviceSynchronize();

		//nochmaliges Herholen der Daten in hc ist wegen dem Update von dc erforderlich
		cudaStatus = cudaMemcpy(hc, dc, csize, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy dc"))return -1;

		changed = false;
		for (int i = 0; i < NUM_K; ++i) {

			//Blieben die Centroide stehen?
			if ((hc[i].x != oc[i].x) || (hc[i].y != oc[i].y))changed = true;

			//speichere bisherige Clusterzuordnung
			oc[i].x = hc[i].x;
			oc[i].y = hc[i].y;
		}
			
		// Begrenzung auf maximale Anzahl von DurchlÃ¤ufen
		loops++;
		if (!changed)printf("\nkeine weitere Bewegung der Zentroide, Beendigung des Algorithmus\n");
		if ((-1 != MAX_LOOPS) && (loops > MAX_LOOPS)) {
			maxlooped = true;
			printf("\nmaximale Anzahl der Schleifendurchlauefe erreicht, Abbruch des Algorithmus\n");
			loops--;
		}
	}

	printf("Anzahl Loops %d\n", loops);

	if (CUDAMALLOCHOST) {
		cudaFreeHost(dp);
		cudaFreeHost(dc);
	}
	else {
		cudaFree(dp);
		cudaFree(dc);
	}
	
	free(hc);
	free(oc);
	free(hp);

	// Zeitmessung Ende
	cudaEventRecord(stop, 0);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("\nAusführungszeit: %f ms (Anzahl Datenpunkte: %d, Anzahl Centroide: %d)", time, NUM_POINTS, NUM_K);

	return reset_device();
}



// ----------------------------------------------------------------------------------------------------------------------------------

// Hilfsroutine für Cuda

cudaError_t reset_device() {
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceReset failed!");
	return cudaStatus;
}

bool cudaFail(cudaError_t cudaStatus, char* dev){
	if (cudaStatus != cudaSuccess) { 
		fprintf(stderr, "Fehler bei "); 
		fprintf(stderr, dev);  
		fprintf(stderr, "\n");
		return true; }
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

			komma = sa.find(",");
			if ((komma == std::string::npos) || (komma < 1) || (komma == sa.length())) {
				printf("Falsches Zahlenformat");
				return -1;
			}

			sz1 = sa.substr(0, komma - 1);
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

			komma = sa.find(",");
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
		file << x[i] << "," << y[i] << std::endl;

	file.close();
}

void savePoints(Point* p) {

	std::ofstream file("C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/points0.dat", std::ios::out);
	if (file.fail()) { printf("\Schreibfehler Datei (Punkte)\n"); return; }

	for (int i = 0; i < NUM_POINTS; ++i)
		file << p[i].x << "," << p[i].y << std::endl;

	file.close();
}

void savePoints(Point* p, int num) {
	char numstr[33];
	itoa(num, numstr, 10);

	std::ofstream file("C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/points" + std::string(numstr) + ".dat", std::ios::out);
	if (file.fail()) { printf("\Schreibfehler Datei (Punkte)\n"); return; }

	for (int i = 0; i < NUM_POINTS; ++i)
		file << p[i].x << "," << p[i].y << "," << p[i].myCentroid << std::endl;

	file.close();
}

void saveK(int* x, int* y, int num) {
	char numstr[33];
	itoa(num, numstr, 10);

	std::string s = "C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/pointsK" + std::string(numstr) + ".dat";
	std::ofstream file(s, std::ios::out);

	if (file.fail()) { printf("\Schreibfehler Datei (K)\n"); return; }

	for (int i = 0; i < NUM_K; ++i)
		file << x[i] << "," << y[i] << std::endl;

	file.close();
}

void saveK(Centroid* c, int num) {
	char numstr[33];
	itoa(num, numstr, 10);

	std::string s = "C:/Users/nbaum/Documents/fu_hagen/parallel/Cluster/Points/pointsK" + std::string(numstr) + ".dat";
	std::ofstream file(s, std::ios::out);

	if (file.fail()) { printf("\Schreibfehler Datei (K)\n"); return; }

	for (int i = 0; i < NUM_K; ++i)
		file << c[i].x << "," << c[i].y << std::endl;

	file.close();
}
