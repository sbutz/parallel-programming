#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>

std::string debug_str;

#define DEBUG true
#define SHARED false
#define SHARED_EXT false
#define SHARED_ALL false
#define CUDAMALLOCHOST true
#define KERNEL_DEBUG_T false
#define SAVE_TO_FILE true
#define SAVE_TO_FILE_STRIDE 1
#define READ_FROM_FILE false
#define CONSOLE_AUSGABE true
#define TRENNZEICHEN " "

#define NUM_POINTS (1024 * 19 + 1111)
#define NUM_K 15
#define MAX_LOOPS 100
#define NUM_THREADS 1024
#define MAX_X 700;
#define MAX_Y 500;

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
void showValues(TYP*, size_t, std::string);


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


	int z = p[id].myCentroid;
	cXSum_arr[id + NUM_POINTS * z] = p[id].x;
	cYSum_arr[id + NUM_POINTS * z] = p[id].y;
	cTotal_arr[id + NUM_POINTS * z] = 1;

}

__global__ void reduce_per_block(const TYP* in, TYP* out) {

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= NUM_K * NUM_POINTS)return;

	extern __shared__ int sdata[NUM_THREADS];

	sdata[tid] = in[id];
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

__global__ void reduce(TYP* in, TYP* out) {

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= NUM_K * NUM_POINTS)return;

	__syncthreads();
	// Führe die Reduction im geteilten Speicher durch	
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			in[id] += in[id + s];
		}
		__syncthreads();
	}
	if (0 == tid)out[blockIdx.x] = in[id];
};




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

__global__ void copyOutToCentroid(Centroid* c, const TYP* cxsum, const TYP* cysum, const TYP* ctotal) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_K)return;

	c[id].total = ctotal[id];
	c[id].xsum = cxsum[id];
	c[id].ysum = cysum[id];
}

__global__ void copyOutToIn3(const TYP* in1, const TYP* in2, const TYP* in3, TYP* out1, TYP* out2, TYP* out3, const int max) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= max)return;

	out1[id] = in1[id];
	out2[id] = in2[id];
	out3[id] = in3[id];
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

	// grundsätzliche Variablen
	cudaError_t cudaStatus;
	int psize = NUM_POINTS * sizeof(Point);
	int csize = NUM_K * sizeof(Centroid);
	int blocks = (NUM_POINTS + NUM_THREADS - 1) / NUM_THREADS;

	//Speicherallokation - Datapoints
	Point* hp = (Point*)malloc(psize), * dp = 0;

	// Centroids
	Centroid* dc;  //cudaStatus = cudaMallocManaged(&c, csize); if (cudaFail(cudaStatus, "Fehler bei cudaMallocMamaged für c"))return -1;
	cudaStatus = cudaMalloc(&dc, csize); if (cudaFail(cudaStatus, "Fehler bei cudaMallocMamaged für c"))return -1;
	Centroid* hc = (Centroid*)malloc(csize);

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
			hp[i].x = rand() % MAX_X; // NotlÃ¶sung
			hp[i].y = rand() % MAX_Y; // NotlÃ¶sung
		}
	} 

	if (SAVE_TO_FILE)savePoints(hp);

	//Initialisierung der ersten Zentroide
#pragma unroll	
	for (int i = 0; i < NUM_K; ++i) {
		hc[i].x = hp[i].x;
		hc[i].y = hp[i].y;
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
	cudaStatus = cudaMemcpy(dc, hc, csize, cudaMemcpyHostToDevice); if (cudaFail(cudaStatus, "memcpy(hc)"))return -1;

	// Variablen f�r die Reduktionsloesung
	int faktor = (NUM_POINTS + NUM_THREADS - 1) / NUM_THREADS;
	int arr_size_typ = NUM_K * NUM_POINTS * sizeof(TYP);
	int out_size_typ = NUM_K * sizeof(TYP) * faktor;

	int blocks_arr = (NUM_POINTS * NUM_K + NUM_THREADS - 1) / NUM_THREADS;		
	TYP* dcxsum_arr, * dcysum_arr, * dctotal_arr, * dcxsum, * dcysum, * dctotal;
	cudaStatus = cudaMalloc(&dcxsum_arr, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dcxsum_arr)"))return -1;
	cudaStatus = cudaMalloc(&dcysum_arr, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dcysum_arr)"))return -1;
	cudaStatus = cudaMalloc(&dctotal_arr, arr_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dctotal_arr)"))return -1;
	cudaStatus = cudaMalloc(&dcxsum, out_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dcxsum)"))return -1;
	cudaStatus = cudaMalloc(&dcysum, out_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dcysum)"))return -1;
	cudaStatus = cudaMalloc(&dctotal, out_size_typ); if (cudaFail(cudaStatus, "cudaMalloc(dctotal)"))return -1;

	// Debug-Variablen f�r Reduce
	TYP* deb_hcxsum_arr = (TYP*)malloc(arr_size_typ);
	TYP* deb_hcysum_arr = (TYP*)malloc(arr_size_typ);
	TYP* deb_hctotal_arr = (TYP*)malloc(arr_size_typ);
	TYP* deb_hcxsum = (TYP*)malloc(out_size_typ);
	TYP* deb_hcysum = (TYP*)malloc(out_size_typ);
	TYP* deb_hctotal = (TYP*)malloc(out_size_typ);

	//Loop, um die Zentroide zu finden
	bool changed = true, maxlooped = false;
	unsigned int loops = 0;
	while (changed && !maxlooped)
	{
		if (CONSOLE_AUSGABE) { //Zeige die aktuellen Centroide
			for (int i = 0; i < NUM_K; ++i)printf("Iteration %d: centroid %d x: %d, y: %d, total: %d, \n", loops, i, hc[i].x, hc[i].y, hc[i].total);
		}

		//*//ordner die Punkte den Centroiden zu
		kMeansClusterAssign << <blocks, NUM_THREADS >> > (dp, dc);
		cudaDeviceSynchronize();

		if (SAVE_TO_FILE && (loops % SAVE_TO_FILE_STRIDE == 0)) {
			cudaStatus = cudaMemcpy(hp, dp, psize, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy dp - save to file"))return -1;
			saveK(hc, loops);
			savePoints(hp, loops);
		}

		//Bestimme die Zentroide neu

		delete_3arr__TYP << <blocks_arr, NUM_THREADS >> > (dcxsum_arr, dcysum_arr, dctotal_arr);
		cudaDeviceSynchronize();

		kMeansCentroidUpdate_reduction_Part1 << <blocks, NUM_THREADS >> > (dp, dcxsum_arr, dcysum_arr, dctotal_arr);
		cudaDeviceSynchronize();
		
		if (NUM_POINTS <= NUM_THREADS) {

			// Jeder Block kuemmert sich um ein Cluster

			reduce_per_block << <NUM_K, NUM_POINTS >> > (dcxsum_arr, dcxsum);
			reduce_per_block << <NUM_K, NUM_POINTS >> > (dcysum_arr, dcysum);
			reduce_per_block << <NUM_K, NUM_POINTS >> > (dctotal_arr, dctotal);

			//reduce << <NUM_K, NUM_POINTS >> > (dcxsum_arr, dcxsum);
			//reduce << <NUM_K, NUM_POINTS >> > (dcysum_arr, dcysum);
			//reduce << <NUM_K, NUM_POINTS >> > (dctotal_arr, dctotal);			
		}
		else {

			int points_per_Block = NUM_POINTS;
			int blocks_per_cluster = NUM_K;			

			do {

				// Zwei oder mehr Bl�cke k�mmern sich um ein Cluster und danach wird dann wieder zusammengefasst.			
				// faktor beschreibt, wie viele Bloecke sich um ein Cluster k�mmern

				faktor = (points_per_Block + NUM_THREADS - 1) / NUM_THREADS;					
				points_per_Block = (points_per_Block + faktor - 1) / faktor;	
				blocks_per_cluster = NUM_K * faktor;

				//printf("%d, %d, %d\n", faktor, points_per_Block, blocks_per_cluster);
				//std::cin >> debug_str;

				reduce_per_block << < blocks_per_cluster, points_per_Block >> > (dcxsum_arr, dcxsum);
				reduce_per_block << < blocks_per_cluster, points_per_Block >> > (dcysum_arr, dcysum);
				reduce_per_block << < blocks_per_cluster, points_per_Block >> > (dctotal_arr, dctotal);

				//reduce << < blocks_per_cluster, points_per_Block >> > (dcxsum_arr, dcxsum);
				//reduce << < blocks_per_cluster, points_per_Block >> > (dcysum_arr, dcysum);
				//reduce << < blocks_per_cluster, points_per_Block >> > (dctotal_arr, dctotal);
								
				cudaDeviceSynchronize();								

				points_per_Block = faktor;	
				
				copyOutToIn3 << < blocks, NUM_THREADS >> > (dcxsum, dcysum, dctotal, dcxsum_arr, dcysum_arr, dctotal_arr, blocks_per_cluster);
				cudaDeviceSynchronize();

			} while (faktor > 1); // bei Faktor == 1 waren wir im obigen finalen Fall
		}

		cudaDeviceSynchronize();	

		copyOutToCentroid << < blocks, NUM_THREADS >> > (dc, dcxsum, dcysum, dctotal);
		cudaDeviceSynchronize();

		kMeansCentroidUpdate_reduction_Part2 << <blocks, NUM_THREADS >> > (dc, true);
		cudaDeviceSynchronize();

		cudaStatus = cudaMemcpy(hc, dc, csize, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "memcpy(dc)")) return -1;


		changed = false;
		for (int i = 0; i < NUM_K; ++i) {

			//Blieben die Centroide stehen?
			if ((0 != hc[i].xdist) && (0 != hc[i].ydist))changed = true;
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

	cudaFree(dc);
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

bool cudaFail(cudaError_t cudaStatus, char* dev) {
	if (cudaStatus != cudaSuccess) {		
		fprintf(stderr, "Fehler bei ");
		fprintf(stderr, dev);
		fprintf(stderr, " - ");
		std::cout << cudaStatus;
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


void showValues(TYP* val, size_t n, std::string text) {
	std::cout << std::endl << text << std::endl;
	for (int i = 0; i < n; ++i)
		printf("%d ", val[i]);
	printf("\n");
}



/*// Debug Anfang
		cudaStatus = cudaMemcpy(deb_hcxsum_arr, dcxsum_arr, arr_size_typ, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy debxsum_arr"))return -1;
		cudaStatus = cudaMemcpy(deb_hcysum_arr, dcysum_arr, arr_size_typ, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy debysum_arr"))return -1;
		cudaStatus = cudaMemcpy(deb_hctotal_arr, dctotal_arr, arr_size_typ, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy debtotal_arr"))return -1;
		cudaStatus = cudaMemcpy(deb_hcxsum, dcxsum, out_size_typ, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy debxsum"))return -1;
		cudaStatus = cudaMemcpy(deb_hcysum, dcysum, out_size_typ, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy debysum"))return -1;
		cudaStatus = cudaMemcpy(deb_hctotal, dctotal, out_size_typ, cudaMemcpyDeviceToHost); if (cudaFail(cudaStatus, "Memcpy debtotal"))return -1;

		showValues(deb_hcxsum_arr, NUM_K * NUM_POINTS, "deb_hcxsum_arr");
		showValues(deb_hcysum_arr, NUM_K * NUM_POINTS, "deb_hc<sum_arr");
		showValues(deb_hctotal_arr, NUM_K * NUM_POINTS, "deb_hctotal_arr");
		showValues(deb_hcxsum, NUM_K, "deb_hcxsum");
		showValues(deb_hcysum, NUM_K, "deb_hcysum");
		showValues(deb_hctotal, NUM_K, "deb_hctotal");

		/// Debug Ende */
