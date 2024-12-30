#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>

#define DEBUG true
#define KERNEL_DEBUG_T false
#define SAVE_TO_FILE true
#define READ_FROM_FILE false

#define NUM_POINTS 1024
#define NUM_K 16
#define MAX_LOOPS 1000
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

cudaError_t reset_device();
cudaError_t init_device(Point*, Point*, Centroid*, Centroid*, int, int);

void saveK(int*, int*, int);
void saveK(Centroid*, int);
void savePoints(int*, int*);
void savePoints(Point*);
void savePoints(Point*, int);
int getPoints(Point*, std::string, int);
int getPoints(int*, int*, std::string, int);

// https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/

__global__ void init_Points(Point* dpoints) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= NUM_POINTS)return;

    dpoints[tid].distance = -1;
    dpoints[tid].myCentroid = -1;    
}

__device__ void distance_dev(TYP x1, TYP y1, TYP x2, TYP y2, LTYP* dist) {
    *dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

__global__ void assign_Points(Point* p, Centroid* c) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= NUM_POINTS)return;

    int cluster = -1;
    TYP pdistx = p[tid].x;
    TYP pdisty = p[tid].y;
    TYP pcentroid = p[tid].myCentroid;
    LTYP pdist = p[tid].distance;
    LTYP* dist = (LTYP*)malloc(sizeof(LTYP));
    for (unsigned int j = 0; j < NUM_K; ++j) {
        distance_dev(pdistx, pdisty, c[j].x, c[j].y, dist);
        if ((*dist < pdist) || (-1 == cluster)) {
            distance_dev(pdistx, pdisty, c[j].x, c[j].y, dist);
            pdist = *dist;
            cluster = j;
        }
    }
    atomicAdd(&c[cluster].total, 1);
    p[tid].myCentroid = cluster;
    atomicAdd(&c[cluster].xsum, p[tid].x);
    atomicAdd(&c[cluster].ysum, p[tid].y);
}


__global__ void assign_PointsTest(Point* p, Centroid* c,int* testwert) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= NUM_POINTS)return;

    //vorläufige Doppelung
    if (tid < NUM_K) {
        c[tid].total = 0;
        c[tid].xsum = 0;
        c[tid].ysum = 0;
    }
        
    int cluster = -1;
    TYP pdistx = p[tid].x;
    TYP pdisty = p[tid].y;
    TYP pcentroid = p[tid].myCentroid;
    LTYP pdist = p[tid].distance;
    LTYP* dist = (LTYP*)malloc(sizeof(LTYP));
    for (unsigned int j = 0; j < NUM_K; ++j) {
        distance_dev(pdistx, pdisty, c[j].x, c[j].y, dist);
        if ((*dist < pdist) || (-1 == cluster)) {            
            distance_dev(pdistx, pdisty, c[j].x, c[j].y, dist);
            pdist = *dist;                                    
            cluster = j;            
        }
    }
    atomicAdd(&c[cluster].total, 1);
    p[tid].myCentroid = cluster;
    atomicAdd(&c[cluster].xsum, p[tid].x);
    atomicAdd(&c[cluster].ysum, p[tid].y);
    if(cluster == 0)testwert[tid] = pdisty; else testwert[tid] = 0;
}


int main()
{
    cudaError_t cudaStatus;
    cudaError_t cudaStatusArr[10];

    int psize = NUM_POINTS * sizeof(Point);
    int csize = NUM_K * sizeof(Centroid);

    //Speicherallokation
    Point* hp = (Point*)malloc(psize), *dp = 0;          // Datapoints
    Centroid* hc = (Centroid*)malloc(csize), *dc = 0;    // Centroids
    Centroid* oc = (Centroid*)malloc(csize);            // former Centroids for comparison purposes

    //Debug
    int* ht = (int*)malloc(sizeof(int)*NUM_POINTS), * dt = 0;
    if(DEBUG)cudaStatus = cudaMalloc(&dt, sizeof(int)*NUM_POINTS);

    //Initialisierung der Datenpunkte
    for (unsigned long i = 0; i < NUM_POINTS; ++i) {
        hp[i].x = rand()%MAX_X; // Notlösung
        hp[i].y = rand()%MAX_Y; // Notlösung
        //hp[i].distance = -1;
        //hp[i].myCentroid = -1;
    }   

    if (READ_FROM_FILE) {
        int anz = getPoints(hp, POINTSFILE, 65);
        if (-1 == anz) {
            printf("Datei konnte nicht gelesen werden\n");
            return 1;
        }
    }
        
    if(SAVE_TO_FILE)savePoints(hp);

    //Initialisierung der ersten Zentroide
    for (int i = 0; i < NUM_K; ++i) {
        oc[i].x = hc[i].x = hp[i].x;
        oc[i].y = hc[i].y = hp[i].y;;
    }     

    if (SAVE_TO_FILE)saveK(hc, 0);

    //Allokation Datenpunkte werden auf Device kopiert    
    cudaStatus = cudaMalloc(&dp, psize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Fehler bei cudaMalloc(dp)\n");
        return 1;
    }

    cudaStatus = cudaMemcpy(dp, hp, psize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Fehler bei cudaMemcpy(dc,hy)\n");
        return 1;
    }

    int blocks = (NUM_POINTS + NUM_THREADS - 1) / NUM_POINTS;
    init_Points << <blocks, NUM_THREADS >> > (dp);

    //Allokation dc
    cudaStatus = cudaMalloc(&dc, csize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Fehler bei cudaMalloc(dc)\n");
        return 1;
    }
    
    //Loop, um die Zentroide zu finden
    bool changed = true;
    unsigned int loops = 0;
    while (changed) {

        //resete Cluster-Eigenschaften
        for (int i = 0; i < NUM_K; ++i) {
            hc[i].xsum = 0;
            hc[i].ysum = 0;
            hc[i].total = 0;
        }

        //Übertragung der Zentroide auf Device
        cudaStatus = cudaMemcpy(dc, hc, csize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)fprintf(stderr, "Fehler bei cudaMemcpy Centroid auf Device\n");

        //Neuzuordnung der Punkte
        if(DEBUG)
            assign_PointsTest<<<blocks, NUM_THREADS>>>(dp,dc, dt);
        else        
            assign_Points << <blocks, NUM_THREADS >> > (dp, dc);

        if (DEBUG) {
            cudaStatus = cudaMemcpy(ht, dt, sizeof(int)*NUM_POINTS, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess)fprintf(stderr, "Fehler bei cudaMemcpy test auf Host\n"); else printf("Test-int: %d\n", ht);
            if (KERNEL_DEBUG_T) {
                printf("\nht:\n");
                for (int i = 0; i < NUM_POINTS; ++i)
                    printf("%d ", ht[i]);
                printf("\nht Ende\n");//*/
            }
        }   

        //Datenübertragung zum Host
        cudaStatus = cudaMemcpy(hc, dc, csize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)fprintf(stderr, "Fehler bei cudaMemcpy Centroid auf Host\n");
       
        //Datenübertragung zum Host
        cudaStatus = cudaMemcpy(hp, dp, psize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)fprintf(stderr, "Fehler bei cudaMemcpy Centroid auf Host\n");

        //Neue Zentroide
        changed = false;
        int j;
        for (int i = 0; i < NUM_K; ++i) {           

            //Berechnung des neuen Zentrums
            if (hc[i].total > 0) {
                hc[i].x = hc[i].xsum / hc[i].total;
                hc[i].y = hc[i].ysum / hc[i].total;

                printf("In Schleife %d:: xsum: %d, ysum: %d, total: %d\n", i, hc[i].xsum, hc[i].ysum, hc[i].total);
            }
            else {
                // kann eigentlich nicht sein
                fprintf(stderr, "Fehler in der Berechnung: hc.total fuer Cluster %d betraegt null (leerer Cluster?).\n", i);
                if(!DEBUG)return 1;
            }            

            //Blieben die Centroide stehen?
            if ((hc[i].x != oc[i].x) || (hc[i].y != oc[i].y))changed = true;
            
            //speichere bisherige Clusterzuordnung
            oc[i].x = hc[i].x;
            oc[i].y = hc[i].y;
        }                        
                
        // Begrenzung auf maximale Anzahl von Durchläufen
        loops++;
        if (SAVE_TO_FILE) {
            saveK(hc, loops);
            savePoints(hp, loops);
        }

        if (DEBUG)
            for (int i = 0; i < NUM_K; ++i) {
                printf("x: %d, y: %d, total: %d\n", hc[i].x, hc[i].y, hc[i].total);
            }

        if ((-1!=MAX_LOOPS)&&(loops > MAX_LOOPS))changed = false;
    }

    printf("Anzahl Loops %d\n", loops);
    
    return reset_device();
}

cudaError_t reset_device() {
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaDeviceReset failed!");
    return cudaStatus;
}

cudaError_t init_device(Point* hp, Point* dp, Centroid* hc, Centroid* dc, int psize, int csize) {

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "(init_device) cudaSetDevice failed! No CUDA-capable GPU installed");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc(&dc, csize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "(init_device) Fehler bei cudaMalloc(dc)\n");        
        return cudaStatus;
    }

    //Datenpunkte werden auf Device kopiert
    int blocks = (NUM_POINTS + NUM_THREADS - 1) / NUM_POINTS;
    cudaStatus = cudaMalloc(&dp, psize);
    if (cudaStatus != cudaSuccess){
        fprintf(stderr, "(init_device) Fehler bei cudaMalloc(dp)\n"); 
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(dp, hp, psize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "(init_device) Fehler bei cudaMemcpy(dc,hy)\n");
        return cudaStatus;
    }
        
    init_Points << <blocks, NUM_THREADS >> > (dp);
    return cudaStatus;
}

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
        std::string sa,sz1, sz2;
        std::size_t komma;
        int zaehler = 0;
        // Read data from the file object and put it into a string.
        while (getline(new_file, sa)&&zaehler<maxLines) {
            
            komma = sa.find(",");
            if ((komma == std::string::npos)||(komma<1)||(komma==sa.length())) {
                printf("Falsches Zahlenformat");
                return -1;
            }

            sz1 = sa.substr(0, komma - 1);
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