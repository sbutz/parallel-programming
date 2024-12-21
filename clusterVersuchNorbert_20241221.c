#define DEBUG false

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#define NUM_POINTS 1024 * 16
#define NUM_K 16
#define MAX_LOOPS 1000
#define NUM_THREADS 1024
#define MAX_X 700;
#define MAX_Y 500;

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
    LTYP xsum;
    LTYP ysum;
}Centroid;

cudaError_t reset_device();
cudaError_t init_device(Point*, Point*, Centroid*, Centroid*, int, int);

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

    if (tid > NUM_POINTS)return;

    int oc, ox, oy; // alter Centroid
    p[tid].myCentroid = -1;
    LTYP* dist = (LTYP*)malloc(sizeof(LTYP));
    for (unsigned int j = 0; j < NUM_K; ++j) {
        distance_dev(p[tid].x, p[tid].y, c[j].x, c[j].y, dist);
        if ((*dist < p[tid].distance) || (-1 == p[tid].myCentroid)) {
            distance_dev(p[tid].x, p[tid].y, c[j].x, c[j].y, dist);
            p[tid].distance = *dist;
            oc = p[tid].myCentroid;
            ox = p[tid].x;
            oy = p[tid].y;
            if (-1 != oc) {
                atomicAdd(&c[oc].total, -1);
                atomicAdd(&c[oc].x, -ox);
                atomicAdd(&c[oc].y, -oy);
            }
            atomicAdd(&c[j].total, 1);
            p[tid].myCentroid = j;
            atomicAdd(&c[j].x, p[tid].x);
            atomicAdd(&c[j].y, p[tid].y);
        }
    }
}


__global__ void assign_PointsTest(Point* p, Centroid* c,int* testwert) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid > NUM_POINTS)return;

    int oc, ox, oy; // alter Centroid
    p[tid].myCentroid = -1;
    LTYP* dist = (LTYP*)malloc(sizeof(LTYP));
    for (unsigned int j = 0; j < NUM_K; ++j) {
        distance_dev(p[tid].x, p[tid].y, c[j].x, c[j].y, dist);
        if ((*dist < p[tid].distance) || (-1 == p[tid].myCentroid)) {            
            distance_dev(p[tid].x, p[tid].y, c[j].x, c[j].y, dist);
            p[tid].distance = *dist;                                    
            oc = p[tid].myCentroid;
            ox = p[tid].x;
            oy = p[tid].y;
            if (-1 != oc) {
                atomicAdd(&c[oc].total, -1);
                atomicAdd(&c[oc].x, -ox);
                atomicAdd(&c[oc].y, -oy);
            }
            atomicAdd(&c[j].total, 1);
            p[tid].myCentroid = j;
            atomicAdd(&c[j].x, p[tid].x);
            atomicAdd(&c[j].y, p[tid].y);
        }
    }   
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
    int ht = 0, *dt = 0;
    cudaStatus = cudaMalloc(&dt, sizeof(int));

    //Initialisierung der Datenpunkte
    for (unsigned long i = 0; i < NUM_POINTS; ++i) {
        hp[i].x = rand()%MAX_X; // Notlösung
        hp[i].y = rand()%MAX_Y; // Notlösung
        //hp[i].distance = -1;
        //hp[i].myCentroid = -1;
    }        

    //Initialisierung der ersten Zentroide
    for (int i = 0; i < NUM_K; ++i) {
        oc[i].x = hc[i].x = hp[i].x;
        oc[i].y = hc[i].y = hp[i].y;;
    }             

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
            cudaStatus = cudaMemcpy(&ht, dt, sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess)fprintf(stderr, "Fehler bei cudaMemcpy test auf Host\n"); else printf("Test-int: %d\n", ht);
        }

        //Datenübertragung zum Host
        cudaStatus = cudaMemcpy(hc, dc, csize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)fprintf(stderr, "Fehler bei cudaMemcpy Centroid auf Host\n");
        
        if(DEBUG)
        for (int i = 0; i < NUM_K; ++i) {
            printf("x: %d, y: %d, total: %d\n", hc[i].x, hc[i].y, hc[i].total);            
        }


        //Neue Zentroide
        changed = false;
        int j;
        for (int i = 0; i < NUM_K; ++i) {           

            //Berechnung des neuen Zentrums
            if (hc[i].total > 0) {
                hc[i].x = hc[i].xsum / hc[i].total;
                hc[i].y = hc[i].ysum / hc[i].total;
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