__global__ void kMeansCentroidUpdateSharedMemory(Point* p, Centroid* c, bool delC)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= NUM_POINTS) return;
	const int tid = threadIdx.x;

	//shared memory fÃƒÂ¼r Datenpunkte und Cluster
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

		//im letzten Block darf die Aufsummierung nicht ÃƒÂ¼ber NUM_POINTS hinausgehen
		int loops = blockDim.x;
		if (((blockIdx.x + 1) * blockDim.x) > NUM_POINTS)loops = NUM_POINTS % blockDim.x;

		for (int j = 0; j < loops; ++j)
		{
			int cid = spC[j];
			cXSum[cid] += spX[j];
			cYSum[cid] += spY[j];
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