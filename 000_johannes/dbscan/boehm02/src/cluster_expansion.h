#ifndef CLUSTER_EXPANSION_H_
#define CLUSTER_EXPANSION_H_

#include "types.h"
#include <vector>

struct DPoints {
  IdxType n;
  float * d_x;
  float * d_y;
};

DPoints copyPointsToDevice(float const * x, float const * y, IdxType n);

void findClusters(
  bool ** d_coreMarkers, IdxType ** d_clusters,
  float * xs, float * ys, IdxType n,
  IdxType coreThreshold, float rsq
);

void unionizeCpu(std::vector<IdxType> & clusters);
void unionizeGpu(IdxType * d_clusters, IdxType n);

#endif