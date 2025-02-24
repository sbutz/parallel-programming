#ifndef CLUSTER_EXPANSION_H_
#define CLUSTER_EXPANSION_H_

#include "c_types.h"
#include <vector>

void findClusters(
  int nSm,
  bool * d_coreMarkers, IdxType * d_clusters,
  float * xs, float * ys, IdxType n,
  IdxType coreThreshold, float rsq
);

#endif