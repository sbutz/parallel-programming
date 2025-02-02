#ifndef CLUSTER_EXPANSION_H_
#define CLUSTER_EXPANSION_H_

#include "types.h"

struct DPoints {
  IdxType n;
  float * d_x;
  float * d_y;
};

DPoints copyPointsToDevice(float const * x, float const * y, IdxType n);


void allocateDeviceMemory(
  IdxType ** d_pointStates,
  IdxType ** d_seedLists, IdxType ** d_seedLengths,
  bool ** d_collisionMatrix,
  int nBlocks,
  IdxType n
);

void findClusters(
  IdxType * d_pointStates, bool * d_collisionMatrix,
  float * xs, float * ys, IdxType n,
  IdxType * d_seedLists, IdxType * d_seedLengths,
  IdxType coreThreshold, float rsq
);

#endif