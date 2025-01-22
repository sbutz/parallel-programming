#ifndef DBSCAN_H_
#define DBSCAN_H_

#include "types.h"
#include <cstddef>


void countNeighborsOnDevice(
  IdxType * d_dcounts,
  float const * d_xs, float const * d_ys, IdxType n,
  IdxType coreThreshold, float r
);

void countNeighbors(
  IdxType * dcounts,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
);

void countNeighborsCpu(
  IdxType * dcounts,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
);

#endif