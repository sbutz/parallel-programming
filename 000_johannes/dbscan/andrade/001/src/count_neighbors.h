#ifndef DBSCAN_H_
#define DBSCAN_H_

#include <cstddef>

using Count = std::size_t;

void countNeighborsOnDevice(
  Count * d_dcounts,
  float const * d_xs, float const * d_ys, Count n,
  float r
);

void countNeighbors(
  Count * dcounts,
  float const * xs, float const * ys, Count n,
  float r
);

void countNeighborsCpu(
  Count * dcounts,
  float * xs, float * ys, Count n,
  float r
);

#endif