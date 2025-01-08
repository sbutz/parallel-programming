#ifndef DBSCAN_H_
#define DBSCAN_H_

#include <cstddef>

using Count = std::size_t;

void countNeighbors(
  Count * dcounts,
  float * xs, float * ys, Count n,
  float r
);

void countNeighborsCpu(
  Count * dcounts,
  float * xs, float * ys, Count n,
  float r
);

#endif