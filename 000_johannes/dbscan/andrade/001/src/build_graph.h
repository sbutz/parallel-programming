#ifndef BUILD_GRAPH_H_
#define BUILD_GRAPH_H_

#include "types.h"
#include <vector>

struct BuildNeighborGraphProfile {
  float timeNeighborCount;
  float timePrefixScan;
  float timeBuildIncidenceList;
};

struct DPoints {
  IdxType n;
  float * d_x;
  float * d_y;
};

struct NeighborGraph {
  std::vector<IdxType> neighborCounts;
  std::vector<IdxType> startIndices;
  std::vector<IdxType> incidenceAry;
};

DPoints copyPointsToDevice(float const * x, float const * y, IdxType n);

NeighborGraph copyDNeighborGraphToHost(DNeighborGraph const & g);

DNeighborGraph buildDNeighborGraphOnDevice(
  BuildNeighborGraphProfile * profile,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
);

NeighborGraph buildNeighborGraph(
  BuildNeighborGraphProfile * profile,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
);

NeighborGraph buildNeighborGraphCpu(
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
);

#endif