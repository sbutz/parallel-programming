#ifndef BUILD_GRAPH_H_
#define BUILD_GRAPH_H_

#include "types.h"
#include <vector>

struct BuildNeighborGraphProfile {
  float timeNeighborCount;
  float timePrefixScan;
  float timeBuildIncidenceList;
};

struct NeighborGraph {
  std::vector<IdxType> neighborCounts;
  std::vector<IdxType> startIndices;
  std::vector<IdxType> incidenceAry;
};

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