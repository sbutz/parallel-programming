#ifndef BUILD_GRAPH_H_
#define BUILD_GRAPH_H_

#include "a_types.h"

struct BuildNeighborGraphProfilingData {
  float timeNeighborCount;
  float timePrefixScan;
  float timeBuildIncidenceList;
};

DNeighborGraph buildNeighborGraph(
  BuildNeighborGraphProfilingData * profile,
  float const * d_xs, float const * d_ys, IdxType n,
  IdxType coreThreshold, float r
);

void freeDNeighborGraph(DNeighborGraph & g);

#endif