#ifndef BUILD_GRAPH_H_
#define BUILD_GRAPH_H_

#include <vector>

using Count = std::size_t;

struct NeighborGraph {
  std::vector<Count> neighborCounts;
  std::vector<Count> startIndices;
  std::vector<Count> incidenceAry;
};

NeighborGraph buildNeighborGraph(
  float const * xs, float const * ys, Count n,
  float r
);

NeighborGraph buildNeighborGraphCpu(
  float const * xs, float const * ys, Count n,
  float r
);

#endif