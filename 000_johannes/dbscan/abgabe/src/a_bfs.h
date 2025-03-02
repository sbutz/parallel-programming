#ifndef BFS_H_
#define BFS_H_

#include "a_types.h"
#include <cstddef>
#include <vector>

struct FindComponentsProfilingData {
  float timeMarkCoreUnvisited;
  float timeFindComponents;
};

void findAllComponents(
  int nSm,
  IdxType * d_visited,
  FindComponentsProfilingData * profile,
  DNeighborGraph const * graph
);

#endif