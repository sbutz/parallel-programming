#ifndef BFS_H_
#define BFS_H_

#include "types.h"
#include <cstddef>
#include <vector>

struct FindComponentsProfilingData {
  float timeMarkNonCore;
  float timeFindComponents;
};

template <int FindNextUnvisitedPolicyKey, int FrontierPolicyKey>
void findAllComponents(
  int nSm,
  IdxType * d_visited,
  FindComponentsProfilingData * profile,
  DNeighborGraph const * graph
);

constexpr int findNextUnvisitedNaivePolicy = 1;
constexpr int findNextUnvisitedSuccessivePolicy = 2;
constexpr int findNextUnvisitedSuccessiveMultWarpPolicy = 3;
constexpr int findNextUnvisitedSuccessiveSimplifiedPolicy = 4;

constexpr int frontierBasicPolicy = 1;
constexpr int frontierSharedPolicy = 2;



#endif