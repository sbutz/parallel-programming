#ifndef CLUSTER_EXPANSION_H_
#define CLUSTER_EXPANSION_H_

#include "b_types.h"
#include <cstddef>

constexpr unsigned int stateReserved        = 0x80000000;
constexpr unsigned int stateUnderInspection = 0x40000000; 
constexpr unsigned int stateCore            = 0x20000000;
constexpr unsigned int stateNoiseOrBorder   = 0x10000000;
constexpr unsigned int stateFree            = 0x00000000;
constexpr unsigned int stateStateBitsMask   = 0xff000000;
constexpr unsigned int stateThreadGroupIdxMask = 0x00ffffff;

void findClusters(
  int nSm,
  unsigned int * d_pointStates, IdxType * d_clusters,
  float * xs, float * ys, IdxType n,
  IdxType coreThreshold, float rsq
);

#endif