#ifndef CLUSTER_EXPANSION_H_
#define CLUSTER_EXPANSION_H_

#include "types.h"

struct DPoints {
  IdxType n;
  float * d_x;
  float * d_y;
};

DPoints copyPointsToDevice(float const * x, float const * y, IdxType n);


void allocateDeviceMemory(
  unsigned int ** d_pointStates, IdxType ** d_clusters,
  IdxType ** d_seedLists, IdxType ** d_seedClusterIds, IdxType ** d_seedLengths,
  unsigned int ** d_syncCounter, unsigned int ** d_collisionMatrix, IdxType ** d_processedIdxs,
  int nBlocks,
  IdxType n
);


constexpr unsigned int stateReserved        = 0x80000000;
constexpr unsigned int stateUnderInspection = 0x40000000; 
constexpr unsigned int stateCore            = 0x20000000;
constexpr unsigned int stateNoiseOrBorder   = 0x10000000;
constexpr unsigned int stateReserved2       = 0x08000000;
constexpr unsigned int stateFree            = 0x00000000;
constexpr unsigned int stateStateBitsMask   = 0xff000000;
constexpr unsigned int stateThreadGroupIdxMask = 0x00ffffff;

void findClusters(
  unsigned int * d_pointStates, IdxType * d_clusters,
  float * xs, float * ys, IdxType n,
  IdxType * d_seedLists, IdxType * d_seedClusterIds, IdxType * d_seedLengths,
  unsigned int * d_syncCounter, unsigned int * d_collisionMatrix, IdxType * d_processedIdxs,
  IdxType coreThreshold, float rsq
);

#endif