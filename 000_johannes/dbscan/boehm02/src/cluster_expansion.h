#ifndef CLUSTER_EXPANSION_H_
#define CLUSTER_EXPANSION_H_

#include "types.h"

struct DPoints {
  IdxType n;
  float * d_x;
  float * d_y;
};

DPoints copyPointsToDevice(float const * x, float const * y, IdxType n);


struct CollisionHandlingData {
  unsigned int * d_mutex;
  volatile IdxType * d_doneWithIdx;
  volatile bool * d_collisionMatrix;

  constexpr static auto calculateSizes(unsigned int nBlocks) {
    struct Result {
      unsigned int szMutex;
      unsigned int szDoneWithIdx;
      unsigned int szCollisionMatrix;
    };
    constexpr size_t nUintBits = 8 * sizeof(unsigned int);
    unsigned int szMutex = 128;
    unsigned int szDoneWithIdx = (nBlocks * sizeof(IdxType) + 127) / 128 * 128;
    unsigned int szCollisionMatrix = nBlocks * nBlocks;
    return Result { szMutex, szDoneWithIdx, szCollisionMatrix };
  }

  constexpr static size_t calculateSize(unsigned int nBlocks) {
    auto sizes = calculateSizes(nBlocks);
    return sizes.szMutex + sizes.szDoneWithIdx + sizes.szCollisionMatrix;
  }
};

void allocateDeviceMemory(
  unsigned int ** d_pointStates, IdxType ** d_clusters,
  CollisionHandlingData * collisionHandlingData,
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
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
);

#endif