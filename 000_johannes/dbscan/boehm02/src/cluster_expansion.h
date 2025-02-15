#ifndef CLUSTER_EXPANSION_H_
#define CLUSTER_EXPANSION_H_

#include "types.h"
#include <vector>

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

void findClusters(
  bool ** d_coreMarkers, IdxType ** d_clusters,
  float * xs, float * ys, IdxType n,
  CollisionHandlingData collisionHandlingData,
  IdxType coreThreshold, float rsq
);

void unionizeCpu(std::vector<IdxType> & clusters);
void unionizeGpu(IdxType * d_clusters, IdxType n);

#endif