#ifndef BFS_H_
#define BFS_H_

#include "types.h"
#include <cstddef>
#include <vector>

struct FindComponentsProfile {
  float timeMarkNonCore;
  float timeFindComponents;
};

struct DeviceGraph {
  DNeighborGraph g;
  DeviceGraph(IdxType nVertices, IdxType lenDestinations, IdxType * d_startIndices, IdxType * destinations);
  DeviceGraph(DeviceGraph const &) = delete;
  ~DeviceGraph();
};

struct ComponentFinder {
  IdxType nVertices = 0;
  IdxType * d_visited = nullptr;
  IdxType * d_frontierBuffer = nullptr;
  struct {
      IdxType * d_cntFrontier;
      IdxType * d_frontier;
  } frontiers[2];
  IdxType nComponentsFound = 0;
  char currentFrontier = 0;

  ComponentFinder(DNeighborGraph const * graph, size_t maxFrontierSize);
  ComponentFinder(ComponentFinder const &) = delete;
  ~ComponentFinder();
  template <int FrontierPolicyKey>
  void findComponent(
      DNeighborGraph const * graph, IdxType startVertex, IdxType visitedTag,
      void (*callback) (void *) = nullptr, void * callbackData = nullptr
  );
  std::vector<IdxType> getComponentTagsVector() const;
};

template <int FindNextUnvisitedPolicyKey, int FrontierPolicyKey>
void doFindAllComponents(
  IdxType * d_resultBuffer, FindComponentsProfile * profile,
  ComponentFinder & cf, DNeighborGraph const * graph, IdxType nextFreeTag, IdxType nextStartIdx, 
  void (*callback) (void *) = nullptr, void * callbackData = nullptr
);

constexpr int findNextUnivisitedNaivePolicy = 1;
constexpr int findNextUnivisitedSuccessivePolicy = 2;
constexpr int findNextUnivisitedSuccessiveMultWarpPolicy = 3;
constexpr int findNextUnivisitedSuccessiveSimplifiedPolicy = 4;

constexpr int frontierBasicPolicy = 1;
constexpr int frontierSharedPolicy = 2;


IdxType * createResultBuffer();
void freeResultBuffer(IdxType * d_resultBuffer);

template <int FindNextUnvisitedPolicyKey, int FrontierPolicyKey, typename Callback>
inline void findAllComponents(FindComponentsProfile * profile, ComponentFinder * cf, DNeighborGraph const * graph, Callback && callback = []{}) {
  IdxType * d_resultBuffer = createResultBuffer();
  doFindAllComponents<FindNextUnvisitedPolicyKey, FrontierPolicyKey>(d_resultBuffer, profile, *cf, graph, 2, 0,
    [](void * callback) { (*(Callback *)callback) (); },
    &callback
  );
  freeResultBuffer(d_resultBuffer);
}

#endif