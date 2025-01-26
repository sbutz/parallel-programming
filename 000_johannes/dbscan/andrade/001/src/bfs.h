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
  void findComponent(
      DNeighborGraph const * graph, IdxType startVertex, IdxType visitedTag,
      void (*callback) (void *) = nullptr, void * callbackData = nullptr
  );
};

struct AllComponentsFinder;

template <int FindNextUnvisitedPolicyKey>
void doFindAllComponents(
  FindComponentsProfile * profile,
  AllComponentsFinder *, DNeighborGraph const *,
  void (*callback) (void *) = nullptr, void * callbackData = nullptr
);

constexpr int findNextUnivisitedNaivePolicy = 1;
constexpr int findNextUnivisitedSuccessivePolicy = 2;
constexpr int findNextUnivisitedSuccessiveMultWarpPolicy = 3;

struct AllComponentsFinder {
  ComponentFinder cf;
  IdxType nextFreeTag;
  IdxType nextStartIndex;
  IdxType * d_resultBuffer;

  AllComponentsFinder(DNeighborGraph const * graph, size_t maxFrontierSize);
  AllComponentsFinder(AllComponentsFinder const &) = delete;
  ~AllComponentsFinder();

  template <int FindNextUnvisitedPolicyKey, typename Callback>
  void findAllComponents(FindComponentsProfile * profile, DNeighborGraph const * graph, Callback && callback = []{}) {
    doFindAllComponents<FindNextUnvisitedPolicyKey>(profile, this, graph,
      [](void * callback) { (*(Callback *)callback) (); },
      &callback
    );
  }

  std::vector<IdxType> getComponentTagsVector() const;
};


#endif