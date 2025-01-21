#ifndef BFS_H_
#define BFS_H_

#include "types.h"
#include <cstddef>
#include <vector>

struct DeviceGraph {
  Graph g;
  DeviceGraph(IdxType nVertices, IdxType lenDestinations, IdxType * incidenceLists, IdxType * destinations);
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

  ComponentFinder(Graph const * graph, size_t maxFrontierSize);
  ComponentFinder(ComponentFinder const &) = delete;
  ~ComponentFinder();
  void findComponent(
      Graph const * graph, IdxType startVertex, IdxType visitedTag,
      void (*callback) (void *) = nullptr, void * callbackData = nullptr
  );
};

struct AllComponentsFinder;

void doFindAllComponents(
  AllComponentsFinder *, Graph const *,
  void (*callback) (void *) = nullptr, void * callbackData = nullptr
);

struct AllComponentsFinder {
  ComponentFinder cf;
  IdxType nextFreeTag;
  IdxType nextStartIndex;
  IdxType * d_resultBuffer;

  AllComponentsFinder(Graph const * graph, size_t maxFrontierSize);
  AllComponentsFinder(AllComponentsFinder const &) = delete;
  ~AllComponentsFinder();

  template <typename Callback>
  void findAllComponents(Graph const * graph, Callback && callback = []{}) {
    doFindAllComponents(this, graph,
      [](void * callback) { (*(Callback *)callback) (); },
      &callback
    );
  }

  std::vector<IdxType> getComponentTagsVector() const;
};


#endif