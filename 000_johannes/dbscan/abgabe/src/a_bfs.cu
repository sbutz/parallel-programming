#include "a_bfs.incl.cu"

void findAllComponents(
    int nSm,
    IdxType * d_visited,
    FindComponentsProfilingData * profile,
    DNeighborGraph const * graph
) {
  findAllComponents<findNextUnvisitedCoreSuccessivePolicy, frontierSharedPolicy> (nSm, d_visited, profile, graph);
}