#include "a_bfs.incl.cu"

void findAllComponents(
    int nSm,
    IdxType * d_visited,
    FindComponentsProfilingData * profile,
    DNeighborGraph const * graph
) {
  findAllComponents<findNextUnvisitedCoreSuccessiveMultWarpPolicy, frontierBasicPolicy> (nSm, d_visited, profile, graph);
}