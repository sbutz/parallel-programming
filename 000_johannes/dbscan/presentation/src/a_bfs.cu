#include "a_bfs.incl.cu"

void findAllComponents(
    int nSm,
    IdxType * d_visited,
    FindComponentsProfilingData * profile,
    DNeighborGraph const * graph,
    std::vector<std::vector<IdxType>> & visitedsStep
) {
  findAllComponents<findNextUnvisitedCoreNaivePolicy, frontierBasicPolicy> (nSm, d_visited, profile, graph, visitedsStep);
}