
using IdxType = unsigned int;
static_assert(sizeof(IdxType) = 4, "");

struct Graph {
    IdxType nVertices;
    IdxType * incidenceLists; // has length nVertices + 1 to simplify code
    IdxType * destinations;
};

struct D_Frontier {
    IdxType * cntFrontier;
    IdxType * frontier;
}
__global__ void kernel_bfs(
    Graph * d_graph,
    bool * d_visited,   // working memory
    D_Frontier frontier,
    D_Frontier newFrontier
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    IdxType stride = 999;

    if (tid == 0) __atomicSet(frontier->cntFrontier, 0);

    for (IdxType i = tid * stride; i < tid * stride + 1; ++i) {
        IdxType incidenceListStart = d_graph->incidenceLists[i];
        IdxType incidenceListEnd = d_graph->incidenceLists[i+1];

        for (IdxType j = incidenceListStart; j < incidenceListEnd; ++j) {
            IdxType destination = graph->destinations[j];
            if(!d_visited[destination]) {
                d_visited[destination] = true;
                appendToFrontier(newFrontier, destination);
            }
        }
    }

    
}