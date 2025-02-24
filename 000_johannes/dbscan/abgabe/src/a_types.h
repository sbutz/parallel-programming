#ifndef TYPES_H_
#define TYPES_H_

using IdxType = unsigned int;
static_assert(sizeof(IdxType) == 4, "");

struct DNeighborGraph {
  IdxType nVertices;
  IdxType lenIncidenceAry;
  IdxType * d_neighborCounts;
  IdxType * d_startIndices;
  IdxType * d_incidenceAry;
};

#endif