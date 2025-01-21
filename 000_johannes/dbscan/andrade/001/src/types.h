#ifndef TYPES_H_
#define TYPES_H_

using IdxType = unsigned int;
static_assert(sizeof(IdxType) == 4, "");

struct Graph {
  IdxType nVertices;
  IdxType * incidenceLists; // has length nVertices + 1 to simplify code
  IdxType * destinations;
};

#endif