#ifndef BUILD_INCIDENCE_LISTS_H_
#define BUILD_INCIDENCE_LISTS_H_

#include "types.h"
#include <cstddef>

void buildIncidenceLists(
  IdxType * listArray,
  float const * xs, float const * ys, IdxType const * cumulative, IdxType n,
  float r
);

void buildIncidenceListsCpu(
  IdxType * listArray,
  float const * xs, float const * ys, IdxType const * cumulative, IdxType n,
  float r
);

void buildIncidenceListsOnDevice(
  IdxType * d_listArray,
  float const * d_xs, float const * d_ys, IdxType const * d_cumulative, IdxType n,
  float r
);

#endif