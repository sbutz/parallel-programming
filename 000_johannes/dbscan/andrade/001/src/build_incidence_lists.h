#ifndef BUILD_INCIDENCE_LISTS_H_
#define BUILD_INCIDENCE_LISTS_H_

#include <cstddef>

using Count = std::size_t;

void buildIncidenceLists(
  Count * listArray,
  float const * xs, float const * ys, Count const * cumulative, Count n,
  float r
);

void buildIncidenceListsCpu(
  Count * listArray,
  float const * xs, float const * ys, Count const * cumulative, Count n,
  float r
);

void buildIncidenceListsOnDevice(
  Count * d_listArray,
  float const * d_xs, float const * d_ys, Count const * d_cumulative, Count n,
  float r
);

#endif