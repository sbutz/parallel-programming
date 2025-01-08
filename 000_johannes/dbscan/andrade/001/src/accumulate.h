#ifndef ACCUMULATE_H_
#define ACCUMULATE_H_

#include <cstddef>

using Count = std::size_t;

void accumulate(Count * dest, Count * src, std::size_t n);
void accumulateCpu(Count * dest, Count * src, std::size_t n);
#endif