#ifndef ACCUMULATE_H_
#define ACCUMULATE_H_

#include "types.h"

IdxType accumulateOnDevice(IdxType * ary, IdxType n);
void accumulate(IdxType * dest, IdxType * src, IdxType n);
void accumulateCpu(IdxType * dest, IdxType * src, IdxType n);
#endif