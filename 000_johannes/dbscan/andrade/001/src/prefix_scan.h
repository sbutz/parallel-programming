#ifndef PREFIX_SCAN_H_
#define PREFIX_SCAN_H_

#include "types.h"

void prefixScanOnDevice(IdxType ** res, IdxType * dest1, IdxType * dest2, IdxType * src, IdxType n);
void prefixScan(IdxType * dest, IdxType * src, IdxType n);
void prefixScanCpu(IdxType * dest, IdxType * src, IdxType n);
#endif