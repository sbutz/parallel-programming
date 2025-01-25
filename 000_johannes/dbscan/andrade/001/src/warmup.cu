#include "warmup.h"
#include "prefix_scan.h"
#include "cuda_helpers.h"
#include <cuda.h>

void warmup() {
  constexpr size_t sz64Mi = (size_t)1 << 26;
  IdxType * tempAry, * temp;
  CUDA_CHECK(cudaMalloc(&tempAry, 2 * sz64Mi * sizeof(IdxType) ))
  for (int i = 0; i < 5; ++i) {
    prefixScanOnDevice(
      &temp, tempAry, tempAry + sz64Mi, tempAry + sz64Mi, sz64Mi
    );
  }
  (void)cudaFree(tempAry);
}