#include "build_graph.h"
#include "count_neighbors.h"
#include "prefix_scan.h"
#include "build_incidence_lists.h"
#include "types.h"
#include "device_vector.h"
#include <cuda.h>

#include <iostream>
NeighborGraph buildNeighborGraph(
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  DeviceVector<float> d_xs (xs, n);
  DeviceVector<float> d_ys (ys, n);
  DeviceVector<IdxType> d_dcounts (UninitializedDeviceVectorTag {}, n);

  countNeighborsOnDevice(d_dcounts.data(), d_xs.data(), d_ys.data(), n, coreThreshold, r);

  DeviceVector<IdxType> d_cumulative (UninitializedDeviceVectorTag {}, 2 * (n + 1));
  CUDA_CHECK(cudaMemset(d_cumulative.data(), 0, sizeof(IdxType)));
  CUDA_CHECK(cudaMemset(d_cumulative.data() + n + 1, 0, sizeof(IdxType)));
  //CUDA_CHECK(cudaMemcpy(d_cumulative.data() + 1, d_dcounts.data(), n * sizeof(IdxType), cudaMemcpyDeviceToDevice));
  auto s = prefixScanOnDevice(
    d_cumulative.data() + 1,
    d_cumulative.data() + n + 2,
    d_dcounts.data(),
    n
  );

  IdxType lenIncidenceAry;
  CUDA_CHECK(cudaMemcpy(&lenIncidenceAry, s + n - 1, sizeof(IdxType), cudaMemcpyDeviceToHost));
  DeviceVector<IdxType> d_incidenceAry (UninitializedDeviceVectorTag{}, lenIncidenceAry);
  buildIncidenceListsOnDevice(d_incidenceAry.data(), d_xs.data(), d_ys.data(), s - 1, n, r);

  std::vector<IdxType> neighborCounts(n);
  std::vector<IdxType> startIndices(n + 1);
  std::vector<IdxType> incidenceAry(lenIncidenceAry);

  d_dcounts.memcpyToHost(neighborCounts.data());
  CUDA_CHECK(cudaMemcpy(startIndices.data(), s - 1, (n + 1) * sizeof(IdxType), cudaMemcpyDeviceToHost));
  d_incidenceAry.memcpyToHost(incidenceAry.data());

  return { std::move(neighborCounts), std::move(startIndices), std::move(incidenceAry) };
}

NeighborGraph buildNeighborGraphCpu(
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  std::vector<IdxType> neighborCounts (n);
  countNeighborsCpu(neighborCounts.data(), xs, ys, n, coreThreshold, r);
std::cerr << "X";

  std::vector<IdxType> cumulative (n + 1);
  prefixScanCpu(cumulative.data(), neighborCounts.data(), n);

std::cerr << "X";
  
  IdxType lenIncidenceAry = cumulative[n];
  std::vector<IdxType> incidenceAry(lenIncidenceAry);
  buildIncidenceListsCpu(incidenceAry.data(), xs, ys, cumulative.data(), n, r);
std::cerr << "X";

  return { std::move(neighborCounts), std::move(cumulative), std::move(incidenceAry) };
}