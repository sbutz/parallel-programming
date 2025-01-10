#include "build_graph.h"
#include "count_neighbors.h"
#include "accumulate.h"
#include "build_incidence_lists.h"
#include "device_vector.h"
#include <cuda.h>

#include <iostream>
NeighborGraph buildNeighborGraph(
  float const * xs, float const * ys, Count n,
  float r
) {
  DeviceVector<float> d_xs (xs, n);
  DeviceVector<float> d_ys (ys, n);
  DeviceVector<Count> d_dcounts (UninitializedDeviceVectorTag {}, n);

  countNeighborsOnDevice(d_dcounts.data(), d_xs.data(), d_ys.data(), n, r);

  DeviceVector<Count> d_cumulative (UninitializedDeviceVectorTag {}, 2 * (n + 1));
  CUDA_CHECK(cudaMemset(d_cumulative.data(), 0, sizeof(Count)));
  CUDA_CHECK(cudaMemcpy(d_cumulative.data() + 1, d_dcounts.data(), n * sizeof(Count), cudaMemcpyDeviceToDevice));
  auto s = accumulateOnDevice(d_cumulative.data(), n + 1);

  Count lenIncidenceAry;
  CUDA_CHECK(cudaMemcpy(&lenIncidenceAry, &d_cumulative.data()[s+n], sizeof(Count), cudaMemcpyDeviceToHost));

  DeviceVector<Count> d_incidenceAry (UninitializedDeviceVectorTag{}, lenIncidenceAry);
  buildIncidenceListsOnDevice(d_incidenceAry.data(), d_xs.data(), d_ys.data(), &d_cumulative.data()[s], n, r);

  std::vector<Count> neighborCounts(n);
  std::vector<Count> startIndices(n);
  std::vector<Count> incidenceAry(lenIncidenceAry);

  d_dcounts.memcpyToHost(neighborCounts.data());
  CUDA_CHECK(cudaMemcpy(startIndices.data(), d_cumulative.data(), n * sizeof(Count), cudaMemcpyDeviceToHost));
  d_incidenceAry.memcpyToHost(incidenceAry.data());

  return { std::move(neighborCounts), std::move(startIndices), std::move(incidenceAry) };
}

NeighborGraph buildNeighborGraphCpu(
  float const * xs, float const * ys, Count n,
  float r
) {
  std::vector<Count> neighborCounts (n);
  countNeighborsOnDevice(neighborCounts.data(), xs, ys, n, r);
std::cerr << "X";

  std::vector<Count> cumulative (n + 1);
  accumulateCpu(cumulative.data(), neighborCounts.data(), n);

std::cerr << "X";
  
  Count lenIncidenceAry = cumulative[n];
  std::vector<Count> incidenceAry(lenIncidenceAry);
  buildIncidenceListsCpu(incidenceAry.data(), xs, ys, cumulative.data(), n, r);
std::cerr << "X";

  cumulative.pop_back();
std::cerr << "X";

  return { std::move(neighborCounts), std::move(cumulative), std::move(incidenceAry) };
}