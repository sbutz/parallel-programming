#include "build_graph.h"
#include "count_neighbors.h"
#include "prefix_scan.h"
#include "build_incidence_lists.h"
#include "types.h"
#include "device_vector.h"
#include <cuda.h>

#include <iostream>

DPoints copyPointsToDevice(float const * x, float const * y, IdxType n) {
  float * d_x, * d_y;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)))
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)))
  CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice))
  CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice))
  return { n, d_x, d_y };
}
DNeighborGraph buildDNeighborGraphOnDevice(
  BuildNeighborGraphProfile * profile,
  float const * d_xs, float const * d_ys, IdxType n,
  IdxType coreThreshold, float r
) {
  IdxType * d_dcounts;
  CUDA_CHECK(cudaMalloc(&d_dcounts, n * sizeof(IdxType)))

  profile->timeNeighborCount = runAndMeasureCuda(
    countNeighborsOnDevice,
    d_dcounts, d_xs, d_ys, n, coreThreshold, r
  );

  IdxType * d_temp;
  CUDA_CHECK(cudaMalloc(&d_temp, 2 * n * sizeof(IdxType)))
  IdxType * d_dest1 = d_temp, * d_dest2 = d_dest1 + n;
  IdxType * s;
  profile->timePrefixScan = runAndMeasureCuda(
    prefixScanOnDevice,
    &s,
    d_dest1,
    d_dest2,
    d_dcounts,
    n
  );

  IdxType * d_startIndices;
  CUDA_CHECK(cudaMalloc(&d_startIndices, (n + 1) * sizeof(IdxType)))
  CUDA_CHECK(cudaMemset(d_startIndices, 0, sizeof(IdxType)))
  CUDA_CHECK(cudaMemcpy(d_startIndices + 1, s, n * (sizeof(IdxType)), cudaMemcpyDeviceToDevice))
  (void)cudaFree(d_temp);

  IdxType lenIncidenceAry;
  CUDA_CHECK(cudaMemcpy(&lenIncidenceAry, d_startIndices + n, sizeof(IdxType), cudaMemcpyDeviceToHost));

  IdxType * d_incidenceAry;
  CUDA_CHECK(cudaMalloc(&d_incidenceAry, lenIncidenceAry * sizeof(IdxType)))
  profile->timeBuildIncidenceList = runAndMeasureCuda(
    buildIncidenceListsOnDevice,
    d_incidenceAry, d_xs, d_ys, d_startIndices, n, r
  );

  return { n, lenIncidenceAry, d_dcounts, d_startIndices, d_incidenceAry };
}

NeighborGraph copyDNeighborGraphToHost(DNeighborGraph const & g) {
  std::vector<IdxType> neighborCounts(g.nVertices);
  std::vector<IdxType> startIndices(g.nVertices + 1);
  std::vector<IdxType> incidenceAry(g.lenIncidenceAry);

  CUDA_CHECK(cudaMemcpy(neighborCounts.data(), g.d_neighborCounts, g.nVertices * sizeof(IdxType), cudaMemcpyDeviceToHost))
  CUDA_CHECK(cudaMemcpy(startIndices.data(), g.d_startIndices, (g.nVertices + 1) * sizeof(IdxType), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(incidenceAry.data(), g.d_incidenceAry, g.lenIncidenceAry * sizeof(IdxType), cudaMemcpyDeviceToHost))

  return { std::move(neighborCounts), std::move(startIndices), std::move(incidenceAry) };
}

void freeDNeighborGraph(DNeighborGraph & g) {
  (void)cudaFree(g.d_neighborCounts);
  (void)cudaFree(g.d_startIndices);
  (void)cudaFree(g.d_incidenceAry);
  g = DNeighborGraph{};
}

NeighborGraph buildNeighborGraph(
  BuildNeighborGraphProfile * profile,
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  DeviceVector<float> d_xs (xs, n);
  DeviceVector<float> d_ys (ys, n);

  DNeighborGraph g = buildDNeighborGraphOnDevice(profile, d_xs.data(), d_ys.data(), n, coreThreshold, r);

  NeighborGraph gg = copyDNeighborGraphToHost(g);

  freeDNeighborGraph(g);

  return gg;
}

NeighborGraph buildNeighborGraphCpu(
  float const * xs, float const * ys, IdxType n,
  IdxType coreThreshold, float r
) {
  std::vector<IdxType> neighborCounts (n);
  countNeighborsCpu(neighborCounts.data(), xs, ys, n, coreThreshold, r);

  std::vector<IdxType> cumulative (n + 1);
  prefixScanCpu(cumulative.data(), neighborCounts.data(), n);

  IdxType lenIncidenceAry = cumulative[n];
  std::vector<IdxType> incidenceAry(lenIncidenceAry);
  buildIncidenceListsCpu(incidenceAry.data(), xs, ys, cumulative.data(), n, r);

  return { std::move(neighborCounts), std::move(cumulative), std::move(incidenceAry) };
}