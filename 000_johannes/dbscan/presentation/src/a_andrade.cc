#include "a_build_graph.h"
#include "a_bfs.h"
#include "warmup.h"

#include "a_types.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

char const * usageText = "Usage: a_andrade input_file n r [w] [iterations]\n";

struct Graph {
  std::vector<IdxType> startIndices;
  std::vector<IdxType> incidenceLists;
};

struct Config {
  char const * inputFilePath;
  unsigned int n;
  float r;
  bool performWarmup;
  int nIterations;
};

static void abortWithUsageMessage() {
	std::cerr << usageText;
	exit(1);
}

static Config parseCommandLineArguments(int argc, char * argv []) {
	Config config {};

	if (argc < 4 || argc > 6) abortWithUsageMessage();

  config.inputFilePath = !strcmp("--", argv[1]) ? nullptr : argv[1];
  config.n = strtol(argv[2], nullptr, 10);
  config.r = strtof(argv[3], nullptr);

  if (config.n <= 0 || config.r <= 0.0f) abortWithUsageMessage();

  if (argc >= 5) {
    char const * argstr = argv[4];
  	for (size_t j = 0; argstr[j]; ++j) {
	  	switch (argstr[j]) {
        case 'w': {
          config.performWarmup = true;
        } break;
        default: abortWithUsageMessage;
      }
    }
  }

  if (argc >= 6) {
    config.nIterations = strtol(argv[5], nullptr, 10);
    if (config.nIterations <= 0) abortWithUsageMessage ();
  } else {
    config.nIterations = 1;
  }

  return config;
}

template <typename F>
static void readFloats(std::istream & strm, F && handle) {
    float v;
    while (strm >> v) std::forward<F>(handle) (v);
}

static void readInput(std::istream & strm, std::vector<float> & x, std::vector<float> & y) {
    auto storeFloats = [&x, &y, c = false] (float v) mutable { (c ? y : x).push_back(v); c = !c; };
    readFloats(strm, storeFloats);
}

static void readInputFile(std::vector<float> & a, std::vector<float> & b, char const * filePath) {
  if (!filePath) {
    readInput(std::cin, a, b);
  } else {
    auto inputStream = std::ifstream(filePath);
    readInput(inputStream, a, b);
  }
}

// *****************************************************************************
// Hilfsfunktionen für die Ausgabe von Arrays im JSON-Format
// *****************************************************************************

void jsonPrintFloatAry(float * ary, size_t n) {
	printf("[ ");
	if (n > 0) {
		size_t i = 0;
		for (;;) {
			printf("%.7f", ary[i]);
			++i;
			if (i == n) break;
			printf(", ");
		}
	}
	printf(" ]");
}

static void jsonPrintIdxTypeAry(IdxType * ary, size_t n) {
	printf("[ ");
	if (n > 0) {
		size_t i = 0;
		for (;;) {
			printf("%lu", static_cast<std::uint64_t> (ary[i]));
			++i;
			if (i == n) break;
			printf(", ");
		}
	}
	printf(" ]");
}

template <typename T>
void jsonPrintUnsignedIntegerVector(std::vector<T> const & vec) {
  auto n = vec.size();
	printf("[ ");
	if (n > 0) {
		size_t i = 0;
		for (;;) {
			printf("%lu", static_cast<std::uint64_t> (vec[i]));
			++i;
			if (i == n) break;
			printf(", ");
		}
	}
	printf(" ]");
}

struct DbscanProfilingData : BuildNeighborGraphProfilingData, FindComponentsProfilingData {
  float timeTotal;
};

// get number of multiprocessors on device
static int getNSm() {
  int nSm;
  CUDA_CHECK(cudaDeviceGetAttribute(&nSm, cudaDevAttrMultiProcessorCount, 0));
  return nSm;
}

static Graph copyGraphToHost(DNeighborGraph const & deviceGraph, IdxType nDataPoints) {
  IdxType lenIncidenceAry;
  CUDA_CHECK(cudaMemcpy(&lenIncidenceAry, deviceGraph.d_startIndices + nDataPoints, sizeof(IdxType), cudaMemcpyDeviceToHost))

  Graph gg {};
  gg.startIndices.resize(nDataPoints + 1);
  gg.incidenceLists.resize(lenIncidenceAry);
  CUDA_CHECK(cudaMemcpy(gg.startIndices.data(), deviceGraph.d_startIndices, (nDataPoints + 1) * sizeof(IdxType), cudaMemcpyDeviceToHost))
  CUDA_CHECK(cudaMemcpy(gg.incidenceLists.data(), deviceGraph.d_incidenceAry, lenIncidenceAry * sizeof(IdxType), cudaMemcpyDeviceToHost))
  return gg;
}

static auto runDbscan (
  DbscanProfilingData * profile,
  float const * h_x, float const * h_y, IdxType nDataPoints,
  IdxType coreThreshold, float r
) {
  int nSm = getNSm();

	cudaEvent_t start; CUDA_CHECK(cudaEventCreate(&start));
	cudaEvent_t stop; CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

  auto && d_x = ManagedDeviceArray<float> (nDataPoints);
  auto && d_y = ManagedDeviceArray<float> (nDataPoints);
  auto && d_visited = ManagedDeviceArray<IdxType> (nDataPoints);
  CUDA_CHECK(cudaMemcpy(d_x.ptr(), h_x, nDataPoints * sizeof(float), cudaMemcpyHostToDevice))
  CUDA_CHECK(cudaMemcpy(d_y.ptr(), h_y, nDataPoints * sizeof(float), cudaMemcpyHostToDevice))
  auto g1 = buildNeighborGraph(
    profile, d_x.ptr(), d_y.ptr(), nDataPoints, coreThreshold, r
  );

  Graph hostGraph = copyGraphToHost(g1, nDataPoints);

  std::vector<std::vector<IdxType>> visitedSteps;
  findAllComponents(nSm, d_visited.ptr(), profile, &g1, visitedSteps);

  auto clusters = std::vector<IdxType> (nDataPoints);
  CUDA_CHECK(cudaMemcpy(clusters.data(), d_visited.ptr(), nDataPoints * sizeof(IdxType), cudaMemcpyDeviceToHost))
  
  std::vector<IdxType> neighborCounts (nDataPoints);
  std::vector<signed char> isCore (nDataPoints);
  static_assert(sizeof(bool) == 1, "");
  CUDA_CHECK(cudaMemcpy(neighborCounts.data(), g1.d_neighborCounts, nDataPoints * sizeof(IdxType), cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < nDataPoints; ++i) isCore[i] = !!neighborCounts[i];

  freeDNeighborGraph(g1);
  
	CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&profile->timeTotal, start, stop));

  struct Result {
    std::vector<signed char> isCore;
    std::vector<IdxType> clusters;
    Graph graph;
    std::vector<std::vector<IdxType>> visitedSteps;
  };
  return Result { std::move(isCore), std::move(clusters), std::move(hostGraph), std::move(visitedSteps) };
}

int main (int argc, char * argv []) {
  Config config = parseCommandLineArguments(argc, argv);

  auto a = std::vector<float> {};
  auto b = std::vector<float> {};
  readInputFile(a, b, config.inputFilePath);

  auto nDataPoints = a.size();
  if (!nDataPoints) {
    std::cerr << "Error: Cannot cluster 0 data points.\n";
    return 1;
  }
  if (config.performWarmup) warmup();

  std::vector<DbscanProfilingData> profiles {};

  std::vector<signed char> isCore {};
  std::vector<IdxType> clusters {};
  Graph graph {};
  std::vector<std::vector<IdxType>> visitedSteps {};
  for (int i = 0;; ++i) {
    DbscanProfilingData profile = {};
    auto res = runDbscan(&profile, a.data(), b.data(), nDataPoints, config.n, config.r);
    profiles.push_back(profile);
    if (i < config.nIterations - 1) continue;

    isCore = std::move(res.isCore);
    clusters = std::move(res.clusters);
    graph = std::move(res.graph);
    visitedSteps = std::move(res.visitedSteps);
    break;
  }

  bool first;

  // print JSON output
  std::cout << "{\n";
    std::cout << "\"output\": {\n";
      std::cout << "\"x\": "; jsonPrintFloatAry(a.data(), a.size()); std::cout << ",\n";
      std::cout << "\"y\": "; jsonPrintFloatAry(b.data(), b.size()); std::cout << ",\n";
      std::cout << "\"is_core\": "; jsonPrintUnsignedIntegerVector(isCore); std::cout << ",\n";
      std::cout << "\"cluster_id\": "; jsonPrintUnsignedIntegerVector(clusters); std::cout << "\n";
    std::cout << "},\n";

    first = true;
    std::cout << "\"graph\": [\n";
      for (int i = 0; i < graph.startIndices.size() - 1; ++i) {
        if (!first) { std::cout << ",\n"; } else { first = false; }
        jsonPrintIdxTypeAry(graph.incidenceLists.data() + graph.startIndices[i], graph.startIndices[i+1] - graph.startIndices[i]);
      }
    std::cout << "\n],\n";

    first = true;
    std::cout << "\"visitedSteps\": [\n";
      for (int i = 0; i < visitedSteps.size(); ++i) {
        if (!first) { std::cout << ",\n"; } else { first = false; }
        jsonPrintUnsignedIntegerVector(visitedSteps[i]);
      }
    std::cout << "\n],\n";

    std::cout << "\"profiles\": [\n";
    first = true;
    for (auto && profile : profiles) {
      if (!first) { std::cout << ",\n"; } else { first = false; }
      std::cout << "{\n";
      std::cout << "\"timeNeighborCount\": " << profile.timeNeighborCount << ",\n";
      std::cout << "\"timePrefixScan\": " << profile.timePrefixScan << ",\n";
      std::cout << "\"timeBuildIncidenceList\": " << profile.timeBuildIncidenceList << ",\n";      
      std::cout << "\"timeMarkCoreUnvisited\": " << profile.timeMarkCoreUnvisited << ",\n";      
      std::cout << "\"timeFindComponents\": " << profile.timeFindComponents << ",\n";      
      std::cout << "\"timeTotal\": " << profile.timeTotal << "\n";
      std::cout << "}";
    }
    std::cout << "\n]\n";
  std::cout << "}\n";

  return 0;
}