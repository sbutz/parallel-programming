#include "build_graph.h"
#include "bfs.h"
#include "warmup.h"

#include "types.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

char const * usageText = "Usage: andrade input_file n r\n";

struct Config {
  char const * inputFilePath;
  unsigned int n;
  float r;
  bool performWarmup;
  bool checkGraph;
};

static void abortWithUsageMessage() {
	std::cerr << usageText;
	exit(1);
}

static Config parseCommandLineArguments(int argc, char * argv []) {
	Config config {};

	if (argc < 4 || argc > 5) abortWithUsageMessage();

  config.inputFilePath = !strcmp("--", argv[1]) ? nullptr : argv[1];
  config.n = strtol(argv[2], nullptr, 10);
  config.r = strtof(argv[3], nullptr);

  if (config.n <= 0 || config.r <= 0.0f) abortWithUsageMessage();

  if (argc == 5) {
    char const * argstr = argv[4];
  	for (size_t j = 0; argstr[j]; ++j) {
	  	switch (argstr[j]) {
        case 'w': {
          config.performWarmup = true;
        } break;
        case 'c': {
          config.checkGraph = true;
        } break;
      }
    }
  }

  return config;
}

template <typename F>
static void readFloats(std::istream & strm, F && handle) {
    float v;
    while (strm >> v) std::forward<F>(handle) (v);
}

static void readInput(std::istream & strm, std::vector<float> & x, std::vector<float> & y) {
    auto storeFloats = [&x, &y, c = false] (float v) mutable { (c ? x : y).push_back(v); c = !c; };
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

struct DbscanProfile : BuildNeighborGraphProfile, FindComponentsProfile {
  float timeTotal;
};

static auto copyPointsToDevice(float const * x, float const * y, IdxType n) {
  float * d_x, * d_y;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)))
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)))
  CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice))
  CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice))
  struct Result { float * d_x; float * d_y; IdxType n; };
  return Result { d_x, d_y, n };
}

static auto runDbscan (
  DbscanProfile * profile,
  float const * h_x, float const * h_y, IdxType nDataPoints,
  IdxType coreThreshold, float r
) {
	cudaEvent_t start; CUDA_CHECK(cudaEventCreate(&start));
	cudaEvent_t stop; CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

  auto points = copyPointsToDevice(h_x, h_y, nDataPoints);
  auto g1 = buildNeighborGraph(
    profile, points.d_x, points.d_y, points.n, coreThreshold, r
  );
  AllComponentsFinder acf(&g1, g1.lenIncidenceAry);
  acf.findAllComponents<findNextUnivisitedNaivePolicy, frontierBasicPolicy>(profile, &g1, []{});
  auto tags = acf.getComponentTagsVector();

  std::vector<IdxType> neighborCounts (nDataPoints);
  std::vector<char> isCore (nDataPoints);
  CUDA_CHECK(cudaMemcpy(neighborCounts.data(), g1.d_neighborCounts, nDataPoints * sizeof(IdxType), cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < nDataPoints; ++i) isCore[i] = !!neighborCounts[i];

	CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&profile->timeTotal, start, stop));

  struct Result {
    std::vector<char> isCore;
    std::vector<IdxType> clusters;
  };
  return Result { std::move(isCore), std::move(tags) };
}

int main (int argc, char * argv []) {
  DbscanProfile profile = {};

  Config config = parseCommandLineArguments(argc, argv);

  auto a = std::vector<float> {};
  auto b = std::vector<float> {};
  readInputFile(a, b, config.inputFilePath);

  auto nDataPoints = a.size();
  if (config.performWarmup) warmup();

  auto res = runDbscan(&profile, a.data(), b.data(), nDataPoints, config.n, config.r);

  // print JSON output
  std::cout << "{\n";
    std::cout << "\"output\": {\n";
      std::cout << "\"x\": "; jsonPrintFloatAry(a.data(), a.size()); std::cout << ",\n";
      std::cout << "\"y\": "; jsonPrintFloatAry(b.data(), b.size()); std::cout << ",\n";
      std::cout << "\"is_core\": "; jsonPrintUnsignedIntegerVector(res.isCore); std::cout << ",\n";
      std::cout << "\"cluster_id\": "; jsonPrintUnsignedIntegerVector(res.clusters); std::cout << "\n";
    std::cout << "},\n";
    std::cout << "\"profile\": {\n";
      std::cout << "\"timeNeighborCount\": " << profile.timeNeighborCount << ",\n";
      std::cout << "\"timePrefixScan\": " << profile.timePrefixScan << ",\n";
      std::cout << "\"timeBuildIncidenceList\": " << profile.timeBuildIncidenceList << ",\n";      
      std::cout << "\"timeMarkNonCore\": " << profile.timeMarkNonCore << ",\n";      
      std::cout << "\"timeFindComponents\": " << profile.timeFindComponents << ",\n";      
      std::cout << "\"timeTotal\": " << profile.timeTotal << "\n";      
    std::cout << "}\n";
  std::cout << "}\n";

  return 0;
}