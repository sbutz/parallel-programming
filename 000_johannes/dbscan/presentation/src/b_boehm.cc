#include "warmup.h"
#include "b_cluster_expansion.h"

#include "b_types.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

char const * usageText = "Usage: b_boehm input_file n r [w] [iterations]\n";

struct Config {
  char const * inputFilePath;
  unsigned int n;
  float r;
  bool performWarmup;
  int nIterations;
};

void abortWithUsageMessage() {
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

struct DbscanProfilingData {
  float timeTotal;
};

// get number of multiprocessors on device
static int getNSm() {
  int nSm;
  CUDA_CHECK(cudaDeviceGetAttribute(&nSm, cudaDevAttrMultiProcessorCount, 0));
  return nSm;
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

  constexpr int nBlocks = 6;

  CUDA_CHECK(cudaMemcpy(d_x.ptr(), h_x, nDataPoints * sizeof(float), cudaMemcpyHostToDevice))
  CUDA_CHECK(cudaMemcpy(d_y.ptr(), h_y, nDataPoints * sizeof(float), cudaMemcpyHostToDevice))

  auto && d_pointStates = ManagedDeviceArray<unsigned int> (nDataPoints);
  auto && d_clusters = ManagedDeviceArray<IdxType> (nDataPoints);

  findClusters(
    nSm,
    d_pointStates.ptr(), d_clusters.ptr(), d_x.ptr(), d_y.ptr(), nDataPoints,
    coreThreshold, r * r
  );

  std::vector<unsigned int> states(nDataPoints);
  CUDA_CHECK(cudaMemcpy(states.data(), d_pointStates.ptr(), nDataPoints * sizeof(unsigned int), cudaMemcpyDeviceToHost))

  std::vector<signed char> isCore(nDataPoints); // avoid vector<bool>
  for (std::size_t i = 0; i < nDataPoints; ++i) isCore[i] = (states[i] & stateCore) != 0;

  std::vector<IdxType> clusters(nDataPoints);
  CUDA_CHECK(cudaMemcpy(clusters.data(), d_clusters.ptr(), nDataPoints * sizeof(IdxType), cudaMemcpyDeviceToHost))

	CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&profile->timeTotal, start, stop));

  struct Result {
    std::vector<signed char> isCore;
    std::vector<IdxType> clusters;
  };
  return Result { std::move(isCore), std::move(clusters) };
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
  for (int i = 0;; ++i) {
    DbscanProfilingData profile = {};
    auto res = runDbscan(&profile, a.data(), b.data(), nDataPoints, config.n, config.r);
    profiles.push_back(profile);
    if (i < config.nIterations - 1) continue;

    isCore = std::move(res.isCore);
    clusters = std::move(res.clusters);
    break;
  }

  // print JSON output
  std::cout << "{\n";
    std::cout << "\"output\": {\n";
      std::cout << "\"x\": "; jsonPrintFloatAry(a.data(), a.size()); std::cout << ",\n";
      std::cout << "\"y\": "; jsonPrintFloatAry(b.data(), b.size()); std::cout << ",\n";
      std::cout << "\"is_core\": "; jsonPrintUnsignedIntegerVector(isCore); std::cout << ",\n";
      std::cout << "\"cluster_id\": "; jsonPrintUnsignedIntegerVector(clusters); std::cout << "\n";
     std::cout << "},\n";
    std::cout << "\"profiles\": [\n";
    bool first = true;
    for (auto && profile : profiles) {
      if (!first) { std::cout << ",\n"; } else { first = false; }
      std::cout << "{\n";
      std::cout << "\"timeTotal\": " << profile.timeTotal << "\n";
      std::cout << "}";
    }
    std::cout << "\n]\n";
  std::cout << "}\n";

  return 0;
}