#include "read_input.h"
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

template <typename L1>
void dumpList(L1 const & lst, std::size_t around) {
  constexpr std::size_t maxEl = 10;

  auto sz = lst.size();
  std::size_t s, e;
  if (sz <= maxEl) {
    s = 0; e = sz;
  } else if (around + 2 >= sz) {
    e = sz; s = e - maxEl;
  } else if (around < maxEl / 2) {
    s = 0; e = maxEl;
  } else {
    s = around - maxEl / 2; e = s + maxEl;
  }

  std::cerr << "[Length: " << sz << "] ";
  auto first = true;
  if (s > 0) std::cerr << "... ";

  for (std::size_t i = 0; i < e; ++i) {
    if (!first) std::cerr << " ";
    first = false;
    if (i == around) {
      std::cerr << "*" << lst[i] << "*";
    } else {
      std::cerr << lst[i];
    }
  }

  if (e < sz) std::cerr << " ...";
  std::cerr << '\n';
}

template <typename L1, typename L2>
bool checkListEquality(char const * info, L1 const & l1, L2 const & l2) {
  auto s1 = l1.size();
  auto s2 = l2.size();
  if (s1 != s2) {
    std::cerr << "[" << info << "] Lengths not equal (" << s1 << " vs. " << s2 << ")\n";
    return false;
  }
  for (std::size_t i = 0; i < s1; ++i) {
    if (l1[i] != l2[i]) {
      std::cerr << "[" << info << "] Elements at position " << i << " not equal.\n";
      dumpList(l1, i);
      dumpList(l2, i);
      return false;
    }
  }
  return true;
}


struct Config {
  char const * inputFilePath;
  unsigned int n;
  float r;
  bool performWarmup;
  bool checkGraph;
};


void abortWithUsageMessage() {
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

void jsonPrintIdxTypeAry(IdxType * ary, size_t n) {
	constexpr char const * formatStr = sizeof(IdxType) == 4? "%u" : "%lu";
	printf("[ ");
	if (n > 0) {
		size_t i = 0;
		for (;;) {
			printf(formatStr, ary[i]);
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

static auto runDbscan (
  DbscanProfile * profile,
  float const * h_x, float const * h_y, IdxType nDataPoints,
  IdxType coreThreshold, float r
) {
	cudaEvent_t start; CUDA_CHECK(cudaEventCreate(&start));
	cudaEvent_t stop; CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

  auto points = copyPointsToDevice(h_x, h_y, nDataPoints);
  auto g1 = buildDNeighborGraphOnDevice(
    profile, points.d_x, points.d_y, points.n, coreThreshold, r
  );
  AllComponentsFinder acf(&g1, g1.lenIncidenceAry);
  acf.findAllComponents(profile, &g1, []{});
  auto tags = acf.getComponentTagsVector();

	CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&profile->timeTotal, start, stop));

  struct Result {
    DNeighborGraph g1;
    std::vector<IdxType> tags;
  };
  return Result { std::move(g1), std::move(tags) };
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

  if (config.checkGraph) {
    auto g = copyDNeighborGraphToHost(res.g1);

    auto gCpu = buildNeighborGraphCpu(a.data(), b.data(), nDataPoints, config.n, config.r);

    bool ok = true;
    ok &= checkListEquality("neighborCounts", g.neighborCounts, gCpu.neighborCounts);
    ok &= checkListEquality("startIndices", g.startIndices, gCpu.startIndices);
    ok &= checkListEquality("incidenceAry", g.incidenceAry, gCpu.incidenceAry);
    if (!ok) { std::cerr << "Error: Graphs generated on CPU and GPU differ.\n"; exit(1); }
  }


  // print JSON output
  std::cout << "{\n";
    std::cout << "\"output\": {\n";
      std::cout << "\"x\": "; jsonPrintFloatAry(a.data(), a.size()); std::cout << ",\n";
      std::cout << "\"y\": "; jsonPrintFloatAry(b.data(), b.size()); std::cout << ",\n";
      std::cout << "\"cluster_id\": "; jsonPrintIdxTypeAry(res.tags.data(), res.tags.size()); std::cout << "\n";
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