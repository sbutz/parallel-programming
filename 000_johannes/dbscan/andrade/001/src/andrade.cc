#include "read_input.h"
#include "build_graph.h"
#include "bfs.h"

#include "types.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>

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
bool checkListEquality(L1 const & l1, L2 const & l2) {
  auto s1 = l1.size();
  auto s2 = l2.size();
  if (s1 != s2) {
    std::cerr << "Lengths not equal (" << s1 << " vs. " << s2 << ")\n";
    return false;
  }
  for (std::size_t i = 0; i < s1; ++i) {
    if (l1[i] != l2[i]) {
      std::cerr << "Elements at position " << i << " not equal.\n";
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
};


void abortWithUsageMessage() {
	std::cerr << usageText;
	exit(1);
}

static Config parseCommandLineArguments(int argc, char * argv []) {
	// defaults
	Config config {};

	if (argc != 4) abortWithUsageMessage();

  config.inputFilePath = !strcmp("--", argv[1]) ? nullptr : argv[1];
  config.n = strtol(argv[2], nullptr, 10);
  config.r = strtof(argv[3], nullptr);

  if (config.n <= 0 || config.r <= 0.0f) abortWithUsageMessage();

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

int main (int argc, char * argv []) {
  Config config = parseCommandLineArguments(argc, argv);

  auto a = std::vector<float> {};
  auto b = std::vector<float> {};
  readInputFile(a, b, config.inputFilePath);

  auto nDataPoints = a.size();
  
  auto g = buildNeighborGraph(a.data(), b.data(), nDataPoints, config.r);
  
  auto gCpu = buildNeighborGraph(a.data(), b.data(), nDataPoints, config.r);

  bool ok = true;
  ok &= checkListEquality(g.neighborCounts, gCpu.neighborCounts);
  ok &= checkListEquality(g.startIndices, gCpu.startIndices);
  ok &= checkListEquality(g.incidenceAry, gCpu.incidenceAry);

  DeviceGraph gg((IdxType)nDataPoints, (IdxType)g.incidenceAry.size(), g.startIndices.data(), g.incidenceAry.data());
  AllComponentsFinder acf(&gg.g, g.incidenceAry.size());
  acf.findAllComponents(&gg.g, []{});
  auto tags = acf.getComponentTagsVector();

  for (std::size_t i = 0; i < a.size(); ++i) {
    std::cout << a[i] << " " << b[i] << " " << tags[i] << '\n';
  }
  return !ok;
}