#include "read_input.h"
#include "build_graph.h"

#include <iostream>
#include <vector>

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

int main () {
  auto a = std::vector<float> {};
  auto b = std::vector<float> {};

  readInput(std::cin, a, b);

  float r = 0.5f;
  auto n = a.size();
  
  auto g = buildNeighborGraph(a.data(), b.data(), n, r);
  
  auto gCpu = buildNeighborGraph(a.data(), b.data(), n, r);

  bool ok = true;
  ok &= checkListEquality(g.neighborCounts, gCpu.neighborCounts);
  ok &= checkListEquality(g.startIndices, gCpu.startIndices);
  ok &= checkListEquality(g.incidenceAry, gCpu.incidenceAry);

  return !ok;
}