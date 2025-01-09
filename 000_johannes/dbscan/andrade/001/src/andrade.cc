#include "read_input.h"
#include "dbscan.h"
#include "accumulate.h"
#include "build_incidence_lists.h"

#include <iostream>
#include <vector>

int main () {
  auto a = std::vector<float> {};
  auto b = std::vector<float> {};

  readInput(std::cin, a, b);

  float r = 1.0f;
  auto n = a.size();
  auto c = std::vector<Count> (n);
  auto ccpu = std::vector<Count> (n);
  countNeighbors(c.data(), a.data(), b.data(), n, r);
  countNeighborsCpu(ccpu.data(), a.data(), b.data(), n, r);

  auto ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (c[i] != ccpu[i]) {
      ok = false;
      std::cerr << "Mismatch\n";
    } 
    //std::cout << c[i] << '\n';
  }

  if (!ok) {
    std::cerr << "There was a mismatch.\n";
    return 1;
  }

  auto d = std::vector<Count> (n + 1);
  auto dcpu = std::vector<Count> (n + 1);
  accumulate(d.data(), c.data(), n);
  accumulateCpu(dcpu.data(), c.data(), n);

  for (std::size_t i = 0; i <= n; ++i) {
    if (d[i] != dcpu[i]) {
      ok = false;
      std::cerr << "Mismatch (accumulation) " << i << " " << d[i] << " " << dcpu[i] << 
       " " << c[0] << " " << c[1] << '\n';
      return 1;
    } 
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::cout << d[i] << '\n';
  }

  if (!ok) {
    std::cerr << "There was a mismatch in accumulation.\n";
    return 1;
  }

  auto il = std::vector<Count> (d[n]);
  auto ilcpu = std::vector<Count> (d[n]);
  buildIncidenceLists(il.data(), a.data(), b.data(), d.data(), n, r);
  buildIncidenceListsCpu(ilcpu.data(), a.data(), b.data(), d.data(), n, r);

  for (std::size_t i = 0; i < il.size(); ++i) {
    if (il[i] != ilcpu[i]) {
      ok = false;
      std::cerr << "Mismatch (incidence lists) " << i << " " << il[i] << " " << ilcpu[i] << '\n';
      return 1;
    } 
  }

  return !ok;
}