#include "read_input.h"
#include "dbscan.h"

#include <iostream>
#include <vector>

int main () {
  auto a = std::vector<float> {};
  auto b = std::vector<float> {};

  readInput(std::cin, a, b);

  auto n = a.size();
  auto c = std::vector<Count> (n);
  auto ccpu = std::vector<Count> (n);
  countNeighbors(c.data(), a.data(), b.data(), n, 1.0f);
  countNeighborsCpu(ccpu.data(), a.data(), b.data(), n, 1.0f);

  auto ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (c[i] != ccpu[i]) {
      ok = false;
      std::cerr << "Mismatch\n";
    } 
    std::cout << c[i] << '\n';
  }

  if (!ok) std::cerr << "There was a mismatch.\n";
  
  return !ok;
}