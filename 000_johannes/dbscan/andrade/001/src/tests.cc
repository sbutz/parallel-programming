#include <iostream>
#include <vector>
#include <array>
#include "build_incidence_lists.h"

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

void performTest(
  std::vector<float> const & xs, std::vector<float> const & ys,
  std::vector<std::size_t> const & cumulative
) {
  float r = 1.0f;
  std::vector<std::size_t> lstArray (cumulative[cumulative.size() - 1]);
  buildIncidenceLists(lstArray.data(), xs.data(), ys.data(), cumulative.data(), xs.size(), r);
  dumpList(lstArray, 0);
}

int main () {
  performTest(
    { 1.0f, 2.0f, 3.0f, 1.5f },
    { 1.0f, 2.0f, 3.0f, 1.0f },
    { 0, 2,    3,    4,    6 }
  );

  performTest(
    { 1.0f, 2.0f, 3.0f },
    { 1.0f, 2.0f, 3.0f },
    { 0, 1,    2,    3 }
  );

  return 0;
}