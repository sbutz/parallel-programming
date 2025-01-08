#ifndef READ_INPUT_H_
#define READ_INPUT_H_

#include <iostream>
#include <vector>

template <typename F>
inline void readFloats(std::istream & strm, F && handle) {
    float v;
    while (strm >> v) std::forward<F>(handle) (v);
}

inline void readInput(std::istream & strm, std::vector<float> & x, std::vector<float> & y) {
    auto storeFloats = [&x, &y, c = false] (float v) mutable { (c ? x : y).push_back(v); c = !c; };
    readFloats(strm, storeFloats);
}

#endif