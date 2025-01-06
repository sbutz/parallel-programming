#include <iostream>
#include <vector>

template <typename F>
void readFloats(std::istream & strm, F && handle) {
    float v;
    while (strm >> v) std::forward<F>(handle) (v);
}

void readInput(std::istream & strm, std::vector<float> & x, std::vector<float> & y) {
    auto storeFloats = [&x, &y, c = false] (float v) mutable { (c ? x : y).push_back(v); c = !c; };
    readFloats(strm, storeFloats);
}

int main () {
    auto a = std::vector<float> {};
    auto b = std::vector<float> {};

    readInput(std::cin, a, b);

    std::cout << "[ "; for (auto && x : a) { std::cout << x << ' '; } std::cout << "]\n";
    std::cout << "[ "; for (auto && x : b) { std::cout << x << ' '; } std::cout << "]\n";
    return 0;
}