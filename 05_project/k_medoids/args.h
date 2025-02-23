#pragma once

#include <cstddef>
#include <vector>

struct Args
{
    std::size_t n_clusters;
    std::vector<double> data;
    std::size_t n_points;
    std::size_t d_points;
};

Args parseArgs(const int argc, const char* argv[]);

void printDoubles(const double* p, std::size_t n_points, std::size_t d_points);
