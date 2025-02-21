#include <iostream>
#include <cstdlib>
#include "args.h"
#include "rapidcsv.h"

static std::tuple<std::vector<double>, std::size_t, std::size_t> parseCsv(const char *path);

Args parseArgs(const int argc, const char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <n_clusters> <data_path>" << std::endl;
        exit(1);
    }

    Args args{};

    args.n_clusters = std::atoi(argv[1]);
    auto csv = parseCsv(argv[2]);
    args.data = std::get<0>(csv);
    args.n_points = std::get<1>(csv);
    args.d_points = std::get<2>(csv);

    return args;
}

std::tuple<std::vector<double>, std::size_t, std::size_t> parseCsv(const char *path) {
    rapidcsv::Document doc(path);
    std::size_t rows = doc.GetRowCnt();
    std::size_t cols = doc.GetColumnCount();
    std::vector<double> data;
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data.push_back(doc.GetCell<double>(j, i));
        }
    }
    return {data, rows, cols};
}

void printDoubles(double* p, std::size_t n_points, std::size_t d_points)
{
    for (std::size_t i = 0; i < n_points; ++i)
    {
        for (std::size_t j = 0; j < d_points; ++j)
        {
            std::printf("%.18f", p[i * d_points + j]);
            if (j < d_points - 1)
            {
                std::printf(",");
            }
        }
        std::cout << std::endl;
    }
}