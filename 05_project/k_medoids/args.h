#pragma once

#include "args.h"
#include "rapidcsv.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

template <typename size_type, typename value_type> struct Args {
    size_type n_clusters;
    std::vector<value_type> data;
    size_type n_points;
    size_type d_points;
    const char *medoids_path;
};

template <class value_type>
std::tuple<std::vector<value_type>, std::size_t, std::size_t> parseCsv(const char *path) {
    rapidcsv::Document doc(path, rapidcsv::LabelParams(-1, -1));
    std::size_t rows = doc.GetRowCnt();
    std::size_t cols = doc.GetColumnCount();
    std::vector<value_type> data;
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data.push_back(doc.GetCell<value_type>(j, i));
        }
    }
    return {data, rows, cols};
}

template <class size_type, class value_type>
Args<size_type, value_type> parseArgs(const int argc, const char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <n_clusters> <points_path> <medoids_path>"
                  << std::endl;
        exit(1);
    }

    Args<size_type, value_type> args{};

    args.n_clusters = std::atoi(argv[1]);
    auto csv = parseCsv<value_type>(argv[2]);
    args.data = std::get<0>(csv);
    args.n_points = std::get<1>(csv);
    args.d_points = std::get<2>(csv);
    args.medoids_path = argv[3];

    return args;
}

template <class size_type, class value_type>
void save_medoids(const Args<size_type, value_type> &args, std::vector<size_type> medoids) {
    std::ofstream file(args.medoids_path);
    assert(file.is_open());
    for (auto &medoid : medoids) {
        for (size_type j = 0; j < args.d_points; j++) {
            file << args.data[medoid * args.d_points + j];
            if (j < args.d_points - 1) {
                file << ",";
            }
        }
        file << std::endl;
    }
    file.close();
}
