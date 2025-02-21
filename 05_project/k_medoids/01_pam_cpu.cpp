#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "args.h"

double euclidean_distance(const double* a, const double* b, std::size_t size)
{
    double distance = 0.0;
    for (std::size_t i = 0; i < size; i++)
    {
        distance += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(distance);
}

double total_cost(const double* points, std::size_t n, const double* medoids, std::size_t k, std::size_t size)
{
    double cost = 0.0;
    for (std::size_t i = 0; i < n; i += 1)
    {
        double min_distance = std::numeric_limits<double>::max();
        for (std::size_t j = 0; j < k; j += 1)
        {
            min_distance = std::min(min_distance, euclidean_distance(&points[i*size], &medoids[j*size], size));
        }
        cost += min_distance;
    }
    return cost;
}

bool find_doubles(const double* p, std::size_t n, const double* v, std::size_t k)
{
    for (std::size_t i = 0; i < n; i += 1)
    {
        bool found = true;
        for (std::size_t j = 0; j < k; ++j)
        {
            if (std::fabs(p[i * k + j] - v[j]) >= std::numeric_limits<double>::epsilon())
            {
                found = false;
                break;
            }
        }
        if (found)
        {
            return true;
        }
    }
    return false;
}

double* pam(const double* points, std::size_t n, std::size_t size, std::size_t k)
{
    double* medoids = new double[k*size];
    double* new_medoids = new double[k*size];
    memcpy(medoids, points, k * size * sizeof(double));
    double cost = total_cost(points, n, medoids, k, size);

    // do till convergence
    do {
        double best_cost = std::numeric_limits<double>::max();
        std::pair<std::size_t, std::size_t> best_swap{0, 0};
        // for each medoid
        for (std::size_t i = 0; i < k; i++)
        {
            // for each non-medoid
            for (std::size_t j = 0; j < n; j++)
            {
                if (find_doubles(medoids, k, &points[j*size], size))
                {
                    continue;
                }

                // consider swap
                memcpy(new_medoids, medoids, k * size * sizeof(double));
                memcpy(&new_medoids[i*size], &points[j*size], size * sizeof(double));
                double tmp = total_cost(points, n, new_medoids, k, size);
                if (tmp < best_cost){
                    best_cost = tmp;
                    best_swap = {i, j};
                }
            }
        }

        // perform best swap
        if (best_cost < cost) {
            memcpy(&medoids[best_swap.first*size], &points[best_swap.second*size], size * sizeof(double));
            cost = best_cost;
        } else {
            break;
        }
    } while (true);

    free(new_medoids);

    return medoids;
}

int main(const int argc, const char* argv[])
{
    const Args args = parseArgs(argc, argv);

    double* medoids = pam(args.data.data(), args.n_points, args.d_points, args.n_clusters);

    printDoubles(medoids, args.n_clusters, args.d_points);

    free(medoids);

    return 0;
}