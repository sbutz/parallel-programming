#include "args.h"
#include "util.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <cuda/std/limits>

static constexpr std::size_t N_THREADS = 512;
static constexpr std::size_t MAX_ITERATIONS = 1000;

struct swap {
    std::size_t point;
    std::size_t medoid;
    double cost;
};

__device__ double euclidean_distance(const double *a, const double *b, std::size_t d_points) {
    double distance = 0.0;
    for (std::size_t i = 0; i < d_points; i++) {
        distance += pow(a[i] - b[i], 2);
    }
    return distance;
}

__global__ void swap_cost(const double *points, const std::size_t n_points,
                          const std::size_t d_points, const std::size_t n_clusters,
                          const std::size_t *medoids, swap *swaps) {
    // Consider swap point[x] with medoid[y]
    const std::size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    const std::size_t idx = x + n_points * y;

    if (!(x < n_points && y < n_clusters)) {
        return;
    }

    swap s{x, y, ::cuda::std::numeric_limits<double>::max()};

    bool is_medoid = false;
    for (std::size_t i = 0; i < n_clusters; i++) {
        if (x == medoids[i]) {
            is_medoid = true;
        }
    }

    if (!is_medoid) {
        // total cost of all points
        s.cost = 0;
        for (std::size_t i = 0; i < n_points; ++i) {
            double min_distance = ::cuda::std::numeric_limits<double>::max();
            for (std::size_t j = 0; j < n_clusters; j++) {
                const double *medoid_addr = &points[medoids[j] * d_points];
                if (j == y) {
                    medoid_addr = &points[x * d_points];
                }
                min_distance = min(
                    min_distance, euclidean_distance(&points[i * d_points], medoid_addr, d_points));
            }
            s.cost += min_distance;
        }
    }

    swaps[idx] = s;
}

std::vector<std::size_t> pam(const Args<std::size_t, double> &args) {
    assert(args.n_clusters <= args.n_points);
    assert(args.n_clusters <= N_THREADS);

    const std::size_t n_swaps = args.n_points * args.n_clusters;

    dim3 blockSize(std::pow(2, std::floor(log2(N_THREADS / args.n_clusters))), args.n_clusters, 1);
    dim3 gridSize((args.n_points + blockSize.x - 1) / blockSize.x,
                  (args.n_clusters + blockSize.y - 1) / blockSize.y, 1);

    // Input
    double *d_points;
    CUDA_ASSERT(cudaMalloc((void **)&d_points, args.n_points * args.d_points * sizeof(double)));
    CUDA_ASSERT(cudaMemcpy(d_points, args.data.data(),
                           args.n_points * args.d_points * sizeof(double), cudaMemcpyHostToDevice));
    std::size_t *d_medoids;
    CUDA_ASSERT(cudaMalloc((void **)&d_medoids, args.n_clusters * sizeof(std::size_t)));
    // Output
    swap *d_swaps;
    CUDA_ASSERT(cudaMalloc((void **)&d_swaps, n_swaps * sizeof(swap)));

    // Build Phase
    std::size_t i;
    std::vector<std::size_t> medoids(args.n_clusters);
    for (i = 0; i < args.n_clusters; i++) {
        medoids[i] = i;
    }

    // Swap Phase
    double cost = ::cuda::std::numeric_limits<double>::max();
    for (i = 0; i < MAX_ITERATIONS; i++) {
        CUDA_ASSERT(cudaMemcpy(d_medoids, medoids.data(), args.n_clusters * sizeof(std::size_t),

                               cudaMemcpyHostToDevice));

        // Determine swap costs
        swap_cost<<<gridSize, blockSize>>>(d_points, args.n_points, args.d_points, args.n_clusters,
                                           d_medoids, d_swaps);
        CUDA_ASSERT(cudaGetLastError());
        CUDA_ASSERT(cudaDeviceSynchronize());

        std::vector<swap> swaps(n_swaps);
        CUDA_ASSERT(
            cudaMemcpy(swaps.data(), d_swaps, n_swaps * sizeof(swap), cudaMemcpyDeviceToHost));

        // Find best swap
        swap best_swap = swaps[0];
        for (std::size_t i = 0; i < n_swaps; i++) {
            if (swaps[i].cost < best_swap.cost) {
                best_swap = swaps[i];
            }
        }

        if (best_swap.cost >= cost) {
            break;
        }

        medoids[best_swap.medoid] = best_swap.point;
        cost = best_swap.cost;
    }

    if (i == MAX_ITERATIONS) {
        printf("Max iterations reached\n");
        std::abort();
    }

    CUDA_ASSERT(cudaFree(d_points));
    CUDA_ASSERT(cudaFree(d_medoids));
    CUDA_ASSERT(cudaFree(d_swaps));

    return medoids;
}

int main(const int argc, const char *argv[]) {
    const auto args = parseArgs<std::size_t, double>(argc, argv);

    cudaInit();

    TRACE(PAM, const auto medoids = pam(args);)

    save_medoids(args, medoids);

    return 0;
}