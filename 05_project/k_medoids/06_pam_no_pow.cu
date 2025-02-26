#include "args.h"
#include "util.h"
#include <cassert>
#include <cstring>
#include <cuda/std/limits>

using size_type = std::uint32_t;
using value_type = float;

static constexpr size_type N_THREADS = 512;
static constexpr size_type MAX_ITERATIONS = 1000;
static constexpr size_type D_POINTS = 2;

struct swap {
    size_type point;
    size_type medoid;
    value_type cost;
};

__device__ value_type euclidean_distance(const value_type *a, const value_type *b) {
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
}

__global__ void swap_cost(const value_type *points, const size_type n_points,
                          const size_type n_clusters, const size_type *medoids, swap *swaps) {
    // Consider swap point[x] with medoid[y]
    const size_type x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_type y = threadIdx.y + blockIdx.y * blockDim.y;

    const size_type idx = x + n_points * y;

    if (!(x < n_points && y < n_clusters)) {
        return;
    }

    swap s{x, y, ::cuda::std::numeric_limits<value_type>::max()};

    bool is_medoid = false;
    for (size_type i = 0; i < n_clusters; i++) {
        if (x == medoids[i]) {
            is_medoid = true;
        }
    }

    if (!is_medoid) {
        // total cost of all points
        s.cost = 0;
        for (size_type i = 0; i < n_points; ++i) {
            value_type min_distance = ::cuda::std::numeric_limits<value_type>::max();
            for (size_type j = 0; j < n_clusters; j++) {
                const value_type *medoid_addr = &points[medoids[j] * D_POINTS];
                if (j == y) {
                    medoid_addr = &points[x * D_POINTS];
                }
                min_distance =
                    min(min_distance, euclidean_distance(&points[i * D_POINTS], medoid_addr));
            }
            s.cost += min_distance;
        }
    }

    swaps[idx] = s;
}

std::vector<size_type> pam(const Args<size_type, value_type> &args) {
    assert(args.n_clusters <= args.n_points);
    assert(args.n_clusters <= N_THREADS);
    assert(args.d_points == D_POINTS);

    const size_type n_swaps = args.n_points * args.n_clusters;

    dim3 blockSize(std::pow(2, std::floor(log2(N_THREADS / args.n_clusters))), args.n_clusters, 1);
    dim3 gridSize((args.n_points + blockSize.x - 1) / blockSize.x,
                  (args.n_clusters + blockSize.y - 1) / blockSize.y, 1);

    // Input
    value_type *d_points;
    CUDA_ASSERT(cudaMalloc((void **)&d_points, args.n_points * args.d_points * sizeof(value_type)));
    CUDA_ASSERT(cudaMemcpy(d_points, args.data.data(),
                           args.n_points * args.d_points * sizeof(value_type),
                           cudaMemcpyHostToDevice));
    size_type *d_medoids;
    CUDA_ASSERT(cudaMalloc((void **)&d_medoids, args.n_clusters * sizeof(size_type)));
    // Output
    swap *d_swaps;
    CUDA_ASSERT(cudaMalloc((void **)&d_swaps, n_swaps * sizeof(swap)));

    // Build Phase
    size_type i;
    std::vector<size_type> medoids(args.n_clusters);
    for (i = 0; i < args.n_clusters; i++) {
        medoids[i] = i;
    }

    // Swap Phase
    value_type cost = ::cuda::std::numeric_limits<value_type>::max();
    for (i = 0; i < MAX_ITERATIONS; i++) {
        CUDA_ASSERT(cudaMemcpy(d_medoids, medoids.data(), args.n_clusters * sizeof(size_type),

                               cudaMemcpyHostToDevice));

        // Determine swap costs
        swap_cost<<<gridSize, blockSize>>>(d_points, args.n_points, args.n_clusters, d_medoids,
                                           d_swaps);
        CUDA_ASSERT(cudaGetLastError());
        CUDA_ASSERT(cudaDeviceSynchronize());

        std::vector<swap> swaps(n_swaps);
        CUDA_ASSERT(
            cudaMemcpy(swaps.data(), d_swaps, n_swaps * sizeof(swap), cudaMemcpyDeviceToHost));

        // Find best swap
        swap best_swap = swaps[0];
        for (size_type i = 0; i < n_swaps; i++) {
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
    const auto args = parseArgs<size_type, value_type>(argc, argv);

    cudaInit();

    TRACE(PAM, const auto medoids = pam(args);)

    save_medoids(args, medoids);

    return 0;
}