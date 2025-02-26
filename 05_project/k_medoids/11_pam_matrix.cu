#include "args.h"
#include "util.h"
#include <cassert>
#include <cstring>
#include <cuda/std/limits>

using size_type = std::uint32_t;
using value_type = float;

static constexpr size_type N_THREADS = 256;
static constexpr size_type MAX_ITERATIONS = 1000;

struct swap {
    size_type point;
    size_type medoid;
    value_type cost;
};

__device__ value_type euclidean_distance(const value_type *a, const value_type *b) {
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
}

__global__ void distance_matrix(const value_type *points, const size_type n_points,
                                value_type *distances) {
    const size_type x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_type y = threadIdx.y + blockIdx.y * blockDim.y;

    if (!(x < n_points && y < n_points)) {
        return;
    }

    distances[x + n_points * y] = euclidean_distance(&points[x], &points[y]);
}

__global__ void swap_cost(const value_type *distances, const size_type n_points,
                          const size_type n_clusters, const size_type *medoids, swap *swaps) {
    // Consider swap point[x] with medoid[y]
    const size_type x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_type y = threadIdx.y + blockIdx.y * blockDim.y;

    if (!(x < n_points && y < n_clusters)) {
        return;
    }

    const size_type idx = x + n_points * y;

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
                const size_type medoid_idx = j == y ? x : medoids[j];
                min_distance = min(min_distance, distances[i + n_points * medoid_idx]);
            }
            s.cost += min_distance;
        }
    }

    swaps[idx] = s;
}

std::vector<size_type> pam(const Args<size_type, value_type> &args) {
    assert(args.n_clusters <= args.n_points);
    assert(args.n_clusters <= N_THREADS);
    assert(args.d_points == 2);

    const size_type n_swaps = args.n_points * args.n_clusters;

    // Input
    value_type *d_points;
    CUDA_ASSERT(cudaMalloc((void **)&d_points, args.n_points * args.d_points * sizeof(value_type)));
    CUDA_ASSERT(cudaMemcpy(d_points, args.data.data(),
                           args.n_points * args.d_points * sizeof(value_type),
                           cudaMemcpyHostToDevice));

    // Output
    value_type *d_distances;
    CUDA_ASSERT(
        cudaMalloc((void **)&d_distances, args.n_points * args.n_points * sizeof(value_type)));

    // Build Distance Matrix
    dim3 blockSize(next_smaller_power_of_two(sqrt(N_THREADS)),
                   next_smaller_power_of_two(sqrt(N_THREADS)), 1);
    dim3 gridSize((args.n_points + blockSize.x - 1) / blockSize.x,
                  (args.n_points + blockSize.y - 1) / blockSize.y, 1);
    distance_matrix<<<gridSize, blockSize>>>(args.data.data(), args.n_points, d_distances);
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetLastError());

    // Input
    size_type *d_medoids;
    CUDA_ASSERT(cudaMalloc((void **)&d_medoids, args.n_clusters * sizeof(size_type)));
    // Output
    swap *d_swaps;
    CUDA_ASSERT(cudaMalloc((void **)&d_swaps, n_swaps * sizeof(swap)));

    blockSize = {next_smaller_power_of_two(N_THREADS / args.n_clusters), args.n_clusters, 1};
    gridSize = {(args.n_points + blockSize.x - 1) / blockSize.x,
                (args.n_clusters + blockSize.y - 1) / blockSize.y, 1};

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
        swap_cost<<<gridSize, blockSize>>>(d_distances, args.n_points, args.n_clusters, d_medoids,
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

    CUDA_ASSERT(cudaFree(d_distances));
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