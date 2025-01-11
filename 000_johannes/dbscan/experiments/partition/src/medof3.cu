#include "cuda_helpers.h"

#include <cuda.h>
#include <iostream>
#include <algorithm>
#include <vector>

static constexpr unsigned int minWarpSize = 32;

static __host__ __device__ void sort3(float & a, float & b, float & c) {
  float tmp;
  if (b > a) { tmp = a; a = b; b = tmp; }
  if (c > b) { tmp = b; b = c; c = tmp; }
  if (b > a) { tmp = c; c = a; a = tmp; }
}

static __host__ __device__ void sort4(float & a, float & b, float & c, float & d) {
  float tmp [4];
  if (b > a) { tmp[0] = a; tmp[1] = b; } else { tmp[0] = b; tmp[1] = a; }
  if (d > c) { tmp[2] = c; tmp[3] = d; } else { tmp[2] = d; tmp[3] = c; }
  int i = 0, j = 2;
  if (tmp[i] < tmp[j]) a = tmp[i++]; else a = tmp[j++];
  if (tmp[i] < tmp[j]) b = tmp[i++]; else b = tmp[j++];
  if (i == 2) { c = tmp[2]; d = tmp[3]; }
  else if (j == 4) { c = tmp[0]; d = tmp[1]; }
  else if (tmp[1] < tmp[3]) { c = tmp[1]; d = tmp[3]; }
  else { c = tmp[3]; d = tmp[1]; }
}

static __global__ void kernel_medsOf3(float * values, std::size_t n) {
  std::size_t t1 = (n + 2u) / 3u;
  std::size_t t0 = (n - t1 + 1u) / 2u;
  std::size_t t2 = n - t0 - t1;

  // now we have t1 >= t0 >= t2 and t0 + t1 + t2 = n4

  std::size_t start1 = t0;
  std::size_t start2 = start1 + t1;

  // TODO: Missing: Handling stride.
  using size_t = std::size_t;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  if (tid < t2) {
    // first case: three blocks
    float a = values[tid];
    float b = values[start1 + tid];
    float c = values[start2 + tid];

    sort3(a, b, c);

    values[tid] = a;
    values[start1 + tid] = b;
    values[start2 + tid] = c;
  }
}

float * selectIth(float * d_values, std::size_t n, std::size_t i);

float * medianOfMediansOfMedians (float * d_values, std::size_t n) {
  float * v;
  std::size_t nn, t0, t1, t2;

  v = d_values;
  nn = n;

  t1 = (nn + 2u) / 3u;
  t0 = (nn - t1 + 1u) / 2u;
  t2 = nn - t0 - t1;

  if (t2 != 0) {

    dim3 dimBlock(256);
    dim3 dimGrid(1024);

    kernel_medsOf3<<<dimGrid,dimBlock>>> (v, nn);

    v += t0;
    nn = t1;
    t1 = (nn + 2u) / 3u;
    t0 = (nn - t1 + 1u) / 2u;
    t2 = nn - t0 - t1;

    if (t2 != 0) {
      kernel_medsOf3<<<dimGrid,dimBlock>>> (v, nn);
    }
  }

  v += t0;
  nn = t1;

  float * d_p = selectIth(v, nn, nn / 2);
  float pval;
  CUDA_CHECK(cudaMemcpy(&pval, d_p, sizeof(float), cudaMemcpyDeviceToHost));
  std::cerr << "Returning from mediansOfMediansOfMedians: " << pval << "\n";
  return d_p;
}


float * partitionPivotCpu(float * s, float * e, float * pivotPtr);

template <typename T>
static __device__ void d_sumShared(T * sValues1, T * sValues2, unsigned int n) {
  unsigned int const tidInBlock = threadIdx.x;
  unsigned int const wid = threadIdx.x / warpSize;

  unsigned int nWarpsRequired = (n + warpSize - 1) / warpSize;
  if (wid < nWarpsRequired) {
    for (;;) {
      T sum1 = tidInBlock < n ? sValues1[tidInBlock] : T{};
      T sum2 = tidInBlock < n ? sValues2[tidInBlock] : T{};
      for (auto w = warpSize / 2; w != 0; w /= 2) {
        sum1 += __shfl_down_sync(0xffffffff, sum1, w);
        sum2 += __shfl_down_sync(0xffffffff, sum2, w);
      }

      __syncwarp();

      if (!(tidInBlock % warpSize)) {
        sValues1[wid] = sum1;
        sValues2[wid] = sum2;
      }

      n = (n + warpSize - 1) / warpSize;
      if (n == 1) break;

      nWarpsRequired = (n + warpSize - 1) / warpSize;
      if (wid >= nWarpsRequired) break;

      __syncthreads();
    }
  }
}

static __global__ void kernel_countLowerUpper(
  std::size_t * lower, std::size_t * upper, unsigned int nBlocks,
  float * values, std::size_t n, float pivot
) {
  using size_t = std::size_t;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  unsigned int wid = threadIdx.x / warpSize;  // index of current warp in current block
  unsigned int lane = threadIdx.x % warpSize; // index of current thread in current warp

  unsigned int const warpsPerBlock = blockDim.x / warpSize;
  extern __shared__ unsigned int sLowerUpper [];
  unsigned int * sLower = sLowerUpper, * sUpper = &sLowerUpper[warpsPerBlock];

  size_t s = 0;
  unsigned int cntLower = 0, cntUpper = 0;
  if (n > stride) {
    for (; s < n - stride; s += stride) {
      float v = values[s + tid];
      cntLower += (v < pivot);
      cntUpper += (v > pivot);
    }
  }
  if (tid < n - s) {
    float v = values[s + tid];
    cntLower += v < pivot;
    cntUpper += v > pivot;
  }
  __syncwarp();

  unsigned int sumLower = cntLower, sumUpper = cntUpper;
  for (auto w = warpSize / 2; w != 0; w /= 2) {
    sumLower += __shfl_down_sync(0xffffffff, sumLower, w);
    sumUpper += __shfl_down_sync(0xffffffff, sumUpper, w);
  }

  if (lane == 0) {
    sLower[wid] = sumLower;
    sUpper[wid] = sumUpper;
  }
  __syncthreads();

  d_sumShared(sLower, sUpper, warpsPerBlock);

  if (threadIdx.x == 0) {
    lower[blockIdx.x] = sLower[0];
    upper[blockIdx.x] = sUpper[0];
  }
}


float * getIthFrom3(float * d_values, std::size_t n, std::size_t i) {
  float ary[4];
  CUDA_CHECK(cudaMemcpy(&ary, d_values, n * sizeof(float), cudaMemcpyDeviceToHost));
  if (n == 1) {
    return (float *)d_values + 0;
  } else if (n == 2) {
    if (ary[0] > ary[1]) return (float *)d_values + 1u - i; else return (float *)d_values + i;
  }
  float nary[3] = { ary[0], ary[1], ary[2] };
  sort3(nary[0], nary[1], nary[2]);
  for (std::size_t j = 0; j < 3; ++j) if (nary[i] == ary[j]) return (float *)d_values + j;

  // unreachable
  exit (1);
}

int run = 0;

float * selectIth(float * d_values, std::size_t n, std::size_t i) {
  std::cerr << "selectIth called with n = " << n << ", i = " << i << "\n";
  if (n <= 3) {
    return getIthFrom3(d_values, n, i);
  }
  
  float * d_pivotPtr = medianOfMediansOfMedians(d_values, n);
  float * h_v; h_v = (float *)malloc(n * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_v, d_values, n * sizeof(float), cudaMemcpyDeviceToHost));
  float * h_pivotPtr = h_v + (d_pivotPtr - (float * )d_values); // ! change if d_values does not point to first element
  float pivot = *h_pivotPtr;
  std::cerr << "Pivot is " << pivot << "\n";

  for (std::size_t j = 0; j < n; ++j) std::cerr << h_v[j] << ", ";
  std::cerr << "*** \n";

  dim3 dimBlock(256);
  dim3 dimGrid(1024);

  std::size_t * d_lower, * d_upper;
  CUDA_CHECK(cudaMalloc(&d_lower, 1024 * sizeof(std::size_t)));
  CUDA_CHECK(cudaMalloc(&d_upper, 1024 * sizeof(std::size_t)));
  kernel_countLowerUpper<<<dimGrid, dimBlock, 2 * dimBlock.x / minWarpSize * sizeof(unsigned int)>>> (
    d_lower, d_upper, 1024, d_values, n, pivot
  );
  CUDA_CHECK(cudaGetLastError());
  std::size_t lower[1024], upper[1024];
  CUDA_CHECK(cudaMemcpy(lower, d_lower, 1024 * sizeof(std::size_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(upper, d_upper, 1024 * sizeof(std::size_t), cudaMemcpyDeviceToHost));
  std::cerr << "n = " << n << "\n";
  std::cerr << "Lower: "; for (auto v : lower) std::cerr << v << " "; std::cerr << "\n";
  std::cerr << "Upper: "; for (auto v : upper) std::cerr << v << " "; std::cerr << "\n";

  float * p = partitionPivotCpu(h_v, h_v + n, h_pivotPtr);
  CUDA_CHECK(cudaMemcpy(d_values, h_v, n * sizeof(float), cudaMemcpyHostToDevice));
  for (std::size_t j = 0; j < n; ++j) std::cerr << h_v[j] << ", ";
  std::cerr << "*** \n";

  std::size_t dist = p - h_v;
  std::cerr << "dist is " << dist << "\n";
  free(h_v);

  ++run;
  if (run == 100) exit(1);

  if (dist == i) {
    std::cerr << "(Found) Returning form selectIth: " << *p << "\n";
    return (float *)d_values + (p - h_v);
  } else if (dist > i) {
    std::size_t nKeep = dist;
    float * retPtr = selectIth(d_values, nKeep, i);
    float ret; CUDA_CHECK(cudaMemcpy(&ret, retPtr, sizeof(float), cudaMemcpyDeviceToHost));
    std::cerr << "Returning from selectIth: " << ret << "\n";
    return retPtr;
  } else {
    std::size_t nCutoff = dist + 1;
    if (nCutoff == 0) std::cerr << "INFINITE\n";
    float * retPtr = selectIth(d_values + nCutoff, n - nCutoff, i - nCutoff);
    float ret; CUDA_CHECK(cudaMemcpy(&ret, retPtr, sizeof(float), cudaMemcpyDeviceToHost));
    std::cerr << "Returning from selectIth: " << ret << "\n";
    return retPtr;
  }
}

float * partitionPivotCpu(float * s, float * e, float * pivotPtr) {
  e--;
  float * first = s; float * last = e;
  auto print = [&] () { for (auto p = first; p <= last; ++p) std::cerr << *p << " "; std::cerr << "\n"; };
  if (s == e) return s;

  float pivot = *pivotPtr;
  std::swap(*e, *pivotPtr);
  for (;;) {
    while (*s < pivot) ++s;
    if (s == e) break;
    std::swap(*s, *e); --e;
//    print();
    while (*e > pivot) --e;
    if (s == e) break;
    std::swap(*s, *e); ++s;
//    print();
  }
  return s;
}

template<typename T, std::size_t N>
static constexpr std::size_t getArraySize(T const (&) [N]) { return N; }

constexpr float sampleValues [] = {
0.675834, 0.549336, 0.117913, 0.199457, 0.218484, 0.198028, 0.143086, 0.818313,
0.198349, 0.492627, 0.133490, 0.122743, 0.232604, 0.013443, 0.428109, 0.375118,
0.724812, 0.637781, 0.426391, 0.832380, 0.601017, 0.023298, 0.859749, 0.798638,
0.983597, 0.890266, 0.918260, 0.975585, 0.610086, 0.776527, 0.331398, 0.101837,
0.370902, 0.712251, 0.598831, 0.028339, 0.794379, 0.425321, 0.163824, 0.058157,
0.358373, 0.298880, 0.619596, 0.720614, 0.996359, 0.822886, 0.300554, 0.723165,
0.569816, 0.808561, 0.390497, 0.734800, 0.384159, 0.140598, 0.515063, 0.901393,
0.747513, 0.056342, 0.664934, 0.359430, 0.955378, 0.298972, 0.692769, 0.483582,
0.925836, 0.192024, 0.464402, 0.631153, 0.812209, 0.896107, 0.195236, 0.725631,
0.803880, 0.326598, 0.763791, 0.444875, 0.121289, 0.194140, 0.168428, 0.783862,
0.007181, 0.047408, 0.890974, 0.325302, 0.331915, 0.673811, 0.571273, 0.597470,
0.036296, 0.806209, 0.008495, 0.061873, 0.761491, 0.085287, 0.395649, 0.958226,
0.392621, 0.743642, 0.603994, 0.995084, 0.195719, 0.521120, 0.936485, 0.456679,
0.626389, 0.149245, 0.665818, 0.781374, 0.053967, 0.909324, 0.028443, 0.489854,
0.261744, 0.664378, 0.776766, 0.022560, 0.869313, 0.559557, 0.424194, 0.223073,
0.775769, 0.411491, 0.699171, 0.063350, 0.998867, 0.048791, 0.359360, 0.798343,
0.320393, 0.952632, 0.982184, 0.494968, 0.741818, 0.852662, 0.430068, 0.710986,
0.064803, 0.814804, 0.178822, 0.322928, 0.458980, 0.114546, 0.507490, 0.823788,
0.583049, 0.790189, 0.588072, 0.184255, 0.612023, 0.329946, 0.863602, 0.927959,
0.222278, 0.247764, 0.814857, 0.268253, 0.525924, 0.235827, 0.962320, 0.787109,
0.127989, 0.224804, 0.304827, 0.312686, 0.397022, 0.838965, 0.136794, 0.741918,
0.883626, 0.448015, 0.130004, 0.820826, 0.931453, 0.103550, 0.070695, 0.402869,
0.090209, 0.984367, 0.925419, 0.806995, 0.385887, 0.425618, 0.874796, 0.943171,
0.636133, 0.151433, 0.030800, 0.034716, 0.732102, 0.454077, 0.361492, 0.309187,
0.778286, 0.727980, 0.750961, 0.733402, 0.659885, 0.076588, 0.339051, 0.872630,
0.013011, 0.744088, 0.990327, 0.856116, 0.229733, 0.838359, 0.133252, 0.505734,
0.461156, 0.014607, 0.713123, 0.101131, 0.815774, 0.812193, 0.609109, 0.328966,
0.026706, 0.018911, 0.888338, 0.027383, 0.538421, 0.417653, 0.280877, 0.930387,
0.657864, 0.684084, 0.552113, 0.454167, 0.673921, 0.581650, 0.204158, 0.017140,
0.580237, 0.652563, 0.028971, 0.747107, 0.204912, 0.014714, 0.872576, 0.840558,
0.358698, 0.916484, 0.592655, 0.159648, 0.329524, 0.183819, 0.627151, 0.451673,
0.206848, 0.193335, 0.440723, 0.836886, 0.584645, 0.952260, 0.827961, 0.747109,
0.080843, 0.731218, 0.852039, 0.747834, 0.739893, 0.944133, 0.294375, 0.431737,
0.568741, 0.754577, 0.367543, 0.287646, 0.095014, 0.595658, 0.107205, 0.793331,
0.062235, 0.031338, 0.975078, 0.116824, 0.576118, 0.580166, 0.547439, 0.867732,
0.033364, 0.925223, 0.801806, 0.330924, 0.792967, 0.742125, 0.454915, 0.873680,
0.781746, 0.292970, 0.493846, 0.207419, 0.965891, 0.069967, 0.272775, 0.852371,
0.777266, 0.274234, 0.519650, 0.659049, 0.225442, 0.082609, 0.186662, 0.684952,
0.387195, 0.679624, 0.660365, 0.749423, 0.097154, 0.225847, 0.793646, 0.667740,
0.504864, 0.545093, 0.012272, 0.210322, 0.906634, 0.169231, 0.995141, 0.602502,
0.536461, 0.056370, 0.325324, 0.805963, 0.880943, 0.100872, 0.409886, 0.200339,
0.710288, 0.833275, 0.845822, 0.595087, 0.739507, 0.218244, 0.008991, 0.548410,
0.095410, 0.132802, 0.223820, 0.235447, 0.339176, 0.149260, 0.604623, 0.727115,
0.244795, 0.820262, 0.735359, 0.018791, 0.072450, 0.809892, 0.491245, 0.893773,
0.177206, 0.267184, 0.676463, 0.608988, 0.371686, 0.428915, 0.208652, 0.746067,
0.226346, 0.993975, 0.779497, 0.391627, 0.940715, 0.391563, 0.140278, 0.909621,
0.431872, 0.016967, 0.086550, 0.973725, 0.307398, 0.005699, 0.832363, 0.291479,
0.835705, 0.772405, 0.090146, 0.566522, 0.397263, 0.921931, 0.770182, 0.243065,
0.893110, 0.933350, 0.245441, 0.310287, 0.621855, 0.536506, 0.629545, 0.008513,
0.320534, 0.541774, 0.181728, 0.660944, 0.724008, 0.445897, 0.155659, 0.823976,
0.522359, 0.262442, 0.102513, 0.219780, 0.447130, 0.483466, 0.192454, 0.932098,
0.682715, 0.591343, 0.399711, 0.208911, 0.203785, 0.684325, 0.229270, 0.031692,
0.864980, 0.037318, 0.386648, 0.627055, 0.907276, 0.734574, 0.029484, 0.192554,
0.212914, 0.349681, 0.140165, 0.266614, 0.871874, 0.941627, 0.074428, 0.564524,
0.102964, 0.745377, 0.889911, 0.789834, 0.463820, 0.780056, 0.267427, 0.852415,
0.064637, 0.988559, 0.580031, 0.359711, 0.184863, 0.370025, 0.257269, 0.777953,
0.503660, 0.797852, 0.685104, 0.293735, 0.666788, 0.621726, 0.691618, 0.063004,
0.309374, 0.515190, 0.078322, 0.560924, 0.571764, 0.597214, 0.752675, 0.148665,
0.185674, 0.029774, 0.092324, 0.714238, 0.959843, 0.878790, 0.089200, 0.216141,
0.561787, 0.396428, 0.279405, 0.389167, 0.943779, 0.744307, 0.057579, 0.391817,
0.038226, 0.321126, 0.963795, 0.984961, 0.253851, 0.211179, 0.913049, 0.149568,
0.349530, 0.483118, 0.634906, 0.333382, 0.317112, 0.482892, 0.275249, 0.804301,
0.277888, 0.177803, 0.559041, 0.005566, 0.944822, 0.701353, 0.626090, 0.131114,
0.738870, 0.502807, 0.753902, 0.810038, 0.529650, 0.012507, 0.519157, 0.189250,
0.165146, 0.986747, 0.845759, 0.513475, 0.852805, 0.577928, 0.897308, 0.482126,
0.298644, 0.275587, 0.657554, 0.133392, 0.243726, 0.184239, 0.338887, 0.973496,
0.715598, 0.687960, 0.039990, 0.577135, 0.901371, 0.288845, 0.130549, 0.954760,
0.868623, 0.271276, 0.821097, 0.744013, 0.514059, 0.614499, 0.094034, 0.930130,
0.506216, 0.028681, 0.030850, 0.324206, 0.083047, 0.229778, 0.348511, 0.125867,
0.848953, 0.298703, 0.961198, 0.454358, 0.424634, 0.353616, 0.996141, 0.119000,
0.593659, 0.193041, 0.773914, 0.380328, 0.491622, 0.731175, 0.632572, 0.612488,
0.061556, 0.033184, 0.881137, 0.056459, 0.126991, 0.305320, 0.849893, 0.565874,
0.371409, 0.764772, 0.938547, 0.571608, 0.642416, 0.961033, 0.696503, 0.810262,
0.682390, 0.007750, 0.902190, 0.550711, 0.879209, 0.776567, 0.234995, 0.861628,
0.183800, 0.548553, 0.776419, 0.533638, 0.543027, 0.525755, 0.829732, 0.501469,
0.339093, 0.641717, 0.422243, 0.335479, 0.881652, 0.715020, 0.804564, 0.764080,
0.939031, 0.662535, 0.776164, 0.222069, 0.220521, 0.763462, 0.537565, 0.043066,
0.704899, 0.997446, 0.640734, 0.359049, 0.557618, 0.101376, 0.632251, 0.093845,
0.265259, 0.128225, 0.907981, 0.726326, 0.226179, 0.382706, 0.169549, 0.113081,
0.676245, 0.087165, 0.942397, 0.226527, 0.740003, 0.967353, 0.034627, 0.135542,
0.039401, 0.369633, 0.546783, 0.688901, 0.819275, 0.577428, 0.674310, 0.973617,
0.926587, 0.462265, 0.645377, 0.203273, 0.175917, 0.937567, 0.381518, 0.598611,
0.250317, 0.559064, 0.504121, 0.330231, 0.453577, 0.932978, 0.229469, 0.751867,
0.737397, 0.645380, 0.986442, 0.832977, 0.405429, 0.616709, 0.826487, 0.181464,
0.476906, 0.864554, 0.424548, 0.523713, 0.176208, 0.613356, 0.234399, 0.789067,
0.959937, 0.639073, 0.408946, 0.343559, 0.576682, 0.091308, 0.167088, 0.372392,
0.570411, 0.296421, 0.747625, 0.598355, 0.735191, 0.984562, 0.060003, 0.191701,
0.966164, 0.197539, 0.008463, 0.453339, 0.153882, 0.931808, 0.808030, 0.061260,
0.604143, 0.704405, 0.202226, 0.064371, 0.672833, 0.736969, 0.272345, 0.300823,
0.563694, 0.129056, 0.245387, 0.669855, 0.988030, 0.144376, 0.306906, 0.404191,
0.335900, 0.536154, 0.640264, 0.736509, 0.357475, 0.635144, 0.097182, 0.001698,
0.364304, 0.654063, 0.900243, 0.193761, 0.201719, 0.791599, 0.837188, 0.829892,
0.712577, 0.072502, 0.911704, 0.667146, 0.123413, 0.421752, 0.844587, 0.889226,
0.844522, 0.572855, 0.031313, 0.329396, 0.706725, 0.053197, 0.177584, 0.182347,
0.507069, 0.733641, 0.347861, 0.581696, 0.175019, 0.599270, 0.505024, 0.911129,
0.658579, 0.875951, 0.631980, 0.278317, 0.434231, 0.452579, 0.326875, 0.081354,
0.304429, 0.602795, 0.599273, 0.856685, 0.807807, 0.438008, 0.398003, 0.726570,
0.898820, 0.517199, 0.668174, 0.975965, 0.333368, 0.073856, 0.705018, 0.206023,
0.554651, 0.439796, 0.691447, 0.132373, 0.566419, 0.679150, 0.464049, 0.852623,
0.580632, 0.586638, 0.183705, 0.959494, 0.807947, 0.666674, 0.247631, 0.674166,
0.058020, 0.071117, 0.230055, 0.366314, 0.125465, 0.363707, 0.020233, 0.235138,
0.642533, 0.618545, 0.575181, 0.101691, 0.537095, 0.456339, 0.286598, 0.487397,
0.026723, 0.392857, 0.197557, 0.366466, 0.527168, 0.313683, 0.068902, 0.144360,
0.951261, 0.330413, 0.635755, 0.409343, 0.643172, 0.321604, 0.357847, 0.836627,
0.566560, 0.788141, 0.455982, 0.156647, 0.346171, 0.132613, 0.797805, 0.066427,
0.382099, 0.157725, 0.668998, 0.590330, 0.388924, 0.052234, 0.358166, 0.739088,
0.410299, 0.558684, 0.678090, 0.696773, 0.473296, 0.304695, 0.224080, 0.878341,
0.360684, 0.643392, 0.028705, 0.986385, 0.899773, 0.549273, 0.859344, 0.831581,
0.218074, 0.416982, 0.795790, 0.010415, 0.756070, 0.112029, 0.555637, 0.787556,
0.081141, 0.317355, 0.828787, 0.046786, 0.098019, 0.886574, 0.935687, 0.500416,
0.988369, 0.338011, 0.228183, 0.016234, 0.843070, 0.689996, 0.907904, 0.200107,
0.235667, 0.059057, 0.517476, 0.762641, 0.387168, 0.735445, 0.206345, 0.695419,
0.093867, 0.246391, 0.654813, 0.870441, 0.869966, 0.824214, 0.232938, 0.257902,
0.702240, 0.305196, 0.787831, 0.668085, 0.801196, 0.122303, 0.677847, 0.262451,
0.752800, 0.017623, 0.243852, 0.071537, 0.379267, 0.534408, 0.806673, 0.319076,
0.653233, 0.877977, 0.815592, 0.469533, 0.760626, 0.663740, 0.775173, 0.391523,
0.139270, 0.939264, 0.147140, 0.482519, 0.635429, 0.461114, 0.740003, 0.371827,
0.404734, 0.707303, 0.665431, 0.980577, 0.299473, 0.353059, 0.082137, 0.414109,
0.340718, 0.525466, 0.681245, 0.762847, 0.349161, 0.082941, 0.944869, 0.409176,
0.169454, 0.597323, 0.924708, 0.238077, 0.944383, 0.583856, 0.679795, 0.106902,
0.692108, 0.529788, 0.228007, 0.197957, 0.113403, 0.714758, 0.244032, 0.972826,
0.153405, 0.886128, 0.496939, 0.275516, 0.214367, 0.555596, 0.496907, 0.605506,
0.238278, 0.550856, 0.915048, 0.542430, 0.719142, 0.645722, 0.221339, 0.125509,
0.037983, 0.954756, 0.062088, 0.513005, 0.817089, 0.251245, 0.410079, 0.692597,
0.602779, 0.674663, 0.405324, 0.498646, 0.546780, 0.350287, 0.288101, 0.094799,
0.326277, 0.964009, 0.852091, 0.700756, 0.882169, 0.656312, 0.561601, 0.431433,
0.122655, 0.685377, 0.918001, 0.431905, 0.082022, 0.277690, 0.622889, 0.133904,
0.193544, 0.768393, 0.036062, 0.255360, 0.269872, 0.603828, 0.896986, 0.014769
};

constexpr std::size_t nSampleValues = getArraySize(sampleValues);

int main() {
  float * d_values = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, nSampleValues * sizeof(float4)));
  CUDA_CHECK(cudaMemcpy(d_values, sampleValues, nSampleValues * sizeof(float), cudaMemcpyHostToDevice));
  float * d_ptr = selectIth(d_values, nSampleValues, nSampleValues / 2);
  float v;
  CUDA_CHECK(cudaMemcpy(&v, d_ptr, sizeof(float), cudaMemcpyDeviceToHost));
  std::cout << v<< '\n';
  std::cout << nSampleValues << '\n';
  cudaFree(d_values);

  float ary [nSampleValues];
  memcpy(&ary, &sampleValues, nSampleValues * sizeof(float));
  std::sort(std::begin(ary), std::end(ary));
  std::cout << ary[nSampleValues / 2] << '\n';
}