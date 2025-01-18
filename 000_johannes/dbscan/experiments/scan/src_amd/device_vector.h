#ifndef DEVICE_VECTOR_H_
#define DEVICE_VECTOR_H_

#include "cuda_helpers.h"
#include <array>
#include <vector>

#include <iostream>

struct UninitializedDeviceVectorTag {};

template <typename T>
struct DeviceVector {
  DeviceVector(UninitializedDeviceVectorTag, std::size_t n)
  : n(n), d(nullptr) {
    if (n > 0) CUDA_CHECK(hipMalloc(&d, n * sizeof(T)));
  }

  DeviceVector(std::size_t n)
  : DeviceVector(UninitializedDeviceVectorTag {}, n) {
    if (n > 0) CUDA_CHECK(hipMemset(d, 0, n * sizeof(T)));
  }

  DeviceVector(std::vector<T> const & vec)
  : DeviceVector(UninitializedDeviceVectorTag {}, vec.size()) {
    if (vec.size() == 0) return;
    CUDA_CHECK(hipMemcpy(d, vec.data(), vec.size() * sizeof(T), hipMemcpyHostToDevice));
  }

  template <std::size_t N>
  DeviceVector(std::array<T, N> const & ary)
  : DeviceVector<T>(UninitializedDeviceVectorTag {}, N) {
    CUDA_CHECK(hipMemcpy(d, ary.data(), N * sizeof(T), hipMemcpyHostToDevice));
  }

  DeviceVector(T const * ptr, std::size_t n)
  : DeviceVector (UninitializedDeviceVectorTag {}, n) {
    if (n == 0) return;
    CUDA_CHECK(hipMemcpy(d, ptr, n * sizeof(T), hipMemcpyHostToDevice));
  }

  DeviceVector(DeviceVector const &) = delete;

  ~DeviceVector() {
    hipFree(d);
    d = nullptr;
  }

  std::size_t size() const { return n; }

  T * data () { return d; }
  T const * data () const { return d; }

//  T & operator [] (std::size_t idx) & { return d[idx]; }
//  T const & operator [] (std::size_t idx) const & { return d[idx]; }

  void memcpyToHost (T * h) const {
    CUDA_CHECK(hipMemcpy(h, d, n * sizeof(T), hipMemcpyDeviceToHost));
  }

private:
  std::size_t n;
  T * d;
};

#endif
