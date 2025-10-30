#pragma once

#include "asgard_compute_cpu.hpp"

namespace asgard::gpu
{

//! cuSolver uses 32-bit int
using direct_int = int;

//! converts CUDA error to a human readable string
std::string error_message(cudaError_t err);
std::string error_message(cublasStatus_t err);
std::string error_message(cusolverStatus_t err);

#define cuda_check_error(_call_) \
  { cudaError_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != cudaSuccess) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

#define cublas_check_error(_call_) \
  { cublasStatus_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != CUBLAS_STATUS_SUCCESS) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

#define cusolver_check_error(_call_) \
  { cusolverStatus_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != CUSOLVER_STATUS_SUCCESS) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

inline void set_device(int id) { cuda_check_error( cudaSetDevice(id) ); }

inline void memfree(void *p) { cudaFree(p); }

template<typename T>
inline T* memalloc(int64_t num) {
  T *data = nullptr;
  cuda_check_error( cudaMalloc((void**)&data, num * sizeof(T)) );
  return data;
}

template<typename T>
inline void memcopy_dev2dev(int64_t num, T const *src, T *dest) {
  cuda_check_error( cudaMemcpy(dest, src, num * sizeof(T), cudaMemcpyDeviceToDevice) );
}

template<typename T>
inline void memcopy_host2dev(int64_t num, T const *src, T *dest) {
  cuda_check_error( cudaMemcpy(dest, src, num * sizeof(T), cudaMemcpyHostToDevice) );
}

template<typename T>
inline void memcopy_dev2host(int64_t num, T const *src, T *dest) {
  cuda_check_error( cudaMemcpy(dest, src, num * sizeof(T), cudaMemcpyDeviceToHost) );
}

//! \brief Transfer data between devices, assumes that compute->set_device(dest_dev)
template<typename T>
void mcopy(int64_t num_entries, device src_dev, T const src[], device dest_dev, T dest[]) {
  cuda_check_error( cudaMemcpyPeer(dest, dest_dev.id, src, src_dev.id,
                                   num_entries * sizeof(T)) );
}
//! \brief Copy an array to the CPU vector
template<typename T>
void copy_to_host(int64_t num_entries, T const x[], std::vector<T> &y) {
  y.resize(num_entries);
  cuda_check_error( cudaMemcpy(y.data(), x, num_entries * sizeof(T), cudaMemcpyDeviceToHost) );
}
//! \brief Copy an CPU vector to a device array
template<typename T>
void copy_to_device(std::vector<T> const &x, T y[]) {
  size_t const num_entries = x.size();
  cuda_check_error( cudaMemcpy(y, x.data(), num_entries * sizeof(T), cudaMemcpyHostToDevice) );
}

inline void device_synchronize() { cudaDeviceSynchronize();  }

template<typename P>
void fill_zeros(int64_t num, P x[]) { cuda_check_error( cudaMemset(x, 0, num * sizeof(P)) ); }

class blas_engine
{
public:
  blas_engine() = default;

  ~blas_engine() {
    if (cublas != nullptr)
      cublasDestroy(cublas);
    if (cusolverdn != nullptr)
      cusolverDnDestroy(cusolverdn);
    if (fone != nullptr) memfree(fone);
    if (done != nullptr) memfree(done);
    if (ftmp != nullptr) memfree(ftmp);
    if (dtmp != nullptr) memfree(dtmp);
  }

  void init() {
    expect(cublas == nullptr);
    expect(cusolverdn == nullptr);
    cublas_check_error( cublasCreate(&cublas) );
    cusolver_check_error( cusolverDnCreate(&cusolverdn) );

    fone = memalloc<float>(1);
    float cpu_fone = 1.0f;
    memcopy_host2dev(1, &cpu_fone, fone);

    done = memalloc<double>(1);
    double cpu_done = 1.0;
    memcopy_host2dev(1, &cpu_done, done);

    ftmp = memalloc<float>(1);
    dtmp = memalloc<double>(1);
  }

  template<typename P>
  void axpy(int num, no_deduce<P> alpha, P const x[], P y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      cublas_check_error( cublasSaxpy(cublas, num, &alpha, x, 1, y, 1) );
    } else {
      cublas_check_error( cublasDaxpy(cublas, num, &alpha, x, 1, y, 1) );
    }
  }
  template<typename P>
  void axpy(int num, P const x[], P y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    cublas_check_error( cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE) );
    if constexpr (is_float<P>) {
      cublas_check_error( cublasSaxpy(cublas, num, fone, x, 1, y, 1) );
    } else {
      cublas_check_error( cublasDaxpy(cublas, num, done, x, 1, y, 1) );
    }
    cublas_check_error( cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST) );
  }
  template<typename P>
  void scal(int num, no_deduce<P> alpha, P x[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      cublas_check_error( cublasSscal(cublas, num, &alpha, x, 1) );
    } else {
      cublas_check_error( cublasDscal(cublas, num, &alpha, x, 1) );
    }
  }

  template<typename P>
  P dot(int num, P const x[], P const y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "dot can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      P res = 0;
      cublas_check_error( cublasSdot(cublas, num, x, 1, y, 1, &res) );
      return res;
    } else {
      P res = 0;
      cublas_check_error( cublasDdot(cublas, num, x, 1, y, 1, &res) );
      return res;
    }
  }

  template<typename P>
  P nrm2(int num, P const x[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "dot can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      P res = 0;
      cublas_check_error( cublasSnrm2(cublas, num, x, 1, &res) );
      return res;
    } else {
      P res = 0;
      cublas_check_error( cublasDnrm2(cublas, num, x, 1, &res) );
      return res;
    }
  }

  template<typename P>
  void gemtv(int m, int n, P alpha, P const A[], P const x[], P beta, P y[]) const {
    if constexpr (is_float<P>) {
      cublas_check_error( cublasSgemv(cublas, CUBLAS_OP_T, m, n, &alpha, A, m, x, 1, &beta, y, 1) );
    } else {
      cublas_check_error( cublasDgemv(cublas, CUBLAS_OP_T, m, n, &alpha, A, m, x, 1, &beta, y, 1) );
    }
  }

  template<typename P>
  void gemv(int m, int n, P alpha, P const A[], P const x[], P beta, P y[]) const {
    if constexpr (is_float<P>) {
      cublas_check_error( cublasSgemv(cublas, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1) );
    } else {
      cublas_check_error( cublasDgemv(cublas, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1) );
    }
  }

  operator cublasHandle_t () const { return cublas; }
  operator cusolverDnHandle_t () const { return cusolverdn; }

private:
  cublasHandle_t cublas = nullptr;
  cusolverDnHandle_t cusolverdn = nullptr;

  float *fone  = nullptr;
  double *done = nullptr;

  float *ftmp  = nullptr;
  double *dtmp = nullptr;
};

}
