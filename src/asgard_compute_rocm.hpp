#pragma once

#include "asgard_compute_cpu.hpp"

namespace asgard::gpu
{

using direct_int = rocblas_int;

//! converts ROCM error to a human readable string
std::string error_message(hipError_t err);
std::string error_message(rocblas_status err);

#define rocm_check_error(_call_) \
  { hipError_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != hipSuccess) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

#define rocblas_check_error(_call_) \
  { rocblas_status __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != rocblas_status_success) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

inline void set_device(int id) { rocm_check_error( hipSetDevice(id) ); }

inline void memfree(void *p) { (void) hipFree(p); }

template<typename T>
inline T* memalloc(int64_t num) {
  T *data = nullptr;
  rocm_check_error( hipMalloc((void**)&data, num * sizeof(T)) );
  return data;
}

template<typename T>
inline void memcopy_dev2dev(int64_t num, T const *src, T *dest) {
  rocm_check_error( hipMemcpy(dest, src, num * sizeof(T), hipMemcpyDeviceToDevice) );
}

template<typename T>
inline void memcopy_host2dev(int64_t num, T const *src, T *dest) {
  rocm_check_error( hipMemcpy(dest, src, num * sizeof(T), hipMemcpyHostToDevice) );
}

template<typename T>
inline void memcopy_dev2host(int64_t num, T const *src, T *dest) {
  rocm_check_error( hipMemcpy(dest, src, num * sizeof(T), hipMemcpyDeviceToHost) );
}

template<typename T>
void mcopy(int64_t num_entries, device src_dev, T const src[], device dest_dev, T dest[]) {
  rocm_check_error( hipSetDevice(src_dev.id) );
  rocm_check_error( hipMemcpy(dest, src, num_entries * sizeof(T), hipMemcpyDeviceToDevice) );
  rocm_check_error( hipSetDevice(dest_dev.id) );
}

template<typename T>
void copy_to_host(int64_t num_entries, T const x[], std::vector<T> &y) {
  y.resize(num_entries);
  rocm_check_error( hipMemcpy(y.data(), x, num_entries * sizeof(T), hipMemcpyDeviceToHost) );
}

template<typename T>
void copy_to_device(std::vector<T> const &x, T y[]) {
  size_t const num_entries = x.size();
  rocm_check_error( hipMemcpy(y, x.data(), num_entries * sizeof(T), hipMemcpyHostToDevice) );
}

inline void device_synchronize() { rocm_check_error( hipDeviceSynchronize() ); }

template<typename P>
void fill_zeros(int64_t num, P x[]) { rocm_check_error( hipMemset(x, 0, num * sizeof(P)) ); }

class blas_engine
{
public:
  blas_engine() = default;

  ~blas_engine() {
    if (rocblas != nullptr)
      rocblas_destroy_handle(rocblas);
    if (fone != nullptr) memfree(fone);
    if (done != nullptr) memfree(done);
    if (ftmp != nullptr) memfree(ftmp);
    if (dtmp != nullptr) memfree(dtmp);
  }

  void init() {
    expect(rocblas == nullptr);
    rocblas_check_error( rocblas_create_handle(&rocblas) );

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
      rocblas_check_error( rocblas_saxpy(rocblas, num, &alpha, x, 1, y, 1) );
    } else {
      rocblas_check_error( rocblas_daxpy(rocblas, num, &alpha, x, 1, y, 1) );
    }
  }

  template<typename P>
  void axpy(int num, P const x[], P y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    rocblas_check_error( rocblas_set_pointer_mode(rocblas, rocblas_pointer_mode_device) );
    if constexpr (is_float<P>) {
      rocblas_check_error( rocblas_saxpy(rocblas, num, fone, x, 1, y, 1) );
    } else {
      rocblas_check_error( rocblas_daxpy(rocblas, num, done, x, 1, y, 1) );
    }
    rocblas_check_error( rocblas_set_pointer_mode(rocblas, rocblas_pointer_mode_host) );
  }

  template<typename P>
  void scal(int num, no_deduce<P> alpha, P x[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "scal can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      rocblas_check_error( rocblas_sscal(rocblas, num, &alpha, x, 1) );
    } else {
      rocblas_check_error( rocblas_dscal(rocblas, num, &alpha, x, 1) );
    }
  }

  template<typename P>
  P dot(int num, P const x[], P const y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "dot can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      rocblas_check_error( rocblas_sdot(rocblas, num, x, 1, y, 1, ftmp) );
      P res = 0;
      memcopy_dev2host(1, ftmp, &res);
      return res;
    } else {
      rocblas_check_error( rocblas_ddot(rocblas, num, x, 1, y, 1, dtmp) );
      P res = 0;
      memcopy_dev2host(1, dtmp, &res);
      return res;
    }
  }

  template<typename P>
  P nrm2(int num, P const x[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "dot can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      P res = 0;
      rocblas_check_error( rocblas_snrm2(rocblas, num, x, 1, &res) );
      return res;
    } else {
      P res = 0;
      rocblas_check_error( rocblas_dnrm2(rocblas, num, x, 1, &res) );
      return res;
    }
  }

  template<typename P>
  void gemtv(int m, int n, P alpha, P const A[], P const x[], P beta, P y[]) const {
    if constexpr (is_float<P>) {
      rocblas_check_error( rocblas_sgemv(rocblas, rocblas_operation_transpose,
                                         m, n, &alpha, A, m, x, 1, &beta, y, 1) );
    } else {
      rocblas_check_error( rocblas_dgemv(rocblas, rocblas_operation_transpose,
                                         m, n, &alpha, A, m, x, 1, &beta, y, 1) );
    }
  }

  template<typename P>
  void gemv(int m, int n, P alpha, P const A[], P const x[], P beta, P y[]) const {
    if constexpr (is_float<P>) {
      rocblas_check_error( rocblas_sgemv(rocblas, rocblas_operation_none,
                                         m, n, &alpha, A, m, x, 1, &beta, y, 1) );
    } else {
      rocblas_check_error( rocblas_dgemv(rocblas, rocblas_operation_none,
                                         m, n, &alpha, A, m, x, 1, &beta, y, 1) );
    }
  }

  operator rocblas_handle () const { return rocblas; }

private:
  rocblas_handle rocblas = nullptr;

  float *fone  = nullptr;
  double *done = nullptr;

  float *ftmp  = nullptr;
  double *dtmp = nullptr;
};

}
