#include "lib_dispatch.hpp"
#include "build_info.hpp"
#include "distribution.hpp"
#include "tensors.hpp"
#include "tools.hpp"

// ==========================================================================
// external declarations for calling blas routines linked with -lblas
// ==========================================================================
//  NOTE: The openblas cblas interfers with the OpenMPI library put these in the
//        implimentation instead of the header to avoid this conflict.
#if defined(ASGARD_ACCELERATE) || \
    defined(__APPLE__) && defined(ASGARD_USE_SCALAPACK)
#include <Accelerate/Accelerate.h>
#else
#ifdef ASGARD_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
extern "C"
{
#ifndef ASGARD_OPENBLAS
  //  Openblas predeclares these from an include in cblas.h
  // --------------------------------------------------------------------------
  // LU decomposition of a general matrix
  // --------------------------------------------------------------------------
  void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);

  void sgetrf_(int *m, int *n, float *A, int *lda, int *ipiv, int *info);
#endif

  // --------------------------------------------------------------------------
  // inverse of a matrix given its LU decomposition
  // --------------------------------------------------------------------------
  void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work, int *lwork,
               int *info);

  void sgetri_(int *n, float *A, int *lda, int *ipiv, float *work, int *lwork,
               int *info);

  // TODO: clean this up when these are added to OpenBLAS
#ifndef dpttrf_
  void dpttrf_(int *n, double *D, double *E, int *info);
#endif
#ifndef spttrf_
  void spttrf_(int *n, float *D, float *E, int *info);
#endif
#ifndef dpttrs_
  void dpttrs_(int *n, int *nrhs, double const *D, double const *E, double *B,
               int *ldb, int *info);
#endif
#ifndef spttrs_
  void spttrs_(int *n, int *nrhs, float const *D, float const *E, float *B,
               int *ldb, int *info);
#endif

#ifndef ASGARD_OPENBLAS
  //  Openblas predeclares these from an include in cblas.h
  void dgesv_(int *n, int *nrhs, double const *A, int *lda, int const *ipiv,
              double *b, int *ldb, int *info);
  void sgesv_(int *n, int *nrhs, float const *A, int *lda, int const *ipiv,
              float *b, int *ldb, int *info);
  void dgetrs_(char *trans, int *n, int *nrhs, double const *A, int *lda,
               int const *ipiv, double *b, int *ldb, int *info);
  void sgetrs_(char *trans, int *n, int *nrhs, float const *A, int *lda,
               int const *ipiv, float *b, int *ldb, int *info);
#endif
}
#endif

#include <cmath>
#include <iostream>
#include <new>
#include <type_traits>

#include "asgard_mpi.h"

#ifdef ASGARD_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#ifdef ASGARD_USE_SCALAPACK
#include "cblacs_grid.hpp"
#include "scalapack_matrix_info.hpp"
extern "C"
{
  void psgesv_(int *n, int *nrhs, float const *a, int *ia, int *ja, int *desca,
               int const *ipiv, float *b, int *ib, int *jb, int *descb,
               int *info);
  void pdgesv_(int *n, int *nrhs, double const *a, int *ia, int *ja, int *desca,
               int const *ipiv, double *b, int *ib, int *jb, int *descb,
               int *info);

  void psgetrs_(const char *trans, int *n, int *nrhs, float const *a, int *ia,
                int *ja, int *desca, int const *ipiv, float *b, int *ib,
                int *jb, int *descb, int *info);
  void pdgetrs_(const char *trans, int *n, int *nrhs, double const *a, int *ia,
                int *ja, int *desca, int const *ipiv, double *b, int *ib,
                int *jb, int *descb, int *info);
}

#endif

namespace asgard::lib_dispatch
{
namespace
{
struct device_handler
{
  device_handler()
  {
#ifdef ASGARD_USE_CUDA
    auto success = cublasCreate(&handle);
    expect(success == CUBLAS_STATUS_SUCCESS);

    success = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    expect(success == CUBLAS_STATUS_SUCCESS);
#endif
  }

  void set_device(int const local_rank)
  {
#ifdef ASGARD_USE_CUDA
    int num_devices;
    auto success = cudaGetDeviceCount(&num_devices);

    expect(success == cudaSuccess);
    expect(local_rank >= 0);
    expect(local_rank < num_devices);

    if (handle)
    {
      auto const cublas_success = cublasDestroy(handle);
      expect(cublas_success == CUBLAS_STATUS_SUCCESS);
    }

    success = cudaSetDevice(local_rank);
    expect(success == cudaSuccess);
    auto const cublas_success = cublasCreate(&handle);
    expect(cublas_success == CUBLAS_STATUS_SUCCESS);

#else
    asgard::ignore(local_rank);
#endif
  }

#ifdef ASGARD_USE_CUDA
  cublasHandle_t const &get_handle() const { return handle; }
#endif
  ~device_handler()
  {
#ifdef ASGARD_USE_CUDA
    cublasDestroy(handle);
#endif
  }

private:
#ifdef ASGARD_USE_CUDA
  cublasHandle_t handle;
#endif
};
static device_handler device;

#ifdef ASGARD_USE_CUDA
inline cublasOperation_t cublas_trans(char trans)
{
  if (trans == 'N' || trans == 'n')
  {
    return CUBLAS_OP_N;
  }
  else
  {
    return CUBLAS_OP_T;
  }
}
#endif
} // namespace

void initialize_libraries(int const local_rank)
{
#ifdef ASGARD_USE_CUDA
  expect(local_rank >= 0);
  int num_devices;
  if (cudaGetDeviceCount(&num_devices) != cudaSuccess)
    throw std::runtime_error("cannot read the number of GPUs");
  device.set_device(local_rank % num_devices);
#else
  asgard::ignore(local_rank);
#endif
}

template<resource resrc, typename P>
void rotg(P *a, P *b, P *c, P *s)
{
  expect(a);
  expect(b);
  expect(c);
  expect(s);
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // function instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success = cublasDrotg(device.get_handle(), a, b, c, s);
      expect(success == 0);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSrotg(device.get_handle(), a, b, c, s);
      expect(success == 0);
    }
    return;
#endif
  }
  // default execution on the host for any resource
  else if constexpr (resrc == resource::host)
  {
    if constexpr (std::is_same_v<P, double>)
    {
      cblas_drotg(a, b, c, s);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      cblas_srotg(a, b, c, s);
    }
  }
}

template<resource resrc, typename P>
P nrm2(int n, P const x[], int incx)
{
  expect(x);
  expect(incx >= 0);
  expect(n >= 0);
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);

  if constexpr (resrc == resource::device)
  {
#ifdef ASGARD_USE_CUDA
    P norm;
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success = cublasDnrm2(device.get_handle(), n, x, incx, &norm);
      expect(success == 0);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSnrm2(device.get_handle(), n, x, incx, &norm);
      expect(success == 0);
    }
    return norm;
#endif
  }
  else if constexpr (resrc == resource::host)
  {
    if constexpr (std::is_same_v<P, double>)
    {
      return cblas_dnrm2(n, x, incx);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      return cblas_snrm2(n, x, incx);
    }
  }
}

template<resource resrc, typename P>
void copy(int n, P const *x, int incx, P *y, int incy)
{
  expect(x);
  expect(y);
  expect(incx >= 0);
  expect(incy >= 0);
  expect(n >= 0);

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
    // function instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success =
          cublasDcopy(device.get_handle(), n, x, incx, y, incy);
      expect(success == 0);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success =
          cublasScopy(device.get_handle(), n, x, incx, y, incy);
      expect(success == 0);
    }
    return;
#endif
  }
  else if constexpr (resrc == resource::host)
  {
    // default execution on the host for any resource
    if constexpr (std::is_same_v<P, double>)
    {
      cblas_dcopy(n, x, incx, y, incy);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      cblas_scopy(n, x, incx, y, incy);
    }
    else
    {
      for (int i = 0; i < n; ++i)
      {
        y[i * incy] = x[i * incx];
      }
    }
  }
}

template<resource resrc, typename P>
P dot(int n, P const *x, int incx, P const *y, int incy)
{
  expect(x);
  expect(y);
  expect(incx >= 0);
  expect(incy >= 0);
  expect(n >= 0);

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
    P result = 0.;
    // instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success =
          cublasDdot(device.get_handle(), n, x, incx, y, incy, &result);
      expect(success == 0);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success =
          cublasSdot(device.get_handle(), n, x, incx, y, incy, &result);
      expect(success == 0);
    }
    return result;
#endif
  }
  else if constexpr (resrc == resource::host)
  {
    // default execution on the host for any resource
    if constexpr (std::is_same_v<P, double>)
    {
      return cblas_ddot(n, x, incx, y, incy);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      return cblas_sdot(n, x, incx, y, incy);
    }
    else
    {
      P ans = 0.0;
      for (int i = 0; i < n; ++i)
      {
        ans += x[i * incx] * y[i * incy];
      }
      return ans;
    }
  }
}

template<resource resrc, typename P>
void axpy(int n, P alpha, const P *x, int incx, P *y, int incy)
{
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  expect(alpha);
  expect(x);
  expect(y);
  expect(incx >= 0);
  expect(incy >= 0);
  expect(n >= 0);

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success =
          cublasDaxpy(device.get_handle(), n, &alpha, x, incx, y, incy);
      expect(success == 0);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success =
          cublasSaxpy(device.get_handle(), n, &alpha, x, incx, y, incy);
      expect(success == 0);
    }
    return;
#endif
  }
  // default execution on the host
  else if constexpr (resrc == resource::host)
  {
    if constexpr (std::is_same_v<P, double>)
    {
      cblas_daxpy(n, alpha, x, incx, y, incy);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      cblas_saxpy(n, alpha, x, incx, y, incy);
    }
  }
}

template<typename P>
void scal(int *n, P *alpha, P *x, int *incx, resource const resrc)
{
  expect(alpha);
  expect(x);
  expect(n && *n >= 0);
  expect(incx && *incx >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDscal(device.get_handle(), *n, alpha, x, *incx);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSscal(device.get_handle(), *n, alpha, x, *incx);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    cblas_dscal(*n, *alpha, x, *incx);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    cblas_sscal(*n, *alpha, x, *incx);
  }
  else
  {
    for (int i = 0; i < *n; ++i)
    {
      x[i * (*incx)] *= *alpha;
    }
  }
}

//
// Simple helpers for non-float types
//
template<typename P>
static void
basic_gemm(P const *A, bool const trans_A, int const lda, P const *B,
           bool trans_B, int const ldb, P *C, int const ldc, int const m,
           int const k, int const n, P const alpha, P const beta)
{
  expect(m > 0);
  expect(k > 0);
  expect(n > 0);
  expect(lda > 0); // FIXME Tyler says these could be more thorough
  expect(ldb > 0);
  expect(ldc > 0);

  for (auto i = 0; i < m; ++i)
  {
    for (auto j = 0; j < n; ++j)
    {
      P result = 0.0;
      for (auto z = 0; z < k; ++z)
      {
        int const A_loc = trans_A ? i * lda + z : z * lda + i;
        int const B_loc = trans_B ? z * ldb + j : j * ldb + z;
        result += A[A_loc] * B[B_loc];
      }
      C[j * ldc + i] = C[j * ldc + i] * beta + alpha * result;
    }
  }
}

template<typename P>
static void basic_gemv(P const *A, bool const trans_A, int const lda,
                       P const *x, int const incx, P *y, int const incy,
                       int const m, int const n, P const alpha, P const beta)
{
  expect(m > 0);
  expect(n > 0);
  expect(lda > 0);
  expect(incx > 0);
  expect(incy > 0);

  for (auto i = 0; i < m; ++i)
  {
    P result = 0.0;
    for (auto j = 0; j < n; ++j)
    {
      int const A_loc = trans_A ? i * lda + j : j * lda + i;
      result += A[A_loc] * x[j * incx];
    }
    y[i * incy] = y[i * incy] * beta + alpha * result;
  }
}

//
//  Translate FORTRAN transpose blas arguments to cblas equivalents.
//
static CBLAS_TRANSPOSE cblas_transpose_type(char const *trans)
{
  if (*trans == 'n' || *trans == 'N')
  {
    return CblasNoTrans;
  }
  else if (*trans == 't' || *trans == 'T')
  {
    return CblasTrans;
  }
  else
  {
    return CblasConjTrans;
  }
}

template<typename P>
void gemv(char const *trans, int *m, int *n, P *alpha, P const *A, int *lda,
          P const *x, int *incx, P *beta, P *y, int *incy, resource const resrc)
{
  expect(alpha);
  expect(A);
  expect(x);
  expect(beta);
  expect(y);
  expect(m && *m >= 0);
  expect(n && *n >= 0);
  expect(lda && *lda >= 0);
  expect(incx && *incx >= 0);
  expect(incy && *incy >= 0);
  expect(trans && (*trans == 't' || *trans == 'n'));

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDgemv(device.get_handle(), cublas_trans(*trans), *m, *n, alpha,
                      A, *lda, x, *incx, beta, y, *incy);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSgemv(device.get_handle(), cublas_trans(*trans), *m, *n, alpha,
                      A, *lda, x, *incx, beta, y, *incy);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    cblas_dgemv(CblasColMajor, cblas_transpose_type(trans), *m, *n, *alpha, A,
                *lda, x, *incx, *beta, y, *incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    cblas_sgemv(CblasColMajor, cblas_transpose_type(trans), *m, *n, *alpha, A,
                *lda, x, *incx, *beta, y, *incy);
  }
  else
  {
    bool const trans_A = (*trans == 't') ? true : false;
    int const rows_A   = trans_A ? *n : *m;
    int const cols_A   = trans_A ? *m : *n;
    basic_gemv(A, trans_A, *lda, x, *incx, y, *incy, rows_A, cols_A, *alpha,
               *beta);
  }
}

template<typename P>
void gemm(char const *transa, char const *transb, int *m, int *n, int *k,
          P *alpha, P const *A, int *lda, P const *B, int *ldb, P *beta, P *C,
          int *ldc, resource const resrc)
{
  expect(alpha);
  expect(A);
  expect(lda && *lda >= 0);
  expect(B);
  expect(ldb && *ldb >= 0);
  expect(beta);
  expect(C);
  expect(ldc && *ldc >= 0);
  expect(m && *m >= 0);
  expect(n && *n >= 0);
  expect(k && *k >= 0);
  expect(transa && (*transa == 't' || *transa == 'n'));
  expect(transb && (*transb == 't' || *transb == 'n'));

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgemm(
          device.get_handle(), cublas_trans(*transa), cublas_trans(*transb), *m,
          *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgemm(
          device.get_handle(), cublas_trans(*transa), cublas_trans(*transb), *m,
          *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    cblas_dgemm(CblasColMajor, cblas_transpose_type(transa),
                cblas_transpose_type(transb), *m, *n, *k, *alpha, A, *lda, B,
                *ldb, *beta, C, *ldc);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    cblas_sgemm(CblasColMajor, cblas_transpose_type(transa),
                cblas_transpose_type(transb), *m, *n, *k, *alpha, A, *lda, B,
                *ldb, *beta, C, *ldc);
  }
  else
  {
    bool const trans_A = (*transa == 't') ? true : false;
    bool const trans_B = (*transb == 't') ? true : false;
    basic_gemm(A, trans_A, *lda, B, trans_B, *ldb, C, *ldc, *m, *k, *n, *alpha,
               *beta);
  }
}

template<typename P>
void getrf(int *m, int *n, P *A, int *lda, int *ipiv, int *info,
           resource const resrc)
{
  expect(A);
  expect(ipiv);
  expect(info);
  expect(lda && *lda >= 0);
  expect(m && *m >= 0);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA

    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);
    expect(*m == *n);
    ignore(m);

    P **A_d;
    if (cudaMalloc((void **)&A_d, sizeof(P *)) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    auto stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgetrfBatched(device.get_handle(), *n, A_d,
                                               *lda, ipiv, info, 1);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgetrfBatched(device.get_handle(), *n, A_d,
                                               *lda, ipiv, info, 1);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    dgetrf_(m, n, A, lda, ipiv, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgetrf_(m, n, A, lda, ipiv, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getrf not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void getri(int *n, P *A, int *lda, int *ipiv, P *work, int *lwork, int *info,
           resource const resrc)
{
  expect(A);
  expect(ipiv);
  expect(work);
  expect(lwork);
  expect(info);
  expect(lda && *lda >= 0);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA

    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    expect(*lwork == (*n) * (*n));
    ignore(lwork);

    P const **A_d;
    P **work_d;
    if (cudaMalloc((void **)&A_d, sizeof(P *)) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    if (cudaMalloc((void **)&work_d, sizeof(P *)) != cudaSuccess)
    {
      throw std::bad_alloc();
    }

    auto stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == 0);
    stat = cudaMemcpy(work_d, &work, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgetriBatched(
          device.get_handle(), *n, A_d, *lda, nullptr, work_d, *n, info, 1);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgetriBatched(
          device.get_handle(), *n, A_d, *lda, nullptr, work_d, *n, info, 1);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    dgetri_(n, A, lda, ipiv, work, lwork, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgetri_(n, A, lda, ipiv, work, lwork, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getri not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void batched_gemm(P **const &a, int *lda, char const *transa, P **const &b,
                  int *ldb, char const *transb, P **const &c, int *ldc, int *m,
                  int *n, int *k, P *alpha, P *beta, int *num_batch,
                  resource const resrc)
{
  expect(alpha);
  expect(a);
  expect(lda && *lda >= 0);
  expect(b);
  expect(ldb && *ldb >= 0);
  expect(beta);
  expect(c);
  expect(ldc && *ldc >= 0);
  expect(m && *m >= 0);
  expect(n && *n >= 0);
  expect(k && *k >= 0);
  expect(transa && (*transa == 't' || *transa == 'n'));
  expect(transb && (*transb == 't' || *transb == 'n'));
  expect(num_batch && *num_batch > 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    P const **a_d;
    P const **b_d;
    P **c_d;
    size_t const list_size = *num_batch * sizeof(P *);

    if (cudaMalloc((void **)&a_d, list_size) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    if (cudaMalloc((void **)&b_d, list_size) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    if (cudaMalloc((void **)&c_d, list_size) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    auto stat = cudaMemcpy(a_d, a, list_size, cudaMemcpyHostToDevice);
    expect(stat == cudaSuccess);
    stat = cudaMemcpy(b_d, b, list_size, cudaMemcpyHostToDevice);
    expect(stat == cudaSuccess);
    stat = cudaMemcpy(c_d, c, list_size, cudaMemcpyHostToDevice);
    expect(stat == cudaSuccess);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgemmBatched(
          device.get_handle(), cublas_trans(*transa), cublas_trans(*transb), *m,
          *n, *k, alpha, a_d, *lda, b_d, *ldb, beta, c_d, *ldc, *num_batch);
      auto const cuda_stat = cudaDeviceSynchronize();
      expect(cuda_stat == 0);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgemmBatched(
          device.get_handle(), cublas_trans(*transa), cublas_trans(*transb), *m,
          *n, *k, alpha, a_d, *lda, b_d, *ldb, beta, c_d, *ldc, *num_batch);
      auto const cuda_stat = cudaDeviceSynchronize();
      expect(cuda_stat == 0);
      expect(success == 0);
    }

    stat = cudaFree(a_d);
    expect(stat == 0);
    stat = cudaFree(b_d);
    expect(stat == 0);
    stat = cudaFree(c_d);
    expect(stat == 0);

    return;
#endif
  }

  // default execution on the host for any resource
  int const end = *num_batch;
#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < end; ++i)
  {
    gemm(transa, transb, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc,
         resource::host);
  }
}

template<typename P>
void gesv(int *n, int *nrhs, P const *A, int *lda, int const *ipiv, P *b,
          int *ldb, int *info)
{
  expect(n);
  expect(nrhs);
  expect(A);
  expect(lda);
  expect(ipiv);
  expect(info);
  expect(b);
  expect(ldb);
  expect(*ldb >= 1);
  expect(*lda >= 1);
  expect(*n >= 0);
  if constexpr (std::is_same<P, double>::value)
  {
    dgesv_(n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgesv_(n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "gesv not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void getrs(char *trans, int *n, int *nrhs, P const *A, int *lda,
           int const *ipiv, P *b, int *ldb, int *info)
{
  expect(trans);
  expect(n);
  expect(nrhs);
  expect(A);
  expect(lda);
  expect(ipiv);
  expect(info);
  expect(b);
  expect(ldb);
  expect(*ldb >= 1);
  expect(*lda >= 1);
  expect(*n >= 0);
  if constexpr (std::is_same<P, double>::value)
  {
    dgetrs_(trans, n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgetrs_(trans, n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getrs not implemented for non-floating types" << '\n';
    assert(false);
  }
}

template<typename P>
void pttrf(int *n, P *D, P *E, int *info, resource const resrc)
{
  expect(D);
  expect(E);
  expect(info);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    throw std::runtime_error("no pttrf support on cuda implemented");
  }

  if constexpr (std::is_same<P, double>::value)
  {
    dpttrf_(n, D, E, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    spttrf_(n, D, E, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "pttrf not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void pttrs(int *n, int *nrhs, P const *D, P const *E, P *B, int *ldb, int *info,
           resource const resrc)
{
  expect(n);
  expect(nrhs);
  expect(D);
  expect(E);
  expect(B);
  expect(ldb);
  expect(info);
  expect(*ldb >= 1);
  expect(*n >= 0);

  if (resrc == resource::device)
  {
    throw std::runtime_error("no pttrs support on cuda implemented");
  }

  if constexpr (std::is_same<P, double>::value)
  {
    dpttrs_(n, nrhs, D, E, B, ldb, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    spttrs_(n, nrhs, D, E, B, ldb, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "spttrs not implemented for non-floating types" << '\n';
    assert(false);
  }
}

#ifdef ASGARD_USE_SCALAPACK

template<typename P>
void scalapack_gesv(int *n, int *nrhs, P const *A, int *descA, int const *ipiv,
                    P *b, int *descB, int *info)
{
  expect(n);
  expect(nrhs);
  expect(A);
  expect(ipiv);
  expect(info);
  expect(b);
  expect(descB);
  expect(*n >= 0);

  int mp{1}, nq{1}, i_one{1};
  if constexpr (std::is_same<P, double>::value)
  {
    pdgesv_(n, nrhs, A, &mp, &nq, descA, ipiv, b, &i_one, &nq, descB, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    psgesv_(n, nrhs, A, &mp, &nq, descA, ipiv, b, &i_one, &nq, descB, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "gesv not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void scalapack_getrs(char *trans, int *n, int *nrhs, P const *A, int *descA,
                     int const *ipiv, P *b, int *descB, int *info)
{
  expect(trans);
  expect(n);
  expect(nrhs);
  expect(A);
  expect(ipiv);
  expect(info);
  expect(b);
  expect(*n >= 0);

  int mp{1}, nq{1}, i_one{1};
  char N{'N'};
  if constexpr (std::is_same<P, double>::value)
  {
    pdgetrs_(&N, n, nrhs, A, &mp, &nq, descA, ipiv, b, &i_one, &nq, descB,
             info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    psgetrs_(&N, n, nrhs, A, &mp, &nq, descA, ipiv, b, &i_one, &nq, descB,
             info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getrs not implemented for non-floating types" << '\n';
    expect(false);
  }
}
#endif

template void rotg<resource::host, float>(float *, float *, float *, float *);
template void
rotg<resource::host, double>(double *, double *, double *, double *);

template float nrm2<resource::host, float>(int, float const[], int);
template double nrm2<resource::host, double>(int, double const[], int);

template void copy<resource::host, float>(int n, float const *x, int incx,
                                          float *y, int incy);
template void copy<resource::host, double>(int n, double const *x, int incx,
                                           double *y, int incy);
template void
copy<resource::host, int>(int n, int const *x, int incx, int *y, int incy);

template float dot<resource::host, float>(int n, float const *x, int incx,
                                          float const *y, int incy);
template double dot<resource::host, double>(int n, double const *x, int incx,
                                            double const *y, int incy);
template int
dot<resource::host, int>(int n, int const *x, int incx, int const *y, int incy);

#ifdef ASGARD_USE_CUDA
template float nrm2<resource::device, float>(int, float const[], int);
template double nrm2<resource::device, double>(int, double const[], int);

template void rotg<resource::device, float>(float *, float *, float *, float *);
template void
rotg<resource::device, double>(double *, double *, double *, double *);

template void copy<resource::device, float>(int n, float const *x, int incx,
                                            float *y, int incy);
template void copy<resource::device, double>(int n, double const *x, int incx,
                                             double *y, int incyc);
template float dot<resource::device, float>(int n, float const *x, int incx,
                                            float const *y, int incy);
template double dot<resource::device, double>(int n, double const *x, int incx,
                                              double const *y, int incy);
template void axpy<resource::device, float>(int n, float alpha, float const *x,
                                            int incx, float *y, int incy);
template void axpy<resource::device, double>(int n, double alpha,
                                             double const *x, int incx,
                                             double *y, int incy);
#endif

template void axpy<resource::host, float>(int n, float alpha, float const *x,
                                          int incx, float *y, int incy);
template void axpy<resource::host, double>(int n, double alpha, double const *x,
                                           int incx, double *y, int incy);

template void
scal(int *n, float *alpha, float *x, int *incx, resource const resrc);
template void
scal(int *n, double *alpha, double *x, int *incx, resource const resrc);
template void scal(int *n, int *alpha, int *x, int *incx, resource const resrc);

template void gemv(char const *trans, int *m, int *n, float *alpha,
                   float const *A, int *lda, float const *x, int *incx,
                   float *beta, float *y, int *incy, resource const resrc);
template void gemv(char const *trans, int *m, int *n, double *alpha,
                   double const *A, int *lda, double const *x, int *incx,
                   double *beta, double *y, int *incy, resource const resrc);
template void gemv(char const *trans, int *m, int *n, int *alpha, int const *A,
                   int *lda, int const *x, int *incx, int *beta, int *y,
                   int *incy, resource const resrc);

template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, float *alpha, float const *A, int *lda,
                   float const *B, int *ldb, float *beta, float *C, int *ldc,
                   resource const resrc);
template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, double *alpha, double const *A, int *lda,
                   double const *B, int *ldb, double *beta, double *C, int *ldc,
                   resource const resrc);
template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, int *alpha, int const *A, int *lda, int const *B,
                   int *ldb, int *beta, int *C, int *ldc, resource const resrc);

template void getrf(int *m, int *n, float *A, int *lda, int *ipiv, int *info,
                    resource const resrc);
template void getrf(int *m, int *n, double *A, int *lda, int *ipiv, int *info,
                    resource const resrc);

template void getri(int *n, float *A, int *lda, int *ipiv, float *work,
                    int *lwork, int *info, resource const resrc);
template void getri(int *n, double *A, int *lda, int *ipiv, double *work,
                    int *lwork, int *info, resource const resrc);

template void batched_gemm(float **const &a, int *lda, char const *transa,
                           float **const &b, int *ldb, char const *transb,
                           float **const &c, int *ldc, int *m, int *n, int *k,
                           float *alpha, float *beta, int *num_batch,
                           resource const resrc);

template void batched_gemm(double **const &a, int *lda, char const *transa,
                           double **const &b, int *ldb, char const *transb,
                           double **const &c, int *ldc, int *m, int *n, int *k,
                           double *alpha, double *beta, int *num_batch,
                           resource const resrc);

template void gesv(int *n, int *nrhs, double const *A, int *lda,
                   int const *ipiv, double *b, int *ldb, int *info);
template void gesv(int *n, int *nrhs, float const *A, int *lda, int const *ipiv,
                   float *b, int *ldb, int *info);

template void getrs(char *trans, int *n, int *nrhs, double const *A, int *lda,
                    int const *ipiv, double *b, int *ldb, int *info);
template void getrs(char *trans, int *n, int *nrhs, float const *A, int *lda,
                    int const *ipiv, float *b, int *ldb, int *info);

template void
pttrf(int *n, double *D, double *E, int *info, resource const resrc);
template void
pttrf(int *n, float *D, float *E, int *info, resource const resrc);

template void pttrs(int *n, int *nrhs, double const *D, double const *E,
                    double *B, int *ldb, int *info, resource const resrc);
template void pttrs(int *n, int *nrhs, float const *D, float const *E, float *B,
                    int *ldb, int *info, resource const resrc);
#ifdef ASGARD_USE_SCALAPACK
template void scalapack_gesv(int *n, int *nrhs, double const *A, int *descA,
                             int const *ipiv, double *b, int *descB, int *info);
template void scalapack_gesv(int *n, int *nrhs, float const *A, int *descA,
                             int const *ipiv, float *b, int *descB, int *info);

template void scalapack_getrs(char *trans, int *n, int *nrhs, double *A,
                              int *descA, int *ipiv, double *b, int *descB,
                              int *info);
template void scalapack_getrs(char *trans, int *n, int *nrhs, float *A,
                              int *descA, int *ipiv, float *b, int *descB,
                              int *info);
#endif
} // namespace asgard::lib_dispatch
