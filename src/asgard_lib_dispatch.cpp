#include "asgard_lib_dispatch.hpp"
#include "asgard_distribution.hpp" // seems needed for MPI
#include "asgard_sparse.hpp"

// ==========================================================================
// external declarations for calling blas routines linked with -lblas
// ==========================================================================
//  NOTE: The openblas cblas interfers with the OpenMPI library put these in the
//        implimentation instead of the header to avoid this conflict.
#if defined(ASGARD_ACCELERATE)
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
  void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv, double *b,
              int *ldb, int *info);
  void sgesv_(int *n, int *nrhs, float *A, int *lda, int *ipiv, float *b,
              int *ldb, int *info);
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

#include "asgard_mpi.hpp"

#ifdef ASGARD_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
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

    auto const cusparse_success = cusparseCreate(&sp_handle);
    expect(cusparse_success == CUSPARSE_STATUS_SUCCESS);
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

      auto const cusparse_success = cusparseDestroy(sp_handle);
      expect(cusparse_success == CUSPARSE_STATUS_SUCCESS);
    }

    success = cudaSetDevice(local_rank);
    expect(success == cudaSuccess);
    auto const cublas_success = cublasCreate(&handle);
    expect(cublas_success == CUBLAS_STATUS_SUCCESS);
    auto const cusparse_success = cusparseCreate(&sp_handle);
    expect(cusparse_success == CUSPARSE_STATUS_SUCCESS);

#else
    asgard::ignore(local_rank);
#endif
  }

#ifdef ASGARD_USE_CUDA
  cublasHandle_t const &get_handle() const { return handle; }
  cusparseHandle_t const &get_sp_handle() const { return sp_handle; }
#endif
  ~device_handler()
  {
#ifdef ASGARD_USE_CUDA
    cublasDestroy(handle);
    cusparseDestroy(sp_handle);
#endif
  }

private:
#ifdef ASGARD_USE_CUDA
  cublasHandle_t handle;
  cusparseHandle_t sp_handle;
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

inline cusparseOperation_t cusparse_trans(char trans)
{
  if (trans == 'N' || trans == 'n')
  {
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  }
  else
  {
    return CUSPARSE_OPERATION_TRANSPOSE;
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
void rot(const int n, P *x, const int incx, P *y, const int incy, const P c,
         const P s)
{
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // function instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success =
          cublasDrot(device.get_handle(), n, x, incx, y, incy, &c, &s);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success =
          cublasSrot(device.get_handle(), n, x, incx, y, incy, &c, &s);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    return;
#endif
  }
  // default execution on the host for any resource
  else if constexpr (resrc == resource::host)
  {
    if constexpr (std::is_same_v<P, double>)
    {
      cblas_drot(n, x, incx, y, incy, c, s);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      cblas_srot(n, x, incx, y, incy, c, s);
    }
  }
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
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSrotg(device.get_handle(), a, b, c, s);
      expect(success == CUBLAS_STATUS_SUCCESS);
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
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSnrm2(device.get_handle(), n, x, incx, &norm);
      expect(success == CUBLAS_STATUS_SUCCESS);
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
  if (n == 0)
    return;

  expect(x);
  expect(y);
  expect(incx >= 0);
  expect(incy >= 0);
  expect(n > 0);

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
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success =
          cublasScopy(device.get_handle(), n, x, incx, y, incy);
      expect(success == CUBLAS_STATUS_SUCCESS);
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
void copy(int64_t n, P const *x, P *y)
{
  if (n == 0)
    return;

  expect(x);
  expect(y);
  expect(n > 0);

  // device-specific specialization if needed
  if constexpr (resrc == resource::device)
  {
#ifdef ASGARD_USE_CUDA
    // function instantiated for these two fp types
    auto const success =
        cudaMemcpy(y, x, n * sizeof(P), cudaMemcpyDeviceToDevice);
    expect(success == cudaSuccess);
#endif
  }
  else if constexpr (resrc == resource::host)
  {
    std::copy_n(x, n, y);
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
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success =
          cublasSdot(device.get_handle(), n, x, incx, y, incy, &result);
      expect(success == CUBLAS_STATUS_SUCCESS);
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
  expect(x != nullptr);
  expect(y != nullptr);
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
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success =
          cublasSaxpy(device.get_handle(), n, &alpha, x, incx, y, incy);
      expect(success == CUBLAS_STATUS_SUCCESS);
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

template<resource resrc, typename P>
void scal(int n, P alpha, P *x, int incx)
{
  expect(n >= 0);
  expect(x);
  expect(incx >= 0);

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
    // instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success = cublasDscal(device.get_handle(), n, &alpha, x, incx);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSscal(device.get_handle(), n, &alpha, x, incx);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    return;
#endif
  }
  else if constexpr (resrc == resource::host)
  {
    // default execution on the host for any resource
    if constexpr (std::is_same_v<P, double>)
    {
      cblas_dscal(n, alpha, x, incx);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      cblas_sscal(n, alpha, x, incx);
    }
    else
    {
      for (int i = 0; i < n; ++i)
      {
        x[i * incx] *= alpha;
      }
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

static CBLAS_TRANSPOSE cblas_transpose_type(char trans)
{
  if (trans == 'n' || trans == 'N')
  {
    return CblasNoTrans;
  }
  else if (trans == 't' || trans == 'T')
  {
    return CblasTrans;
  }
  else
  {
    return CblasConjTrans;
  }
}

template<resource resrc, typename P>
void gemv(char trans, int m, int n, P alpha, P const *A, int lda, P const *x,
          int incx, P beta, P *y, int incy)
{
  // instantiated for these two fp types
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  expect(A);
  expect(x);
  expect(y);
  expect(m >= 0);
  expect(n >= 0);
  expect(lda >= 0);
  expect(incx >= 0);
  expect(incy >= 0);
  expect(trans == 't' || trans == 'T' || trans == 'n' || trans == 'N');

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success =
          cublasDgemv(device.get_handle(), cublas_trans(trans), m, n, &alpha, A,
                      lda, x, incx, &beta, y, incy);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success =
          cublasSgemv(device.get_handle(), cublas_trans(trans), m, n, &alpha, A,
                      lda, x, incx, &beta, y, incy);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    return;
#endif
  }
  else if constexpr (resrc == resource::host)
  {
    // default execution on the host
    if constexpr (std::is_same_v<P, double>)
    {
      cblas_dgemv(CblasColMajor, cblas_transpose_type(trans), m, n, alpha, A,
                  lda, x, incx, beta, y, incy);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      cblas_sgemv(CblasColMajor, cblas_transpose_type(trans), m, n, alpha, A,
                  lda, x, incx, beta, y, incy);
    }
  }
}

template<resource resrc, typename P>
void gemm(char transa, char transb, int m, int n, int k, P alpha, P const *A,
          int lda, P const *B, int ldb, P beta, P *C, int ldc)
{
  expect(A);
  expect(lda >= 0);
  expect(B);
  expect(ldb >= 0);
  expect(C);
  expect(ldc >= 0);
  expect(m >= 0);
  expect(n >= 0);
  expect(k >= 0);
  expect(transa == 't' || transa == 'T' || transa == 'n' || transa == 'N');
  expect(transb == 't' || transb == 'T' || transb == 'n' || transb == 'N');

  if constexpr (resrc == resource::device)
  {
    static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success = cublasDgemm(
          device.get_handle(), cublas_trans(transa), cublas_trans(transb), m, n,
          k, &alpha, A, lda, B, ldb, &beta, C, ldc);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSgemm(
          device.get_handle(), cublas_trans(transa), cublas_trans(transb), m, n,
          k, &alpha, A, lda, B, ldb, &beta, C, ldc);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    return;
#endif
  }
  else if constexpr (resrc == resource::host)
  {
    // default execution on the host for any resource
    if constexpr (std::is_same_v<P, double>)
    {
      cblas_dgemm(CblasColMajor, cblas_transpose_type(transa),
                  cblas_transpose_type(transb), m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      cblas_sgemm(CblasColMajor, cblas_transpose_type(transa),
                  cblas_transpose_type(transb), m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc);
    }
    else
    {
      bool const trans_A = (transa == 't') ? true : false;
      bool const trans_B = (transb == 't') ? true : false;
      basic_gemm(A, trans_A, lda, B, trans_B, ldb, C, ldc, m, k, n, alpha,
                 beta);
    }
  }
}

template<resource resrc, typename P>
int getrf(int m, int n, P *A, int lda, int *ipiv)
{
  // instantiated for these two fp types
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  expect(A);
  expect(ipiv);
  expect(lda >= 0);
  expect(m >= 0);
  expect(n >= 0);

  int info{1};
  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    expect(m == n);
    ignore(m);

    P **A_d;
    if (cudaMalloc((void **)&A_d, sizeof(P *)) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    auto stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == cudaSuccess);
    int *info_d;
    if (cudaMalloc((void **)&info_d, sizeof(int)) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    // instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success = cublasDgetrfBatched(device.get_handle(), n, A_d, lda,
                                               ipiv, info_d, 1);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSgetrfBatched(device.get_handle(), n, A_d, lda,
                                               ipiv, info_d, 1);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    stat = cudaMemcpy(&info, info_d, sizeof(int), cudaMemcpyDeviceToHost);
    expect(stat == cudaSuccess);
    stat = cudaFree(A_d);
    expect(stat == cudaSuccess);
    stat = cudaFree(info_d);
    expect(stat == cudaSuccess);

#endif
  }
  if constexpr (resrc == resource::host)
  {
    // default execution on the host for any resource
    if constexpr (std::is_same_v<P, double>)
    {
      dgetrf_(&m, &n, A, &lda, ipiv, &info);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      sgetrf_(&m, &n, A, &lda, ipiv, &info);
    }
  }
  return info;
}

template<resource resrc, typename P>
int getri(int n, P *A, int lda, int *ipiv, P *work, int lwork)
{
  // instantiated for these two fp types
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  expect(A);
  expect(ipiv);
  expect(work);
  expect(lwork == n * n);
  expect(lda >= 0);
  expect(n >= 0);

  int info{1};
  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    P **A_d;
    if (cudaMalloc((void **)&A_d, sizeof(P *)) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    P **work_d;
    if (cudaMalloc((void **)&work_d, sizeof(P *)) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    int *info_d;
    if (cudaMalloc((void **)&info_d, sizeof(int)) != cudaSuccess)
    {
      throw std::bad_alloc();
    }
    auto stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == 0);
    stat = cudaMemcpy(work_d, &work, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success = cublasDgetriBatched(device.get_handle(), n, A_d, lda,
                                               nullptr, work_d, n, info_d, 1);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSgetriBatched(device.get_handle(), n, A_d, lda,
                                               nullptr, work_d, n, info_d, 1);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    stat = cudaMemcpy(&info, info_d, sizeof(int), cudaMemcpyDeviceToHost);
    expect(stat == cudaSuccess);

    stat = cudaFree(A_d);
    expect(stat == cudaSuccess);
    stat = cudaFree(work_d);
    expect(stat == cudaSuccess);
    stat = cudaFree(info_d);
    expect(stat == cudaSuccess);
#endif
  }
  else if constexpr (resrc == resource::host)
  {
    // default execution on the host for any resource
    if constexpr (std::is_same_v<P, double>)
    {
      dgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      sgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }
  }
  return info;
}

template<resource resrc, typename P>
void batched_gemm(P **const &a, int lda, char transa, P **const &b, int ldb,
                  char transb, P **const &c, int ldc, int m, int n, int k,
                  P alpha, P beta, int num_batch)
{
  expect(a);
  expect(lda >= 0);
  expect(b);
  expect(ldb >= 0);
  expect(c);
  expect(ldc >= 0);
  expect(m >= 0);
  expect(n >= 0);
  expect(k >= 0);
  expect(transa == 't' || transa == 'T' || transa == 'n' || transa == 'N');
  expect(transb == 't' || transb == 'T' || transb == 'n' || transb == 'N');
  expect(num_batch > 0);

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    P const **a_d;
    P const **b_d;
    P **c_d;
    size_t const list_size = num_batch * sizeof(P *);

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
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success = cublasDgemmBatched(
          device.get_handle(), cublas_trans(transa), cublas_trans(transb), m, n,
          k, &alpha, a_d, lda, b_d, ldb, &beta, c_d, ldc, num_batch);
      auto const cuda_stat = cudaDeviceSynchronize();
      expect(cuda_stat == 0);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasSgemmBatched(
          device.get_handle(), cublas_trans(transa), cublas_trans(transb), m, n,
          k, &alpha, a_d, lda, b_d, ldb, &beta, c_d, ldc, num_batch);
      auto const cuda_stat = cudaDeviceSynchronize();
      expect(cuda_stat == 0);
      expect(success == CUBLAS_STATUS_SUCCESS);
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
  else if constexpr (resrc == resource::host)
  {
// default execution on the host for any resource
#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < num_batch; ++i)
    {
      gemm(transa, transb, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i],
           ldc);
    }
  }
}

template<typename P>
int gesv(int n, int nrhs, P *A, int lda, int *ipiv, P *b, int ldb)
{
  expect(A);
  expect(ipiv);
  expect(b);
  expect(ldb >= 1);
  expect(lda >= 1);
  expect(n >= 0);

  int info{1};
  if constexpr (std::is_same_v<P, double>)
  {
    dgesv_(&n, &nrhs, A, &lda, ipiv, b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<P, float>)
  {
    sgesv_(&n, &nrhs, A, &lda, ipiv, b, &ldb, &info);
  }
  return info;
}

#ifdef ASGARD_USE_CUDA
static cublasOperation_t cublas_operation_type(char trans)
{
  if (trans == 'n' || trans == 'N')
  {
    return CUBLAS_OP_N;
  }
  else if (trans == 't' || trans == 'T')
  {
    return CUBLAS_OP_T;
  }
  else
  {
    return CUBLAS_OP_C;
  }
}

static cublasFillMode_t cublas_fillmode_type(char trans)
{
  if (trans == 'U' || trans == 'u')
  {
    return CUBLAS_FILL_MODE_UPPER;
  }
  else
  {
    expect(trans == 'L' || trans == 'l');
    return CUBLAS_FILL_MODE_LOWER;
  }
}

static cublasDiagType_t cublas_diag_type(char trans)
{
  if (trans == 'U' || trans == 'u')
  {
    return CUBLAS_DIAG_UNIT;
  }
  else
  {
    expect(trans == 'N' || trans == 'n');
    return CUBLAS_DIAG_NON_UNIT;
  }
}
#endif

static CBLAS_UPLO cblas_uplo_type(char trans)
{
  if (trans == 'U' || trans == 'u')
  {
    return CblasUpper;
  }
  else
  {
    expect(trans == 'L' || trans == 'l');
    return CblasLower;
  }
}

static CBLAS_DIAG cblas_diag_type(char trans)
{
  if (trans == 'U' || trans == 'u')
  {
    return CblasUnit;
  }
  else
  {
    expect(trans == 'N' || trans == 'n');
    return CblasNonUnit;
  }
}

template<resource resrc, typename P>
void tpsv(const char uplo, const char trans, const char diag, const int n,
          const P *ap, P *x, const int incx)
{
  expect(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l');
  expect(trans == 't' || trans == 'T' || trans == 'n' || trans == 'N');
  expect(diag == 'U' || diag == 'u' || diag == 'N' || diag == 'n');
  expect(n >= 0);
  expect(ap);
  expect(x);
  expect(incx > 0);

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // instantiated for these two fp types
    if constexpr (std::is_same_v<P, double>)
    {
      auto const success = cublasDtpsv(
          device.get_handle(), cublas_fillmode_type(uplo),
          cublas_operation_type(trans), cublas_diag_type(diag), n, ap, x, incx);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      auto const success = cublasStpsv(
          device.get_handle(), cublas_fillmode_type(uplo),
          cublas_operation_type(trans), cublas_diag_type(diag), n, ap, x, incx);
      expect(success == CUBLAS_STATUS_SUCCESS);
    }
#endif
  }
  // default execution on the host
  else if constexpr (resrc == resource::host)
  {
    if constexpr (std::is_same_v<P, double>)
    {
      cblas_dtpsv(CblasColMajor, cblas_uplo_type(uplo),
                  cblas_transpose_type(trans), cblas_diag_type(diag), n, ap, x,
                  incx);
    }
    else if constexpr (std::is_same_v<P, float>)
    {
      cblas_stpsv(CblasColMajor, cblas_uplo_type(uplo),
                  cblas_transpose_type(trans), cblas_diag_type(diag), n, ap, x,
                  incx);
    }
  }
}

template<typename P>
int getrs(char trans, int n, int nrhs, P const *A, int lda, int const *ipiv,
          P *b, int ldb)
{
  expect(A);
  expect(ipiv);
  expect(b);
  expect(ldb >= 1);
  expect(lda >= 1);
  expect(n >= 0);

  int info{1};
  // the const_cast below is needed due to bad header under OSX
  if constexpr (std::is_same_v<P, double>)
  {
    dgetrs_(&trans, &n, &nrhs, const_cast<P *>(A), &lda,
            const_cast<int *>(ipiv), b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<P, float>)
  {
    sgetrs_(&trans, &n, &nrhs, const_cast<P *>(A), &lda,
            const_cast<int *>(ipiv), b, &ldb, &info);
  }
  return info;
}

template<typename P>
int pttrf(int n, P *D, P *E)
{
  expect(D);
  expect(E);
  expect(n >= 0);

  int info{1};
  if constexpr (std::is_same_v<P, double>)
  {
    dpttrf_(&n, D, E, &info);
  }
  else if constexpr (std::is_same_v<P, float>)
  {
    spttrf_(&n, D, E, &info);
  }
  return info;
}

template<typename P>
int pttrs(int n, int nrhs, P const *D, P const *E, P *B, int ldb)
{
  expect(D);
  expect(E);
  expect(B);
  expect(ldb >= 1);
  expect(n >= 0);

  int info{1};
  // the const_cast below is needed due to bad header under OSX
  if constexpr (std::is_same_v<P, double>)
  {
    dpttrs_(&n, &nrhs, const_cast<P *>(D), const_cast<P *>(E), B, &ldb, &info);
  }
  else if constexpr (std::is_same_v<P, float>)
  {
    spttrs_(&n, &nrhs, const_cast<P *>(D), const_cast<P *>(E), B, &ldb, &info);
  }
  return info;
}

template<resource resrc, typename P>
void sparse_gemv(char const trans, int rows, int cols, int nnz,
                 const int *row_offsets, const int *col_indices, const P *vals,
                 P alpha, const P *x, P beta, P *y)
{
  expect(row_offsets);
  expect(col_indices);
  expect(vals);
  expect(x);
  expect(y);
  expect(rows >= 0);
  expect(cols >= 0);
  expect(nnz >= 0);
  expect(trans == 't' || trans == 'n');

  if constexpr (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // set the correct cuda data type based on P=double or P=float
    cudaDataType_t constexpr fp_prec =
        std::is_same_v<P, double> ? CUDA_R_64F : CUDA_R_32F;

    // create cuSPARSE descriptors
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vecX, vecY;
    auto status =
        cusparseCreateCsr(&mat, rows, cols, nnz, const_cast<int *>(row_offsets),
                          const_cast<int *>(col_indices), const_cast<P *>(vals),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, fp_prec);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    status = cusparseCreateDnVec(&vecX, cols, const_cast<P *>(x), fp_prec);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    status = cusparseCreateDnVec(&vecY, rows, const_cast<P *>(y), fp_prec);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    // find tmp buffer size if needed
    size_t buffer_size = 0;
    status             = cusparseSpMV_bufferSize(
        device.get_sp_handle(), cusparse_trans(trans), &alpha, mat, vecX, &beta,
        vecY, fp_prec, CUSPARSE_SPMV_CSR_ALG2, &buffer_size);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    void *sp_buffer = NULL;
    auto success    = cudaMalloc(&sp_buffer, buffer_size);
    expect(success == cudaSuccess);

    // call the sparse grid dense vector multiply
    // using CSR_ALG2 since it provides deterministic bit-wise results for
    // each run
    status = cusparseSpMV(device.get_sp_handle(), cusparse_trans(trans), &alpha,
                          mat, vecX, &beta, vecY, fp_prec,
                          CUSPARSE_SPMV_CSR_ALG2, sp_buffer);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    success = cudaFree(sp_buffer);
    expect(success == cudaSuccess);

    // destroy the cuSparse desriptors
    status = cusparseDestroyDnVec(vecX);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    status = cusparseDestroyDnVec(vecY);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    status = cusparseDestroySpMat(mat);
    expect(status == CUSPARSE_STATUS_SUCCESS);
#endif
    return;
  }
  if constexpr (resrc == resource::host)
  {
    // TODO: only non-transpose case implemented
    expect(trans == 'n');

    for (int i = 0; i < rows; i++)
    {
      y[i] *= beta;
      // sparse case, iterate over number of column entries in the current row
      for (int col = row_offsets[i]; col < row_offsets[i + 1]; col++)
      {
        y[i] += alpha * vals[col] * x[col_indices[col]];
      }
    }
  }
}

#ifdef ASGARD_ENABLE_FLOAT
template void rot<resource::host, float>(const int n, float *x, const int incx,
                                         float *y, const int incy,
                                         const float c, const float s);
template void rotg<resource::host, float>(float *, float *, float *, float *);
template float nrm2<resource::host, float>(int, float const[], int);
template void copy<resource::host, float>(int n, float const *x, int incx,
                                          float *y, int incy);
template void copy<resource::host, float>(int64_t n, float const *x, float *y);
template float dot<resource::host, float>(int n, float const *x, int incx,
                                          float const *y, int incy);
template void axpy<resource::host, float>(int n, float alpha, float const *x,
                                          int incx, float *y, int incy);
template void
scal<resource::host, float>(int n, float alpha, float *x, int incx);
template void gemv<resource::host, float>(char trans, int m, int n, float alpha,
                                          float const *A, int lda,
                                          float const *x, int incx, float beta,
                                          float *y, int incy);
template void gemm<resource::host, float>(char transa, char transb, int m,
                                          int n, int k, float alpha,
                                          float const *A, int lda,
                                          float const *B, int ldb, float beta,
                                          float *C, int ldc);
template int
getrf<resource::host, float>(int m, int n, float *A, int lda, int *ipiv);
template int getri<resource::host, float>(int n, float *A, int lda, int *ipiv,
                                          float *work, int lwork);
template void
batched_gemm<resource::host, float>(float **const &a, int lda, char transa,
                                    float **const &b, int ldb, char transb,
                                    float **const &c, int ldc, int m, int n,
                                    int k, float alpha, float beta,
                                    int num_batch);
template int
gesv(int n, int nrhs, float *A, int lda, int *ipiv, float *b, int ldb);
template int getrs(char trans, int n, int nrhs, float const *A, int lda,
                   int const *ipiv, float *b, int ldb);
template void tpsv<resource::host, float>(const char uplo, const char trans,
                                          const char diag, const int n,
                                          const float *ap, float *x,
                                          const int incx);
template int pttrf(int n, float *D, float *E);
template int
pttrs(int n, int nrhs, float const *D, float const *E, float *B, int ldb);

template void
sparse_gemv<resource::host, float>(char const trans, int rows, int cols,
                                   int nnz, const int *row_offsets,
                                   const int *col_indices, const float *vals,
                                   float alpha, const float *x, float beta,
                                   float *y);

#endif

// always enable for quadrature calculations
template void copy<resource::host, double>(int n, double const *x, int incx,
                                           double *y, int incy);
template void
copy<resource::host, double>(int64_t n, double const *x, double *y);
template void
scal<resource::host, double>(int n, double alpha, double *x, int incx);

#ifdef ASGARD_ENABLE_DOUBLE
template void
rot<resource::host, double>(const int n, double *x, const int incx, double *y,
                            const int incy, const double c, const double s);
template void
rotg<resource::host, double>(double *, double *, double *, double *);
template double nrm2<resource::host, double>(int, double const[], int);
template double dot<resource::host, double>(int n, double const *x, int incx,
                                            double const *y, int incy);
template void axpy<resource::host, double>(int n, double alpha, double const *x,
                                           int incx, double *y, int incy);
template void gemv<resource::host, double>(char trans, int m, int n,
                                           double alpha, double const *A,
                                           int lda, double const *x, int incx,
                                           double beta, double *y, int incy);
template void gemm<resource::host, double>(char transa, char transb, int m,
                                           int n, int k, double alpha,
                                           double const *A, int lda,
                                           double const *B, int ldb,
                                           double beta, double *C, int ldc);
template int
getrf<resource::host, double>(int m, int n, double *A, int lda, int *ipiv);
template int getri<resource::host, double>(int n, double *A, int lda, int *ipiv,
                                           double *work, int lwork);
template void
batched_gemm<resource::host, double>(double **const &a, int lda, char transa,
                                     double **const &b, int ldb, char transb,
                                     double **const &c, int ldc, int m, int n,
                                     int k, double alpha, double beta,
                                     int num_batch);
template int
gesv(int n, int nrhs, double *A, int lda, int *ipiv, double *b, int ldb);
template void tpsv<resource::host, double>(const char uplo, const char trans,
                                           const char diag, const int n,
                                           const double *ap, double *x,
                                           const int incx);
template int getrs(char trans, int n, int nrhs, double const *A, int lda,
                   int const *ipiv, double *b, int ldb);
template int pttrf(int n, double *D, double *E);
template int
pttrs(int n, int nrhs, double const *D, double const *E, double *B, int ldb);

template void
sparse_gemv<resource::host, double>(char const trans, int rows, int cols,
                                    int nnz, const int *row_offsets,
                                    const int *col_indices, const double *vals,
                                    double alpha, const double *x, double beta,
                                    double *y);

#endif

template void
copy<resource::host, int>(int n, int const *x, int incx, int *y, int incy);
template void copy<resource::host, int>(int64_t n, int const *x, int *y);
template int
dot<resource::host, int>(int n, int const *x, int incx, int const *y, int incy);
template void scal<resource::host, int>(int n, int alpha, int *x, int incx);
template void gemm<resource::host, int>(char transa, char transb, int m, int n,
                                        int k, int alpha, int const *A, int lda,
                                        int const *B, int ldb, int beta, int *C,
                                        int ldc);

#ifdef ASGARD_USE_CUDA

template void copy<resource::device, int>(int64_t n, int const *x, int *y);

#ifdef ASGARD_ENABLE_FLOAT
template float nrm2<resource::device, float>(int, float const[], int);
template void
rot<resource::device, float>(const int n, float *x, const int incx, float *y,
                             const int incy, const float c, const float s);
template void rotg<resource::device, float>(float *, float *, float *, float *);
template void copy<resource::device, float>(int n, float const *x, int incx,
                                            float *y, int incy);
template void
copy<resource::device, float>(int64_t n, float const *x, float *y);
template float dot<resource::device, float>(int n, float const *x, int incx,
                                            float const *y, int incy);
template void axpy<resource::device, float>(int n, float alpha, float const *x,
                                            int incx, float *y, int incy);
template void
scal<resource::device, float>(int n, float alpha, float *x, int incx);
template void gemv<resource::device, float>(char trans, int m, int n,
                                            float alpha, float const *A,
                                            int lda, float const *x, int incx,
                                            float beta, float *y, int incy);
template void gemm<resource::device, float>(char transa, char transb, int m,
                                            int n, int k, float alpha,
                                            float const *A, int lda,
                                            float const *B, int ldb, float beta,
                                            float *C, int ldc);
template void
batched_gemm<resource::device, float>(float **const &a, int lda, char transa,
                                      float **const &b, int ldb, char transb,
                                      float **const &c, int ldc, int m, int n,
                                      int k, float alpha, float beta,
                                      int num_batch);
template int
getrf<resource::device, float>(int m, int n, float *A, int lda, int *ipiv);
template int getri<resource::device, float>(int n, float *A, int lda, int *ipiv,
                                            float *work, int lwork);
template void tpsv<resource::device, float>(const char uplo, const char trans,
                                            const char diag, const int n,
                                            const float *ap, float *x,
                                            const int incx);

template void
sparse_gemv<resource::device, float>(char const trans, int rows, int cols,
                                     int nnz, const int *row_offsets,
                                     const int *col_indices, const float *vals,
                                     float alpha, const float *x, float beta,
                                     float *y);
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template double nrm2<resource::device, double>(int, double const[], int);
template void
rot<resource::device, double>(const int n, double *x, const int incx, double *y,
                              const int incy, const double c, const double s);
template void
rotg<resource::device, double>(double *, double *, double *, double *);
template void copy<resource::device, double>(int n, double const *x, int incx,
                                             double *y, int incyc);
template void
copy<resource::device, double>(int64_t n, double const *x, double *y);
template double dot<resource::device, double>(int n, double const *x, int incx,
                                              double const *y, int incy);
template void axpy<resource::device, double>(int n, double alpha,
                                             double const *x, int incx,
                                             double *y, int incy);
template void
scal<resource::device, double>(int n, double alpha, double *x, int incx);
template void gemv<resource::device, double>(char trans, int m, int n,
                                             double alpha, double const *A,
                                             int lda, double const *x, int incx,
                                             double beta, double *y, int incy);
template void gemm<resource::device, double>(char transa, char transb, int m,
                                             int n, int k, double alpha,
                                             double const *A, int lda,
                                             double const *B, int ldb,
                                             double beta, double *C, int ldc);
template void
batched_gemm<resource::device, double>(double **const &a, int lda, char transa,
                                       double **const &b, int ldb, char transb,
                                       double **const &c, int ldc, int m, int n,
                                       int k, double alpha, double beta,
                                       int num_batch);
template int
getrf<resource::device, double>(int m, int n, double *A, int lda, int *ipiv);
template int getri<resource::device, double>(int n, double *A, int lda,
                                             int *ipiv, double *work,
                                             int lwork);
template void tpsv<resource::device, double>(const char uplo, const char trans,
                                             const char diag, const int n,
                                             const double *ap, double *x,
                                             const int incx);
template void
sparse_gemv<resource::device, double>(char const trans, int rows, int cols,
                                      int nnz, const int *row_offsets,
                                      const int *col_indices,
                                      const double *vals, double alpha,
                                      const double *x, double beta, double *y);

#endif
#endif

} // namespace asgard::lib_dispatch
