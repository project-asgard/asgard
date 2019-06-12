#pragma once
#include <cassert>

enum class resource
{
  host,
  device
};

// ==========================================================================
// external declarations for calling blas routines linked with -lblas
// ==========================================================================
extern "C"
{
  /* --------------------------------------------------------------------------
     DCOPy copies a vector, x, to a vector, y.
     uses unrolled loops for increments equal to one.
     --------------------------------------------------------------------------
   */
  void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
  void scopy_(int *n, float *x, int *incx, float *y, int *incy);
  // --------------------------------------------------------------------------
  // vector-vector multiply
  // d = x*y
  // --------------------------------------------------------------------------
  double ddot_(int *n, double *x, int *incx, double *y, int *incy);
  float sdot_(int *n, float *x, int *incx, float *y, int *incy);

  // --------------------------------------------------------------------------
  // vector-vector addition
  // y := ax + y
  // --------------------------------------------------------------------------
  double
  daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
  float saxpy_(int *n, float *alpha, float *x, int *incx, float *y, int *incy);

  // --------------------------------------------------------------------------
  // vector-scalar multiply
  // y := y*alpha
  // --------------------------------------------------------------------------
  double dscal_(int *n, double *alpha, double *x, int *incx);
  float sscal_(int *n, float *alpha, float *x, int *incx);

  // --------------------------------------------------------------------------
  // matrix-vector multiply
  // y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
  // --------------------------------------------------------------------------
  void dgemv_(char const *trans, int *m, int *n, double *alpha, double *A,
              int *lda, double *x, int *incx, double *beta, double *y,
              int *incy);
  void sgemv_(char const *trans, int *m, int *n, float *alpha, float *A,
              int *lda, float *x, int *incx, float *beta, float *y, int *incy);
  // --------------------------------------------------------------------------
  // matrix-matrix multiply
  // C := alpha*A*B + beta*C
  // --------------------------------------------------------------------------
  void dgemm_(char const *transa, char const *transb, int *m, int *n, int *k,
              double *alpha, double *A, int *lda, double *B, int *ldb,
              double *beta, double *C, int *ldc);
  void sgemm_(char const *transa, char const *transb, int *m, int *n, int *k,
              float *alpha, float *A, int *lda, float *B, int *ldb, float *beta,
              float *C, int *ldc);

  // --------------------------------------------------------------------------
  // LU decomposition of a general matrix
  // --------------------------------------------------------------------------
  void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);

  void sgetrf_(int *m, int *n, float *A, int *lda, int *ipiv, int *info);

  // --------------------------------------------------------------------------
  // inverse of a matrix given its LU decomposition
  // --------------------------------------------------------------------------
  void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work, int *lwork,
               int *info);

  void sgetri_(int *n, float *A, int *lda, int *ipiv, float *work, int *lwork,
               int *info);
}

// -- precision/execution resource wrapper for blas --

namespace lib_dispatch
{
template<typename P>
void copy(int *n, P *x, int *incx, P *y, int *incy,
          resource const res = resource::host);

template<typename P>
P dot(int *n, P *x, int *incx, P *y, int *incy,
      resource const res = resource::host);

template<typename P>
void axpy(int *n, P *alpha, P *x, int *incx, P *y, int *incy,
          resource const res = resource::host);

template<typename P>
void scal(int *n, P *alpha, P *x, int *incx,
          resource const res = resource::host);

template<typename P>
void gemv(char const *trans, int *m, int *n, P *alpha, P *A, int *lda, P *x,
          int *incx, P *beta, P *y, int *incy,
          resource const res = resource::host);

template<typename P>
void gemm(char const *transa, char const *transb, int *m, int *n, int *k,
          P *alpha, P *A, int *lda, P *B, int *ldb, P *beta, P *C, int *ldc,
          resource const res = resource::host);

template<typename P>
void getrf(int *m, int *n, P *A, int *lda, int *ipiv, int *info,
           resource const res = resource::host);

template<typename P>
void getri(int *n, P *A, int *lda, int *ipiv, P *work, int *lwork, int *info,
           resource const res = resource::host);

extern template void
copy(int *n, float *x, int *incx, float *y, int *incy, resource const res);
extern template void
copy(int *n, double *x, int *incx, double *y, int *incy, resource const res);
extern template void
copy(int *n, int *x, int *incx, int *y, int *incy, resource const res);

extern template float
dot(int *n, float *x, int *incx, float *y, int *incy, resource const res);
extern template double
dot(int *n, double *x, int *incx, double *y, int *incy, resource const res);
extern template int
dot(int *n, int *x, int *incx, int *y, int *incy, resource const res);

extern template void axpy(int *n, float *alpha, float *x, int *incx, float *y,
                          int *incy, resource const res);
extern template void axpy(int *n, double *alpha, double *x, int *incx,
                          double *y, int *incy, resource const res);
extern template void axpy(int *n, int *alpha, int *x, int *incx, int *y,
                          int *incy, resource const res);

extern template void
scal(int *n, float *alpha, float *x, int *incx, resource const res);
extern template void
scal(int *n, double *alpha, double *x, int *incx, resource const res);
extern template void
scal(int *n, int *alpha, int *x, int *incx, resource const res);

extern template void gemv(char const *trans, int *m, int *n, float *alpha,
                          float *A, int *lda, float *x, int *incx, float *beta,
                          float *y, int *incy, resource const res);
extern template void gemv(char const *trans, int *m, int *n, double *alpha,
                          double *A, int *lda, double *x, int *incx,
                          double *beta, double *y, int *incy,
                          resource const res);
extern template void gemv(char const *trans, int *m, int *n, int *alpha, int *A,
                          int *lda, int *x, int *incx, int *beta, int *y,
                          int *incy, resource const res);

extern template void gemm(char const *transa, char const *transb, int *m,
                          int *n, int *k, float *alpha, float *A, int *lda,
                          float *B, int *ldb, float *beta, float *C, int *ldc,
                          resource const res);
extern template void gemm(char const *transa, char const *transb, int *m,
                          int *n, int *k, double *alpha, double *A, int *lda,
                          double *B, int *ldb, double *beta, double *C,
                          int *ldc, resource const res);
extern template void gemm(char const *transa, char const *transb, int *m,
                          int *n, int *k, int *alpha, int *A, int *lda, int *B,
                          int *ldb, int *beta, int *C, int *ldc,
                          resource const res);

extern template void getrf(int *m, int *n, float *A, int *lda, int *ipiv,
                           int *info, resource const res);
extern template void getrf(int *m, int *n, double *A, int *lda, int *ipiv,
                           int *info, resource const res);

extern template void getri(int *n, float *A, int *lda, int *ipiv, float *work,
                           int *lwork, int *info, resource const res);
extern template void getri(int *n, double *A, int *lda, int *ipiv, double *work,
                           int *lwork, int *info, resource const res);
} // namespace lib_dispatch
