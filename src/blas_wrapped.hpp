#pragma once
#include <cassert>

enum class environment
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
     DCOPY copies a vector, x, to a vector, y.
     uses unrolled loops for increments equal to one.
     --------------------------------------------------------------------------
   */
  void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
  void scopy_(int *n, float *x, int *incx, float *y, int *incy);
  // --------------------------------------------------------------------------
  // vector-vector multiply
  // d = x*y
  // --------------------------------------------------------------------------
  double ddot_(int *n, double *X, int *incx, double *Y, int *incy);
  float sdot_(int *n, float *X, int *incx, float *Y, int *incy);

  // --------------------------------------------------------------------------
  // vector-vector addition
  // y := ax + y
  // --------------------------------------------------------------------------
  double
  daxpy_(int *n, double *alpha, double *X, int *incx, double *Y, int *incy);
  float saxpy_(int *n, float *alpha, float *X, int *incx, float *Y, int *incy);

  // --------------------------------------------------------------------------
  // vector-scalar multiply
  // y := y*alpha
  // --------------------------------------------------------------------------
  double dscal_(int *n, double *alpha, double *X, int *incx);
  float sscal_(int *n, float *alpha, float *X, int *incx);

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

//
// Simple matrix multiply for non-float types
// FIXME we will probably eventually need a version that does transpose
//
template<typename P>
static void igemm_(P *A, int const lda, P *B, int const ldb, P *C,
                   int const ldc, int const m, int const k, int const n)
{
  assert(m > 0);
  assert(k > 0);
  assert(n > 0);
  assert(lda > 0); // FIXME Tyler says these could be more thorough
  assert(ldb > 0);
  assert(ldc > 0);

  for (auto i = 0; i < m; ++i)
  {
    for (auto j = 0; j < n; ++j)
    {
      P result = 0.0;
      for (auto z = 0; z < k; ++z)
      {
        // result += A[i,k] * B[k,j]
        result += A[z * lda + i] * B[j * ldb + z];
      }
      // C[i,j] += result
      C[j * ldc + i] += result;
    }
  }
}

// -- precision/execution environment wrapper for blas --

template<typename P>
void copy(int *n, P *x, int *incx, P *y, int *incy, environment const environ);

template<typename P>
P dot(int *n, P *X, int *incx, P *Y, int *incy, environment const environ);

template<typename P>
void axpy(int *n, P *alpha, P *X, int *incx, P *Y, int *incy,
          environment const environ);

template<typename P>
void scal(int *n, P *alpha, P *X, int *incx, environment const environ);

template<typename P>
void gemv(char const *trans, int *m, int *n, P *alpha, P *A, int *lda, P *x,
          int *incx, P *beta, P *y, int *incy, environment const environ);

template<typename P>
void gemm(char const *transa, char const *transb, int *m, int *n, int *k,
          P *alpha, P *A, int *lda, P *B, int *ldb, P *beta, P *C, int *ldc,
          environment const environ);

template<typename P>
void getrf(int *m, int *n, P *A, int *lda, int *ipiv, int *info,
           environment const environ);

template<typename P>
void getri(int *n, P *A, int *lda, int *ipiv, P *work, int *lwork, int *info,
           environment const environ);
