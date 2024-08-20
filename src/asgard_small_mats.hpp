#pragma once

#include "./device/asgard_kronmult.hpp"

#include "asgard_lib_dispatch.hpp"

/*!
 * \file asgard_small_mats.hpp
 * \brief Private header containing small matrix algorithms
 * \author The ASGarD Team
 * \ingroup asgard_smallmat
 */

/*!
 * \internal
 * \defgroup asgard_smallmat Small matrix operations
 *
 * Many algorithms in ASGarD revolve around matrix-vector or matrix-matrix operations
 * where the matrices have size around degree + 1.
 * Working with small degree of 2 - 3 (quadratic cubic) or even moderate (10 - 20),
 * using BLAS is inefficient due to the optimizations that assume large scale
 * problems.
 * When the data is small enough to fir into the CPU L1 cache (e.g., 128K for most
 * small CPUs and 256K for large ones), then a "naive" approach using OpenMP simd
 * will produce better results than BLAS.
 *
 * Making a dedicated matrix class and containing the size is also inefficient.
 * We want matrices to be packed in place (on stack or larger heap data-structures),
 * so we always assume non-owning reference to the matrix (i.e., get the pointer)
 * and separate size variables that can be common for all matrices, as opposed
 * to storing redundant num-rows/cols for a large number of small matrices.
 *
 * The matrices can still be packaged into a large data-structure that uses RAII,
 * here we implement the low level algorithms for performing matrix operations.
 *
 * The signature mimics BLAS/LAPACK as much as reasonable, while assuming contiguous
 * data, i.e., vector stride is always 1 and lda is always the number of rows.
 *
 * Finally, we want to make the header private, i.e., do not expose into user
 * code, unless the user explicitly wants to have it included.
 * \endinternal
 */

/*!
 * \ingroup asgard_smallmat
 * \brief Guards against inclusion into the public header stream
 */
#define ASGARD_SMALL_MATRIX_METHODS

/*!
 * \ingroup asgard_smallmat
 * \brief namespace for the small matrix operations
 */
namespace asgard::smmat
{
//! debug purposes, print a small vector of size n
template<typename P>
void print(int const &n, P const x[])
{
  std::cout.precision(8);
  std::cout << std::scientific;
  for (int j = 0; j < n; j++)
    std::cout << std::setw(18) << x[j];
  std::cout << '\n';
}
//! debug purposes, print a small matrix of size nr by nc
template<typename P>
void print(int const &nr, int const &nc, P const A[])
{
  std::cout.precision(8);
  std::cout << std::scientific;
  for (int i = 0; i < nr; i++)
  {
    for (int j = 0; j < nc; j++)
      std::cout << std::setw(18) << A[j * nr + i];
    std::cout << '\n';
  }
}
//! scale x by alpha, n is the size of x
template<typename P>
void scal(int const &n, P alpha, P x[])
{
  ASGARD_OMP_SIMD
  for (int i = 0; i < n; i++)
    x[i] *= alpha;
}
//! matrix-vector multiplication y += A * x, A has size nr X nc
template<typename P>
void gemv1(int const &nr, int const &nc, P const A[], P const x[], P y[])
{
  for (int i = 0; i < nc; i++)
    ASGARD_OMP_SIMD
    for (int j = 0; j < nr; j++)
      y[j] += A[i * nr + j] * x[i];
}
//! matrix-vector multiplication y = A * x, A has size nr X nc
template<typename P>
void gemv(int const &nr, int const &nc, P const A[], P const x[], P y[])
{
  ASGARD_OMP_SIMD
  for (int j = 0; j < nr; j++)
    y[j] = A[j] * x[0];

  ASGARD_PRAGMA_OMP_SIMD(collapse(2))
  for (int i = 1; i < nc; i++)
    for (int j = 0; j < nr; j++)
      y[j] += A[i * nr + j] * x[i];
}
//! triple-matrix-matrix product, C = A * diag(d) * B, A is n by m, C is n by n
template<typename P>
void gemm3(int const &n, int const &m, P const A[], P const d[], P const B[], P C[])
{
  for (int c = 0; c < n; c++) // for each of the r columns of the output
  {
    P *cc = &C[c * n];
    P x   = d[0] * B[c * m];
    ASGARD_OMP_SIMD
    for (int j = 0; j < n; j++)
      cc[j] = A[j] * x;

    for (int i = 1; i < m; i++)
    {
      x = d[i] * B[c * m + i];
      ASGARD_OMP_SIMD
      for (int j = 0; j < n; j++)
        cc[j] += A[i * n + j] * x;
    }
  }
}
//! invert a 2x2 matrix
template<typename P>
void inv2by2(P A[])
{
  scal(4, P{1} / (A[0] * A[3] - A[1] * A[2]), A);
  std::swap(A[0], A[3]);
  A[1] = -A[1];
  A[2] = -A[2];
}
//! multiply in place by a 2x2 matrix
template<typename P>
void gemv2by2(P const A[], P x[])
{
  P t1 = x[0];
  P t2 = x[1];
  x[0] = A[0] * t1 + A[2] * t2;
  x[1] = A[1] * t1 + A[3] * t2;
}
//! multiply in place by a 2x2 matrix
template<typename P>
void gemm2by2(P const A[], P B[])
{
  gemv2by2(A, B);
  gemv2by2(A, B + 2);
}
//! cholesky factorize
template<typename P>
void potrf(int const &n, P A[])
{
  for (int i = 0; i < n; i++)
  {
    P sum = 0;
    for (int j = 0; j < i; j++)
      sum += A[i * n + j] * A[i * n + j];

    A[i * n + i] = std::sqrt(A[i * n + i] - sum);

    for (int j = i + 1; j < n; j++)
    {
      sum = 0;
      for (int k = 0; k < i; k++)
        sum += A[i * n + k] * A[j * n + k];
      A[j * n + i] = (A[j * n + i] - sum) / A[i * n + i];
    }
  }
}
//! cholesky solve
template<typename P>
void posv(int const &n, P const A[], P x[])
{
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < i; j++)
      x[i] -= A[i * n + j] * x[j];

    x[i] /= A[i * n + i];
  }
  for(int i = n - 1; i >= 0; i--){
    x[i] /= A[i * n + i];
    ASGARD_OMP_SIMD
    for(int j = i - 1; j >= 0; j--)
      x[j] -= x[i] * A[i * n + j];
  }
}
//! cholesky solve
template<typename P>
void posvm(int const &n, P const A[], P B[])
{
  for (int i = 0; i < n; i++)
    posv(n, A, B + i * n);
}

//! C += (dir) A^T B, dir must be +/-1
template<int dir = +1, typename P>
void gemm_tn(int const &nrc, int const &nk, P const A[], P const B[], P C[])
{
  static_assert(dir == 1 or dir == -1);
  // TODO figure out the simd logic here
  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int c = 0; c < nrc; c++)
    for (int r = 0; r < nrc; r++)
      for (int k = 0; k < nk; k++)
        if constexpr (dir == 1)
          C[c * nrc + r] += A[r * nk + k] * B[c * nk + k];
        else
          C[c * nrc + r] -= A[r * nk + k] * B[c * nk + k];
}
template<typename P>
void neg_transp(int const &n, P A[])
{
  for (int c = 0; c < n; c++)
  {
    A[c * n + c] = -A[c * n + c];
    for (int r = c + 1; r < n; r++)
    {
      P t = -A[c * n + r];
      A[c * n + r] = -A[r * n + c];
      A[r * n + c] = t;
    }
  }
}
template<typename P>
void neg_transp_swap(int const &n, P A[], P B[])
{
  for (int c = 0; c < n; c++)
  {
    for (int r = 0; r < n; r++)
    {
      P t = -A[c * n + r];
      A[c * n + r] = -B[r * n + c];
      B[r * n + c] = t;
    }
  }
}

//! C += (dir) A B, dir must be +1/0/-1
template<int dir = 0, typename P>
void gemm(int const &n, P const A[], P const B[], P C[])
{
  static_assert(dir == 1 or dir == 0 or dir == -1);
  if constexpr (dir == 0)
    ASGARD_PRAGMA_OMP_SIMD(collapse(2))
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        C[c * n + r] = A[r] * B[c * n];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int k = (dir == 0) ? 1 : 0; k < n; k++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        if constexpr (dir == 1 or dir == 0)
          C[c * n + r] += A[k * n + r] * B[c * n + k];
        else
          C[c * n + r] -= A[k * n + r] * B[c * n + k];
}

//! R = a0 * t0^T
template<typename P>
void gemm_nt(int const &n, P const a0[], P const t0[], P R[])
{
  ASGARD_PRAGMA_OMP_SIMD(collapse(2))
  for (int c = 0; c < n; c++)
    for (int r = 0; r < n; r++)
      R[c * n + r] = a0[r] * t0[c];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 1; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a0[i * n + r] * t0[i * n + c];
}

//! R = a0 * t0 + a1 * t1, all matrices are n by n (column major)
template<typename P>
void gemm_pair(int const &n, P const a0[], P const t0[], P const a1[], P const t1[], P R[])
{
  ASGARD_PRAGMA_OMP_SIMD(collapse(2))
  for (int c = 0; c < n; c++)
    for (int r = 0; r < n; r++)
      R[c * n + r] = a0[r] * t0[c * n];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 1; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a0[i * n + r] * t0[c * n + i];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 0; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a1[i * n + r] * t1[c * n + i];
}
//! R = a0 * t0 + a1 * t1, all matrices are n by n (column major), constexpr variant
template<int n, typename P>
void gemm_pair(P const a0[], P const t0[], P const a1[], P const t1[], P R[])
{
  ASGARD_PRAGMA_OMP_SIMD(collapse(2))
  for (int c = 0; c < n; c++)
    for (int r = 0; r < n; r++)
      R[c * n + r] = a0[r] * t0[c * n];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 1; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a0[i * n + r] * t0[c * n + i];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 0; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a1[i * n + r] * t1[c * n + i];
}
//! R = a0 * t0^T + a1 * t1^T, all matrices are n by n (column major)
template<typename P>
void gemm_pairt(int const &n, P const a0[], P const t0[], P const a1[], P const t1[], P R[])
{
  ASGARD_PRAGMA_OMP_SIMD(collapse(2))
  for (int c = 0; c < n; c++)
    for (int r = 0; r < n; r++)
      R[c * n + r] = a0[r] * t0[c];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 1; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a0[i * n + r] * t0[i * n + c];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 0; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a1[i * n + r] * t1[i * n + c];
}
//! R = a0 * t0^T + a1 * t1^T, all matrices are n by n (column major), constexpr variant
template<int n, typename P>
void gemm_pairt(P const a0[], P const t0[], P const a1[], P const t1[], P R[])
{
  ASGARD_PRAGMA_OMP_SIMD(collapse(2))
  for (int c = 0; c < n; c++)
    for (int r = 0; r < n; r++)
      R[c * n + r] = a0[r] * t0[c];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 1; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a0[i * n + r] * t0[i * n + c];

  ASGARD_PRAGMA_OMP_SIMD(collapse(3))
  for (int i = 0; i < n; i++)
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        R[c * n + r] += a1[i * n + r] * t1[i * n + c];
}

} // namespace asgard::smmat
