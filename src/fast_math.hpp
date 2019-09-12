#pragma once
#include "lib_dispatch.hpp"
#include "tensors.hpp"

namespace fm
{
// a non-matlab one-liner that had no better home - compute 2^arg
inline int two_raised_to(int exponent)
{
  assert(exponent >= 0);
  return 1 << exponent;
}

// axpy - y += a*x
template<typename P, mem_type mem, mem_type omem, resource resrc>
fk::vector<P, mem, resrc> &
axpy(fk::vector<P, omem, resrc> const &x, fk::vector<P, mem, resrc> &y,
     P const alpha = 1.0)
{
  assert(x.size() == y.size());
  int n    = x.size();
  int one  = 1;
  P alpha_ = alpha;
  lib_dispatch::axpy(&n, &alpha_, x.data(), &one, y.data(), &one, resrc);
  return y;
}

// copy(x,y) - copy vector x into y
template<typename P, mem_type mem, mem_type omem, resource resrc>
fk::vector<P, mem, resrc> &
copy(fk::vector<P, omem, resrc> const &x, fk::vector<P, mem, resrc> &y)
{
  assert(y.size() >= x.size());
  int n   = x.size();
  int one = 1;
  lib_dispatch::copy(&n, x.data(), &one, y.data(), &one, resrc);
  return y;
}

// scal - scale a vector
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc> &scal(P const alpha, fk::vector<P, mem, resrc> &x)
{
  int one  = 1;
  int n    = x.size();
  P alpha_ = alpha;
  lib_dispatch::scal(&n, &alpha_, x.data(), &one, resrc);
  return x;
}

// gemv - matrix vector multiplication
template<typename P, mem_type amem, mem_type xmem, mem_type ymem,
         resource resrc>
fk::vector<P, ymem, resrc> &
gemv(fk::matrix<P, amem, resrc> const &A, fk::vector<P, xmem, resrc> const &x,
     fk::vector<P, ymem, resrc> &y, bool const trans_A = false,
     P const alpha = 1.0, P const beta = 0.0)
{
  int const rows_A = trans_A ? A.ncols() : A.nrows();
  int const cols_A = trans_A ? A.nrows() : A.ncols();

  assert(rows_A == y.size());
  assert(cols_A == x.size());

  int lda           = A.stride();
  int one           = 1;
  P alpha_          = alpha;
  P beta_           = beta;
  char const transa = trans_A ? 't' : 'n';
  int m             = A.nrows();
  int n             = A.ncols();

  lib_dispatch::gemv(&transa, &m, &n, &alpha_, A.data(), &lda, x.data(), &one,
                     &beta_, y.data(), &one, resrc);

  return y;
}

// gemm - matrix matrix multiplication
template<typename P, mem_type amem, mem_type bmem, mem_type cmem,
         resource resrc>
fk::matrix<P, cmem, resrc> &
gemm(fk::matrix<P, amem, resrc> const &A, fk::matrix<P, bmem, resrc> const &B,
     fk::matrix<P, cmem, resrc> &C, bool const trans_A = false,
     bool const trans_B = false, P const alpha = 1.0, P const beta = 0.0)
{
  int const rows_A = trans_A ? A.ncols() : A.nrows();
  int const cols_A = trans_A ? A.nrows() : A.ncols();

  int const rows_B = trans_B ? B.ncols() : B.nrows();
  int const cols_B = trans_B ? B.nrows() : B.ncols();

  assert(C.nrows() == rows_A);
  assert(C.ncols() == cols_B);
  assert(cols_A == rows_B);

  int lda           = A.stride();
  int ldb           = B.stride();
  int ldc           = C.stride();
  P alpha_          = alpha;
  P beta_           = beta;
  char const transa = trans_A ? 't' : 'n';
  char const transb = trans_B ? 't' : 'n';
  int m             = rows_A;
  int n             = cols_B;
  int k             = rows_B;

  lib_dispatch::gemm(&transa, &transb, &m, &n, &k, &alpha_, A.data(), &lda,
                     B.data(), &ldb, &beta_, C.data(), &ldc, resrc);

  return C;
}

// gesv - Solve Ax=B using LU decomposition
// template void gesv( int* n, int* nrhs, float* A, int* lda, int* ipiv,
//                    float* b, int* ldb, int* info );
template<typename P, mem_type amem, mem_type bmem>
void gesv(fk::matrix<P, amem> const &A, fk::vector<P, bmem> &B,
          std::vector<int> &ipiv)
{
  int rows_A = A.nrows();
  int cols_A = A.ncols();

  int rows_B = B.size();
  int cols_B = 1;

  int rows_ipiv = ipiv.size();
  assert(cols_A == rows_B);
  assert(rows_ipiv == rows_A);

  int lda = A.stride();
  int ldb = B.size();

  int info;
  lib_dispatch::gesv(&rows_A, &cols_B, A.data(), &lda, ipiv.data(), B.data(),
                     &ldb, &info);
  if (info > 0)
  {
    printf("The diagonal element of the triangular factor of A,\n");
    printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
    printf("the solution could not be computed.\n");
    exit(1);
  }
}

// getrs - Solve Ax=B using LU factorization
// A is assumed to have already beem factored using a
// previous call to gesv() or getrf() where ipiv is
// computed.
// void getrs(char *trans, int *n, int *nrhs, double *A,
//            int *lda, int *ipiv, double *b, int *ldb,
//            int *info);
//
template<typename P, mem_type amem, mem_type bmem>
void getrs(fk::matrix<P, amem> const &A, fk::vector<P, bmem> &B,
           std::vector<int> &ipiv)
{
  int rows_A = A.nrows();
  int cols_A = A.ncols();

  int rows_B = B.size();
  int cols_B = 1;

  int rows_ipiv = ipiv.size();
  assert(cols_A == rows_B);
  assert(rows_ipiv == rows_A);

  char trans = 'N';
  int lda    = A.stride();
  int ldb    = B.size();

  int info;
  lib_dispatch::getrs(&trans, &rows_A, &cols_B, A.data(), &lda, ipiv.data(),
                      B.data(), &ldb, &info);
  if (info < 0)
  {
    printf("Argument %d in call to getrs() has an illegal value\n", -info);
    exit(1);
  }
}

} // namespace fm
