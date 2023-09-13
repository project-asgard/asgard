#pragma once
#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "lib_dispatch.hpp"
#include "program_options.hpp"
#include "sparse.hpp"
#include "tools.hpp"
#ifdef ASGARD_USE_SCALAPACK
#include "scalapack_vector_info.hpp"
#endif
#include <numeric>

namespace asgard::fm
{
// a non-matlab one-liner that had no better home - compute 2^arg
template<typename T>
inline constexpr T two_raised_to(T const exponent)
{
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, unsigned> ||
                std::is_same_v<T, long> || std::is_same_v<T, unsigned long> ||
                std::is_same_v<T, long long> ||
                std::is_same_v<T, unsigned long long>);
  expect(exponent >= 0);
  expect(exponent < std::numeric_limits<T>::digits);
  return T{1} << exponent;
}

template<typename P, mem_type mem, resource resrc>
P nrm2(fk::vector<P, mem, resrc> const &x)
{
#ifndef ASGARD_USE_CUDA
  static_assert(resrc == resource::host);
#endif
  if (x.empty())
    return 0.0;
  return lib_dispatch::nrm2<resrc>(x.size(), x.data(), 1);
}

/* Frobenius norm of owner matrix */
template<typename P, resource resrc>
P frobenius(fk::matrix<P, mem_type::owner, resrc> const &m)
{
  if (m.empty())
  {
    return 0.0;
  }

  else if constexpr (std::is_floating_point_v<P>)
  {
    return lib_dispatch::nrm2<resrc>(m.size(), m.data(), 1);
  }

  /* create a view of the matrix and pass it to the non-owner overload of the
   * function */
  else
  {
    fk::matrix<P, mem_type::const_view, resrc> const m_view(m);
    frobenius(m_view);
  }
}

/* with matrix views, contiguous raw data cannot be assumed - calculate manually
 */
template<typename P, mem_type mem, resource resrc, mem_type m_ = mem,
         typename = enable_for_all_views<m_>>
P frobenius(fk::matrix<P, mem, resrc> const &m)
{
  if (m.empty())
  {
    return 0.0;
  }

  /* if the matrix is on the device, copy it to host */
  else if constexpr (resrc == resource::device)
  {
    fk::matrix<P, mem_type::owner, resource::host> m_host = m.clone_onto_host();

    return std::sqrt(std::accumulate(m_host.begin(), m_host.end(), 0,
                                     [](P const sum_of_squares, P const value) {
                                       return sum_of_squares + value * value;
                                     }));
  }

  else if constexpr (resrc == resource::host)
  {
    return std::sqrt(std::accumulate(m.begin(), m.end(), 0,
                                     [](P const sum_of_squares, P const value) {
                                       return sum_of_squares + value * value;
                                     }));
  }
}

// axpy - y += a*x
template<typename P, mem_type mem, mem_type omem, resource resrc>
fk::vector<P, mem, resrc> &
axpy(fk::vector<P, omem, resrc> const &x, fk::vector<P, mem, resrc> &y,
     P const alpha = 1.0)
{
  expect(x.size() == y.size());
  int n    = x.size();
  int one  = 1;
  P alpha_ = alpha;
  lib_dispatch::axpy<resrc>(n, alpha_, x.data(), one, y.data(), one);
  return y;
}

// copy(x,y) - copy vector x into y
template<typename P, mem_type mem, mem_type omem, resource resrc>
fk::vector<P, mem, resrc> &
copy(fk::vector<P, omem, resrc> const &x, fk::vector<P, mem, resrc> &y)
{
  expect(y.size() >= x.size());
  int64_t n = x.size();
  lib_dispatch::copy<resrc>(n, x.data(), y.data());
  return y;
}

// scal - scale a vector
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc> &scal(P const alpha, fk::vector<P, mem, resrc> &x)
{
  int one  = 1;
  int n    = x.size();
  P alpha_ = alpha;
  lib_dispatch::scal<resrc>(n, alpha_, x.data(), one);
  return x;
}

// scal - scale a matrix
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc> &scal(P const alpha, fk::matrix<P, mem, resrc> &x)
{
  int one  = 1;
  int n    = x.size();
  P alpha_ = alpha;
  lib_dispatch::scal<resrc>(n, alpha_, x.data(), one);
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

  expect(rows_A == y.size());
  expect(cols_A == x.size());

  int lda           = A.stride();
  int one           = 1;
  P alpha_          = alpha;
  P beta_           = beta;
  char const transa = trans_A ? 't' : 'n';
  int m             = A.nrows();
  int n             = A.ncols();

  lib_dispatch::gemv<resrc>(transa, m, n, alpha_, A.data(), lda, x.data(), one,
                            beta_, y.data(), one);

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

  expect(C.nrows() == rows_A);
  expect(C.ncols() == cols_B);
  expect(cols_A == rows_B);

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

  lib_dispatch::gemm<resrc>(transa, transb, m, n, k, alpha_, A.data(), lda,
                            B.data(), ldb, beta_, C.data(), ldc);

  return C;
}

/** gesv - Solve Ax=B using LU decomposition
 *
 * \param A  n-by-n coefficient matrix
 * \param B  n-by-1 right hand side matrix
 * \param ipiv pivot indices, size >= max(1, n)
 */
template<typename P, mem_type amem, mem_type bmem>
void gesv(fk::matrix<P, amem> &A, fk::vector<P, bmem> &B,
          std::vector<int> &ipiv)
{
  static_assert(amem != mem_type::const_view,
                "cannot factorize a const-view of a matrix");

  int rows_A = A.nrows();
  int cols_A = A.ncols();

  int rows_B = B.size();
  int cols_B = 1;

  int rows_ipiv = ipiv.size();
  expect(cols_A == rows_B);
  expect(rows_ipiv >= rows_A);

  int lda = A.stride();
  int ldb = B.size();

  int info = lib_dispatch::gesv(rows_A, cols_B, A.data(), lda, ipiv.data(),
                                B.data(), ldb);
  if (info < 0)
  {
    throw std::runtime_error(
        std::string("Argument " + std::to_string(info) +
                    " in call to gesv() has an illegal value\n"));
  }
  else if (info > 0)
  {
    std::ostringstream msg;
    msg << "The diagonal element of the triangular factor of A,\n";
    msg << "U(" << info << "," << info << ") is zero, so that A is singular;\n";
    msg << "the solution could not be computed.\n";
    throw std::runtime_error(msg.str());
  }
}

/** gesv - Solve Ax=B using LU decomposition
 *
 * \param A  n-by-n coefficient matrix
 * \param B  n-by-nrhs right hand side matrix
 * \param ipiv pivot indices, size >= max(1, n)
 */
template<typename P, mem_type amem, mem_type bmem>
void gesv(fk::matrix<P, amem> &A, fk::matrix<P, bmem> &B,
          std::vector<int> &ipiv)
{
  int rows_A = A.nrows();
  int cols_A = A.ncols();

  int rows_B = B.nrows();
  int cols_B = B.ncols();

  int rows_ipiv = ipiv.size();
  expect(cols_A == rows_B);
  expect(rows_ipiv >= rows_A);

  int lda = A.stride();
  int ldb = B.stride();

  int info = lib_dispatch::gesv(rows_A, cols_B, A.data(), lda, ipiv.data(),
                                B.data(), ldb);
  if (info < 0)
  {
    throw std::runtime_error(
        std::string("Argument " + std::to_string(info) +
                    " in call to gesv() has an illegal value\n"));
  }
  else if (info > 0)
  {
    std::ostringstream msg;
    msg << "The diagonal element of the triangular factor of A,\n";
    msg << "U(" << info << "," << info << ") is zero, so that A is singular;\n";
    msg << "the solution could not be computed.\n";
    throw std::runtime_error(msg.str());
  }
}

/** tpsv - Solve Ax=B using LU decomposition
 *
 * \param A  n-by-n coefficient matrix
 * \param B  n-by-1 right hand side matrix
 * \param uplo whether the matrix A is upper or lower triangular
 * \param trans whether matrix A is transposed
 * \param diag whether the matrix A is unit triangular
 */
template<typename P, mem_type amem, mem_type bmem, resource resrc>
void tpsv(fk::vector<P, amem, resrc> const &A, fk::vector<P, bmem, resrc> &B,
          char uplo = 'U', char trans = 'N', char diag = 'N')
{
  int rows_B = B.size();
  expect(A.size() == rows_B * (rows_B + 1) / 2);

  lib_dispatch::tpsv<resrc>(uplo, trans, diag, rows_B, A.data(), B.data(), 1);
}

#ifdef ASGARD_USE_SCALAPACK
// gesv - Solve Ax=B using LU decomposition
// template void gesv( int* n, int* nrhs, float* A, int* lda, int* ipiv,
//                    float* b, int* ldb, int* info );
template<typename P, mem_type amem, mem_type bmem>
void gesv(fk::matrix<P, amem> &A, fk::scalapack_matrix_info &ainfo,
          fk::vector<P, bmem> &B, fk::scalapack_vector_info &binfo,
          std::vector<int> &ipiv)
{
  expect(ainfo.local_rows() == A.nrows());
  expect(ainfo.local_cols() == A.ncols());
  expect(binfo.local_size() == B.size());

  int rows_ipiv = ipiv.size();
  expect(rows_ipiv == ainfo.local_rows() + ainfo.mb());

  int rows_A = ainfo.nrows();
  int cols_B = 1;
  int info;
  lib_dispatch::scalapack_gesv(&rows_A, &cols_B, A.data(), ainfo.get_desc(),
                               ipiv.data(), B.data(), binfo.get_desc(), &info);
  if (info > 0)
  {
    std::cout << "The diagonal element of the triangular factor of A,\n";
    std::cout << "U(" << info << "," << info
              << ") is zero, so that A is singular;\n";
    std::cout << "the solution could not be computed.\n";
    exit(1);
  }
}
#endif

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
  expect(cols_A == rows_B);
  expect(rows_ipiv == rows_A);

  char trans = 'N';
  int lda    = A.stride();
  int ldb    = B.size();

  int info = lib_dispatch::getrs(trans, rows_A, cols_B, A.data(), lda,
                                 ipiv.data(), B.data(), ldb);
  if (info < 0)
  {
    printf("Argument %d in call to getrs() has an illegal value\n", -info);
    exit(1);
  }
}

/** pttrf - computes the L*D*L**T factorization of a real symmetric positive
 * definite tridiagonal matrix A.
 * \param D diagonal entries of tridiagonal matrix A. On exit, n diagonal elements D from the L*D*L**T factorization of A.
 * \param E the (n-1) subdiagonal elements of matrix A. On exit, subdiagonal elements of the unit bidiagonal factor L from the L*D*L**T factorization of A.
 */
template<typename P, mem_type dmem, mem_type emem>
void pttrf(fk::vector<P, dmem> &D, fk::vector<P, emem> &E)
{
  int N = D.size();

  expect(N >= 0);
  expect(E.size() == N - 1);

  int info = lib_dispatch::pttrf(N, D.data(), E.data());
  if (info < 0)
  {
    throw std::runtime_error(
        std::string("Argument " + std::to_string(info) +
                    " in call to pttrf() has an illegal value\n"));
  }
}

/** pttrs - solves a tridiagonal system of the form A * X = B using the L*D*L**T
 * factoration of A computed by pttrf.
 * \param D diagonal elements of the diagonal matrix D from the L*D*L**T factorization of A
 * \param E subdiagonal (n-1) elements of the unit bidiagonal factor L from the L*D*L**T factorization of A
 * \param B RHS vectors B for the system of linear equations. On exit, the solution vectors, X.
 */
template<typename P, mem_type dmem, mem_type emem, mem_type bmem>
void pttrs(fk::vector<P, dmem> const &D, fk::vector<P, emem> const &E,
           fk::matrix<P, bmem> &B)
{
  int N    = D.size();
  int nrhs = B.ncols();
  int ldb  = B.stride();

  expect(N >= 0);
  expect(nrhs >= 0);
  expect(E.size() == N - 1);

  int info = lib_dispatch::pttrs(N, nrhs, D.data(), E.data(), B.data(), ldb);
  if (info < 0)
  {
    throw std::runtime_error(
        std::string("Argument " + std::to_string(info) +
                    " in call to pttrs() has an illegal value\n"));
  }
}

/** pttrs - solves a tridiagonal system of the form A * X = B using the L*D*L**T
 * factoration of A computed by pttrf. Overload with B as a vector and NRHS = 1.
 * \param D diagonal elements of the diagonal matrix D from the L*D*L**T factorization of A
 * \param E subdiagonal (n-1) elements of the unit bidiagonal factor L from the L*D*L**T factorization of A
 * \param B RHS vectors B for the system of linear equations. On exit, the solution vectors, X.
 */
template<typename P, mem_type dmem, mem_type emem, mem_type bmem>
void pttrs(fk::vector<P, dmem> const &D, fk::vector<P, emem> const &E,
           fk::vector<P, bmem> &B)
{
  int N    = D.size();
  int nrhs = 1;
  int ldb  = B.size();

  expect(N >= 0);
  expect(nrhs >= 0);
  expect(E.size() == N - 1);
  expect(ldb == N);

  int info = lib_dispatch::pttrs(N, nrhs, D.data(), E.data(), B.data(), ldb);
  if (info < 0)
  {
    throw std::runtime_error(
        std::string("Argument " + std::to_string(info) +
                    " in call to pttrs() has an illegal value\n"));
  }
}

#ifdef ASGARD_USE_SCALAPACK
// getrs - Solve Ax=B using LU factorization
// A is assumed to have already beem factored using a
// previous call to gesv() or getrf() where ipiv is
// computed.
// void getrs(char *trans, int *n, int *nrhs, double *A,
//            int *lda, int *ipiv, double *b, int *ldb,
//            int *info);
//
template<typename P, mem_type amem, mem_type bmem>
void getrs(fk::matrix<P, amem> const &A, fk::scalapack_matrix_info &ainfo,
           fk::vector<P, bmem> &B, fk::scalapack_vector_info &binfo,
           std::vector<int> &ipiv)
{
  expect(ainfo.local_rows() == A.nrows());
  expect(ainfo.local_cols() == A.ncols());
  expect(binfo.local_size() == B.size());

  int rows_ipiv = ipiv.size();
  expect(rows_ipiv == ainfo.local_rows() + ainfo.mb());
  int rows_A = ainfo.nrows();
  int cols_B = 1;
  char trans = 'N';
  int info;
  lib_dispatch::scalapack_getrs(&trans, &rows_A, &cols_B, A.data(),
                                ainfo.get_desc(), ipiv.data(), B.data(),
                                binfo.get_desc(), &info);
  if (info < 0)
  {
    printf("Argument %d in call to getrs() has an illegal value\n", -info);
    exit(1);
  }
}
#endif

// sparse gemv - sparse matrix dense vector multiplication
template<typename P, mem_type xmem, mem_type ymem, resource resrc>
fk::vector<P, ymem, resrc> &
sparse_gemv(fk::sparse<P, resrc> const &A, fk::vector<P, xmem, resrc> const &x,
            fk::vector<P, ymem, resrc> &y, bool const trans_A = false,
            P const alpha = 1.0, P const beta = 0.0)
{
  int const rows_opA = trans_A ? A.ncols() : A.nrows();
  int const cols_A   = trans_A ? A.nrows() : A.ncols();

  expect(rows_opA == y.size());
  expect(cols_A == x.size());

  char const transa = trans_A ? 't' : 'n';
  lib_dispatch::sparse_gemv<resrc>(transa, A.nrows(), A.ncols(), A.nnz(),
                                   A.offsets(), A.columns(), A.data(), alpha,
                                   x.data(), beta, y.data());

  return y;
}

template<typename P, mem_type mem, resource resrc>
void ilu(fk::sparse<P, mem, resrc> const &S, fk::matrix<P, mem, resrc> &A)
{
  auto tmp = S.to_dense();
  expect(tmp.nrows() >= A.nrows());
  expect(tmp.ncols() >= A.ncols());

  A = tmp.extract_submatrix(0, 0, A.nrows(), A.ncols());

  int rows_A = A.nrows();
  int cols_A = A.ncols();
  int lda    = A.stride();

  lib_dispatch::sp_ilu(rows_A, cols_A, A.data(), lda, S.data(), S.offsets(),
                       S.columns(), S.nnz());
}

} // namespace asgard::fm
