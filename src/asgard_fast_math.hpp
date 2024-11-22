#pragma once

#include "asgard_sparse.hpp"

namespace asgard::fm
{
//! computes 2^exponent using bit-shift operations, only for int-like types
template<typename T>
inline constexpr T ipow2(T const exponent)
{
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, unsigned> ||
                std::is_same_v<T, long> || std::is_same_v<T, unsigned long> ||
                std::is_same_v<T, long long> ||
                std::is_same_v<T, unsigned long long>);
  expect(exponent >= 0);
  expect(exponent < std::numeric_limits<T>::digits);
  return T{1} << exponent;
}

template<typename T = int64_t>
inline constexpr T ipow(T base, int exponent)
{
  expect(exponent >= 1);
  T result = base;
  for (int e = 1; e < exponent; e++)
    result *= base;
  return result;
}

// computes std::floor( std::log2(x) ), returns 0 for x = 0
// uses only bit-wise and integer operations
inline int intlog2(int x)
{
  int result = 0;
  while (x >>= 1)
    result++;
  return result;
}
// computes std::pow( 2, std::floor( std::log2(x) ) )
// using only integer bit-shifts
inline int int2_raised_to_log2(int x)
{
  int result = 1;
  while (x >>= 1)
    result <<= 1;
  return result;
}
// two outputs in one operation
// computes int2_raised_to_log2(i) and std::pow(std::sqrt(2.0), intlog2(i))
inline void intlog2_pow2pows2(int x, int &i2l2, double &is2l2)
{
  i2l2  = 1;
  is2l2 = 1.0;
  while (x >>= 1)
  {
    i2l2 <<= 1;
    is2l2 *= 1.41421356237309505; // sqrt(2.0)
  }
}
// computes std::pow( 0.5, std::floor( std::log2(x) ) )
// uses faster logic
inline double half_raided_to_intlog2(int x)
{
  double result = 1.0;
  while (x >>= 1)
    result *= 0.5;
  return result;
}

template<typename vec_type>
auto nrminf(vec_type const &x)
{
  return std::abs(
      *std::max_element(x.begin(), x.end(), [](auto a, auto b) {
        return (std::abs(a) < std::abs(b));
      }));
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

template<typename P>
P nrm2(std::vector<P> const &x)
{
  if (x.empty())
    return 0.0;
  return lib_dispatch::nrm2<resource::host>(x.size(), x.data(), 1);
}

/*!
 * \brief Comptues the l-inf norm of the difference between x and y
 *
 * This works with all std::vector, std::array and fk::vector.
 * Does not work with GPU vectors and does not check if the data is on the device.
 */
template<typename vecx, typename vecy>
auto diff_inf(vecx const &x, vecy const &y)
{
  using precision = typename vecx::value_type;
  using index     = decltype(x.size());
  expect(x.size() == static_cast<index>(y.size()));

  precision m{0};
  for (index i = index{0}; i < x.size(); i++)
    m = std::max(m, std::abs(x[i] - y[i]));
  return m;
}

/*!
 * \brief Comptues the root-mean-square-error between two vectors
 *
 * This works with all std::vector, std::array and fk::vector.
 * Does not work with GPU vectors and does not check if the data is on the device.
 */
template<typename vecx, typename vecy>
auto rmserr(vecx const &x, vecy const &y)
{
  using precision = typename vecx::value_type;
  using index     = decltype(x.size());
  expect(x.size() == y.size());

  precision err{0};
  for (index i = index{0}; i < x.size(); i++)
  {
    precision const d = x[i] - y[i];
    err += d * d;
  }
  return std::sqrt(err / x.size());
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

  int lda = A.stride();
  int ldb = B.stride();
  int ldc = C.stride();

  char const transa = trans_A ? 't' : 'n';
  char const transb = trans_B ? 't' : 'n';

  lib_dispatch::gemm<resrc>(transa, transb, rows_A, cols_B, rows_B, alpha, A.data(), lda,
                            B.data(), ldb, beta, C.data(), ldc);

  return C;
}

// gemm - shortcut giving alpha/beta before the matrices and skips the transposes
template<typename P, mem_type amem, mem_type bmem, mem_type cmem,
         resource resrc>
fk::matrix<P, cmem, resrc> &
gemm(P const alpha, fk::matrix<P, amem, resrc> const &A, fk::matrix<P, bmem, resrc> const &B,
     P const beta, fk::matrix<P, cmem, resrc> &C)
{
  return gemm(A, B, C, false, false, alpha, beta);
}

/** gesv - Solve Ax=B using LU decomposition
 *
 * \param A  n-by-n coefficient matrix
 * \param B  n-by-1 right hand side matrix
 * \param ipiv pivot indices, size >= max(1, n)
 *
 * (modified to allow for the use of std::vector in place of B)
 */
template<typename P, mem_type amem, typename vec_type>
void gesv(fk::matrix<P, amem> &A, vec_type &B,
          std::vector<int> &ipiv)
{
  static_assert(amem != mem_type::const_view,
                "cannot factorize a const-view of a matrix");

  static_assert(std::is_same_v<typename vec_type::value_type, P>,
                "precision mismatch between the vectors");

  int rows_A = A.nrows();
  int cols_A = A.ncols();

  int rows_B = static_cast<int>(B.size());
  int cols_B = 1;

  int rows_ipiv = ipiv.size();
  expect(cols_A == rows_B);
  expect(rows_ipiv >= rows_A);

  int lda = A.stride();
  int ldb = static_cast<int>(B.size());

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

// getrf - Computes the LU factorization of a general m-by-n matrix.
template<typename P, mem_type amem>
void getrf(fk::matrix<P, amem> &A, std::vector<int> &ipiv)
{
  int rows_ipiv = ipiv.size();
  expect(rows_ipiv == std::max(1, std::min(A.nrows(), A.ncols())));
  int info = lib_dispatch::getrf(A.nrows(), A.ncols(), A.data(), A.stride(),
                                 ipiv.data());
  if (info != 0)
  {
    std::stringstream sout;
    if (info < 0)
    {
      sout << "The " << -info << "-th parameter had an illegal value!\n";
    }
    else
    {
      sout << "The diagonal element of the triangular factor of A,\n";
      sout << "U(" << info << ',' << info
           << ") is zero, so that A is singular;\n";
      sout << "the solution could not be computed.\n";
    }
    throw std::runtime_error(sout.str());
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
template<typename P, mem_type amem, typename vec_type>
void getrs(fk::matrix<P, amem> const &A, vec_type &B,
           std::vector<int> const &ipiv)
{
  int rows_A = A.nrows();
  int cols_A = A.ncols();

  int rows_B = static_cast<int>(B.size());
  int cols_B = 1;

  int rows_ipiv = ipiv.size();
  expect(cols_A == rows_B);
  expect(rows_ipiv == rows_A);

  char trans = 'N';
  int lda    = A.stride();
  int ldb    = static_cast<int>(B.size());

  int info = lib_dispatch::getrs(trans, rows_A, cols_B, A.data(), lda,
                                 ipiv.data(), B.data(), ldb);
  if (info < 0)
  {
    printf("Argument %d in call to getrs() has an illegal value\n", -info);
    exit(1);
  }
  expect(info == 0);
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

} // namespace asgard::fm
