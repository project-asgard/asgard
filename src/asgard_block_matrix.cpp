#include "asgard_block_matrix.hpp"
#include "asgard_small_mats.hpp"

namespace asgard
{

template<typename P>
void dense_matrix<P>::factorize()
{
  tools::time_event timing_("dense-matrix::factorize");
  expect(nrows_ == ncols_);
  ipiv.resize(nrows_);
  int info = lib_dispatch::getrf(nrows_, ncols_, data_.data(), nrows_,
                                 ipiv.data());

  if (info != 0)
  {
    std::stringstream sout;
    if (info < 0)
    {
      sout << "getrf(): the " << -info << "-th parameter had an illegal value!\n";
    }
    else
    {
      sout << "getrf(): the diagonal element of the triangular factor of A,\n";
      sout << "U(" << info << ',' << info << ") is zero, so that A is singular;\n";
      sout << "the matrix could not be factorized.\n";
    }
    throw std::runtime_error(sout.str());
  }
}

template<typename P>
void dense_matrix<P>::solve(std::vector<P> &b) const
{
  tools::time_event timing_("dense-matrix::solve");
  expect(is_factorized());
  int info = lib_dispatch::getrs('N', nrows_, 1, data_.data(), nrows_,
                                  ipiv.data(), b.data(), nrows_);
  expect(info == 0);
}

template<typename P>
void gemm1(int const n, block_matrix<P> const &A, block_matrix<P> const &B, block_matrix<P> &C)
{
  int M = A.nrows();
  int N = B.ncols();
  int K = A.ncols();

  expect(C.nrows() == M);
  expect(C.ncols() == N);
  expect(B.nrows() == K);

  expect(A.nblock() == n * n);
  expect(B.nblock() == n * n);
  expect(C.nblock() == n * n);

#pragma omp parallel for
  for (int c = 0; c < N; c++) {
    for (int r = 0; r < M; r++) {
      for (int k = 0; k < K; k++)
        smmat::gemm<1>(n, A(r, k), B(k, c), C(r, c));
    }
  }
}

template<typename P>
void block_diag_matrix<P>::spd_factorize(int const n)
{
  expect(n * n == nblock());
  switch (n)
  {
  case 1:
    ASGARD_OMP_PARFOR_SIMD
    for (int64_t r = 0; r < nrows_; r++)
      data_[r][0] = P{1} / data_[r][0];
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
      smmat::inv2by2(data_[r]);
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
      smmat::potrf(n, data_[r]);
    break;
  }
}

template<typename P>
void block_diag_matrix<P>::solve(int const n, P rhs[]) const
{
  expect(n * n == nblock());
  switch (n)
  {
  case 1:
    ASGARD_OMP_PARFOR_SIMD
    for (int64_t r = 0; r < nrows_; r++)
      rhs[r] *= data_[r][0];
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
      smmat::gemv2by2(data_[r], rhs + 2 * r);
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
      smmat::posv(n, data_[r], rhs + n * r);
    break;
  }
}

template<typename P>
void block_diag_matrix<P>::solve(int const n, block_diag_matrix<P> &rhs) const
{
  switch (n)
  {
  case 1:
    ASGARD_OMP_PARFOR_SIMD
    for (int64_t r = 0; r < nrows_; r++)
      rhs[r][0] *= data_[r][0];
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
      smmat::gemm2by2(data_[r], rhs[r]);
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
      smmat::posvm(n, data_[r], rhs[r]);
    break;
  }
}

template<typename P>
void block_diag_matrix<P>::solve(int const n, block_tri_matrix<P> &rhs) const
{
  switch (n)
  {
  case 1:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
    {
      rhs.lower(r)[0] *= data_[r][0];
      rhs.diag(r)[0] *= data_[r][0];
      rhs.upper(r)[0] *= data_[r][0];
    }
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
    {
      smmat::gemm2by2(data_[r], rhs.lower(r));
      smmat::gemm2by2(data_[r], rhs.diag(r));
      smmat::gemm2by2(data_[r], rhs.upper(r));
    }
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows_; r++)
    {
      smmat::posvm(n, data_[r], rhs.lower(r));
      smmat::posvm(n, data_[r], rhs.diag(r));
      smmat::posvm(n, data_[r], rhs.upper(r));
    }
    break;
  }
}

template<typename P>
void block_diag_matrix<P>::inplace_gemv(int n, std::vector<P> &vec, std::vector<P> &work) const
{
  expect(nblock() == n * n);
  expect(vec.size() == static_cast<size_t>(n * nrows_));
  if (work.size() < vec.size())
    work.resize(vec.size());

  std::copy(vec.begin(), vec.end(), work.begin());

  span2d<P> x(n, nrows_, work.data());
  span2d<P> y(n, nrows_, vec.data());

#pragma omp parallel for
  for (int64_t r = 1; r < nrows_ - 1; r++) {
    smmat::gemv(n, n, data_[r], x[r], y[r]);
  }
}

template<typename P>
void block_tri_matrix<P>::inplace_gemv(int n, std::vector<P> &vec, std::vector<P> &work) const
{
  expect(nblock() == n * n);
  expect(vec.size() == static_cast<size_t>(n * nrows_));
  if (work.size() < vec.size())
    work.resize(vec.size());

  std::copy(vec.begin(), vec.end(), work.begin());

  span2d<P> x(n, nrows_, work.data());
  span2d<P> y(n, nrows_, vec.data());

  if (nrows_ == 1) {
    smmat::gemv(n, n, diag(0), x[0], y[0]);
    return;
  }

  int const s = nrows_ - 1; // stop row

  smmat::gemv(n, n, diag(0), x[0], y[0]);
  smmat::gemv1(n, n, lower(0), x[s], y[0]);
  smmat::gemv1(n, n, upper(0), x[1], y[0]);

#pragma omp parallel for
  for (int64_t r = 1; r < nrows_ - 1; r++) {
    smmat::gemv(n, n, diag(r), x[r], y[r]);
    smmat::gemv1(n, n, lower(r), x[r - 1], y[r]);
    smmat::gemv1(n, n, upper(r), x[r + 1], y[r]);
  }

  smmat::gemv(n, n, diag(s), x[s], y[s]);
  smmat::gemv1(n, n, lower(s), x[s - 1], y[s]);
  smmat::gemv1(n, n, upper(s), x[0], y[s]);
}

template<typename P>
void gemm_block_tri_ul(
    int const n, block_tri_matrix<P> const &A, block_tri_matrix<P> const &B,
    block_tri_matrix<P> &C)
{
  int64_t const M = A.nrows();
  expect(A.nblock() == n * n);
  expect(A.nblock() == B.nblock());
  expect(A.nblock() == C.nblock());
  expect(B.nrows() == M);
  expect(C.nrows() == M);

  // lower(r) -> (r, r - 1), diag(r) -> (r, r), upper(r) -> (r, r + 1)
  // lower(0) -> (0, n - 1), upper(n - 1) -> (n - 1, 0)
  // c_i,j = sum_k a_i,k * b_k,j

  smmat::gemm<0>(n, A.diag(0), B.lower(0), C.lower(0));

  smmat::gemm<0>(n, A.diag(0), B.diag(0), C.diag(0));
  smmat::gemm<1>(n, A.upper(0), B.lower(1), C.diag(0));

  smmat::gemm<0>(n, A.upper(0), B.diag(1), C.upper(0));

#pragma omp parallel for
  for (int64_t r = 1; r < M - 1; r++)
  {
    smmat::gemm<0>(n, A.diag(r), B.lower(r), C.lower(r));

    smmat::gemm<0>(n, A.diag(r), B.diag(r), C.diag(r));
    smmat::gemm<1>(n, A.upper(r), B.lower(r + 1), C.diag(r));

    smmat::gemm<0>(n, A.diag(r), B.upper(r), C.upper(r));
    smmat::gemm<1>(n, A.upper(r), B.diag(r + 1), C.upper(r));
  }

  smmat::gemm<0>(n, A.upper(M - 1), B.diag(0), C.upper(M - 1));

  smmat::gemm<0>(n, A.diag(M - 1), B.lower(M - 1), C.lower(M - 1));

  smmat::gemm<0>(n, A.diag(M - 1), B.diag(M - 1), C.diag(M - 1));
  smmat::gemm<1>(n, A.upper(M - 1), B.lower(0), C.diag(M - 1));
}

template<typename P>
void gemm_block_tri_lu(
    int const n, block_tri_matrix<P> const &A, block_tri_matrix<P> const &B,
    block_tri_matrix<P> &C)
{
  int const M = A.nrows();
  expect(A.nblock() == B.nblock());
  expect(A.nblock() == C.nblock());
  expect(A.nblock() == n * n);
  expect(B.nrows() == M);
  expect(C.nrows() == M);

  smmat::gemm<0>(n, A.lower(0), B.diag(M - 1), C.lower(0));
  smmat::gemm<0>(n, A.lower(0), B.upper(M - 1), C.diag(0));
  smmat::gemm<1>(n, A.diag(0), B.diag(0), C.diag(0));
  smmat::gemm<0>(n, A.diag(0), B.upper(0), C.upper(0));

#pragma omp parallel for
  for (int64_t r = 1; r < M - 1; r++)
  {
    smmat::gemm<0>(n, A.lower(r), B.diag(r - 1), C.lower(r));
    smmat::gemm<0>(n, A.lower(r), B.upper(r - 1), C.diag(r));
    smmat::gemm<1>(n, A.diag(r), B.diag(r), C.diag(r));
    smmat::gemm<0>(n, A.diag(r), B.upper(r), C.upper(r));
  }

  smmat::gemm<0>(n, A.lower(M - 1), B.diag(M - 2), C.lower(M - 1));
  smmat::gemm<0>(n, A.lower(M - 1), B.upper(M - 2), C.diag(M - 1));
  smmat::gemm<1>(n, A.diag(M - 1), B.diag(M - 1), C.diag(M - 1));
  smmat::gemm<0>(n, A.diag(M - 1), B.upper(M - 1), C.upper(M - 1));
}

template<typename P>
void gemm_block_tri(int const n, block_tri_matrix<P> const &A, block_tri_matrix<P> const &B,
                    block_tri_matrix<P> &C)
{
  int const M = A.nrows();
  expect(M >= 1);
  expect(A.nblock() == B.nblock());
  expect(A.nblock() == C.nblock());
  expect(A.nblock() == n * n);
  expect(B.nrows() == M);
  expect(C.nrows() == M);

  if (M == 1) {
    smmat::gemm<0>(n, A.diag(0), B.diag(0), C.diag(0));
    return;
  }

  smmat::gemm<0>(n, A.diag(0), B.lower(0), C.lower(0));
  smmat::gemm<1>(n, A.lower(0), B.diag(M - 1), C.lower(0));

  smmat::gemm<0>(n, A.lower(0), B.upper(M - 1), C.diag(0));
  smmat::gemm<1>(n, A.diag(0), B.diag(0), C.diag(0));
  smmat::gemm<1>(n, A.upper(0), B.lower(1), C.diag(0));

  smmat::gemm<0>(n, A.diag(0), B.upper(0), C.upper(0));
  smmat::gemm<1>(n, A.upper(0), B.diag(1), C.upper(0));

#pragma omp parallel for
  for (int64_t r = 1; r < M - 1; r++)
  {
    smmat::gemm<0>(n, A.diag(r), B.lower(r), C.lower(r));
    smmat::gemm<1>(n, A.lower(r), B.diag(r - 1), C.lower(r));

    smmat::gemm<0>(n, A.lower(r), B.upper(r - 1), C.diag(r));
    smmat::gemm<1>(n, A.diag(r), B.diag(r), C.diag(r));
    smmat::gemm<1>(n, A.upper(r), B.lower(r + 1), C.diag(r));

    smmat::gemm<0>(n, A.diag(r), B.upper(r), C.upper(r));
    smmat::gemm<1>(n, A.upper(r), B.diag(r + 1), C.upper(r));
  }

  smmat::gemm<0>(n, A.lower(M - 1), B.diag(M - 2), C.lower(M - 1));
  smmat::gemm<1>(n, A.diag(M - 1), B.lower(M - 1), C.lower(M - 1));

  smmat::gemm<0>(n, A.lower(M - 1), B.upper(M - 2), C.diag(M - 1));
  smmat::gemm<1>(n, A.diag(M - 1), B.diag(M - 1), C.diag(M - 1));
  smmat::gemm<1>(n, A.upper(M - 1), B.lower(0), C.diag(M - 1));

  smmat::gemm<0>(n, A.diag(M - 1), B.upper(M - 1), C.upper(M - 1));
  smmat::gemm<1>(n, A.upper(M - 1), B.diag(0), C.upper(M - 1));
}

template<typename P>
void gemm_diag_tri(
    int const n, block_diag_matrix<P> const &A, block_tri_matrix<P> const &B,
    block_tri_matrix<P> &C)
{
  int64_t const M = A.nrows();
  expect(A.nblock() == n * n);
  expect(A.nblock() == B.nblock());
  expect(A.nblock() == C.nblock());
  expect(B.nrows() == M);
  expect(C.nrows() == M);

#pragma omp parallel for
  for (int64_t r = 0; r < M; r++)
  {
    smmat::gemm<0>(n, A[r], B.lower(r), C.lower(r));
    smmat::gemm<0>(n, A[r], B.diag(r), C.diag(r));
    smmat::gemm<0>(n, A[r], B.upper(r), C.upper(r));
  }
}

template<typename P>
void gemm_tri_diag(
    int const n, block_tri_matrix<P> const &A, block_diag_matrix<P> const &B,
    block_tri_matrix<P> &C)
{
  int64_t const M = A.nrows();
  expect(A.nblock() == n * n);
  expect(A.nblock() == B.nblock());
  expect(A.nblock() == C.nblock());
  expect(B.nrows() == M);
  expect(C.nrows() == M);

  smmat::gemm<0>(n, A.diag(0), B[0], C.diag(0));
  smmat::gemm<0>(n, A.lower(1), B[0], C.lower(1));
  smmat::gemm<0>(n, A.upper(M-1), B[0], C.upper(M-1));

#pragma omp parallel for
  for (int64_t r = 1; r < M - 2; r++)
  {
    smmat::gemm<0>(n, A.upper(r-1), B[r], C.upper(r-1));
    smmat::gemm<0>(n, A.diag(r), B[r], C.diag(r));
    smmat::gemm<0>(n, A.lower(r+1), B[r], C.lower(r+1));
  }

  smmat::gemm<0>(n, A.lower(0), B[M - 1], C.lower(0));
  smmat::gemm<0>(n, A.upper(M - 2), B[M - 1], C.upper(M - 2));
  smmat::gemm<0>(n, A.diag(M - 1), B[M - 1], C.diag(M - 1));
}

template<typename P>
void gemm_block_diag(int const n, block_diag_matrix<P> const &A, block_diag_matrix<P> const &B, block_diag_matrix<P> &C)
{
  int64_t const M = A.nrows();
  expect(A.nblock() == n * n);
  expect(A.nblock() == B.nblock());
  expect(A.nblock() == C.nblock());
  expect(B.nrows() == M);
  expect(C.nrows() == M);

#pragma omp parallel for
  for (int64_t r = 0; r < M; r++)
    smmat::gemm<0>(n, A[r], B[r], C[r]);
}

template<typename P>
void invert_mass(int const n, mass_matrix<P> const &mass, block_tri_matrix<P> &op)
{
  expect(mass.nblock() == op.nblock());
  int64_t nr = op.nrows();
  expect(mass.nrows() == nr);

  switch (n)
  {
  case 1:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
    {
      *op.lower(r) *= *mass[r];
      *op.diag(r) *= *mass[r];
      *op.upper(r) *= *mass[r];
    }
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
    {
      smmat::gemm2by2(mass[r], op.lower(r));
      smmat::gemm2by2(mass[r], op.diag(r));
      smmat::gemm2by2(mass[r], op.upper(r));
    }
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
    {
      smmat::posvm(n, mass[r], op.lower(r));
      smmat::posvm(n, mass[r], op.diag(r));
      smmat::posvm(n, mass[r], op.upper(r));
    }
    break;
  }
}

template<typename P>
void invert_mass(int const n, mass_matrix<P> const &mass, block_diag_matrix<P> &op)
{
  expect(mass.nblock() == op.nblock());
  int64_t const nr = op.nrows();
  expect(mass.nrows() == nr);

  switch (n)
  {
  case 1:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
      *op[r] *= *mass[r];
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
      smmat::gemm2by2(mass[r], op[r]);
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
      smmat::posvm(n, mass[r], op[r]);
    break;
  }
}

template<typename P>
void block_sparse_matrix<P>::gemv(int const n, int const level, connection_patterns const &conns, P const x[], P y[]) const
{
  expect(n * n == nblock());

  connect_1d const &conn = conns(htype_);
  int const nrows        = fm::ipow2(level);

  expect(nrows <= conn.num_rows());

#pragma omp parallel for
  for (int r = 0; r < nrows; r++)
  {
    P *out = y + r * n;
    std::fill_n(out, n, P{0});
    for (int j = conn.row_begin(r); j < conn.row_end(r); j++)
    {
      int const c = conn[j]; // column
      if (c >= nrows)
        break;
      smmat::gemv1(n, n, data_[j], x + c * n, out);
    }
  }
}

template<typename P>
void invert_mass(int const n, mass_matrix<P> const &mass, P x[])
{
  expect(mass.nblock() == n * n);
  int64_t const nr = mass.nrows();

  switch (n)
  {
  case 1:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
      x[r] *= *mass[r];
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
      smmat::gemv2by2(mass[r], &x[2 * r]);
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nr; r++)
      smmat::posv(n, mass[r], &x[r * n]);
    break;
  }
}

template<typename P>
void to_euler(int const n, P alpha, block_diag_matrix<P> &A) {
  expect(A.nblock() == n * n);

  int const nrows = A.nrows();
  int const n2    = n * n;

#pragma omp parallel for
  for (int i = 0; i < nrows; i++) {
    P *r = A[i];
    smmat::scal(n2, alpha, r);
    for (int j = 0; j < n; j++)
      r[j * n + j] += P{1};
  }
}

template<typename P>
void to_euler(int const n, P alpha, block_tri_matrix<P> &A) {
  expect(A.nblock() == n * n);

  int const nrows = A.nrows();
  int const n2    = n * n;

  #pragma omp parallel for
  for (int i = 0; i < nrows; i++) {
    P *r = A.diag(i);
    smmat::scal(n2, alpha, A.lower(i));
    smmat::scal(n2, alpha, r);
    smmat::scal(n2, alpha, A.upper(i));
    for (int j = 0; j < n; j++)
      r[j * n + j] += P{1};
  }
}

template<typename P>
void psedoinvert(int const n, block_diag_matrix<P> &A,
                 block_diag_matrix<P> &iA)
{
  expect(A.nblock() == n * n);
  iA.resize_and_zero(n * n, A.nrows());

  int const nrows = A.nrows();

#pragma omp parallel for
  for (int i = 0; i < nrows; i++) {
    smmat::getrf(n, A[i]);
    smmat::set_eye(n, iA[i]);
    smmat::getrs_l(n, A[i], iA[i]);
    smmat::getrs_u(n, A[i], iA[i]);
  }
}

template<typename P>
void psedoinvert(int const n, block_tri_matrix<P> &A,
                 block_tri_matrix<P> &iA)
{
  expect(A.nblock() == n * n);
  iA.resize_and_zero(n * n, A.nrows());

  int const nrows = A.nrows();

  for (int i = 0; i < nrows; i++)
    smmat::set_eye(n, iA.diag(i));

  smmat::getrf(n, A.diag(0));

  if (nrows <= 2) {
    if (nrows == 1) {
      smmat::getrs_l(n, A.diag(0), iA.diag(0));
      smmat::getrs_u(n, A.diag(0), iA.diag(0));
    } else {
      // merge the periodic blocks
      smmat::axpy(n * n, P{1}, A.lower(0), A.upper(0));
      smmat::axpy(n * n, P{1}, A.upper(1), A.lower(1));
      // factorize the rest of the blocks
      smmat::getrs_l(n, A.diag(0), A.upper(0));
      smmat::getrs_u_right(n, A.diag(0), A.lower(1));
      smmat::gemm<-1>(n, A.lower(1), A.upper(0), A.diag(1));
      smmat::getrf(n, A.diag(1));
      // invert the L factor
      smmat::getrs_l(n, A.diag(0), iA.diag(0));
      smmat::gemm<-1>(n, A.lower(1), iA.diag(0), iA.lower(1));
      smmat::getrs_l(n, A.diag(1), iA.lower(1));
      smmat::getrs_l(n, A.diag(1), iA.diag(1));
      // invert the U factor
      smmat::getrs_u(n, A.diag(1), iA.lower(1));
      smmat::getrs_u(n, A.diag(1), iA.diag(1));
      smmat::gemm<-1>(n, A.upper(0), iA.lower(1), iA.diag(0));
      smmat::gemm<-1>(n, A.upper(0), iA.diag(1), iA.upper(0));
      smmat::getrs_u(n, A.diag(0), iA.diag(1));
      smmat::getrs_u(n, A.diag(0), iA.upper(1));
    }
    return;
  }

  // factorization
  smmat::getrs_l(n, A.diag(0), A.upper(0));
  smmat::getrs_u_right(n, A.diag(0), A.lower(1));

  for (int i = 1; i < nrows - 1; i++)
  {
    smmat::gemm<-1>(n, A.lower(i), A.upper(i - 1), A.diag(i));
    smmat::getrf(n, A.diag(i));
    smmat::getrs_l(n, A.diag(i), A.upper(i));
    smmat::getrs_u_right(n, A.diag(i), A.lower(i + 1));
  }

  int const r = nrows - 1;
  smmat::getrs_l(n, A.diag(0), A.upper(r));
  smmat::getrs_u_right(n, A.diag(0), A.lower(0));

  smmat::gemm<-1>(n, A.lower(r), A.upper(r - 1), A.diag(r));
  smmat::gemm<-1>(n, A.upper(r), A.lower(0), A.diag(r));
  smmat::getrf(n, A.diag(r));

  // inversion of L
  smmat::getrs_l(n, A.diag(0), iA.diag(0));
  for (int i = 1; i < nrows; i++) {
    smmat::gemm<-1>(n, A.lower(i), iA.diag(i - 1), iA.lower(i));
    smmat::getrs_l(n, A.diag(i), iA.lower(i));
    smmat::getrs_l(n, A.diag(i), iA.diag(i));
  }
  smmat::gemm<-1>(n, A.upper(r), iA.diag(0), iA.upper(r));

  // inversion of U
  smmat::getrs_u(n, A.diag(r), iA.lower(r));
  smmat::getrs_u(n, A.diag(r), iA.diag(r));
  smmat::getrs_u(n, A.diag(r), iA.upper(r));

  for (int i = r - 1; i > 0; --i) {
    smmat::gemm<-1>(n, A.upper(i), iA.lower(i + 1), iA.diag(i));
    smmat::gemm<-1>(n, A.upper(i), iA.diag(i + 1), iA.upper(i));

    smmat::getrs_u(n, A.diag(i), iA.lower(i));
    smmat::getrs_u(n, A.diag(i), iA.diag(i));
    smmat::getrs_u(n, A.diag(i), iA.upper(i));
  }

  smmat::gemm<-1>(n, A.upper(0), iA.lower(1), iA.diag(0));
  smmat::gemm<-1>(n, A.upper(0), iA.upper(r), iA.diag(0));
  smmat::gemm<-1>(n, A.upper(0), iA.diag(1), iA.upper(0));

  smmat::getrs_u(n, A.diag(0), iA.lower(0));
  smmat::getrs_u(n, A.diag(0), iA.diag(0));
  smmat::getrs_u(n, A.diag(0), iA.upper(0));
}

#ifdef ASGARD_ENABLE_DOUBLE
template class dense_matrix<double>;
template class block_diag_matrix<double>;
template class block_tri_matrix<double>;

template void gemm1(int const n, block_matrix<double> const &A, block_matrix<double> const &B,
                    block_matrix<double> &C);

template void gemm_block_tri_ul<double>(
    int const n, block_tri_matrix<double> const &A, block_tri_matrix<double> const &B,
    block_tri_matrix<double> &C);
template void gemm_block_tri_lu<double>(
    int const n, block_tri_matrix<double> const &A, block_tri_matrix<double> const &B,
    block_tri_matrix<double> &C);
template void gemm_block_tri<double>(
    int const n, block_tri_matrix<double> const &A, block_tri_matrix<double> const &B,
    block_tri_matrix<double> &C);
template void gemm_diag_tri<double>(
    int const n, block_diag_matrix<double> const &A, block_tri_matrix<double> const &B,
    block_tri_matrix<double> &C);
template void gemm_tri_diag<double>(
    int const n, block_tri_matrix<double> const &A, block_diag_matrix<double> const &B,
    block_tri_matrix<double> &C);
template void gemm_block_diag<double>(
    int const n, block_diag_matrix<double> const &A, block_diag_matrix<double> const &B,
    block_diag_matrix<double> &C);

template void invert_mass(int const, mass_matrix<double> const &, block_tri_matrix<double> &);
template void invert_mass(int const, mass_matrix<double> const &, block_diag_matrix<double> &);
template void invert_mass(int const, mass_matrix<double> const &, double[]);

template void to_euler<double>(int const n, double alpha, block_diag_matrix<double> &A);
template void to_euler<double>(int const n, double alpha, block_tri_matrix<double> &A);

template void psedoinvert<double>(int const, block_diag_matrix<double> &, block_diag_matrix<double> &);
template void psedoinvert<double>(int const, block_tri_matrix<double> &, block_tri_matrix<double> &);

template void block_sparse_matrix<double>::gemv(
    int const, int const, connection_patterns const &, double const[], double[]) const;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class dense_matrix<float>;
template class block_diag_matrix<float>;
template class block_tri_matrix<float>;

template void gemm1(int const n, block_matrix<float> const &A, block_matrix<float> const &B,
                    block_matrix<float> &C);

template void gemm_block_tri_ul<float>(
    int const n, block_tri_matrix<float> const &A, block_tri_matrix<float> const &B,
    block_tri_matrix<float> &C);
template void gemm_block_tri_lu<float>(
    int const n, block_tri_matrix<float> const &A, block_tri_matrix<float> const &B,
    block_tri_matrix<float> &C);
template void gemm_block_tri<float>(
    int const n, block_tri_matrix<float> const &A, block_tri_matrix<float> const &B,
    block_tri_matrix<float> &C);
template void gemm_diag_tri<float>(
    int const n, block_diag_matrix<float> const &A, block_tri_matrix<float> const &B,
    block_tri_matrix<float> &C);
template void gemm_tri_diag<float>(
    int const n, block_tri_matrix<float> const &A, block_diag_matrix<float> const &B,
    block_tri_matrix<float> &C);
template void gemm_block_diag<float>(
    int const n, block_diag_matrix<float> const &A, block_diag_matrix<float> const &B,
    block_diag_matrix<float> &C);

template void invert_mass(int const, mass_matrix<float> const &, block_tri_matrix<float> &);
template void invert_mass(int const, mass_matrix<float> const &, block_diag_matrix<float> &);
template void invert_mass(int const, mass_matrix<float> const &, float[]);

template void to_euler<float>(int const n, float alpha, block_diag_matrix<float> &A);
template void to_euler<float>(int const n, float alpha, block_tri_matrix<float> &A);

template void psedoinvert<float>(int const, block_diag_matrix<float> &, block_diag_matrix<float> &);
template void psedoinvert<float>(int const, block_tri_matrix<float> &, block_tri_matrix<float> &);

template void block_sparse_matrix<float>::gemv(
    int const, int const, connection_patterns const &, float const[], float[]) const;
#endif

} // namespace asgard
