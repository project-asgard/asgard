#include "asgard_block_matrix.hpp"
#include "asgard_small_mats.hpp"

namespace asgard
{

template<typename P>
void dense_matrix<P>::factorize()
{
  assert(nrows_ == ncols_);

  #ifdef ASGARD_USE_GPU
  tools::time_event timing_("dense-matrix::factorize-gpu");
  gpu_factor = data_;
  compute->getrf(nrows_, gpu_factor, gpu_ipiv);
  #else
  tools::time_event timing_("dense-matrix::factorize");
  compute->getrf(nrows_, data_, ipiv);
  #endif
}

template<typename P>
void dense_matrix<P>::solve(std::vector<P> &b) const
{
  assert(is_factorized());

  #ifdef ASGARD_USE_GPU
  tools::time_event timing_("dense-matrix::solve-gpu");
  compute->getrs(nrows_, gpu_factor, gpu_ipiv, b);
  #else
  tools::time_event timing_("dense-matrix::solve");
  compute->getrs(nrows_, data_, ipiv, b);
  #endif
}

#ifdef ASGARD_USE_GPU
template<typename P>
void dense_matrix<P>::solve(gpu::vector<P> &b) const
{
  tools::time_event timing_("dense-matrix::solve-gpu");
  assert(is_factorized());

  compute->getrs(nrows_, gpu_factor, gpu_ipiv, b);
}
template<typename P>
void dense_matrix<P>::solve(P b[]) const
{
  tools::time_event timing_("dense-matrix::solve-gpu");
  assert(is_factorized());

  compute->getrs(nrows_, gpu_factor, gpu_ipiv, b);
}
#endif

template<typename P>
void dense_matrix<P>::print(std::ostream &os) {
  for (int64_t r = 0; r < nrows_; r++) {
    for (int64_t c = 0; c < ncols_; c++)
      os << std::setw(16) << data_[c * nrows_ + r];
    os << '\n';
  }
}

template<typename P>
void block_matrix<P>::print(std::ostream &os, int br, int bc, int oswidth)
{
  if (br == -1)
  {
    int const nb = data_.stride();
    br = 0;
    while (br < nb and br * br != nb)
      ++br;
    assert(br * br == nb);
    bc = br;
  }
  assert(br * bc == data_.stride());
  for (auto r : indexof(nrows_))
  {
    for (int i = 0; i < br; i++)
    {
      for (auto c : indexof(ncols_))
      {
        for (int j = 0; j < bc; j++)
          os << std::setw(oswidth) << data_[c * nrows_ + r][j * br + i];
        os << std::setw(oswidth / 2) << "  ";
      }
      os << '\n';
    }
    os << '\n';
  }
}

template<typename P>
void block_matrix<P>::printc(std::ostream &os, int c, int oswidth)
{
  int const nb = data_.stride();
  int br = 0;
  while (br < nb and br * br != nb)
    ++br;
  assert(br * br == nb);
  int bc = br;
  assert(br * bc == data_.stride());
  for (auto r : indexof(nrows_))
  {
    for (int i = 0; i < br; i++)
    {
      for (int j = 0; j < bc; j++)
        os << std::setw(oswidth) << data_[c * nrows_ + r][j * br + i];
      os << std::setw(oswidth / 2) << "  ";

      os << '\n';
    }
    os << '\n';
  }
}

template<typename P>
void block_matrix<P>::printr(std::ostream &os, int r, int oswidth)
{
  int const nb = data_.stride();
  int br = 0;
  while (br < nb and br * br != nb)
    ++br;
  assert(br * br == nb);
  int bc = br;
  assert(br * bc == data_.stride());
  for (int i = 0; i < br; i++)
  {
    for (auto c : indexof(ncols_))
    {
      for (int j = 0; j < bc; j++)
        os << std::setw(oswidth) << data_[c * nrows_ + r][j * br + i];
      os << std::setw(oswidth / 2) << "  ";
    }
    os << '\n';
  }
  os << '\n';
}

template<typename P>
P block_matrix<P>::max_diff(block_matrix<P> const &other) {
  assert(nrows_ == other.nrows_);
  assert(ncols_ == other.ncols_);
  assert(nblock() == other.nblock());
  int64_t const size = nrows_ * ncols_ * nblock();
  P const *v1 = data_[0];
  P const *v2 = other.data_[0];
  P err = 0;
  for (auto i : indexof(size))
    err = std::max(err, std::abs(v1[i] - v2[i]));
  return err;
}

template<typename P>
dense_matrix<P> block_matrix<P>::to_dense_matrix(int const n) const
{
  assert(n * n == data_.stride());
  dense_matrix<P> mat(n * nrows_, n * ncols_);
  #pragma omp parallel for
  for (int r = 0; r < nrows_; r++)
    for (int c = 0; c < ncols_; c++)
      for (int k = 0; k < n; k++)
        std::copy_n(data_[c * nrows_ + r] + n * k , n, mat.data(n * r, n * c + k));
  return mat;
}

template<typename P>
void gemm1(int const n, block_matrix<P> const &A, block_matrix<P> const &B, block_matrix<P> &C)
{
  int M = A.nrows();
  int N = B.ncols();
  int K = A.ncols();

  assert(C.nrows() == M);
  assert(C.ncols() == N);
  assert(B.nrows() == K);

  assert(A.nblock() == n * n);
  assert(B.nblock() == n * n);
  assert(C.nblock() == n * n);

#pragma omp parallel for
  for (int c = 0; c < N; c++) {
    for (int r = 0; r < M; r++) {
      for (int k = 0; k < K; k++)
        smmat::gemm<1>(n, A(r, k), B(k, c), C(r, c));
    }
  }
}

template<typename P>
block_matrix<P> mass_matrix<P>::to_full() const
{
  int const n = nblock();
  block_matrix<P> full(n, nrows(), nrows());
  #pragma omp parallel for
  for (int64_t r = 0; r < nrows(); r++)
    std::copy_n(data_[r], n, full(r, r));
  return full;
}

template<typename P>
block_matrix<P> block_diag_matrix<P>::to_full() const
{
  int const n = nblock();
  block_matrix<P> full(n, nrows(), nrows());
  #pragma omp parallel for
  for (int64_t r = 0; r < nrows(); r++)
    std::copy_n(data_[r], n, full(r, r));
  return full;
}

template<typename P>
void block_diag_matrix<P>::spd_factorize(int const n)
{
  assert(n * n == nblock());
  switch (n)
  {
  case 1:
    ASGARD_OMP_PARFOR_SIMD
    for (int64_t r = 0; r < nrows(); r++)
      data_[r][0] = P{1} / data_[r][0];
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows(); r++)
      smmat::inv2by2(data_[r]);
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows(); r++)
      smmat::potrf(n, data_[r]);
    break;
  }
}

template<typename P>
void block_diag_matrix<P>::solve(int const n, P rhs[]) const
{
  assert(n * n == nblock());
  switch (n)
  {
  case 1:
    ASGARD_OMP_PARFOR_SIMD
    for (int64_t r = 0; r < nrows(); r++)
      rhs[r] *= data_[r][0];
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows(); r++)
      smmat::gemv2by2(data_[r], rhs + 2 * r);
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows(); r++)
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
    for (int64_t r = 0; r < nrows(); r++)
      rhs[r][0] *= data_[r][0];
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows(); r++)
      smmat::gemm2by2(data_[r], rhs[r]);
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows(); r++)
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
    for (int64_t r = 0; r < nrows(); r++)
    {
      rhs.lower(r)[0] *= data_[r][0];
      rhs.diag(r)[0] *= data_[r][0];
      rhs.upper(r)[0] *= data_[r][0];
    }
    break;
  case 2:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows(); r++)
    {
      smmat::gemm2by2(data_[r], rhs.lower(r));
      smmat::gemm2by2(data_[r], rhs.diag(r));
      smmat::gemm2by2(data_[r], rhs.upper(r));
    }
    break;
  default:
#pragma omp parallel for
    for (int64_t r = 0; r < nrows(); r++)
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
  assert(nblock() == n * n);
  assert(vec.size() == static_cast<size_t>(n * nrows()));
  if (work.size() < vec.size())
    work.resize(vec.size());

  std::copy(vec.begin(), vec.end(), work.begin());

  span2d<P> x(n, nrows(), work.data());
  span2d<P> y(n, nrows(), vec.data());

#pragma omp parallel for
  for (int64_t r = 0; r < nrows(); r++) {
    smmat::gemv(n, n, data_[r], x[r], y[r]);
  }
}

template<typename P>
void block_tri_matrix<P>::inplace_gemv(int n, std::vector<P> &vec, std::vector<P> &work) const
{
  assert(nblock() == n * n);
  assert(vec.size() == static_cast<size_t>(n * nrows_));
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
block_tri_matrix<P> &block_tri_matrix<P>::operator += (block_tri_matrix<P> const &other) {
  assert(nrows_ == other.nrows_);
  assert(data_.stride() == other.data_.stride());

  int64_t const num_entries = data_.total_size();

  P *dest = data_[0];
  P const *src = other.data_[0];

  ASGARD_OMP_PARFOR_SIMD
  for(int64_t i = 0; i < num_entries; i++)
    dest[i] += src[i];

  return *this;
}

template<typename P>
block_tri_matrix<P> &block_tri_matrix<P>::operator += (block_diag_matrix<P> const &other)
{
  assert(nrows_ == other.nrows());
  assert(data_.stride() == other.nblock());

  int const n = data_.stride();

  #pragma omp parallel for
  for (int64_t r = 0; r < nrows_; r++)
    smmat::axpy1(n, other[r], (*this)[r]);

  return *this;
}

template<typename P>
block_matrix<P> block_tri_matrix<P>::to_full() const
{
  int const n = nblock();
  block_matrix<P> full(n, nrows_, nrows_);
  std::copy_n(diag(0), n, full(0, 0));
  if (nrows_ == 1)
    return full;
  std::copy_n(lower(0), n, full(0, nrows_ - 1));
  std::copy_n(upper(0), n, full(0, 1));
  for (int64_t r = 1; r < nrows_ - 1; r++)
  {
    std::copy_n(lower(r), n, full(r, r - 1));
    std::copy_n(diag(r), n, full(r, r));
    std::copy_n(upper(r), n, full(r, r + 1));
  }
  std::copy_n(lower(nrows_ - 1), n, full(nrows_ - 1, nrows_ - 2));
  std::copy_n(diag(nrows_ - 1), n, full(nrows_ - 1, nrows_ - 1));
  std::copy_n(upper(nrows_ - 1), n, full(nrows_ - 1, 0));
  if (nrows_ == 2)
  {
    for (int i : indexof<int>(data_.stride()))
      full(0, 1)[i] += lower(0)[i];
    for (int i : indexof<int>(data_.stride()))
      full(1, 0)[i] += lower(nrows_ - 1)[i];
  }
  return full;
}

template<typename P>
block_matrix<P> block_sparse_matrix<P>::to_full(connect_1d const &conn) const
{
  int const n     = nblock();
  int const nrows = conn.num_rows();
  int mcol = 0;
  for (int j = 0; j < conn.num_connections(); j++)
    mcol = std::max(mcol, conn[j]);
  block_matrix<P> full(n, nrows, mcol + 1);

  for (int r = 0; r < nrows; r++)
    for (int j = conn.row_begin(r); j < conn.row_end(r); j++)
      std::copy_n(data_[j], n, full(r, conn[j]));

  return full;
}

template<typename P>
void fill_pattern(P const pattern[], block_diag_matrix<P> &A)
{
  int const rows   = A.nrows();
  int const nblock = A.nblock();

  #pragma omp parallel for
  for (int r = 0; r < rows; r++)
    std::copy_n(pattern, nblock, A[r]);
}

template<typename P>
void gemm_block_tri_ul(
    int const n, block_tri_matrix<P> const &A, block_tri_matrix<P> const &B,
    block_tri_matrix<P> &C)
{
  int64_t const M = A.nrows();
  assert(A.nblock() == n * n);
  assert(A.nblock() == B.nblock());
  assert(A.nblock() == C.nblock());
  assert(B.nrows() == M);
  assert(C.nrows() == M);

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
  assert(A.nblock() == B.nblock());
  assert(A.nblock() == C.nblock());
  assert(A.nblock() == n * n);
  assert(B.nrows() == M);
  assert(C.nrows() == M);

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
  assert(M >= 1);
  assert(A.nblock() == B.nblock());
  assert(A.nblock() == C.nblock());
  assert(A.nblock() == n * n);
  assert(B.nrows() == M);
  assert(C.nrows() == M);

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
  assert(A.nblock() == n * n);
  assert(A.nblock() == B.nblock());
  assert(A.nblock() == C.nblock());
  assert(B.nrows() == M);
  assert(C.nrows() == M);

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
  assert(A.nblock() == n * n);
  assert(A.nblock() == B.nblock());
  assert(A.nblock() == C.nblock());
  assert(B.nrows() == M);
  assert(C.nrows() == M);

  smmat::gemm<0>(n, A.diag(0), B[0], C.diag(0));
  if (M == 1)
    return;

  smmat::gemm<0>(n, A.lower(1), B[0], C.lower(1));
  smmat::gemm<0>(n, A.upper(M-1), B[0], C.upper(M-1));

#pragma omp parallel for
  for (int64_t r = 1; r < M - 1; r++)
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
  assert(A.nblock() == n * n);
  assert(A.nblock() == B.nblock());
  assert(A.nblock() == C.nblock());
  assert(B.nrows() == M);
  assert(C.nrows() == M);

#pragma omp parallel for
  for (int64_t r = 0; r < M; r++)
    smmat::gemm<0>(n, A[r], B[r], C[r]);
}

template<typename P>
void invert_mass(int const n, mass_matrix<P> const &mass, block_tri_matrix<P> &op)
{
  assert(mass.nblock() == op.nblock());
  int64_t nr = op.nrows();
  assert(mass.nrows() == nr);

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
  assert(mass.nblock() == op.nblock());
  int64_t const nr = op.nrows();
  assert(mass.nrows() == nr);

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
  assert(n * n == nblock());

  connect_1d const &conn = conns(htype_);
  int const nrows        = fm::ipow2(level);

  assert(nrows <= conn.num_rows());

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
void block_sparse_matrix<P>::scal(P v)
{
  P *x = data();
  size_t const num = data_vector().size();
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < num; i++)
    x[i] *= v;
}

template<typename P>
void invert_mass(int const n, mass_matrix<P> const &mass, P x[])
{
  assert(mass.nblock() == n * n);
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

#ifdef ASGARD_ENABLE_DOUBLE
template class dense_matrix<double>;
template class block_matrix<double>;
template class block_diag_matrix<double>;
template class block_tri_matrix<double>;
template class block_sparse_matrix<double>;

template void gemm1(int const n, block_matrix<double> const &A, block_matrix<double> const &B,
                    block_matrix<double> &C);

template void fill_pattern<double>(double const pattern[], block_diag_matrix<double> &A);

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
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class dense_matrix<float>;
template class block_matrix<float>;
template class block_diag_matrix<float>;
template class block_tri_matrix<float>;
template class block_sparse_matrix<float>;

template void gemm1(int const n, block_matrix<float> const &A, block_matrix<float> const &B,
                    block_matrix<float> &C);

template void fill_pattern<float>(float const pattern[], block_diag_matrix<float> &A);

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
#endif

} // namespace asgard
