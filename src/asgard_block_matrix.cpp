#include "asgard_block_matrix.hpp"
#include "asgard_small_mats.hpp"

namespace asgard
{

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

  // std::cout << " this one\n"; // FIX THIS ALGORITHM

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

#ifdef ASGARD_ENABLE_DOUBLE
template void gemm_block_tri_ul<double>(
    int const n, block_tri_matrix<double> const &A, block_tri_matrix<double> const &B,
    block_tri_matrix<double> &C);
template void gemm_block_tri_lu<double>(
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

template void block_sparse_matrix<double>::gemv(
    int const, int const, connection_patterns const &, double const[], double[]) const;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void gemm_block_tri_ul<float>(
    int const n, block_tri_matrix<float> const &A, block_tri_matrix<float> const &B,
    block_tri_matrix<float> &C);
template void gemm_block_tri_lu<float>(
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

template void block_sparse_matrix<float>::gemv(
    int const, int const, connection_patterns const &, float const[], float[]) const;
#endif

} // namespace asgard
