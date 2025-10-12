
#include "asgard_kronmult.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace asgard::kronmult
{

template<typename precision, int num_dimensions, int dim, int n>
void gbkron_mult_add(precision const A[], precision const x[], precision y[])
{
  if constexpr (n == 1) // dimension does not matter here
  {
    y[0] += A[0] * x[0];
    return;
  }

  static_assert(num_dimensions >= 1 and num_dimensions <= 6);
  if constexpr (num_dimensions == 1)
  {
    ASGARD_PRAGMA_OMP_SIMD(collapse(2))
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        y[k] += A[j * n + k] * x[j];
  }
  else if constexpr (num_dimensions == 2)
  {
    if constexpr (dim == 1)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(3))
      for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
          for (int s = 0; s < n; s++)
            y[s + k * n] += A[j * n + s] * x[j + k * n];
    }
    else
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(3))
      for (int j = 0; j < n; j++)
        for (int s = 0; s < n; s++)
          for (int k = 0; k < n; k++)
            y[k + s * n] += A[j * n + s] * x[k + j * n];
    }
  }
  else if constexpr (num_dimensions == 3)
  {
    if constexpr (dim == 2)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(4))
      for (int l = 0; l < n; l++)
        for (int k = 0; k < n; k++)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              y[l * n * n + k * n + s] += A[j * n + s] * x[l * n * n + n * k + j];
    }
    else if constexpr (dim == 1)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(4))
      for (int l = 0; l < n; l++)
        for (int j = 0; j < n; j++)
          for (int s = 0; s < n; s++)
            for (int k = 0; k < n; k++)
              y[l * n * n + s * n + k] += x[l * n * n + j * n + k] * A[j * n + s];
    }
    else // dim == 0
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(4))
      for (int j = 0; j < n; j++)
        for (int s = 0; s < n; s++)
          for (int l = 0; l < n; l++)
            for (int k = 0; k < n; k++)
              y[s * n * n + l * n + k] += x[n * n * j + n * l + k] * A[j * n + s];
    }
  }
  else if constexpr (num_dimensions == 4)
  {
    if constexpr (dim == 3)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(5))
      for (int p = 0; p < n; p++)
        for (int l = 0; l < n; l++)
          for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                y[p * ipow<n, 3>() + l * n * n + k * n + s] += A[j * n + s] * x[p * ipow<n, 3>() + l * n * n + k * n + j];
    }
    else if constexpr (dim == 2)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(5))
      for (int p = 0; p < n; p++)
        for (int l = 0; l < n; l++)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int k = 0; k < n; k++)
                y[p * ipow<n, 3>() + l * n * n + s * n + k] += x[p * ipow<n, 3>() + l * n * n + j * n + k] * A[j * n + s];
    }
    else if constexpr (dim == 1)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(5))
      for (int p = 0; p < n; p++)
        for (int j = 0; j < n; j++)
          for (int s = 0; s < n; s++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                y[p * ipow<n, 3>() + s * n * n + l * n + k] += x[p * ipow<n, 3>() + j * n * n + l * n + k] * A[j * n + s];
    }
    else // dim == 0
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(5))
      for (int j = 0; j < n; j++)
        for (int s = 0; s < n; s++)
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                y[s * n * n * n + p * n * n + l * n + k] +=
                    x[n * n * n * j + n * n * p + n * l + k] *
                    A[j * n + s];
    }
  }
  else if constexpr (num_dimensions == 5)
  {
    if constexpr (dim == 4)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(6))
      for (int v = 0; v < n; v++)
        for (int p = 0; p < n; p++)
          for (int l = 0; l < n; l++)
            for (int k = 0; k < n; k++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  y[v * ipow<n, 4>() + p * ipow<n, 3>() + l * n * n + k * n + s] += A[j * n + s] * x[v * ipow<n, 4>() + p * ipow<n, 3>() + l * n * n + k * n + j];
    }
    else if constexpr (dim == 3)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(6))
      for (int v = 0; v < n; v++)
        for (int p = 0; p < n; p++)
          for (int l = 0; l < n; l++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int k = 0; k < n; k++)
                  y[v * ipow<n, 4>() + p * ipow<n, 3>() + l * n * n + s * n + k] += x[v * ipow<n, 4>() + p * ipow<n, 3>() + l * n * n + j * n + k] * A[j * n + s];
    }
    else if constexpr (dim == 2)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(6))
      for (int v = 0; v < n; v++)
        for (int p = 0; p < n; p++)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  y[v * ipow<n, 4>() + p * ipow<n, 3>() + s * n * n + l * n + k] += x[v * ipow<n, 4>() + p * ipow<n, 3>() + j * n * n + l * n + k] * A[j * n + s];
    }
    else if constexpr (dim == 1)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(6))
      for (int v = 0; v < n; v++)
        for (int j = 0; j < n; j++)
          for (int s = 0; s < n; s++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  y[v * ipow<n, 4>() + s * ipow<n, 3>() + p * n * n + l * n + k] += x[v * ipow<n, 4>() + j * ipow<n, 3>() + p * n * n + l * n + k] * A[j * n + s];
    }
    else // dim == 0
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(6))
      for (int j = 0; j < n; j++)
        for (int s = 0; s < n; s++)
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  y[s * ipow<n, 4>() + v * ipow<n, 3>() + p * n * n + l * n + k] +=
                      x[ipow<n, 4>() * j + ipow<n, 3>() * v + n * n * p +
                        n * l + k] *
                      A[j * n + s];
    }
  }
  else if constexpr (num_dimensions == 6)
  {
    if constexpr (dim == 5)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(7))
      for (int w = 0; w < n; w++)
        for (int v = 0; v < n; v++)
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    y[ipow<n, 5>() * w + ipow<n, 4>() * v + ipow<n, 3>() * p + n * n * l + n * k + s] +=
                        A[j * n + s] * x[ipow<n, 5>() * w + ipow<n, 4>() * v + ipow<n, 3>() * p + n * n * l + n * k + j];
    }
    else if constexpr (dim == 4)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(7))
      for (int w = 0; w < n; w++)
        for (int v = 0; v < n; v++)
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int k = 0; k < n; k++)
                    y[ipow<n, 5>() * w + ipow<n, 4>() * v + ipow<n, 3>() * p + n * n * l + n * s + k] +=
                        x[ipow<n, 5>() * w + ipow<n, 4>() * v + ipow<n, 3>() * p + n * n * l + n * j + k] * A[j * n + s];
    }
    else if constexpr (dim == 3)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(7))
      for (int w = 0; w < n; w++)
        for (int v = 0; v < n; v++)
          for (int p = 0; p < n; p++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    y[ipow<n, 5>() * w + ipow<n, 4>() * v + ipow<n, 3>() * p + n * n * s + n * l + k] +=
                        x[ipow<n, 5>() * w + ipow<n, 4>() * v + ipow<n, 3>() * p + n * n * j + n * l + k] * A[j * n + s];
    }
    else if constexpr (dim == 2)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(7))
      for (int w = 0; w < n; w++)
        for (int v = 0; v < n; v++)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    y[ipow<n, 5>() * w + ipow<n, 4>() * v + ipow<n, 3>() * s + n * n * p + n * l + k] +=
                        x[ipow<n, 5>() * w + ipow<n, 4>() * v + ipow<n, 3>() * j + n * n * p + n * l + k] * A[j * n + s];
    }
    else if constexpr (dim == 1)
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(7))
      for (int w = 0; w < n; w++)
        for (int j = 0; j < n; j++)
          for (int s = 0; s < n; s++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    y[ipow<n, 5>() * w + ipow<n, 4>() * s + ipow<n, 3>() * v + n * n * p + n * l + k] +=
                        x[ipow<n, 5>() * w + ipow<n, 4>() * j + ipow<n, 3>() * v + n * n * p + n * l + k] * A[j * n + s];
    }
    else
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(7))
      for (int j = 0; j < n; j++)
        for (int s = 0; s < n; s++)
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    y[ipow<n, 5>() * s + ipow<n, 4>() * w + ipow<n, 3>() * v + n * n * p + n * l + k] +=
                        x[ipow<n, 5>() * j + ipow<n, 4>() * w + ipow<n, 3>() * v + n * n * p + n * l + k] *
                        A[j * n + s];
    }
  }
}

inline int64_t asgard_kronmult_nblocks_ = 0;

template<typename precision, conn_fill fill, int num_dimensions, int dim, int n>
void block_cpu(sparse_grid const &grid, connect_1d const &conn,
               precision const vals[], precision const x[], precision y[],
               std::vector<std::vector<int64_t>> &row_wspace)
{
  constexpr int n2 = n * n;

  constexpr int64_t block_size = ipow<n, num_dimensions>();

  dimension_sort const &dsort = grid.dsort();

  int const num_vecs = dsort.num_vecs(dim);

#ifdef _OPENMP
  int const max_threads = omp_get_max_threads();
#else
  int const max_threads = 1;
#endif

  if (static_cast<int>(row_wspace.size()) < max_threads)
    row_wspace.resize(max_threads);

  int threadid = 0;
#pragma omp parallel
  {
    int64_t my_block_count = 0;

    int tid;
#pragma omp critical
    tid = threadid++;

    // xidx holds indexes for the entries of the current
    // sparse row that are present in the current ilist
    std::vector<int64_t> &xidx = row_wspace[tid];
    if (static_cast<int>(xidx.size()) < conn.num_rows())
      xidx.resize(conn.num_rows(), -1);

#pragma omp for schedule(dynamic)
    for (int vec_id = 0; vec_id < num_vecs; vec_id++)
    {
      int const vec_begin = dsort.vec_begin(dim, vec_id);
      int const vec_end   = dsort.vec_end(dim, vec_id);
      // map the indexes of present entries
      for (int j = vec_begin; j < vec_end; j++)
        xidx[grid.dsorted(dim, j)] = dsort.map(dim, j) * block_size;

      // matrix-vector product using xidx as a row
      for (int rj = vec_begin; rj < vec_end; rj++)
      {
        // row in the 1d pattern
        int const row = grid.dsorted(dim, rj);

        precision *const local_y = y + xidx[row];

        // columns for the 1d pattern
        int col_begin = (fill == conn_fill::upper)
                         ? conn.row_diag(row) : conn.row_begin(row);
        int col_end   = (fill == conn_fill::lower or fill == conn_fill::lower_udiag)
                         ? conn.row_diag(row) : conn.row_end(row);

        if constexpr (n != -1) {
          if constexpr (fill == conn_fill::lower_udiag) {
            std::copy_n(x + xidx[row], block_size, local_y);
          } else {
            for (int j = 0; j < block_size; j++)
              local_y[j] = precision{0};
          }
        }

        for (int c = col_begin; c < col_end; c++)
        {
          int64_t const xj = xidx[conn[c]];
          if (xj != -1)
          {
            // std::cout << " (iy, ix) = (" << xidx[row] / block_size << ", " << xidx[conn[c]] / block_size
            //           << ")   (ir, ic) = " << row << ", " << conn[c] << ")  "
            //           << "  " << (vals + n2 * c)[0] << "    " << (x + xj)[0] << "    " << local_y[0] << "\n";

            if constexpr (n == -1)
              my_block_count += 1;
            else
              gbkron_mult_add<precision, num_dimensions, dim, n>(vals + n2 * c, x + xj, local_y);
          }
        }
      }

      // restore the entries
      for (int j = vec_begin; j < vec_end; j++)
        xidx[grid.dsorted(dim, j)] = -1;
    }

    if constexpr (n == -1)
#pragma omp atomic
      asgard_kronmult_nblocks_ += my_block_count;
  } // pragma parallel
}

template<typename precision, conn_fill fill, int num_dimensions, int dim>
void block_cpu(int n, sparse_grid const &grid,
               connect_1d const &conn, precision const vals[],
               precision const x[], precision y[],
               std::vector<std::vector<int64_t>> &row_wspace)
{
  static_assert(dim < num_dimensions);
  switch (n)
  {
  case -1: // special case: count the number of flops
    block_cpu<precision, fill, num_dimensions, dim, -1>(grid, conn, vals, x, y, row_wspace);
    break;
  case 1: // pwconstant
    block_cpu<precision, fill, num_dimensions, dim, 1>(grid, conn, vals, x, y, row_wspace);
    break;
  case 2: // linear
    block_cpu<precision, fill, num_dimensions, dim, 2>(grid, conn, vals, x, y, row_wspace);
    break;
  case 3: // quadratic
    block_cpu<precision, fill, num_dimensions, dim, 3>(grid, conn, vals, x, y, row_wspace);
    break;
  case 4: // cubic
    block_cpu<precision, fill, num_dimensions, dim, 4>(grid, conn, vals, x, y, row_wspace);
    break;
  default:
    throw std::runtime_error("(kronmult) unimplemented n for given -degree");
  };
}

template<typename precision, conn_fill fill, int num_dimensions>
void block_cpu(int n, sparse_grid const &grid, int dim, connect_1d const &conn,
               precision const vals[], precision const x[], precision y[],
               std::vector<std::vector<int64_t>> &row_wspace)
{
  expect(dim < num_dimensions);
  switch (dim)
  {
  case 0:
    block_cpu<precision, fill, num_dimensions, 0>(n, grid, conn, vals, x, y, row_wspace);
    break;
  case 1:
    if constexpr (num_dimensions >= 2) {
      block_cpu<precision, fill, num_dimensions, 1>(n, grid, conn, vals, x, y, row_wspace);
      break;
    }
  case 2:
    if constexpr (num_dimensions >= 3) {
      block_cpu<precision, fill, num_dimensions, 2>(n, grid, conn, vals, x, y, row_wspace);
      break;
    }
  case 3:
    if constexpr (num_dimensions >= 4) {
      block_cpu<precision, fill, num_dimensions, 3>(n, grid, conn, vals, x, y, row_wspace);
      break;
    }
  case 4:
    if constexpr (num_dimensions >= 5) {
      block_cpu<precision, fill, num_dimensions, 4>(n, grid, conn, vals, x, y, row_wspace);
      break;
    }
  case 5:
    if constexpr (num_dimensions >= 6) {
      block_cpu<precision, fill, num_dimensions, 5>(n, grid, conn, vals, x, y, row_wspace);
      break;
    }
  default:
    throw std::runtime_error("incorrect dim, incompatible with num_dimensions");
  }
  static_assert(1 <= num_dimensions and num_dimensions <= max_num_dimensions);
}

template<typename precision, conn_fill fill>
void block_cpu(int num_dimensions, int n, sparse_grid const &grid,
               int dim, connect_1d const &conn,
               precision const vals[], precision const x[], precision y[],
               std::vector<std::vector<int64_t>> &row_wspace)
{
  switch (num_dimensions)
  {
  case 1:
    block_cpu<precision, fill, 1>(n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  case 2:
    block_cpu<precision, fill, 2>(n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  case 3:
    block_cpu<precision, fill, 3>(n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  case 4:
    block_cpu<precision, fill, 4>(n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  case 5:
    block_cpu<precision, fill, 5>(n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  case 6:
    block_cpu<precision, fill, 6>(n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  default:
    throw std::runtime_error("(kronmult) works with only up to 6 dimensions");
  };
}

template<typename precision>
void block_cpu(int num_dimensions, int n, sparse_grid const &grid,
               int dim, conn_fill fill, connect_1d const &conn,
               precision const vals[], precision const x[],
               precision y[], std::vector<std::vector<int64_t>> &row_wspace)
{
  switch (fill)
  {
  case conn_fill::lower:
    block_cpu<precision, conn_fill::lower>(
        num_dimensions, n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  case conn_fill::lower_udiag:
    block_cpu<precision, conn_fill::lower_udiag>(
        num_dimensions, n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  case conn_fill::upper:
    block_cpu<precision, conn_fill::upper>(
        num_dimensions, n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  default: // case permutes::matrix_fill::both:
    block_cpu<precision, conn_fill::both>(
        num_dimensions, n, grid, dim, conn, vals, x, y, row_wspace);
    break;
  }
}

template<typename precision>
void block_cpu(
    int n, sparse_grid const &grid, connection_patterns const &conns,
    permutes const &perm,
    std::array<block_sparse_matrix<precision>, max_num_dimensions> const &cmats,
    precision alpha, precision const x[], precision beta, precision y[],
    workspace<precision> &work)
{
  tools::time_event performance_("block_cpu");

  precision *w1 = work.w1.data();
  precision *w2 = work.w2.data();

  auto get_connect_1d = [&](conn_fill const fill)
      -> connect_1d const & {
    // if the term has flux, i.e., fdir != -1
    // then the direction using fill::both will use the flux+volume connectivity
    // otherwise we will use only the volume connectivity
    if (perm.flux_dir != -1 and fill == conn_fill::both)
      return conns[connect_1d::hierarchy::full];
    else
      return conns[connect_1d::hierarchy::volume];
  };

  int const num_dims    = grid.num_dims();
  int const active_dims = perm.num_dimensions();
  expect(active_dims > 0);

  for (size_t i = 0; i < perm.fill.size(); i++)
  {
    int dir = perm.direction[i][0];

    block_cpu(num_dims, n, grid, dir, perm.fill[i][0],
                get_connect_1d(perm.fill[i][0]),
                cmats[dir].data(), x, w1, work.row_map);

    for (int d = 1; d < active_dims; d++)
    {
      dir = perm.direction[i][d];
      block_cpu(num_dims, n, grid, dir, perm.fill[i][d],
                get_connect_1d(perm.fill[i][d]),
                cmats[dir].data(), w1, w2, work.row_map);
      std::swap(w1, w2);
    }

    int64_t num_entries = static_cast<int64_t>(work.w1.size());

    if (i == 0) {
      if (beta == 0) {
        if (alpha == 1) {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t j = 0; j < num_entries; j++)
            y[j] = w1[j];
        } else {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t j = 0; j < num_entries; j++)
            y[j] = alpha * w1[j];
        }
      } else {
        if (alpha == 1) {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t j = 0; j < num_entries; j++)
            y[j] = beta * y[j] + w1[j];
        } else {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t j = 0; j < num_entries; j++)
            y[j] = beta * y[j] + alpha * w1[j];
        }
      }
    } else {
      if (alpha == 1) {
        ASGARD_OMP_PARFOR_SIMD
        for (int64_t j = 0; j < num_entries; j++)
          y[j] += w1[j];
      } else if (alpha == -1) {
        ASGARD_OMP_PARFOR_SIMD
        for (int64_t j = 0; j < num_entries; j++)
          y[j] -= w1[j];
      } else {
        ASGARD_OMP_PARFOR_SIMD
        for (int64_t j = 0; j < num_entries; j++)
          y[j] += alpha * w1[j];
      }
    }
  }
}

template<typename precision>
void block_cpu(
    int n, sparse_grid const &grid, connection_patterns const &conns,
    permutes const &perm, block_sparse_matrix<precision> const &cmat,
    precision alpha, precision const x[], precision beta, precision y[],
    workspace<precision> &work)
{
  tools::time_event performance_("block_cpu");
  precision *w1 = work.w1.data();
  precision *w2 = work.w2.data();

  auto get_connect_1d = [&](conn_fill const fill)
      -> connect_1d const & {
    // if the term has flux, i.e., fdir != -1
    // then the direction using fill::both will use the flux+volume connectivity
    // otherwise we will use only the volume connectivity
    if (perm.flux_dir != -1 and fill == conn_fill::both)
      return conns[connect_1d::hierarchy::full];
    else
      return conns[connect_1d::hierarchy::volume];
  };

  int const num_dims    = grid.num_dims();
  int const active_dims = perm.num_dimensions();
  expect(active_dims > 0);

  for (size_t i = 0; i < perm.fill.size(); i++)
  {
    int dir = perm.direction[i][0];

    block_cpu(num_dims, n, grid, dir, perm.fill[i][0],
              get_connect_1d(perm.fill[i][0]),
              cmat.data(), x, w1, work.row_map);

    for (int d = 1; d < active_dims; d++)
    {
      dir = perm.direction[i][d];
      block_cpu(num_dims, n, grid, dir, perm.fill[i][d],
                get_connect_1d(perm.fill[i][d]),
                cmat.data(), w1, w2, work.row_map);
      std::swap(w1, w2);
    }

    int64_t num_entries = static_cast<int64_t>(work.w1.size());

    if (i == 0) {
      if (beta == 0) {
        if (alpha == 1) {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t j = 0; j < num_entries; j++)
            y[j] = w1[j];
        } else {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t j = 0; j < num_entries; j++)
            y[j] = alpha * w1[j];
        }
      } else {
        if (alpha == 1) {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t j = 0; j < num_entries; j++)
            y[j] = beta * y[j] + w1[j];
        } else {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t j = 0; j < num_entries; j++)
            y[j] = beta * y[j] + alpha * w1[j];
        }
      }
    } else {
      if (alpha == 1) {
        ASGARD_OMP_PARFOR_SIMD
        for (int64_t j = 0; j < num_entries; j++)
          y[j] += w1[j];
      } else if (alpha == -1) {
        ASGARD_OMP_PARFOR_SIMD
        for (int64_t j = 0; j < num_entries; j++)
          y[j] -= w1[j];
      } else {
        ASGARD_OMP_PARFOR_SIMD
        for (int64_t j = 0; j < num_entries; j++)
          y[j] += alpha * w1[j];
      }
    }
  }
}

#ifdef ASGARD_USE_FLOPCOUNTER

template<typename precision>
int64_t block_cpu(
    int n, sparse_grid const &grid, connection_patterns const &conns,
    permutes const &perm, precision alpha, precision beta, workspace<precision> &work)
{
  auto get_connect_1d = [&](conn_fill const fill)
      -> connect_1d const & {
    if (perm.flux_dir != -1 and fill == conn_fill::both)
      return conns[connect_1d::hierarchy::full];
    else
      return conns[connect_1d::hierarchy::volume];
  };

  int const num_dims    = grid.num_dims();
  int const active_dims = perm.num_dimensions();
  expect(active_dims > 0);

  int64_t const num_entries = static_cast<int64_t>(work.w1.size());

  asgard_kronmult_nblocks_ = 0;

  int64_t num_scal = 0;

  for (size_t i = 0; i < perm.fill.size(); i++)
  {
    int dir = perm.direction[i][0];

    block_cpu<precision>(num_dims, -1, grid, dir, perm.fill[i][0],
                         get_connect_1d(perm.fill[i][0]),
                         nullptr, nullptr, nullptr, work.row_map);

    for (int d = 1; d < active_dims; d++)
    {
      dir = perm.direction[i][d];
      block_cpu<precision>(num_dims, -1, grid, dir, perm.fill[i][d],
                           get_connect_1d(perm.fill[i][d]),
                           nullptr, nullptr, nullptr, work.row_map);
    }

    if (i == 0) {
      if (beta == 0) {
        num_scal += num_entries;
      } else {
        num_scal += 2 * num_entries;
      }
    } else {
      if (alpha == 1 or alpha == -1) {
        num_scal += num_entries;
      } else {
        num_scal += 2 * num_entries;
      }
    }
  }

  return 2 * asgard_kronmult_nblocks_ * fm::ipow(n, num_dims + 1) + num_scal;
}

#endif

#ifdef ASGARD_ENABLE_DOUBLE

template void block_cpu<double>(
    int, sparse_grid const &, connection_patterns const &, permutes const &,
    std::array<block_sparse_matrix<double>, max_num_dimensions> const &,
    double, double const[], double, double[], workspace<double> &);

template void block_cpu<double>(
      int, sparse_grid const &, connection_patterns const &, permutes const &,
      block_sparse_matrix<double> const &,
      double, double const[], double, double[], workspace<double> &);

#ifdef ASGARD_USE_FLOPCOUNTER
template int64_t block_cpu<double>(
    int, sparse_grid const &, connection_patterns const &,
    permutes const &, double, double, workspace<double> &);
#endif

#endif

#ifdef ASGARD_ENABLE_FLOAT

template void block_cpu<float>(
    int, sparse_grid const &, connection_patterns const &, permutes const &,
    std::array<block_sparse_matrix<float>, max_num_dimensions> const &,
    float, float const[], float, float[], workspace<float> &);

template void block_cpu<float>(
      int, sparse_grid const &, connection_patterns const &, permutes const &,
      block_sparse_matrix<float> const &,
      float, float const[], float, float[], workspace<float> &);

#ifdef ASGARD_USE_FLOPCOUNTER
template int64_t block_cpu<float>(
    int, sparse_grid const &, connection_patterns const &,
    permutes const &, float, float, workspace<float> &);
#endif

#endif

} // namespace asgard::kronmult
