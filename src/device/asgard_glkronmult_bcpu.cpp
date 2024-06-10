
#include "asgard_kronmult.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace asgard::kronmult
{
#ifdef KRON_MODE_GLOBAL_BLOCK

template<typename precision, int num_dimensions, int dim, int n>
void gbkron_mult_add(precision const A[], precision const x[], precision y[])
{
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
        for (int s = 0; s < n; s++)
          for (int k = 0; k < n; k++)
            y[s * n + k] += A[j * n + s] * x[n * j + k];
    }
    else
    {
      ASGARD_PRAGMA_OMP_SIMD(collapse(3))
      for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
          for (int s = 0; s < n; s++)
            y[k * n + s] += A[j * n + s] * x[k * n + j];
    }
  }
}

template<typename precision, permutes::matrix_fill fill, int num_dimensions, int dim, int n>
void global_cpu(int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                connect_1d const &conn,
                std::vector<precision> const &vals,
                precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  constexpr int n2 = n * n;

  int const num_vecs = dsort.num_vecs(dim);

  int const max_threads = omp_get_max_threads();
  if (static_cast<int>(row_wspace.size()) < max_threads)
    row_wspace.resize(max_threads);

  int threadid = 0;
#pragma omp parallel
  {
    int tid;
#pragma omp critical
    tid = threadid++;

    // xidx holds indexes for the entries of the current
    // sparse row that are present in the current ilist
    std::vector<int64_t> &xidx = row_wspace[tid];
    if (static_cast<int>(xidx.size()) < conn.num_rows())
      xidx.resize(conn.num_rows(), -1);

#pragma omp parallel for schedule(dynamic)
    for (int vec_id = 0; vec_id < num_vecs; vec_id++)
    {
      int const vec_begin = dsort.vec_begin(dim, vec_id);
      int const vec_end   = dsort.vec_end(dim, vec_id);
      // map the indexes of present entries
      for (int j=vec_begin; j<vec_end; j++)
        xidx[ dsort(ilist, dim, j) ] = dsort.map(dim, j) * block_size;

      // matrix-vector product using xidx as a row
      for (int rj=vec_begin; rj<vec_end; rj++)
      {
        // row in the 1d pattern
        int const row = dsort(ilist, dim, rj);
        precision *local_y = &y[ xidx[row] ];

        // columns for the 1d pattern
        int col_begin = (fill == permutes::matrix_fill::upper) ? conn.row_diag(row) : conn.row_begin(row);
        int col_end   = (fill == permutes::matrix_fill::lower) ? conn.row_diag(row) : conn.row_end(row);

        for (int c = col_begin; c < col_end; c++)
        {
          int const j = conn[c];
          if (x[ xidx[j] ] != -1)
            gbkron_mult_add<precision, num_dimensions, dim, n>(&vals[n2 * j], &x[ xidx[j] ], local_y);
        }
      }

      // restore the entries
      for(int j=vec_begin; j<vec_end; j++)
        xidx[ dsort(ilist, dim, j) ] = -1;
    }
  }
}

template<typename precision, permutes::matrix_fill fill, int num_dimensions, int dim>
void global_cpu(int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                connect_1d const &conn,
                std::vector<precision> const &vals,
                precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  switch(n)
  {
    case 1: // pwconstant
      global_cpu<precision, fill, num_dimensions, dim, 1>(block_size, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 2: // linear
      global_cpu<precision, fill, num_dimensions, dim, 2>(block_size, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 3: // quadratic
      global_cpu<precision, fill, num_dimensions, dim, 3>(block_size, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 4: // cubic
      global_cpu<precision, fill, num_dimensions, dim, 4>(block_size, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    default:
      throw std::runtime_error("(kronmult) unimplemented n for given number of dims");
  };
}

template<typename precision, permutes::matrix_fill fill, int num_dimensions>
void global_cpu(int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                int dim, connect_1d const &conn,
                std::vector<precision> const &vals,
                precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  expect(dim < num_dimensions);
  if constexpr (num_dimensions == 1)
  {
    global_cpu<precision, fill, num_dimensions, 0>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
  }
  else if constexpr (num_dimensions == 2)
  {
    if (dim == 0)
      global_cpu<precision, fill, num_dimensions, 0>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
    else
      global_cpu<precision, fill, num_dimensions, 1>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
  }
  else if constexpr (num_dimensions == 3)
  {
    switch(dim)
    {
      case 0:
        global_cpu<precision, fill, num_dimensions, 0>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 1:
        global_cpu<precision, fill, num_dimensions, 1>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      default: // case 2:
        global_cpu<precision, fill, num_dimensions, 2>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
    }
  }
  else if constexpr (num_dimensions == 4)
  {
    switch(dim)
    {
      case 0:
        global_cpu<precision, fill, num_dimensions, 0>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 1:
        global_cpu<precision, fill, num_dimensions, 1>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 2:
        global_cpu<precision, fill, num_dimensions, 2>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      default: // case 3:
        global_cpu<precision, fill, num_dimensions, 3>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
    }
  }
  else if constexpr (num_dimensions == 5)
  {
    switch(dim)
    {
      case 0:
        global_cpu<precision, fill, num_dimensions, 0>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 1:
        global_cpu<precision, fill, num_dimensions, 1>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 2:
        global_cpu<precision, fill, num_dimensions, 2>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 3:
        global_cpu<precision, fill, num_dimensions, 3>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      default: // case 4:
        global_cpu<precision, fill, num_dimensions, 4>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
    }
  }
  else // num_dimensions == 6
  {
    switch(dim)
    {
      case 0:
        global_cpu<precision, fill, num_dimensions, 0>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 1:
        global_cpu<precision, fill, num_dimensions, 1>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 2:
        global_cpu<precision, fill, num_dimensions, 2>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 3:
        global_cpu<precision, fill, num_dimensions, 3>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      case 4:
        global_cpu<precision, fill, num_dimensions, 4>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
      default: // case 5:
        global_cpu<precision, fill, num_dimensions, 5>(n, block_size, ilist, dsort, conn, vals, x, y, row_wspace);
        break;
    }
  }
}

template<typename precision, permutes::matrix_fill fill>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                int dim, connect_1d const &conn,
                std::vector<precision> const &vals,
                precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  switch(num_dimensions)
  {
    case 1:
      global_cpu<precision, fill, 1>(n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
    case 2:
      global_cpu<precision, fill, 2>(n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
    case 3:
      global_cpu<precision, fill, 3>(n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
    case 4:
      global_cpu<precision, fill, 4>(n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
    case 5:
      global_cpu<precision, fill, 5>(n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
    case 6:
      global_cpu<precision, fill, 6>(n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
    default:
      throw std::runtime_error("(kronmult) works with only up to 6 dimensions");
  };
}

template<typename precision>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                int dim, permutes::matrix_fill fill, connect_1d const &conn,
                std::vector<precision> const &vals,
                precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  switch(fill)
  {
    case permutes::matrix_fill::lower:
      global_cpu<precision, permutes::matrix_fill::lower>
        (num_dimensions, n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
    case permutes::matrix_fill::upper:
      global_cpu<precision, permutes::matrix_fill::upper>
        (num_dimensions, n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
    default: // case permutes::matrix_fill::both:
      global_cpu<precision, permutes::matrix_fill::both>
        (num_dimensions, n, block_size, ilist, dsort, dim, conn, vals, x, y, row_wspace);
      break;
  }
}

template<typename precision>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                std::vector<permutes> const &perms,
                std::vector<bool> const &has_flux,
                connect_1d const &conn_volumes, connect_1d const &conn_full,
                std::vector<std::vector<precision>> const &gvals,
                std::vector<int> const &terms,
                precision const x[], precision y[],
                block_global_workspace<precision> &workspace)
{
  int64_t const num_entries = block_size * ilist.total_size();

  if (static_cast<int64_t>(workspace.w1.size()) < num_entries)
    workspace.w1.resize(num_entries);
  if (static_cast<int64_t>(workspace.w2.size()) < num_entries)
    workspace.w2.resize(num_entries);

  precision *w1 = workspace.w1.data();
  precision *w2 = workspace.w2.data();

  for (int t : terms)
  {
    // terms can have different effective dimension, since some of them are identity
    permutes const &perm = perms[t];
    int const dims       = perm.num_dimensions();
    if (dims == 0)
      continue;

    for (size_t i = 0; i < perm.fill.size(); i++)
    {
      int dir = perm.direction[i][0];

      global_cpu(num_dimensions, n, block_size, ilist, dsort, dir, perm.fill[i][0],
                 (perm.fill[i][0] == permutes::matrix_fill::both and has_flux[t]) ? conn_full : conn_volumes,
                 gvals[t * num_dimensions + dir], x, w1, workspace.row_map);

      for (int d = 1; d < dims; d++)
      {
        dir = perm.direction[i][d];
        global_cpu(num_dimensions, n, block_size, ilist, dsort, dir, perm.fill[i][d],
                   (perm.fill[i][d] == permutes::matrix_fill::both and has_flux[t]) ? conn_full : conn_volumes,
                   gvals[t * num_dimensions + dir], w1, w2, workspace.row_map);
        std::swap(w1, w2);
      }

#pragma omp parallel for
      for (int64_t j = 0; j < num_entries; j++)
        y[j] += w1[j];
    }
  }
}


#ifdef ASGARD_ENABLE_DOUBLE

template void global_cpu<double>(int, int, int64_t,
                                 vector2d<int> const &, dimension_sort const &,
                                 std::vector<permutes> const &,
                                 std::vector<bool> const &, connect_1d const &,
                                 connect_1d const &, std::vector<std::vector<double>> const &,
                                 std::vector<int> const &, double const[], double[],
                                 block_global_workspace<double> &);

#endif

#ifdef ASGARD_ENABLE_FLOAT

template void global_cpu<float>(int, int, int64_t, std::vector<permutes> const &,
                                vector2d<int> const &, dimension_sort const &,
                                std::vector<bool> const &, connect_1d const &,
                                connect_1d const &, std::vector<std::vector<float>> const &,
                                std::vector<int> const &, float const[], float[],
                                block_global_workspace<float> &);
#endif

#endif
} // namespace asgard::kronmult
