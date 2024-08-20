
#include "asgard_kronmult.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace asgard::kronmult
{
#ifdef KRON_MODE_GLOBAL

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

int64_t number_of_blocks_;

template<typename precision, permutes::matrix_fill fill, int num_dimensions, int dim, int n>
void global_cpu(vector2d<int> const &ilist,
                dimension_sort const &dsort, connect_1d const &conn,
                precision const vals[], precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  constexpr int n2 = n * n;

  constexpr int64_t block_size = ipow<n, num_dimensions>();

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
        xidx[dsort(ilist, dim, j)] = dsort.map(dim, j) * block_size;

      // matrix-vector product using xidx as a row
      for (int rj = vec_begin; rj < vec_end; rj++)
      {
        // row in the 1d pattern
        int const row = dsort(ilist, dim, rj);

        precision *const local_y = y + xidx[row];

        // columns for the 1d pattern
        int col_begin = (fill == permutes::matrix_fill::upper) ? conn.row_diag(row) : conn.row_begin(row);
        int col_end   = (fill == permutes::matrix_fill::lower) ? conn.row_diag(row) : conn.row_end(row);

        if constexpr (n != -1)
          for (int j = 0; j < block_size; j++)
            local_y[j] = precision{0};

        for (int c = col_begin; c < col_end; c++)
        {
          int64_t const xj = xidx[conn[c]];
          if (xj != -1)
          {
            if constexpr (n == -1)
#pragma omp atomic
              number_of_blocks_ += 1;
            else
              gbkron_mult_add<precision, num_dimensions, dim, n>(vals + n2 * c, x + xj, local_y);
          }
        }
      }

      // restore the entries
      for (int j = vec_begin; j < vec_end; j++)
        xidx[dsort(ilist, dim, j)] = -1;
    }
  }
}

template<typename precision, int num_dimensions, int dim, int n>
void globalsv_cpu(vector2d<int> const &ilist, dimension_sort const &dsort,
                  connect_1d const &conn, precision const vals[],
                  precision y[], std::vector<std::vector<int64_t>> &row_wspace)
{
  constexpr int n2 = n * n;

  constexpr int64_t block_size = ipow<n, num_dimensions>();

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
        xidx[dsort(ilist, dim, j)] = dsort.map(dim, j) * block_size;

      // matrix-vector product using xidx as a row
      for (int rj = vec_begin; rj < vec_end; rj++)
      {
        // row in the 1d pattern
        int const row = dsort(ilist, dim, rj);

        precision *const local_y = y + xidx[row];

        // columns for the 1d pattern, lower part only
        int col_begin = conn.row_begin(row);
        int col_end   = conn.row_diag(row);

        for (int c = col_begin; c < col_end; c++)
        {
          int64_t const xj = xidx[conn[c]];
          if (xj != -1)
          {
            if constexpr (n == -1)
#pragma omp atomic
              ++number_of_blocks_;
            else
              gbkron_mult_add<precision, num_dimensions, dim, n>(vals + n2 * c, y + xj, local_y);
          }
        }
      }

      // restore the entries
      for (int j = vec_begin; j < vec_end; j++)
        xidx[dsort(ilist, dim, j)] = -1;
    }
  }
}

template<typename precision, permutes::matrix_fill fill, int num_dimensions, int dim>
void global_cpu(int n, vector2d<int> const &ilist, dimension_sort const &dsort,
                connect_1d const &conn, precision const vals[],
                precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  switch (n)
  {
  case -1: // special case: count the number of flops
    global_cpu<precision, fill, num_dimensions, dim, -1>(ilist, dsort, conn, vals, x, y, row_wspace);
    break;
  case 1: // pwconstant
    global_cpu<precision, fill, num_dimensions, dim, 1>(ilist, dsort, conn, vals, x, y, row_wspace);
    break;
  case 2: // linear
    global_cpu<precision, fill, num_dimensions, dim, 2>(ilist, dsort, conn, vals, x, y, row_wspace);
    break;
  case 3: // quadratic
    global_cpu<precision, fill, num_dimensions, dim, 3>(ilist, dsort, conn, vals, x, y, row_wspace);
    break;
  case 4: // cubic
    global_cpu<precision, fill, num_dimensions, dim, 4>(ilist, dsort, conn, vals, x, y, row_wspace);
    break;
  default:
    throw std::runtime_error("(kronmult) unimplemented n for given number of dims");
  };
}

template<typename precision, int num_dimensions, int dim>
void globalsv_cpu(int n, vector2d<int> const &ilist, dimension_sort const &dsort,
                  connect_1d const &conn, precision const vals[],
                  precision y[], std::vector<std::vector<int64_t>> &row_wspace)
{
  switch (n)
  {
  case -1: // special case: count the number of flops
    globalsv_cpu<precision, num_dimensions, dim, -1>(ilist, dsort, conn, vals, y, row_wspace);
    break;
  case 1: // pwconstant
    globalsv_cpu<precision, num_dimensions, dim, 1>(ilist, dsort, conn, vals, y, row_wspace);
    break;
  case 2: // linear
    globalsv_cpu<precision, num_dimensions, dim, 2>(ilist, dsort, conn, vals, y, row_wspace);
    break;
  case 3: // quadratic
    globalsv_cpu<precision, num_dimensions, dim, 3>(ilist, dsort, conn, vals, y, row_wspace);
    break;
  case 4: // cubic
    globalsv_cpu<precision, num_dimensions, dim, 4>(ilist, dsort, conn, vals, y, row_wspace);
    break;
  default:
    throw std::runtime_error("(kronmult-sv) unimplemented n for given number of dims");
  };
}

template<typename precision, permutes::matrix_fill fill, int num_dimensions>
void global_cpu(int n, vector2d<int> const &ilist,
                dimension_sort const &dsort, int dim, connect_1d const &conn,
                precision const vals[], precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  expect(dim < num_dimensions);
  if constexpr (num_dimensions == 1)
  {
    global_cpu<precision, fill, num_dimensions, 0>(n, ilist, dsort, conn, vals, x, y, row_wspace);
  }
  else if constexpr (num_dimensions == 2)
  {
    if (dim == 0)
      global_cpu<precision, fill, num_dimensions, 0>(n, ilist, dsort, conn, vals, x, y, row_wspace);
    else
      global_cpu<precision, fill, num_dimensions, 1>(n, ilist, dsort, conn, vals, x, y, row_wspace);
  }
  else if constexpr (num_dimensions == 3)
  {
    switch (dim)
    {
    case 0:
      global_cpu<precision, fill, num_dimensions, 0>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 1:
      global_cpu<precision, fill, num_dimensions, 1>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    default: // case 2:
      global_cpu<precision, fill, num_dimensions, 2>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    }
  }
  else if constexpr (num_dimensions == 4)
  {
    switch (dim)
    {
    case 0:
      global_cpu<precision, fill, num_dimensions, 0>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 1:
      global_cpu<precision, fill, num_dimensions, 1>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 2:
      global_cpu<precision, fill, num_dimensions, 2>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    default: // case 3:
      global_cpu<precision, fill, num_dimensions, 3>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    }
  }
  else if constexpr (num_dimensions == 5)
  {
    switch (dim)
    {
    case 0:
      global_cpu<precision, fill, num_dimensions, 0>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 1:
      global_cpu<precision, fill, num_dimensions, 1>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 2:
      global_cpu<precision, fill, num_dimensions, 2>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 3:
      global_cpu<precision, fill, num_dimensions, 3>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    default: // case 4:
      global_cpu<precision, fill, num_dimensions, 4>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    }
  }
  else // num_dimensions == 6
  {
    switch (dim)
    {
    case 0:
      global_cpu<precision, fill, num_dimensions, 0>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 1:
      global_cpu<precision, fill, num_dimensions, 1>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 2:
      global_cpu<precision, fill, num_dimensions, 2>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 3:
      global_cpu<precision, fill, num_dimensions, 3>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    case 4:
      global_cpu<precision, fill, num_dimensions, 4>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    default: // case 5:
      global_cpu<precision, fill, num_dimensions, 5>(n, ilist, dsort, conn, vals, x, y, row_wspace);
      break;
    }
  }
}

template<typename precision, int num_dimensions>
void globalsv_cpu(int n, vector2d<int> const &ilist, dimension_sort const &dsort,
                  int dim, connect_1d const &conn, precision const vals[],
                  precision y[], std::vector<std::vector<int64_t>> &row_wspace)
{
  expect(dim < num_dimensions);
  if constexpr (num_dimensions == 1)
  {
    globalsv_cpu<precision, num_dimensions, 0>(n, ilist, dsort, conn, vals, y, row_wspace);
  }
  else if constexpr (num_dimensions == 2)
  {
    if (dim == 0)
      globalsv_cpu<precision, num_dimensions, 0>(n, ilist, dsort, conn, vals, y, row_wspace);
    else
      globalsv_cpu<precision, num_dimensions, 1>(n, ilist, dsort, conn, vals, y, row_wspace);
  }
  else if constexpr (num_dimensions == 3)
  {
    switch (dim)
    {
    case 0:
      globalsv_cpu<precision, num_dimensions, 0>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 1:
      globalsv_cpu<precision, num_dimensions, 1>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    default: // case 2:
      globalsv_cpu<precision, num_dimensions, 2>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    }
  }
  else if constexpr (num_dimensions == 4)
  {
    switch (dim)
    {
    case 0:
      globalsv_cpu<precision, num_dimensions, 0>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 1:
      globalsv_cpu<precision, num_dimensions, 1>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 2:
      globalsv_cpu<precision, num_dimensions, 2>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    default: // case 3:
      globalsv_cpu<precision, num_dimensions, 3>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    }
  }
  else if constexpr (num_dimensions == 5)
  {
    switch (dim)
    {
    case 0:
      globalsv_cpu<precision, num_dimensions, 0>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 1:
      globalsv_cpu<precision, num_dimensions, 1>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 2:
      globalsv_cpu<precision, num_dimensions, 2>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 3:
      globalsv_cpu<precision, num_dimensions, 3>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    default: // case 4:
      globalsv_cpu<precision, num_dimensions, 4>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    }
  }
  else // num_dimensions == 6
  {
    switch (dim)
    {
    case 0:
      globalsv_cpu<precision, num_dimensions, 0>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 1:
      globalsv_cpu<precision, num_dimensions, 1>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 2:
      globalsv_cpu<precision, num_dimensions, 2>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 3:
      globalsv_cpu<precision, num_dimensions, 3>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    case 4:
      globalsv_cpu<precision, num_dimensions, 4>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    default: // case 5:
      globalsv_cpu<precision, num_dimensions, 5>(n, ilist, dsort, conn, vals, y, row_wspace);
      break;
    }
  }
}

template<typename precision, permutes::matrix_fill fill>
void global_cpu(int num_dimensions, int n, vector2d<int> const &ilist,
                dimension_sort const &dsort, int dim, connect_1d const &conn,
                precision const vals[], precision const x[], precision y[],
                std::vector<std::vector<int64_t>> &row_wspace)
{
  switch (num_dimensions)
  {
  case 1:
    global_cpu<precision, fill, 1>(n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  case 2:
    global_cpu<precision, fill, 2>(n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  case 3:
    global_cpu<precision, fill, 3>(n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  case 4:
    global_cpu<precision, fill, 4>(n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  case 5:
    global_cpu<precision, fill, 5>(n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  case 6:
    global_cpu<precision, fill, 6>(n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  default:
    throw std::runtime_error("(kronmult) works with only up to 6 dimensions");
  };
}

template<typename precision>
void globalsv_cpu(int num_dimensions, int n, vector2d<int> const &ilist,
                  dimension_sort const &dsort, int dim, connect_1d const &conn,
                  precision const vals[], precision y[],
                  std::vector<std::vector<int64_t>> &row_wspace)
{
  switch (num_dimensions)
  {
  case 1:
    globalsv_cpu<precision, 1>(n, ilist, dsort, dim, conn, vals, y, row_wspace);
    break;
  case 2:
    globalsv_cpu<precision, 2>(n, ilist, dsort, dim, conn, vals, y, row_wspace);
    break;
  case 3:
    globalsv_cpu<precision, 3>(n, ilist, dsort, dim, conn, vals, y, row_wspace);
    break;
  case 4:
    globalsv_cpu<precision, 4>(n, ilist, dsort, dim, conn, vals, y, row_wspace);
    break;
  case 5:
    globalsv_cpu<precision, 5>(n, ilist, dsort, dim, conn, vals, y, row_wspace);
    break;
  case 6:
    globalsv_cpu<precision, 6>(n, ilist, dsort, dim, conn, vals, y, row_wspace);
    break;
  default:
    throw std::runtime_error("(kronmult) works with only up to 6 dimensions");
  };
}

template<typename precision>
void global_cpu(int num_dimensions, int n, vector2d<int> const &ilist,
                dimension_sort const &dsort, int dim, permutes::matrix_fill fill,
                connect_1d const &conn, precision const vals[], precision const x[],
                precision y[], std::vector<std::vector<int64_t>> &row_wspace)
{
  switch (fill)
  {
  case permutes::matrix_fill::lower:
    global_cpu<precision, permutes::matrix_fill::lower>(num_dimensions, n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  case permutes::matrix_fill::upper:
    global_cpu<precision, permutes::matrix_fill::upper>(num_dimensions, n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  default: // case permutes::matrix_fill::both:
    global_cpu<precision, permutes::matrix_fill::both>(num_dimensions, n, ilist, dsort, dim, conn, vals, x, y, row_wspace);
    break;
  }
}

template<typename precision>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                std::vector<permutes> const &perms,
                std::vector<int> const &flux_dir,
                connect_1d const &conn_volumes, connect_1d const &conn_full,
                std::vector<std::vector<precision>> const &gvals,
                std::vector<int> const &terms,
                precision const x[], precision y[],
                block_global_workspace<precision> &workspace)
{
  int64_t const num_entries = block_size * ilist.num_strips();

  if (static_cast<int64_t>(workspace.w1.size()) < num_entries)
    workspace.w1.resize(num_entries);
  if (static_cast<int64_t>(workspace.w2.size()) < num_entries)
    workspace.w2.resize(num_entries);

  precision *w1 = workspace.w1.data();
  precision *w2 = workspace.w2.data();

  auto get_connect_1d = [&](int const &fdir, permutes::matrix_fill const &fill)
      -> connect_1d const & {
    // if the term has flux, i.e., fdir != -1
    // then the direction using fill::both will use the flux+volume connectivity
    // otherwise we will use only the volume connectivity
    if (fdir != -1 and fill == permutes::matrix_fill::both)
      return conn_full;
    else
      return conn_volumes;
  };

  for (int t : terms)
  {
    // terms can have different effective dimension, since some of them are identity
    permutes const &perm  = perms[t];
    int const active_dims = perm.num_dimensions();
    if (active_dims == 0)
      continue;

    for (size_t i = 0; i < perm.fill.size(); i++)
    {
      int dir = perm.direction[i][0];

      global_cpu(num_dimensions, n, ilist, dsort, dir, perm.fill[i][0],
                 get_connect_1d(flux_dir[t], perm.fill[i][0]),
                 gvals[t * num_dimensions + dir].data(), x, w1, workspace.row_map);

      for (int d = 1; d < active_dims; d++)
      {
        dir = perm.direction[i][d];
        global_cpu(num_dimensions, n, ilist, dsort, dir, perm.fill[i][d],
                   get_connect_1d(flux_dir[t], perm.fill[i][d]),
                   gvals[t * num_dimensions + dir].data(), w1, w2, workspace.row_map);
        std::swap(w1, w2);
      }

#pragma omp parallel for
      for (int64_t j = 0; j < num_entries; j++)
        y[j] += w1[j];
    }
  }
}

template<typename precision>
int64_t block_global_count_flops(
    int num_dimensions, int64_t block_size,
    vector2d<int> const &ilist, dimension_sort const &dsort,
    std::vector<permutes> const &perms,
    std::vector<int> const &flux_dir,
    connect_1d const &conn_volumes, connect_1d const &conn_full,
    std::vector<int> const &terms,
    block_global_workspace<precision> &workspace)
{
  number_of_blocks_ = 0;

  for (int const t : terms)
  {
    // terms can have different effective dimension, since some of them are identity
    permutes const &perm  = perms[t];
    int const active_dims = perm.num_dimensions();
    if (active_dims == 0)
      continue;

    for (size_t i = 0; i < perm.fill.size(); i++)
    {
      for (int d = 0; d < active_dims; d++)
      {
        global_cpu<precision>(num_dimensions, -1, ilist, dsort, perm.direction[i][d], perm.fill[i][d],
                              (perm.fill[i][d] == permutes::matrix_fill::both and flux_dir[t] != -1) ? conn_full : conn_volumes,
                              nullptr, nullptr, nullptr, workspace.row_map);
      }
    }
  }

  return number_of_blocks_ * block_size;
}

template<typename precision>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                permutes const &perm, connect_1d const &vconn,
                precision const gvals[], precision alpha, precision const x[],
                precision y[], block_global_workspace<precision> &workspace)
{
  int64_t const num_entries = block_size * ilist.num_strips();

  if (static_cast<int64_t>(workspace.w1.size()) < num_entries)
    workspace.w1.resize(num_entries);
  if (static_cast<int64_t>(workspace.w2.size()) < num_entries)
    workspace.w2.resize(num_entries);

  precision *w1 = workspace.w1.data();
  precision *w2 = workspace.w2.data();

  // terms can have different effective dimension, since some of them are identity
  expect(num_dimensions == perm.num_dimensions());

  size_t const num_perms = perm.fill.size();
  for (size_t i = 0; i < num_perms; i++)
  {
    auto &dir = perm.direction[i];

    global_cpu(num_dimensions, n, ilist, dsort, dir[0], perm.fill[i][0],
               vconn, gvals, x, w1, workspace.row_map);

    for (int d = 1; d < num_dimensions; d++)
    {
      global_cpu(num_dimensions, n, ilist, dsort, dir[d], perm.fill[i][d],
                 vconn, gvals, w1, w2, workspace.row_map);
      std::swap(w1, w2);
    }

    if (i == 0)
    {
      if (perm.fill.size() == 1)
#pragma omp parallel for
        for (int64_t j = 0; j < num_entries; j++)
          y[j] = alpha * w1[j];
      else
        std::copy_n(w1, num_entries, y);
    }
    else
    {
      if (i + 1 == num_perms)
#pragma omp parallel for
        for (int64_t j = 0; j < num_entries; j++)
          y[j] = alpha * (y[j] + w1[j]);
      else
#pragma omp parallel for
        for (int64_t j = 0; j < num_entries; j++)
          y[j] += w1[j];
    }
  }
}

template<typename precision>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                permutes const &perm, connect_1d const &vconn,
                precision const gvals[], precision const x[], precision y[],
                block_global_workspace<precision> &workspace)
{
  int64_t const num_entries = block_size * ilist.num_strips();

  if (static_cast<int64_t>(workspace.w1.size()) < num_entries)
    workspace.w1.resize(num_entries);
  if (static_cast<int64_t>(workspace.w2.size()) < num_entries)
    workspace.w2.resize(num_entries);

  precision *w1 = workspace.w1.data();
  precision *w2 = workspace.w2.data();

  // terms can have different effective dimension, since some of them are identity
  expect(num_dimensions == perm.num_dimensions());

  for (size_t i = 0; i < perm.fill.size(); i++)
  {
    auto &dir = perm.direction[i];

    global_cpu(num_dimensions, n, ilist, dsort, dir[0], perm.fill[i][0],
               vconn, gvals, x, w1, workspace.row_map);

    for (int d = 1; d < num_dimensions; d++)
    {
      global_cpu(num_dimensions, n, ilist, dsort, dir[d], perm.fill[i][d],
                 vconn, gvals, w1, w2, workspace.row_map);
      std::swap(w1, w2);
    }

    if (i == 0)
      std::copy_n(w1, num_entries, y);
    else
#pragma omp parallel for
      for (int64_t j = 0; j < num_entries; j++)
        y[j] += w1[j];
  }
}

template<typename precision>
void globalsv_cpu(int num_dimensions, int n, vector2d<int> const &ilist,
                  dimension_sort const &dsort, connect_1d const &vconn,
                  precision const gvals[], precision y[],
                  block_global_workspace<precision> &workspace)
{
  for (int d = 0; d < num_dimensions; d++)
    globalsv_cpu(num_dimensions, n, ilist, dsort, d, vconn, gvals, y, workspace.row_map);
}

#ifdef ASGARD_ENABLE_DOUBLE

template void global_cpu<double>(int, int, int64_t,
                                 vector2d<int> const &, dimension_sort const &,
                                 std::vector<permutes> const &,
                                 std::vector<int> const &, connect_1d const &,
                                 connect_1d const &, std::vector<std::vector<double>> const &,
                                 std::vector<int> const &, double const[], double[],
                                 block_global_workspace<double> &);

template int64_t block_global_count_flops<double>(
    int num_dimensions, int64_t block_size,
    vector2d<int> const &ilist, dimension_sort const &dsort,
    std::vector<permutes> const &perms,
    std::vector<int> const &flux_dir,
    connect_1d const &conn_volumes, connect_1d const &conn_full,
    std::vector<int> const &terms,
    block_global_workspace<double> &workspace);

template void global_cpu<double>(
    int, int, int64_t, vector2d<int> const &, dimension_sort const &,
    permutes const &, connect_1d const &, double const[], double const[],
    double[], block_global_workspace<double> &);

template void global_cpu<double>(
    int, int, int64_t, vector2d<int> const &, dimension_sort const &,
    permutes const &, connect_1d const &, double const[], double,
    double const[], double[], block_global_workspace<double> &);

template void globalsv_cpu(
    int, int, vector2d<int> const &, dimension_sort const &,
    connect_1d const &, double const[], double[],
    block_global_workspace<double> &workspace);
#endif

#ifdef ASGARD_ENABLE_FLOAT

template void global_cpu<float>(int, int, int64_t,
                                vector2d<int> const &, dimension_sort const &,
                                std::vector<permutes> const &,
                                std::vector<int> const &, connect_1d const &,
                                connect_1d const &, std::vector<std::vector<float>> const &,
                                std::vector<int> const &, float const[], float[],
                                block_global_workspace<float> &);

template int64_t block_global_count_flops<float>(
    int num_dimensions, int64_t block_size,
    vector2d<int> const &ilist, dimension_sort const &dsort,
    std::vector<permutes> const &perms,
    std::vector<int> const &flux_dir,
    connect_1d const &conn_volumes, connect_1d const &conn_full,
    std::vector<int> const &terms,
    block_global_workspace<float> &workspace);

template void global_cpu<float>(
    int, int, int64_t, vector2d<int> const &, dimension_sort const &,
    permutes const &, connect_1d const &, float const[], float const[],
    float[], block_global_workspace<float> &);

template void global_cpu<float>(
    int, int, int64_t, vector2d<int> const &, dimension_sort const &,
    permutes const &, connect_1d const &, float const[], float,
    float const[], float[], block_global_workspace<float> &);

template void globalsv_cpu(
    int, int, vector2d<int> const &, dimension_sort const &,
    connect_1d const &, float const[], float[],
    block_global_workspace<float> &workspace);
#endif

#endif // KRON_MODE_GLOBAL
} // namespace asgard::kronmult
