
#include "asgard_kronmult.hpp"

namespace asgard::kronmult
{
#ifdef KRON_MODE_GLOBAL
template<typename precision>
void global_cpu_one(vector2d<int> const &ilist, dimension_sort const &dsort,
                    int dim, permutes::matrix_fill fill,
                    connect_1d const &conn, precision const *vals,
                    precision const *x, precision *y)
{
  std::fill_n(y, ilist.num_strips(), precision{0});

  // the sort splits the index set into sparse vectors
  int const num_vecs = dsort.num_vecs(dim);

  // loop over all the sparse vectors
#pragma omp parallel for
  for (int vec_id = 0; vec_id < num_vecs; vec_id++)
  {
    // the vector is between vec_begin(dim, vec_id) and vec_end(dim, vec_id)
    // the vector has to be multiplied by the upper/lower/both portion
    // of a sparse matrix
    // sparse matrix times a sparse vector requires pattern matching
    int const vec_begin = dsort.vec_begin(dim, vec_id);
    int const vec_end   = dsort.vec_end(dim, vec_id);

    for (int j = vec_begin; j < vec_end; j++)
    {
      int row = dsort(ilist, dim, j); // 1D index of this output in y

      int mat_begin = (fill == permutes::matrix_fill::upper) ? conn.row_diag(row) : conn.row_begin(row);
      int mat_end   = (fill == permutes::matrix_fill::lower) ? conn.row_diag(row) : conn.row_end(row);

      // loop over the matrix row and the vector looking for matching non-zeros
      int mat_j = mat_begin;
      int vec_j = vec_begin;
      while (mat_j < mat_end and vec_j < vec_end)
      {
        int const vec_index = dsort(ilist, dim, vec_j); // pattern index 1d
        int const mat_index = conn[mat_j];
        // the sort helps here, since indexes are in order, it is easy to
        // match the index patterns
        if (vec_index < mat_index)
        {
          vec_j += 1;
          // search_counter += 1;
        }
        else if (mat_index < vec_index)
        {
          mat_j += 1;
          //search_counter += 1;
        }
        else // mat_index == vec_index, found matching entry, add to output
        {
          // flop_counter += 1;
          y[dsort.map(dim, j)] += x[dsort.map(dim, vec_j)] * vals[mat_j];

          // entry match, increment both indexes for the pattern
          vec_j += 1;
          mat_j += 1;
        }
      }
    }
  }
}

template<typename precision>
void global_cpu(permutes const &perms,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                connect_1d const &conn, std::vector<int> const &terms,
                std::vector<std::vector<precision>> const &vals,
                precision alpha, precision const *x, precision *y,
                precision *w1, precision *w2)
{
  int const num_dimensions = ilist.stride();
  if (num_dimensions == 1) // no need to split anything
  {
    for (int t : terms)
    {
      global_cpu_one(ilist, dsort, 0, permutes::matrix_fill::both, conn,
                     vals[t].data(), x, w1);
      lib_dispatch::axpy<resource::host>(ilist.num_strips(), alpha, w1, 1, y, 1);
    }
  }
  else
  {
    for (int t : terms)
    {
      for (size_t i = 0; i < perms.fill.size(); i++)
      {
        global_cpu_one(ilist, dsort, perms.direction[i][0], perms.fill[i][0],
                       conn, vals[t * num_dimensions + perms.direction[i][0]].data(), x, w1);
        for (int d = 1; d < num_dimensions; d++)
        {
          global_cpu_one(ilist, dsort, perms.direction[i][d], perms.fill[i][d],
                         conn, vals[t * num_dimensions + perms.direction[i][d]].data(), w1, w2);
          std::swap(w1, w2);
        }
        lib_dispatch::axpy<resource::host>(ilist.num_strips(), alpha, w1, 1, y, 1);
      }
    }
  }
}

template<typename precision>
void global_cpu_one(permutes::matrix_fill fill, int64_t num_rows,
                    std::vector<int> const &pntr,
                    std::vector<int> const &indx,
                    std::vector<int> const &diag,
                    std::vector<precision> const &vals,
                    precision const *x, precision *y)
{
  switch (fill)
  {
  case permutes::matrix_fill::upper:
#pragma omp parallel for
    for (int64_t r = 0; r < num_rows; r++)
    {
      y[r] = 0;
      ASGARD_PRAGMA_OMP_SIMD()
      for (int j = diag[r]; j < pntr[r + 1]; j++)
        y[r] += vals[j] * x[indx[j]];
    }
    break;
  case permutes::matrix_fill::lower:
#pragma omp parallel for
    for (int64_t r = 0; r < num_rows; r++)
    {
      y[r] = 0;
      ASGARD_PRAGMA_OMP_SIMD()
      for (int j = pntr[r]; j < diag[r]; j++)
        y[r] += vals[j] * x[indx[j]];
    }
    break;
  case permutes::matrix_fill::both:
#pragma omp parallel for
    for (int64_t r = 0; r < num_rows; r++)
    {
      y[r] = 0;
      ASGARD_PRAGMA_OMP_SIMD()
      for (int j = pntr[r]; j < pntr[r + 1]; j++)
        y[r] += vals[j] * x[indx[j]];
    }
    break;
  }
}

template<typename precision>
void global_cpu(int num_dimensions,
                std::vector<permutes> const &perms,
                std::vector<std::vector<int>> const &gpntr,
                std::vector<std::vector<int>> const &gindx,
                std::vector<std::vector<int>> const &gdiag,
                std::vector<std::vector<precision>> const &gvals,
                std::vector<int> const &terms, precision const *x, precision *y,
                precision *w1, precision *w2)
{
  int64_t const num_rows = static_cast<int64_t>(gpntr.front().size() - 1);

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
      global_cpu_one(perm.fill[i][0], num_rows, gpntr[dir], gindx[dir],
                     gdiag[dir], gvals[t * num_dimensions + dir], x, w1);

      for (int d = 1; d < dims; d++)
      {
        dir = perm.direction[i][d];
        global_cpu_one(perm.fill[i][d], num_rows, gpntr[dir], gindx[dir],
                       gdiag[dir], gvals[t * num_dimensions + dir], w1, w2);
        std::swap(w1, w2);
      }

#pragma omp parallel for
      for (int64_t j = 0; j < num_rows; j++)
        y[j] += w1[j];
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template void global_cpu(int, std::vector<permutes> const &,
                         std::vector<std::vector<int>> const &,
                         std::vector<std::vector<int>> const &,
                         std::vector<std::vector<int>> const &,
                         std::vector<std::vector<double>> const &,
                         std::vector<int> const &, double const *, double *,
                         double *, double *);

template void global_cpu(permutes const &perms,
                         vector2d<int> const &ilist, dimension_sort const &dsort,
                         connect_1d const &conn, std::vector<int> const &terms,
                         std::vector<std::vector<double>> const &vals,
                         double alpha, double const *x, double *y,
                         double *, double *);
#endif
#ifdef ASGARD_ENABLE_FLOAT
template void global_cpu(int, std::vector<permutes> const &,
                         std::vector<std::vector<int>> const &,
                         std::vector<std::vector<int>> const &,
                         std::vector<std::vector<int>> const &,
                         std::vector<std::vector<float>> const &,
                         std::vector<int> const &, float const *, float *,
                         float *, float *);
template void global_cpu(permutes const &perms,
                         vector2d<int> const &ilist, dimension_sort const &dsort,
                         connect_1d const &conn, std::vector<int> const &terms,
                         std::vector<std::vector<float>> const &vals,
                         float alpha, float const *x, float *y,
                         float *, float *);
#endif

#endif
} // namespace asgard::kronmult
