#include "asgard_kronmult_matrix.hpp"
#include "batch.hpp"
#include "lib_dispatch.hpp"
#include "tools.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include <limits.h>
#include <mutex>
#include <vector>

namespace asgard
{
template<typename precision>
kronmult_matrix<precision>
make_kronmult_dense(PDE<precision> const &pde,
                    adapt::distributed_grid<precision> const &discretization,
                    options const &program_options, imex_flag const imex)
{
  // convert pde to kronmult dense matrix
  auto const &grid         = discretization.get_subgrid(get_rank());
  int const num_dimensions = pde.num_dims;
  int const kron_size      = pde.get_dimensions()[0].get_degree();
  int const num_terms      = pde.num_terms;
  int const num_rows       = grid.row_stop - grid.row_start + 1;
  int const num_cols       = grid.col_stop - grid.col_start + 1;

  int64_t lda = kron_size * fm::two_raised_to((program_options.do_adapt_levels)
                                                  ? program_options.max_level
                                                  : pde.max_level);

  int64_t osize = 0;
  std::vector<int64_t> dim_term_offset(num_terms * pde.num_dims + 1);
  for (int t = 0; t < num_terms; t++)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      dim_term_offset[t * num_dimensions + d] = osize;
      osize += pde.get_coefficients(t, d).size();
    }
  }

  fk::vector<int, mem_type::owner, resource::host> iA(
      num_rows * num_cols * num_terms * num_dimensions);
  fk::vector<precision, mem_type::owner, resource::host> vA(osize);

#ifdef ASGARD_USE_CUDA
  std::cout << "  kronmult dense matrix allocation (MB): "
            << get_MB<int>(iA.size()) + get_MB<precision>(vA.size()) << "\n";
#endif
  // will print the command to use for performance testing
  // std::cout << "./asgard_kronmult_benchmark " << num_dimensions << " " <<
  // kron_size
  //          << " " << num_rows << " " << num_terms
  //          << " " << osize / (kron_size * kron_size) << "\n";

  // load the matrices into a contiguous data-structure
  // keep the column major format
  // but stack on the columns so the new leading dimensions is kron_size
  precision *pA = vA.data();
  for (int t = 0; t < num_terms; t++)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      if (!program_options.use_imex_stepping ||
          (program_options.use_imex_stepping &&
           pde.get_terms()[t][d].flag == imex))
      {
        auto const &ops   = pde.get_coefficients(t, d); // this is an fk::matrix
        int const num_ops = ops.nrows() / kron_size;

        // the matrices of the kron products are organized into blocks
        // of a large matrix, the matrix is square with size num-ops by
        // kron-size rearrange in a sequential way (by columns) to avoid the lda
        for (int ocol = 0; ocol < num_ops; ocol++)
          for (int orow = 0; orow < num_ops; orow++)
            for (int i = 0; i < kron_size; i++)
              pA = std::copy_n(ops.data() + kron_size * orow +
                                   lda * (kron_size * ocol + i),
                               kron_size, pA);
      }
      else
      {
        pA = std::fill_n(pA, pde.get_coefficients(t, d).size(), precision{0});
      }
    }
  }

  // compute the indexes for the matrices for the kron-products
  int const *const flattened_table =
      discretization.get_table().get_active_table().data();
  std::vector<int> oprow(num_dimensions);
  std::vector<int> opcol(num_dimensions);
  auto ia = iA.begin();
  for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
  {
    int const *const row_coords = flattened_table + 2 * num_dimensions * row;
    for (int i = 0; i < num_dimensions; i++)
      oprow[i] =
          (row_coords[i] == 0)
              ? 0
              : ((1 << (row_coords[i] - 1)) + row_coords[i + num_dimensions]);

    for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
    {
      int const *const col_coords = flattened_table + 2 * num_dimensions * col;
      for (int i = 0; i < num_dimensions; i++)
        opcol[i] =
            (col_coords[i] == 0)
                ? 0
                : ((1 << (col_coords[i] - 1)) + col_coords[i + num_dimensions]);

      for (int t = 0; t < num_terms; t++)
      {
        for (int d = 0; d < num_dimensions; d++)
        {
          int64_t const num_ops =
              pde.get_coefficients(t, d).nrows() / kron_size;
          *ia++ = dim_term_offset[t * num_dimensions + d] +
                  (oprow[d] + opcol[d] * num_ops) * kron_size * kron_size;
        }
      }
    }
  }

#ifdef ASGARD_USE_CUDA
  int tensor_size = kron_size;
  for(int d=1; d<num_dimensions; d++)
    tensor_size *= kron_size;

  fk::vector<int> row_indx(num_rows * num_cols);
  fk::vector<int> col_indx(num_rows * num_cols);

  for(int i=0; i<num_rows; i++)
  {
    for(int j=0; j<num_cols; j++)
    {
      row_indx[i * num_cols + j] = i * tensor_size;
      col_indx[i * num_cols + j] = j * tensor_size;
    }
  }

  // if using CUDA, copy the matrices onto the GPU
  return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                    num_cols, num_terms,
                                    row_indx.clone_onto_device(),
                                    col_indx.clone_onto_device(),
                                    iA.clone_onto_device(),
                                    vA.clone_onto_device());
#else
  fk::vector<int> pntr(num_rows + 1);
  fk::vector<int> indx(num_rows * num_cols);

  for(int i=0; i<num_rows; i++)
  {
    pntr[i] = i * num_cols;
    for(int j=0; j<num_cols; j++)
      indx[pntr[i] + j]= j;
  }
  pntr[num_rows] = indx.size();

  // if using the CPU, move the vectors into the matrix structure
  return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                    num_cols, num_terms, std::move(pntr),
                                    std::move(indx), std::move(iA),
                                    std::move(vA));
#endif
}

template kronmult_matrix<float>
make_kronmult_dense<float>(PDE<float> const &,
                           adapt::distributed_grid<float> const &,
                           options const &, imex_flag const);
template kronmult_matrix<double>
make_kronmult_dense<double>(PDE<double> const &,
                            adapt::distributed_grid<double> const &,
                            options const &, imex_flag const);

/*!
 * \bried Returns true if the 1D elements are connected
 *
 * The two elements are defined by (level L, index within the level p),
 * the first point is (L1, p1) and we assume that L1 <= L2.
 */
inline bool check_connected(int L1, int p1, int L2, int p2)
{
  expect(L1 <= L2);

  int side = (p2 % 2 == 0) ? -1 : 1;
  while(L2 > L1)
  {
    L2--;
    p2 /= 2;
    // check is the elements on the edge of the ancestry block
    if (p2 % 2 == 0)
      side = (side == -1) ? -1 : 0;
    else
      side = (side == 1) ? 1 : 0;
  }
  // p2 == p1, then (L1, p1) is ancestor of (L2, p2) and support overlaps
  // p2 + side == p1, then the elements share a side
  return (p2 == p1) or (p2 + side == p1);
}

inline bool check_connected(int const num_dimensions, int const *const row, int const *const col)
{
  for(int j=0; j<num_dimensions; j++)
    if (row[j] <= col[j])
    {
      if (not check_connected(row[j], row[j + num_dimensions], col[j], col[j + num_dimensions]))
        return false;
    }
    else
    {
      if (not check_connected(col[j], col[j + num_dimensions], row[j], row[j + num_dimensions]))
        return false;
    }

  return true;
}

/*!
 * \brief Maps four integers to an integer.
 */
class quad2int_map
{
public:
  quad2int_map() = default;

  int find(int term, int dim, int prow, int pcol)
  {
    auto res = indexes.find(std::array<int, 4>{term, dim, prow, pcol});
    return (res != indexes.end()) ? res->second : -1;
  }

  void add(int term, int dim, int prow, int pcol, int idx)
  {
    indexes[std::array<int, 4>{term, dim, prow, pcol}] = idx;
  }

private:
  struct lex_less{
    bool operator() (std::array<int, 4> const &lhs, std::array<int, 4> const &rhs) const
    {
      for(int i=0; i<4; i++)
        if (lhs[i] < rhs[i])
          return true;
      return false;
    }
  };
  std::map<std::array<int, 4>, int, lex_less> indexes;
};

template<typename precision>
kronmult_matrix<precision>
make_kronmult_sparse(PDE<precision> const &pde,
                     adapt::distributed_grid<precision> const &discretization,
                     options const &program_options, imex_flag const imex)
{
  // convert pde to kronmult dense matrix
  auto const &grid         = discretization.get_subgrid(get_rank());
  int const num_dimensions = pde.num_dims;
  int const kron_size      = pde.get_dimensions()[0].get_degree();
  int const num_terms      = pde.num_terms;
  int const num_rows       = grid.row_stop - grid.row_start + 1;
  int const num_cols       = grid.col_stop - grid.col_start + 1;

  int64_t lda = kron_size * fm::two_raised_to((program_options.do_adapt_levels)
                                                  ? program_options.max_level
                                                  : pde.max_level);

  int const *const flattened_table =
      discretization.get_table().get_active_table().data();

// This is a bad algorithm as it loops over all possible pairs of multi-indexes
// The correct algorithm is to infer the connectivity from the sparse grid
// graph hierarchy and avoid doing so many comparisons, but that requires messy
// work with the way the indexes are stored in memory.
// To do this properly, I need a fast map from a multi-index to the matrix row
// associated with the multi-index (or the no-row if it's missing).
// The unordered_map does not provide this functionality and addition of
// the flattened table in element.hpp is not a good answer.
// Will fix in a future PR ...
  std::vector<int> ccount(num_rows, 0); // counts the connections
  for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
  {
    int const *const row_coords = flattened_table + 2 * num_dimensions * row;
    // (L, p) = (row_coords[i], row_coords[i + num_dimensions])
    for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
    {
      int const *const col_coords = flattened_table + 2 * num_dimensions * col;
      if (check_connected(num_dimensions, row_coords, col_coords))
        ccount[row]++;
    }
  }

  // num_connect is number of non-zeros of the sparse matrix
  int num_connect = std::accumulate(ccount.begin(), ccount.end(), 0);
  std::cout << "  kronmult sparse matrix fill: " << 100.0 * double(num_connect) / (double(num_rows) * double(num_cols)) << "%\n";

  std::vector<precision> vA; // dynamically copy the matrices
  fk::vector<int> iA(num_connect * num_dimensions * num_terms);
  quad2int_map indexes;

  int tensor_size = kron_size;
  for(int d=1; d<num_dimensions; d++)
    tensor_size *= kron_size;

#ifdef ASGARD_USE_CUDA
  fk::vector<int> row_indx(num_connect);
  fk::vector<int> col_indx(num_connect);
#else
  fk::vector<int> pntr(num_rows + 1);
  fk::vector<int> indx(num_connect);
  pntr[0] = 0;
  for(int i=0; i<num_rows; i++)
    pntr[i+1] = pntr[i] + ccount[i];
#endif

  int c = 0; // index over row_indx/col_indx
  std::vector<int> oprow(num_dimensions); // keeps serial point indexes
  std::vector<int> opcol(num_dimensions);

  for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
  {
    int const *const row_coords = flattened_table + 2 * num_dimensions * row;
    // (L, p) = (row_coords[i], row_coords[i + num_dimensions])
    for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
    {
      int const *const col_coords = flattened_table + 2 * num_dimensions * col;
      if (check_connected(num_dimensions, row_coords, col_coords))
      {
#ifdef ASGARD_USE_CUDA
        row_indx[c] = (row - grid.row_start) * tensor_size;
        col_indx[c] = (col - grid.col_start) * tensor_size;
#else
        indx[c] = col - grid.col_start;
#endif
        int ia = num_dimensions * num_terms * c++;

        for (int j = 0; j < num_dimensions; j++)
        {
          oprow[j] =
              (row_coords[j] == 0)
                  ? 0
                  : ((1 << (row_coords[j] - 1))
                     + row_coords[j + num_dimensions]);
          opcol[j] =
              (col_coords[j] == 0)
                  ? 0
                  : ((1 << (col_coords[j] - 1))
                     + col_coords[j + num_dimensions]);
        }

        for (int t = 0; t < num_terms; t++)
        {
          for (int d = 0; d < num_dimensions; d++)
          {
            auto const &ops   = pde.get_coefficients(t, d); // this is an fk::matrix

            int offset = indexes.find(t, d, oprow[d], opcol[d]);
            if (offset == -1)
            {
              // matrix missing, insert into the back of vA
              offset = static_cast<int>(vA.size());
              iA[ia++] = offset;
              indexes.add(t, d, oprow[d], opcol[d], offset);
              vA.resize(vA.size() + kron_size * kron_size, 0);
              if (!program_options.use_imex_stepping ||
                  (program_options.use_imex_stepping &&
                   pde.get_terms()[t][d].flag == imex))
              {
                auto pA = &vA[offset];
                for (int j = 0; j < num_dimensions; j++)
                  pA = std::copy_n(ops.data() + kron_size * oprow[d] +
                                       lda * (kron_size * opcol[j] + j),
                                   kron_size, pA);
              }
              // else, do not include due to imex flags, already padded with 0
            }
            else
            {
              iA[ia++] = offset;
            }
          }
        }
      }
    }
  }

#ifdef ASGARD_USE_CUDA
  fk::vector<precision> valsA(vA); // copy should not be needed here, but it is
  return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                    num_cols, num_terms,
                                    row_indx.clone_onto_device(),
                                    col_indx.clone_onto_device(),
                                    iA.clone_onto_device(),
                                    valsA.clone_onto_device());
#else
  // if using the CPU, move the vectors into the matrix structure
  return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                    num_cols, num_terms, std::move(pntr),
                                    std::move(indx), std::move(iA),
                                    fk::vector<precision>(vA));
#endif
}

template kronmult_matrix<float>
make_kronmult_sparse<float>(PDE<float> const &,
                            adapt::distributed_grid<float> const &,
                            options const &, imex_flag const);
template kronmult_matrix<double>
make_kronmult_sparse<double>(PDE<double> const &,
                             adapt::distributed_grid<double> const &,
                             options const &, imex_flag const);

} // namespace asgard
