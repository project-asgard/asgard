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
//! \brief Extract the actual set of terms based on options and imex flags
template<typename precision>
std::vector<int> get_used_terms(PDE<precision> const &pde, options const &opts,
                                imex_flag const imex)
{
  if (not opts.use_imex_stepping)
  {
    std::vector<int> terms(pde.num_terms);
    std::iota(terms.begin(), terms.end(), 0); // fills with 0, 1, 2, 3 ...
    return terms;
  }
  else
  {
    std::vector<int> terms;
    terms.reserve(pde.num_terms);
    for (int t = 0; t < pde.num_terms; t++)
      if (pde.get_terms()[t][0].flag == imex)
        terms.push_back(t);

    std::cerr << " terms reduced: " << pde.num_terms << " -> " << terms.size() << "\n";
    return terms;
  }
};

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
  int const num_rows       = grid.row_stop - grid.row_start + 1;
  int const num_cols       = grid.col_stop - grid.col_start + 1;

  int64_t lda = kron_size * fm::two_raised_to((program_options.do_adapt_levels)
                                                  ? program_options.max_level
                                                  : pde.max_level);

  // take into account the terms that will be skipped due to the imex_flag
  std::vector<int> const used_terms = get_used_terms(pde, program_options,
                                                     imex);
  int const num_terms = static_cast<int>(used_terms.size());

  if (used_terms.size() == 0)
    throw std::runtime_error("no terms selected in the current combination of "
                             "imex flags and options, thus must be wrong");

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

  // check if the matrix size will overflow the 'int' indexing
  int64_t size_of_indexes =
      int64_t{num_rows} * int64_t{num_cols} * num_terms * num_dimensions;
  if (size_of_indexes > (int64_t{1} << 31) - 1) // 2^31 -1 is the largest 'int'
    throw std::runtime_error(
        "the storage required for the dense matrix is too large for the 32-bit "
        "signed indexing, try running with '--kron-mode sparse'");

  fk::vector<int, mem_type::owner, resource::host> iA(
      num_rows * num_cols * num_terms * num_dimensions);
  fk::vector<precision, mem_type::owner, resource::host> vA(osize);

  // will print the command to use for performance testing
  // std::cout << "./asgard_kronmult_benchmark " << num_dimensions << " " <<
  // kron_size
  //          << " " << num_rows << " " << num_terms
  //          << " " << osize / (kron_size * kron_size) << "\n";

  // load the matrices into a contiguous data-structure
  // keep the column major format
  // but stack on the columns so the new leading dimensions is kron_size
  precision *pA = vA.data();
  for (int t : used_terms)
  {
    for (int d = 0; d < num_dimensions; d++)
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
              pde.get_coefficients(used_terms[t], d).nrows() / kron_size;
          *ia++ = dim_term_offset[t * num_dimensions + d] +
                  (oprow[d] + opcol[d] * num_ops) * kron_size * kron_size;
        }
      }
    }
  }

  int64_t flops = kronmult_matrix<precision>::compute_flops(
      num_dimensions, kron_size, num_terms, num_rows * num_cols);

  std::cout << "  kronmult dense matrix: " << num_rows << " by " << num_cols
            << "\n";
  std::cout << "        Gflops per call: " << flops * 1.E-9 << "\n";

#ifdef ASGARD_USE_CUDA
  std::cout << "  kronmult dense matrix allocation (MB): "
            << get_MB<int>(iA.size()) + get_MB<precision>(vA.size()) << "\n";

  // if using CUDA, copy the matrices onto the GPU
  return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                    num_cols, num_terms, iA.clone_onto_device(),
                                    vA.clone_onto_device());
#else
  // if using the CPU, move the vectors into the matrix structure
  return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                    num_cols, num_terms, std::move(iA),
                                    std::move(vA));
#endif
}

/*!
 * \bried Returns true if the 1D elements are connected
 *
 * The two elements are defined by (level L, index within the level is p),
 * the first point is (L1, p1) and we assume that L1 <= L2.
 */
inline bool check_connected(int L1, int p1, int L2, int p2)
{
  expect(L1 <= L2);

  // periodic boundary conditions
  // if these are left-most and right-most cells in respective levels
  // assume connected due to the periodic boundary conditions
  if ((p1 == 0 and p2 == ((1 << (L2 - 1)) - 1)) or
      (p2 == 0 and p1 == ((1 << (L1 - 1)) - 1)))
    return true;

  // same level, either same cell or neighbors
  if (L1 == L2)
    return std::abs(p1 - p2) <= 1;

  // At this point, we know that the two points live at different levels
  //   and since we require that L1 <= L2 we know that L1 < L2.
  // Now we look for the cells that connect to (L2, p2) and live at level L1.
  // The "parent" cell is obtained by recursively decreasing the level
  //   and dividing p2 by 2, when we reach L1 we will have the parent with
  //   overlapping support.
  // However, if the original point (L2, p2) lives at the edge of the support
  //   of the parent, it is also connect to the parent neighbor (left or right).
  // Left of the support means p2 % 2 == 0 and right means p2 % 2 == 1,
  //   while the neighbor is at -1 or +1 respectively.
  // (L2, p2) is at the left/right side of the parent, iff the entire ancestry
  //   is consistently at the left/right.
  // side here is initialized with dummy values and will be checked every time
  //   the level is decremented. When side ends up as -1, it means the cell is
  //   is at the left edge, +1 means right edge, 0 means not at the edge.
  int side = (p2 % 2 == 0) ? -1 : 1;
  while (L2 > L1)
  {
    // check is the elements on the edge of the ancestry block
    if (p2 % 2 == 0)
      side = (side == -1) ? -1 : 0;
    else
      side = (side == 1) ? 1 : 0;

    L2--;
    p2 /= 2;
  }
  // p2 == p1, then (L1, p1) is ancestor of (L2, p2) and support overlaps
  // p2 + side == p1, then the elements share a side
  return (p2 == p1) or (p2 + side == p1);
}

//! \brief Processes two multi-index and returns true if they are connected for all dimensions.
inline bool check_connected(int const num_dimensions, int const *const row,
                            int const *const col)
{
  for (int j = 0; j < num_dimensions; j++)
    if (row[j] <= col[j])
    {
      if (not check_connected(row[j], row[j + num_dimensions], col[j],
                              col[j + num_dimensions]))
        return false;
    }
    else
    {
      if (not check_connected(col[j], col[j + num_dimensions], row[j],
                              row[j + num_dimensions]))
        return false;
    }

  return true;
}

/*!
 * \brief Keeps track of the connectivity of the elements in the 1d hierarchy.
 *
 * Constructs a sparse matrix-like structure with row-compressed format and
 * ordered indexes within each row, so that the 1D connectivity can be verified
 * with a simple binary search.
 *
 * The advantage of the structure is to provide:
 * - easier check if two 1D cell are connected or not
 * - index of the connection, so operator coefficient matrices can be easily
 *   referenced
 *
 * TODO: This logic should be moved inside the coefficients class.
 */
class connect_1d
{
public:
  /*!
   *  \brief Constructor, makes the connectivity up to and including the given
   *         max-level.
   */
  connect_1d(int const max_level)
      : levels(max_level), cells(1 << levels), pntr(cells + 1, 0),
        indx(2 * cells)
  {
    std::vector<int> cell_per_level(levels + 2, 1);
    for (int l = 2; l < levels + 2; l++)
      cell_per_level[l] = 2 * cell_per_level[l - 1];

    // first two cells are connected to everything
    pntr[1] = cells;
    pntr[2] = 2 * cells;
    for (int i = 0; i < cells; i++)
      indx[i] = i;
    for (int i = 0; i < cells; i++)
      indx[i + cells] = i;

    // for the remaining, loop level by level, cell by cell
    for (int l = 2; l < levels + 1; l++)
    {
      int level_size = cell_per_level[l]; // number of cells in this level

      // for each cell in this level, look at all cells connected
      // look at previous levels, this level, follow on levels

      // start with the first cell, on the left edge
      int i = level_size; // index of the first cell
      // always connected to cells 0 and 1
      indx.push_back(0);
      indx.push_back(1);
      // look at cells above
      for (int upl = 2; upl < l; upl++)
      {
        // edge cell is connected to both edge cells on each level
        indx.push_back(cell_per_level[upl]);
        indx.push_back(cell_per_level[upl + 1] - 1);
      }
      // look at this level
      indx.push_back(i);
      indx.push_back(i + 1);
      // connect also to the right-most cell (periodic boundary)
      if (l > 2) // at level l = 2, i+1 is the right-most cell
        indx.push_back(cell_per_level[l + 1] - 1);
      // look at follow on levels
      for (int downl = l + 1; downl < levels + 1; downl++)
      {
        // connect to the first bunch of cell, i.e., with overlapping support
        // going on by 2, 4, 8 ... and one more for touching boundary
        // also connect to the right-most cell
        int lstart = cell_per_level[downl];
        for (int downp = 0; downp < cell_per_level[downl - l + 1] + 1; downp++)
          indx.push_back(lstart + downp);
        indx.push_back(cell_per_level[downl + 1] - 1);
      }
      pntr[i + 1] = static_cast<int>(indx.size()); // done with point

      // handle middle cells
      for (int p = 1; p < level_size - 1; p++)
      {
        i++;
        // always connected to the first two cells
        indx.push_back(0);
        indx.push_back(1);
        // ancestors on previous levels
        for (int upl = 2; upl < l; upl++)
        {
          int segment_size = cell_per_level[l - upl + 1];
          int ancestor     = p / segment_size;
          int edge         = p - ancestor * segment_size; // p % segment_size
          // if on the left edge of the ancestor
          if (edge == 0)
            indx.push_back(cell_per_level[upl] + ancestor - 1);
          indx.push_back(cell_per_level[upl] + ancestor);
          // if on the right edge of the ancestor
          if (edge == segment_size - 1)
            indx.push_back(cell_per_level[upl] + ancestor + 1);
        }
        // on this level
        indx.push_back(i - 1);
        indx.push_back(i);
        indx.push_back(i + 1);
        // kids on further levels
        int left_kid = p; // initialize, will be updated on first iteration
        int num_kids = 1;
        for (int downl = l + 1; downl < levels + 1; downl++)
        {
          left_kid *= 2;
          num_kids *= 2;
          for (int j = left_kid - 1; j < left_kid + num_kids + 1; j++)
            indx.push_back(cell_per_level[downl] + j);
        }
        pntr[i + 1] = static_cast<int>(indx.size()); // done with cell i
      }

      // right edge cell
      i++;
      // always connected to 0 and 1
      indx.push_back(0);
      indx.push_back(1);
      for (int upl = 2; upl < l; upl++)
      {
        // edge cell is connected to both edge cells on each level
        indx.push_back(cell_per_level[upl]);
        indx.push_back(cell_per_level[upl + 1] - 1);
      }
      // at this level
      // connect also to the left-most cell (periodic boundary)
      if (l > 2) // at level l = 2, left-most cell is i-1, don't double add
        indx.push_back(cell_per_level[l]);
      indx.push_back(i - 1);
      indx.push_back(i);
      // look at follow on levels
      for (int downl = l + 1; downl < levels + 1; downl++)
      {
        // left edge on the level
        indx.push_back(cell_per_level[downl]);
        // get the last bunch of cells at the level
        int lend = cell_per_level[downl + 1] - 1;
        for (int downp = cell_per_level[downl - l + 1]; downp > -1; downp--)
          indx.push_back(lend - downp);
      }
      pntr[i + 1] = static_cast<int>(indx.size()); // done with the right edge
    } // done with level, move to the next level
  }   // close the constructor

  int get_offset(int row, int col) const
  {
    // first two levels are large and trivial, no need to search
    if (row == 0)
      return col;
    else if (row == 1)
      return cells + col;
    // if not on the first or second row, do binary search
    int sstart = pntr[row], send = pntr[row + 1] - 1;
    int current = (sstart + send) / 2;
    while (sstart <= send)
    {
      if (indx[current] < col)
      {
        sstart = current + 1;
      }
      else if (indx[current] > col)
      {
        send = current - 1;
      }
      else
      {
        return current;
      };
      current = (sstart + send) / 2;
    }
    return -1;
  }

  int num_connections() const { return static_cast<int>(indx.size()); }

  int num_cells() const { return cells; }

  int row_begin(int row) const { return pntr[row]; }

  int row_end(int row) const { return pntr[row + 1]; }

  int operator[](int j) const { return indx[j]; }

private:
  int levels;
  int cells;
  std::vector<int> pntr;
  std::vector<int> indx;
};

template<typename precision>
kronmult_matrix<precision>
make_kronmult_sparse(PDE<precision> const &pde,
                     adapt::distributed_grid<precision> const &discretization,
                     options const &program_options, imex_flag const imex)
{
  auto const form_id = tools::timer.start("make-kronmult-sparse");
  // convert pde to kronmult dense matrix
  auto const &grid         = discretization.get_subgrid(get_rank());
  int const num_dimensions = pde.num_dims;
  int const kron_size      = pde.get_dimensions()[0].get_degree();
  int const num_rows       = grid.row_stop - grid.row_start + 1;
  int const num_cols       = grid.col_stop - grid.col_start + 1;

  int64_t lda = kron_size * fm::two_raised_to((program_options.do_adapt_levels)
                                                  ? program_options.max_level
                                                  : pde.max_level);

  // take into account the terms that will be skipped due to the imex_flag
  std::vector<int> const used_terms = get_used_terms(pde, program_options,
                                                     imex);
  int const num_terms = static_cast<int>(used_terms.size());

  // size of the small kron matrices
  int const kron_squared = kron_size * kron_size;

  // holds the 1D sparsity structure for the coefficient matrices
  connect_1d cells1d(pde.max_level);
  int const num_1d = cells1d.num_connections();

  // storing the 1D operator matrices by 1D row and column
  // each connected pair of 1D cells will be associated with a block
  //  of operator coefficients
  int const block1D_size = num_dimensions * num_terms * kron_squared;
  fk::vector<precision> vA(num_1d * block1D_size);
  auto pA = vA.begin();
  for (int row = 0; row < cells1d.num_cells(); row++)
  {
    for (int j = cells1d.row_begin(row); j < cells1d.row_end(row); j++)
    {
      int col = cells1d[j];
      for (int const t : used_terms)
      {
        for (int d = 0; d < num_dimensions; d++)
        {
          precision const *const ops = pde.get_coefficients(t, d).data();
          for (int k = 0; k < kron_size; k++)
            pA = std::copy_n(ops + kron_size * row +
                                 lda * (kron_size * col + k),
                             kron_size, pA);
        }
      }
    }
  }

  int const *const flattened_table =
      discretization.get_table().get_active_table().data();

  // This is a bad algorithm as it loops over all possible pairs of
  // multi-indexes The correct algorithm is to infer the connectivity from the
  // sparse grid graph hierarchy and avoid doing so many comparisons, but that
  // requires messy work with the way the indexes are stored in memory. To do
  // this properly, I need a fast map from a multi-index to the matrix row
  // associated with the multi-index (or indicate if it's missing).
  // The unordered_map does not provide this functionality and the addition of
  // the flattened table in element.hpp is not a good answer.
  // Will fix in a future PR ...

  // the algorithm use two stages, counts the non-zeros in the global matrix,
  // then fills in the associated indexes for the rows, columns and iA
  std::vector<int> ccount(num_rows, 0); // first counts the connections
#pragma omp parallel for
  for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
  {
    int const *const row_coords = flattened_table + 2 * num_dimensions * row;
    // (L, p) = (row_coords[i], row_coords[i + num_dimensions])
    for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
    {
      int const *const col_coords = flattened_table + 2 * num_dimensions * col;
      if (check_connected(num_dimensions, row_coords, col_coords))
        ccount[row - grid.row_start]++;
    }
  }

  // right now, ccount[row] is the number of connections for each row
  // later we will replace ccount by cumulative count similar to pntr
  // num_connect is the total number of non-zeros of the sparse matrix
  int num_connect = std::accumulate(ccount.begin(), ccount.end(), 0);

  fk::vector<int> iA(num_connect * num_dimensions * num_terms);

#ifdef ASGARD_USE_CUDA
  int tensor_size = kronmult_matrix<precision>::compute_tensor_size(
      num_dimensions, kron_size);

  fk::vector<int> row_indx(num_connect);
  fk::vector<int> col_indx(num_connect);
  {
    int cumulative = 0;
    for (int i = 0; i < num_rows; i++)
    {
      int current = cumulative;
      cumulative += ccount[i];
      ccount[i] = current;
    }
  }
#else
  fk::vector<int> pntr(num_rows + 1);
  fk::vector<int> indx(num_connect);
  pntr[0] = 0;
  for (int i = 0; i < num_rows; i++)
    pntr[i + 1] = pntr[i] + ccount[i];
  for (int i = 0; i < num_rows; i++)
    ccount[i] = pntr[i];
#endif

#pragma omp parallel
{
  std::vector<int> offsets(num_dimensions); // find the 1D offsets

  #pragma omp for
  for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
  {
    int c = ccount[row - grid.row_start];
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
          int const oprow = (row_coords[j] == 0)
                                ? 0
                                : ((1 << (row_coords[j] - 1)) +
                                   row_coords[j + num_dimensions]);
          int const opcol = (col_coords[j] == 0)
                                ? 0
                                : ((1 << (col_coords[j] - 1)) +
                                   col_coords[j + num_dimensions]);

          offsets[j] = cells1d.get_offset(oprow, opcol);
        }

        for (int t = 0; t < num_terms; t++)
        {
          for (int d = 0; d < num_dimensions; d++)
          {
            iA[ia++] = offsets[d] * block1D_size +
                       (t * num_dimensions + d) * kron_squared;
          }
        }
      }
    }
  }
}

  tools::timer.stop(form_id);

  std::cout << "  kronmult sparse matrix fill: "
            << 100.0 * double(num_connect) /
                   (double(num_rows) * double(num_cols))
            << "%\n";

#ifdef ASGARD_USE_CUDA
  int64_t flops = kronmult_matrix<precision>::compute_flops(
      num_dimensions, kron_size, num_terms, col_indx.size());
  std::cout << "              Gflops per call: " << flops * 1.E-9 << "\n";
  std::cout << "  kronmult sparse matrix allocation (MB): "
            << get_MB<int>(iA.size()) + get_MB<precision>(vA.size()) +
                   get_MB<int>(2 * row_indx.size())
            << "\n";

  return kronmult_matrix<precision>(
      num_dimensions, kron_size, num_rows, num_cols, num_terms,
      row_indx.clone_onto_device(), col_indx.clone_onto_device(),
      iA.clone_onto_device(), vA.clone_onto_device());
#else
  int64_t flops = kronmult_matrix<precision>::compute_flops(
      num_dimensions, kron_size, num_terms, indx.size());
  std::cout << "              Gflops per call: " << flops * 1.E-9 << "\n";

  // if using the CPU, move the vectors into the matrix structure
  return kronmult_matrix<precision>(
      num_dimensions, kron_size, num_rows, num_cols, num_terms, std::move(pntr),
      std::move(indx), std::move(iA), std::move(vA));
#endif
}

template<typename P>
kronmult_matrix<P>
make_kronmult_matrix(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                     options const &cli_opts, imex_flag const imex)
{
  if (cli_opts.kmode == kronmult_mode::dense)
  {
    return make_kronmult_dense<P>(pde, grid, cli_opts, imex);
  }
  else
  {
    return make_kronmult_sparse<P>(pde, grid, cli_opts, imex);
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template kronmult_matrix<double>
make_kronmult_matrix<double>(PDE<double> const &,
                             adapt::distributed_grid<double> const &,
                             options const &, imex_flag const);

#endif

#ifdef ASGARD_ENABLE_FLOAT
template kronmult_matrix<float>
make_kronmult_matrix<float>(PDE<float> const &,
                            adapt::distributed_grid<float> const &,
                            options const &, imex_flag const);

#endif

} // namespace asgard
