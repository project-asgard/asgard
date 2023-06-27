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

    return terms;
  }
}

void check_available_memory(int64_t baseline_memory, int64_t available_MB)
{
  if (available_MB < 2)
  { // less then 2MB
    throw std::runtime_error(
        "the problem is too large to fit in the specified memory limit, "
        "this problem requires at least " +
        std::to_string(baseline_memory + 2) + "MB and minimum recommended is " +
        std::to_string(baseline_memory + 512) + "MB but the more the better");
  }
  else if (available_MB < 512)
  { // less than 512MB
    std::cerr
        << "  -- warning: low memory, recommended for this problem size is: "
        << std::to_string(baseline_memory + 512) << "\n";
  }
}

template<typename precision>
kronmult_matrix<precision>
make_kronmult_dense(PDE<precision> const &pde,
                    adapt::distributed_grid<precision> const &discretization,
                    options const &program_options,
                    memory_usage const &mem_stats, imex_flag const imex)
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
  std::vector<int> const used_terms =
      get_used_terms(pde, program_options, imex);
  int const num_terms = static_cast<int>(used_terms.size());

  if (used_terms.size() == 0)
    throw std::runtime_error("no terms selected in the current combination of "
                             "imex flags and options, this must be wrong");

  int64_t osize = 0;
  std::vector<int64_t> dim_term_offset(num_terms * pde.num_dims + 1);
  for (int t = 0; t < num_terms; t++)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      dim_term_offset[t * num_dimensions + d] = osize;
      osize += pde.get_coefficients(used_terms[t], d).size();
    }
  }

  fk::vector<precision> vA(osize);

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

  // all indexes that the matrix will need
  int64_t size_of_indexes =
      int64_t{num_rows} * int64_t{num_cols} * num_terms * num_dimensions;

  int list_row_stride = 0;

  std::vector<fk::vector<int>> list_iA;
  if (mem_stats.kron_call == memory_usage::one_call)
  {
    list_iA.push_back(fk::vector<int>(size_of_indexes));
  }
  else
  {
    // break the work into chunks
    int64_t kron_unit_size = num_terms * num_dimensions * num_cols;
    // work_size fits in 32-bit int so there is no overflow here
    list_row_stride = static_cast<int>(mem_stats.work_size / kron_unit_size);

    if (list_row_stride < 1) // many billions of dof
    {
      if (mem_stats.mem_limit == memory_usage::environment)
        throw std::runtime_error("problem size is too large for the memory "
                                 "specified memory limit");
      else
        throw std::runtime_error("problem is too large for ASGarD "
                                 "(int overflow issues)");
    }

    list_iA.resize((num_rows + list_row_stride - 1) / list_row_stride);
    for (size_t i = 0; i < list_iA.size() - 1; i++)
    {
      list_iA[i] = fk::vector<int>(kron_unit_size * list_row_stride);
    }
    list_iA.back() =
        fk::vector<int>(size_of_indexes - (list_iA.size() - 1) *
                                              list_row_stride * kron_unit_size);
  }

  int64_t used_entries = 0;
  for (auto const &list : list_iA)
    used_entries += list.size();

  // compute the indexes for the matrices for the kron-products
  int const *const flattened_table =
      discretization.get_table().get_active_table().data();
  std::vector<int> oprow(num_dimensions);
  std::vector<int> opcol(num_dimensions);
  auto ilist = list_iA.begin();
  auto ia    = ilist->begin();
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
    // if reached the end of the list and there is more to go
    if (ia == ilist->end())
    {
      ilist++;
      if (ilist != list_iA.end())
        ia = ilist->begin();
    }
  }

  int64_t flops = kronmult_matrix<precision>::compute_flops(
      num_dimensions, kron_size, num_terms, num_rows * num_cols);

  std::cout << "  kronmult dense matrix: " << num_rows << " by " << num_cols
            << "\n";
  std::cout << "        Gflops per call: " << flops * 1.E-9 << "\n";

#ifdef ASGARD_USE_CUDA
  if (mem_stats.kron_call == memory_usage::one_call)
  {
    std::cout << "  kronmult dense matrix allocation (MB): "
              << get_MB<int>(list_iA[0].size()) + get_MB<precision>(vA.size())
              << "\n";

    // if using CUDA, copy the matrices onto the GPU
    return kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols, num_terms,
        list_iA[0].clone_onto_device(), vA.clone_onto_device());
  }
  else
  {
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    std::cout << "  kronmult dense matrix coefficient allocation (MB): "
              << get_MB<precision>(vA.size()) << "\n";
    std::cout << "  kronmult dense matrix common workspace allocation (MB): "
              << get_MB<int>(2 * mem_stats.work_size) << "\n";

    return kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols, num_terms,
        list_row_stride, std::move(list_iA), vA.clone_onto_device());
#else
    std::vector<fk::vector<int, mem_type::owner, resource::device>> gpu_iA(
        list_iA.size());
    int64_t num_ints = 0;
    for (size_t i = 0; i < gpu_iA.size(); i++)
    {
      gpu_iA[i] = list_iA[i].clone_onto_device();
      num_ints += gpu_iA[i].size();
    }
    std::cout << "        memory usage (MB): "
              << get_MB<precision>(vA.size()) + get_MB<int>(num_ints) << "\n";

    return kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols, num_terms,
        list_row_stride, std::move(gpu_iA), vA.clone_onto_device());
#endif
  }

#else
  if (mem_stats.kron_call == memory_usage::one_call)
  {
    // if using the CPU, move the vectors into the matrix structure
    return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                      num_cols, num_terms,
                                      std::move(list_iA[0]), std::move(vA));
  }
  else
  {
    return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                      num_cols, num_terms, list_row_stride,
                                      std::move(list_iA), std::move(vA));
  }
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

void compute_coefficient_offsets(kron_sparse_cache const &spcache,
                                 int const *const row_coords,
                                 int const *const col_coords,
                                 std::vector<int> &offsets)
{
  size_t num_dimensions = offsets.size();
  for (size_t j = 0; j < num_dimensions; j++)
  {
    int const oprow =
        (row_coords[j] == 0)
            ? 0
            : ((1 << (row_coords[j] - 1)) + row_coords[j + num_dimensions]);
    int const opcol =
        (col_coords[j] == 0)
            ? 0
            : ((1 << (col_coords[j] - 1)) + col_coords[j + num_dimensions]);

    offsets[j] = spcache.cells1d.get_offset(oprow, opcol);
  }
}

template<typename precision>
kronmult_matrix<precision>
make_kronmult_sparse(PDE<precision> const &pde,
                     adapt::distributed_grid<precision> const &discretization,
                     options const &program_options,
                     memory_usage const &mem_stats, imex_flag const imex,
                     kron_sparse_cache &spcache)
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
  std::vector<int> const used_terms =
      get_used_terms(pde, program_options, imex);
  int const num_terms = static_cast<int>(used_terms.size());

  // size of the small kron matrices
  int const kron_squared = kron_size * kron_size;

  int const num_1d = spcache.cells1d.num_connections();

#ifdef ASGARD_USE_CUDA
  int const tensor_size = kronmult_matrix<precision>::compute_tensor_size(
      num_dimensions, kron_size);
#endif

  // storing the 1D operator matrices by 1D row and column
  // each connected pair of 1D cells will be associated with a block
  //  of operator coefficients
  int const block1D_size = num_dimensions * num_terms * kron_squared;
  fk::vector<precision> vA(num_1d * block1D_size);
  auto pA = vA.begin();
  for (int row = 0; row < spcache.cells1d.num_cells(); row++)
  {
    for (int j = spcache.cells1d.row_begin(row);
         j < spcache.cells1d.row_end(row); j++)
    {
      int col = spcache.cells1d[j];
      for (int const t : used_terms)
      {
        for (int d = 0; d < num_dimensions; d++)
        {
          precision const *const ops = pde.get_coefficients(t, d).data();
          for (int k = 0; k < kron_size; k++)
            pA =
                std::copy_n(ops + kron_size * row + lda * (kron_size * col + k),
                            kron_size, pA);
        }
      }
    }
  }

  int const *const flattened_table =
      discretization.get_table().get_active_table().data();

#ifndef ASGARD_USE_CUDA
  std::vector<int> row_group_pntr; // group rows in the CPU case
#endif

  std::vector<fk::vector<int>> list_iA;
  std::vector<fk::vector<int>> list_row_indx;
  std::vector<fk::vector<int>> list_col_indx;

  if (mem_stats.kron_call == memory_usage::one_call)
  {
    list_iA.push_back(
        fk::vector<int>(spcache.num_nonz * num_dimensions * num_terms));

#ifdef ASGARD_USE_CUDA
    list_row_indx.push_back(fk::vector<int>(spcache.num_nonz));
    list_col_indx.push_back(fk::vector<int>(spcache.num_nonz));
#else
    list_row_indx.push_back(fk::vector<int>(num_rows + 1));
    list_col_indx.push_back(fk::vector<int>(spcache.num_nonz));
    std::copy_n(spcache.cconnect.begin(), num_rows, list_row_indx[0].begin());
    list_row_indx[0][num_rows] = spcache.num_nonz;
#endif

// load the entries in the one-call mode, can be done in parallel
#pragma omp parallel
    {
      std::vector<int> offsets(num_dimensions); // find the 1D offsets

#pragma omp for
      for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
      {
        int const c = spcache.cconnect[row - grid.row_start];

#ifdef ASGARD_USE_CUDA
        auto iy = list_row_indx[0].begin() + c;
#endif
        auto ix = list_col_indx[0].begin() + c;

        auto ia = list_iA[0].begin() + num_dimensions * num_terms * c;

        int const *const row_coords =
            flattened_table + 2 * num_dimensions * row;
        // (L, p) = (row_coords[i], row_coords[i + num_dimensions])
        for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
        {
          int const *const col_coords =
              flattened_table + 2 * num_dimensions * col;

          if (check_connected(num_dimensions, row_coords, col_coords))
          {
#ifdef ASGARD_USE_CUDA
            *iy++ = (row - grid.row_start) * tensor_size;
            *ix++ = (col - grid.col_start) * tensor_size;
#else
            *ix++ = col - grid.col_start;
#endif

            compute_coefficient_offsets(spcache, row_coords, col_coords,
                                        offsets);

            for (int t = 0; t < num_terms; t++)
              for (int d = 0; d < num_dimensions; d++)
                *ia++ = offsets[d] * block1D_size +
                        (t * num_dimensions + d) * kron_squared;
          }
        }
      }
    }
  }
  else
  { // split the problem into multiple chunks
    // size of the indexes for each pair of (row, col) indexes (for x and y)
    int kron_unit_size = num_dimensions * num_terms;
    // number of pairs that fit in the work-size
    int max_units = mem_stats.work_size / kron_unit_size;
#ifdef ASGARD_USE_CUDA
    // CUDA case, split evenly since parallelism is per kron-product
    int num_chunks = (spcache.num_nonz + max_units - 1) / max_units;
    list_iA.resize(num_chunks);
    list_row_indx.resize(num_chunks);
    list_col_indx.resize(num_chunks);

    for (size_t i = 0; i < list_iA.size() - 1; i++)
    {
      list_iA[i]       = fk::vector<int>(max_units * kron_unit_size);
      list_row_indx[i] = fk::vector<int>(max_units);
      list_col_indx[i] = fk::vector<int>(max_units);
    }
    list_iA.back() = fk::vector<int>(
        (spcache.num_nonz - (num_chunks - 1) * max_units) * kron_unit_size);
    list_row_indx.back() =
        fk::vector<int>(spcache.num_nonz - (num_chunks - 1) * max_units);
    list_col_indx.back() =
        fk::vector<int>(spcache.num_nonz - (num_chunks - 1) * max_units);

    auto list_itra = list_iA.begin();
    auto list_ix   = list_col_indx.begin();
    auto list_iy   = list_row_indx.begin();

    auto ia = list_itra->begin();
    auto ix = list_ix->begin();
    auto iy = list_iy->begin();

    std::vector<int> offsets(num_dimensions); // find the 1D offsets

    for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
    {
      int const *const row_coords = flattened_table + 2 * num_dimensions * row;
      // (L, p) = (row_coords[i], row_coords[i + num_dimensions])
      for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
      {
        int const *const col_coords =
            flattened_table + 2 * num_dimensions * col;

        if (check_connected(num_dimensions, row_coords, col_coords))
        {
          *iy++ = (row - grid.row_start) * tensor_size;
          *ix++ = (col - grid.col_start) * tensor_size;

          compute_coefficient_offsets(spcache, row_coords, col_coords, offsets);

          for (int t = 0; t < num_terms; t++)
            for (int d = 0; d < num_dimensions; d++)
              *ia++ = offsets[d] * block1D_size +
                      (t * num_dimensions + d) * kron_squared;

          if (ix == list_ix->end() and list_ix < list_col_indx.end())
          {
            ia = (++list_itra)->begin();
            ix = (++list_ix)->begin();
            iy = (++list_iy)->begin();
          }
        }
      }
    }

#else
    // CPU case, combine rows together into large groups but don't exceed the
    // work-size
    row_group_pntr.push_back(0);
    int64_t num_units = 0;
    for (int i = 0; i < num_rows; i++)
    {
      int nz_per_row =
          ((i + 1 < num_rows) ? spcache.cconnect[i + 1] : spcache.num_nonz) -
          spcache.cconnect[i];
      if (num_units + nz_per_row > max_units)
      {
        // begin new chunk
        list_iA.push_back(fk::vector<int>(num_units * kron_unit_size));
        list_row_indx.push_back(fk::vector<int>(i - row_group_pntr.back() + 1));
        list_col_indx.push_back(fk::vector<int>(num_units));

        row_group_pntr.push_back(i);
        num_units = nz_per_row;
      }
      else
      {
        num_units += nz_per_row;
      }
    }
    if (num_units > 0)
    {
      list_iA.push_back(fk::vector<int>(num_units * kron_unit_size));
      list_row_indx.push_back(
          fk::vector<int>(num_rows - row_group_pntr.back() + 1));
      list_col_indx.push_back(fk::vector<int>(num_units));
    }
    row_group_pntr.push_back(num_rows);

    std::vector<int> offsets(num_dimensions);

    auto iconn       = spcache.cconnect.begin();
    int64_t shift_iy = 0;

    for (size_t i = 0; i < row_group_pntr.size() - 1; i++)
    {
      auto ia = list_iA[i].begin();
      auto ix = list_col_indx[i].begin();
      auto iy = list_row_indx[i].begin();

      for (int64_t row = row_group_pntr[i]; row < row_group_pntr[i + 1]; row++)
      {
        *iy++ = *iconn++ - shift_iy; // copy the pointer index

        int const *const row_coords =
            flattened_table + 2 * num_dimensions * row;
        // (L, p) = (row_coords[i], row_coords[i + num_dimensions])
        for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
        {
          int const *const col_coords =
              flattened_table + 2 * num_dimensions * col;

          if (check_connected(num_dimensions, row_coords, col_coords))
          {
            *ix++ = (col - grid.col_start);

            compute_coefficient_offsets(spcache, row_coords, col_coords,
                                        offsets);

            for (int t = 0; t < num_terms; t++)
              for (int d = 0; d < num_dimensions; d++)
                *ia++ = offsets[d] * block1D_size +
                        (t * num_dimensions + d) * kron_squared;
          }
        }
      }

      if (i + 2 < row_group_pntr.size())
      {
        *iy++    = *iconn - shift_iy;
        shift_iy = *iconn;
      }
      else
      {
        *iy++ = spcache.num_nonz - shift_iy;
      }
    }

#endif
  }

  tools::timer.stop(form_id);

  std::cout << "  kronmult sparse matrix fill: "
            << 100.0 * double(spcache.num_nonz) /
                   (double(num_rows) * double(num_cols))
            << "%\n";

  int64_t flops = kronmult_matrix<precision>::compute_flops(
      num_dimensions, kron_size, num_terms, spcache.num_nonz);
  std::cout << "              Gflops per call: " << flops * 1.E-9 << "\n";

#ifdef ASGARD_USE_CUDA
  if (mem_stats.kron_call == memory_usage::one_call)
  {
    std::cout << "        memory usage (unique): "
              << get_MB<int>(list_row_indx[0].size()) +
                     get_MB<int>(list_col_indx[0].size()) +
                     get_MB<int>(list_iA[0].size()) +
                     get_MB<precision>(vA.size())
              << "\n";
    std::cout << "        memory usage (shared): 0\n";
    return kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols, num_terms,
        list_row_indx[0].clone_onto_device(),
        list_col_indx[0].clone_onto_device(), list_iA[0].clone_onto_device(),
        vA.clone_onto_device());
  }
  else
  {
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    std::cout << "        memory usage (unique): "
              << get_MB<precision>(vA.size()) << "\n";
    std::cout << "        memory usage (shared): "
              << 2 * get_MB<int>(mem_stats.work_size) +
                     4 * get_MB<int>(mem_stats.row_work_size)
              << "\n";
    return kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols, num_terms,
        std::move(list_row_indx), std::move(list_col_indx), std::move(list_iA),
        vA.clone_onto_device());
#else
    std::vector<fk::vector<int, mem_type::owner, resource::device>> gpu_iA(
        list_iA.size());
    std::vector<fk::vector<int, mem_type::owner, resource::device>> gpu_col(
        list_col_indx.size());
    std::vector<fk::vector<int, mem_type::owner, resource::device>> gpu_row(
        list_row_indx.size());
    int64_t num_ints = 0;
    for (size_t i = 0; i < gpu_iA.size(); i++)
    {
      gpu_iA[i]  = list_iA[i].clone_onto_device();
      gpu_col[i] = list_col_indx[i].clone_onto_device();
      gpu_row[i] = list_row_indx[i].clone_onto_device();
      num_ints += gpu_iA[i].size() + gpu_col[i].size() + gpu_row[i].size();
    }
    std::cout << "        memory usage (MB): "
              << get_MB<precision>(vA.size()) + get_MB<int>(num_ints) << "\n";
    return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows,
                                      num_cols, num_terms, std::move(gpu_row),
                                      std::move(gpu_col), std::move(gpu_iA),
                                      vA.clone_onto_device());
#endif
  }
#else

  return kronmult_matrix<precision>(
      num_dimensions, kron_size, num_rows, num_cols, num_terms,
      std::move(list_row_indx), std::move(list_col_indx), std::move(list_iA),
      std::move(vA));

#endif
}

template<typename P>
kronmult_matrix<P>
make_kronmult_matrix(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                     options const &cli_opts, memory_usage const &mem_stats,
                     imex_flag const imex, kron_sparse_cache &spcache,
                     bool force_sparse)
{
  if (cli_opts.kmode == kronmult_mode::dense and not force_sparse)
  {
    return make_kronmult_dense<P>(pde, grid, cli_opts, mem_stats, imex);
  }
  else
  {
    return make_kronmult_sparse<P>(pde, grid, cli_opts, mem_stats, imex,
                                   spcache);
  }
}

template<typename P>
void update_kronmult_coefficients(PDE<P> const &pde,
                                  options const &program_options,
                                  imex_flag const imex,
                                  kron_sparse_cache &spcache,
                                  kronmult_matrix<P> &mat)
{
  auto const form_id       = tools::timer.start("kronmult-update-coefficients");
  int const num_dimensions = pde.num_dims;
  int const kron_size      = pde.get_dimensions()[0].get_degree();

  int64_t lda = kron_size * fm::two_raised_to((program_options.do_adapt_levels)
                                                  ? program_options.max_level
                                                  : pde.max_level);

  // take into account the terms that will be skipped due to the imex_flag
  std::vector<int> const used_terms =
      get_used_terms(pde, program_options, imex);
  int const num_terms = static_cast<int>(used_terms.size());

  // size of the small kron matrices
  int const kron_squared = kron_size * kron_size;
  fk::vector<P> vA;

  if (mat.is_dense())
  {
    int64_t osize = 0;
    for (int t = 0; t < num_terms; t++)
      for (int d = 0; d < num_dimensions; d++)
        osize += pde.get_coefficients(used_terms[t], d).size();

    vA = fk::vector<P>(osize);

    auto pA = vA.begin();
    for (int t : used_terms)
    {
      for (int d = 0; d < num_dimensions; d++)
      {
        auto const &ops   = pde.get_coefficients(t, d);
        int const num_ops = ops.nrows() / kron_size;

        for (int ocol = 0; ocol < num_ops; ocol++)
          for (int orow = 0; orow < num_ops; orow++)
            for (int i = 0; i < kron_size; i++)
              pA = std::copy_n(ops.data() + kron_size * orow +
                                   lda * (kron_size * ocol + i),
                               kron_size, pA);
      }
    }
  }
  else
  {
    // holds the 1D sparsity structure for the coefficient matrices
    int const num_1d = spcache.cells1d.num_connections();

    // storing the 1D operator matrices by 1D row and column
    // each connected pair of 1D cells will be associated with a block
    //  of operator coefficients
    int const block1D_size = num_dimensions * num_terms * kron_squared;
    vA                     = fk::vector<P>(num_1d * block1D_size);
    auto pA                = vA.begin();
    for (int row = 0; row < spcache.cells1d.num_cells(); row++)
    {
      for (int j = spcache.cells1d.row_begin(row);
           j < spcache.cells1d.row_end(row); j++)
      {
        int col = spcache.cells1d[j];
        for (int const t : used_terms)
        {
          for (int d = 0; d < num_dimensions; d++)
          {
            P const *const ops = pde.get_coefficients(t, d).data();
            for (int k = 0; k < kron_size; k++)
              pA = std::copy_n(ops + kron_size * row +
                                   lda * (kron_size * col + k),
                               kron_size, pA);
          }
        }
      }
    }
  }

#ifdef ASGARD_USE_CUDA
  mat.update_stored_coefficients(vA.clone_onto_device());
#else
  mat.update_stored_coefficients(std::move(vA));
#endif

  tools::timer.stop(form_id);
}

template<typename P>
memory_usage
compute_mem_usage(PDE<P> const &pde,
                  adapt::distributed_grid<P> const &discretization,
                  options const &program_options, imex_flag const imex,
                  kron_sparse_cache &spcache, int memory_limit_MB,
                  int64_t index_limit, bool force_sparse)
{
  auto const &grid         = discretization.get_subgrid(get_rank());
  int const num_dimensions = pde.num_dims;
  int const kron_size      = pde.get_dimensions()[0].get_degree();
  int const num_rows       = grid.row_stop - grid.row_start + 1;
  int const num_cols       = grid.col_stop - grid.col_start + 1;

  memory_usage stats;

#ifdef ASGARD_USE_GPU_MEM_LIMIT
  if (memory_limit_MB == 0)
    memory_limit_MB = program_options.memory_limit;
#else
  ignore(memory_limit_MB);
#endif

  // parameters common to the dense and sparse cases
  // matrices_per_prod is the number of matrices per Kronecker product
  int64_t matrices_per_prod = pde.num_terms * num_dimensions;

  // base_line_entries are the entries that must always be loaded in GPU memory
  // first we compute the size of the state vectors (x and y) and then we
  // add the size of the coefficients (based on sparse/dense mode)
  int64_t base_line_entries =
      (num_rows + num_cols) *
      kronmult_matrix<P>::compute_tensor_size(num_dimensions, kron_size);

  if (program_options.kmode == kronmult_mode::dense and not force_sparse)
  {
    // assume all terms will be loaded into the GPU, as one IMEX flag or another
    for (int t = 0; t < pde.num_terms; t++)
      for (int d = 0; d < num_dimensions; d++)
        base_line_entries += pde.get_coefficients(t, d).size();

    stats.baseline_memory = 1 + static_cast<int>(get_MB<P>(base_line_entries));

#ifdef ASGARD_USE_GPU_MEM_LIMIT
    int64_t available_MB = memory_limit_MB - stats.baseline_memory;
    check_available_memory(stats.baseline_memory, available_MB);

    int64_t available_entries = (int64_t{available_MB} * 1024 * 1024) /
                                static_cast<int64_t>(sizeof(int));
#else
    int64_t available_entries = index_limit;
#endif

    int64_t size_of_indexes =
        int64_t{num_rows} * int64_t{num_cols} * matrices_per_prod;

    if (size_of_indexes <= available_entries and size_of_indexes <= index_limit)
    {
      stats.kron_call = memory_usage::one_call;
    }
    else
    {
      stats.kron_call = memory_usage::multi_calls;

      if (size_of_indexes > index_limit)
      {
        stats.mem_limit = memory_usage::overflow;
        stats.work_size = index_limit;
      }
      else
      {
        stats.mem_limit = memory_usage::environment;
        stats.work_size = available_entries / 2;
      }

#ifdef ASGARD_USE_GPU_MEM_LIMIT
      if (2 * stats.work_size > available_entries)
        stats.work_size = available_entries / 2;
#endif
    }
  }
  else
  { // sparse mode
    // if possible, keep the 1d connectivity matrix
    if (pde.max_level != spcache.cells1d.max_loaded_level())
      spcache.cells1d = connect_1d(pde.max_level);

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

    if (spcache.cconnect.size() == static_cast<size_t>(num_rows))
      std::fill(spcache.cconnect.begin(), spcache.cconnect.end(), 0);
    else
      spcache.cconnect = std::vector<int>(num_rows);

#pragma omp parallel for
    for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
    {
      int const *const row_coords = flattened_table + 2 * num_dimensions * row;
      // (L, p) = (row_coords[i], row_coords[i + num_dimensions])
      for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
      {
        int const *const col_coords =
            flattened_table + 2 * num_dimensions * col;
        if (check_connected(num_dimensions, row_coords, col_coords))
          spcache.cconnect[row - grid.row_start]++;
      }
    }

    spcache.num_nonz = 0; // total number of connected cells
    for (int i = 0; i < num_rows; i++)
    {
      int c               = spcache.cconnect[i];
      spcache.cconnect[i] = spcache.num_nonz;
      spcache.num_nonz += c;
    }

    base_line_entries += spcache.cells1d.num_connections() * num_dimensions *
                         pde.num_terms * kron_size * kron_size;

    stats.baseline_memory = 1 + static_cast<int>(get_MB<P>(base_line_entries));

#ifdef ASGARD_USE_GPU_MEM_LIMIT
    int64_t available_MB = memory_limit_MB - stats.baseline_memory;
    check_available_memory(stats.baseline_memory, available_MB);

    int64_t available_entries =
        (available_MB * 1024 * 1024) / static_cast<int64_t>(sizeof(int));
#else
    int64_t available_entries = index_limit;
#endif

    int64_t size_of_indexes = spcache.num_nonz * (matrices_per_prod + 2);

    if (size_of_indexes <= available_entries and size_of_indexes <= index_limit)
    {
      stats.kron_call = memory_usage::one_call;
    }
    else
    {
      int min_terms = pde.num_terms;
      if (imex != imex_flag::unspecified)
      {
        std::vector<int> const explicit_terms =
            get_used_terms(pde, program_options, imex_flag::imex_explicit);
        std::vector<int> const implicit_terms =
            get_used_terms(pde, program_options, imex_flag::imex_implicit);
        min_terms = std::min(explicit_terms.size(), implicit_terms.size());
      }

      stats.kron_call = memory_usage::multi_calls;
      if (size_of_indexes > index_limit)
      {
        stats.mem_limit     = memory_usage::overflow;
        stats.work_size     = index_limit;
        stats.row_work_size = index_limit / (num_dimensions * min_terms);
      }
      else
      {
        stats.mem_limit = memory_usage::environment;
        int64_t work_products =
            available_entries / (min_terms * num_dimensions + 2);
        stats.work_size     = min_terms * num_dimensions * (work_products / 2);
        stats.row_work_size = work_products / 2;
      }

#ifdef ASGARD_USE_GPU_MEM_LIMIT
      if (2 * stats.work_size + 2 * stats.row_work_size > available_entries)
      {
        int64_t work_products =
            available_entries / (min_terms * num_dimensions + 2);
        stats.work_size     = min_terms * num_dimensions * (work_products / 2);
        stats.row_work_size = work_products / 2;
      }
#endif
    }
  }

  stats.initialized = true;

  return stats;
}

#ifdef ASGARD_ENABLE_DOUBLE
template kronmult_matrix<double>
make_kronmult_matrix<double>(PDE<double> const &,
                             adapt::distributed_grid<double> const &,
                             options const &, memory_usage const &,
                             imex_flag const, kron_sparse_cache &, bool);
template void
update_kronmult_coefficients<double>(PDE<double> const &, options const &,
                                     imex_flag const, kron_sparse_cache &,
                                     kronmult_matrix<double> &);
template memory_usage
compute_mem_usage<double>(PDE<double> const &,
                          adapt::distributed_grid<double> const &,
                          options const &, imex_flag const, kron_sparse_cache &,
                          int, int64_t, bool);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template kronmult_matrix<float>
make_kronmult_matrix<float>(PDE<float> const &,
                            adapt::distributed_grid<float> const &,
                            options const &, memory_usage const &,
                            imex_flag const, kron_sparse_cache &, bool);
template void
update_kronmult_coefficients<float>(PDE<float> const &, options const &,
                                    imex_flag const, kron_sparse_cache &,
                                    kronmult_matrix<float> &);
template memory_usage
compute_mem_usage<float>(PDE<float> const &,
                         adapt::distributed_grid<float> const &,
                         options const &, imex_flag const, kron_sparse_cache &,
                         int, int64_t, bool);
#endif

} // namespace asgard
