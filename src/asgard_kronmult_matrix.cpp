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
  std::vector<int> const used_terms = get_used_terms(pde, program_options, imex);
  int const num_terms               = static_cast<int>(used_terms.size());

  if (used_terms.empty())
    return asgard::kronmult_matrix<precision>(num_dimensions, kron_size,
                                              num_rows, num_cols);

  constexpr resource mode = resource::host;

  std::vector<fk::vector<precision, mem_type::owner, mode>> terms(num_terms);
  int const num_1d_blocks =
      pde.get_coefficients(used_terms[0], 0).nrows() / kron_size;

  for (int t = 0; t < num_terms; t++)
  {
    terms[t] = fk::vector<precision, mem_type::owner, mode>(
        num_dimensions * num_1d_blocks * num_1d_blocks * kron_size * kron_size);
    auto pA = terms[t].begin();
    for (int d = 0; d < num_dimensions; d++)
    {
      auto const &ops = pde.get_coefficients(used_terms[t], d);

      // the matrices of the kron products are organized into blocks
      // of a large matrix, the matrix is square with size num-ops by
      // kron-size rearrange in a sequential way (by columns) to avoid the lda
      for (int ocol = 0; ocol < num_1d_blocks; ocol++)
        for (int orow = 0; orow < num_1d_blocks; orow++)
          for (int i = 0; i < kron_size; i++)
            pA = std::copy_n(ops.data() + kron_size * orow +
                                 lda * (kron_size * ocol + i),
                             kron_size, pA);
    }
  }

  int const *const ftable =
      discretization.get_table().get_active_table().data();

  int const num_indexes = 1 + std::max(grid.row_stop, grid.col_stop);
  fk::vector<int, mem_type::owner, mode> elem(num_dimensions * num_indexes);
  for (int i = 0; i < num_indexes; i++)
  {
    int const *const idx = ftable + 2 * num_dimensions * i;

    for (int d = 0; d < num_dimensions; d++)
    {
      elem[i * num_dimensions + d] =
          (idx[d] == 0)
              ? 0
              : (fm::two_raised_to(idx[d] - 1) + idx[d + num_dimensions]);
    }
  }

  int64_t flps = kronmult_matrix<precision>::compute_flops(
      num_dimensions, kron_size, num_terms, int64_t{num_rows} * num_cols);

  std::cout << "  kronmult dense matrix: " << num_rows << " by " << num_cols
            << "\n";
  std::cout << "        Gflops per call: " << flps * 1.E-9 << "\n";

  std::cout << "        memory usage (MB): "
            << get_MB<precision>(terms.size()) + get_MB<int>(elem.size())
            << "\n";

#ifdef ASGARD_USE_CUDA
  std::vector<fk::vector<precision, mem_type::owner, resource::device>>
      gpu_terms(num_terms);
  for (int t = 0; t < num_terms; t++)
    gpu_terms[t] = terms[t].clone_onto_device();

  auto gpu_elem = elem.clone_onto_device();

  return asgard::kronmult_matrix<precision>(
      num_dimensions, kron_size, num_rows, num_cols, num_terms,
      std::move(gpu_terms), std::move(gpu_elem), grid.row_start, grid.col_start,
      num_1d_blocks);
#else
  return asgard::kronmult_matrix<precision>(
      num_dimensions, kron_size, num_rows, num_cols, num_terms,
      std::move(terms), std::move(elem), grid.row_start, grid.col_start,
      num_1d_blocks);
#endif
}

//! \brief Processes two multi-index and returns true if they are connected for all dimensions.
inline bool check_connected_edge(int const num_dimensions, int const *const row,
                                 int const *const col)
{
  // different levels, check if the points are connected by volume
  auto check_diff_volumes = [](int l1, int p1, int l2, int p2)
      -> bool {
    if (l1 < l2)
    {
      while (l1 < l2)
      {
        l2--;
        p2 /= 2;
      }
    }
    else
    {
      while (l2 < l1)
      {
        l1--;
        p1 /= 2;
      }
    }
    return p1 == p2;
  };

  int edge_conn = 0;
  for (int j = 0; j < num_dimensions; j++)
  {
    if (row[j] == col[j]) // same level, consider only edge connections
    {
      if (row[num_dimensions + j] != col[num_dimensions + j])
      {
        if ((row[num_dimensions + j] == 0 and col[num_dimensions + j] == ((1 << (col[j] - 1)) - 1)) or
            (col[num_dimensions + j] == 0 and row[num_dimensions + j] == ((1 << (row[j] - 1)) - 1)))
          edge_conn += 1; // periodic boundary
        else
        {
          if (std::abs(row[num_dimensions + j] - col[num_dimensions + j]) == 1)
            edge_conn += 1; // adjacent elements
          else
            return false; // same level and not connected by edge or volume
        }
      }
    }
    else // different level, consider volume connection only
    {
      // if not connected by volume in higher d, then not connected at all
      if (not(row[j] <= 1 or col[j] <= 1 or
              check_diff_volumes(row[j], row[num_dimensions + j], col[j], col[num_dimensions + j])))
        return false;
    }

    if (edge_conn > 1)
      return false;
  }
  return (edge_conn == 1);
}

/*!
 * \brief Returns true if the 1D elements are connected
 *
 * The two elements are defined by (level L, index within the level is p),
 * the first point is (L1, p1) and we assume that L1 <= L2.
 */
inline bool check_connected(int L1, int p1, int L2, int p2)
{
  expect(L1 <= L2);

  // levels 0 and 1 are connected to everything
  if (L1 <= 1 or L2 <= 1)
    return true;

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

void compute_coefficient_offsets(connect_1d const &cells1d,
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

    offsets[j] = cells1d.get_offset(oprow, opcol);
  }
}
void compute_coefficient_offsets(kron_sparse_cache const &spcache,
                                 int const *const row_coords,
                                 int const *const col_coords,
                                 std::vector<int> &offsets)
{
  compute_coefficient_offsets(spcache.cells1d, row_coords, col_coords, offsets);
}

template<typename precision>
kronmult_matrix<precision>
make_kronmult_sparse(PDE<precision> const &pde,
                     adapt::distributed_grid<precision> const &discretization,
                     options const &program_options,
                     memory_usage const &mem_stats, imex_flag const imex,
                     kron_sparse_cache &spcache)
{
  tools::time_event performance_("make-kronmult-sparse");
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

  if (used_terms.empty())
    return asgard::kronmult_matrix<precision>(num_dimensions, kron_size,
                                              num_rows, num_cols);

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
  for (int row = 0; row < spcache.cells1d.num_rows(); row++)
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
      num_ints += int64_t{gpu_iA[i].size()} + int64_t{gpu_col[i].size()} +
                  int64_t{gpu_row[i].size()};
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
    return make_kronmult_dense<P>(pde, grid, cli_opts, imex);
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
  tools::time_event kron_time_("kronmult-update-coefficients");
  int const num_dimensions = pde.num_dims;
  int const kron_size      = pde.get_dimensions()[0].get_degree();

  int64_t lda = kron_size * fm::two_raised_to((program_options.do_adapt_levels)
                                                  ? program_options.max_level
                                                  : pde.max_level);

  // take into account the terms that will be skipped due to the imex_flag
  std::vector<int> const used_terms =
      get_used_terms(pde, program_options, imex);
  int const num_terms = static_cast<int>(used_terms.size());

  if (num_terms == 0)
    return;

  // size of the small kron matrices
  int const kron_squared = kron_size * kron_size;
  fk::vector<P> vA;

  if (mat.is_dense())
  {
    constexpr resource mode = resource::host;

    std::vector<fk::vector<P, mem_type::owner, mode>> terms(num_terms);
    int const num_1d_blocks =
        pde.get_coefficients(used_terms[0], 0).nrows() / kron_size;

    for (int t = 0; t < num_terms; t++)
    {
      terms[t] = fk::vector<P, mem_type::owner, mode>(
          num_dimensions * num_1d_blocks * num_1d_blocks * kron_squared);
      auto pA = terms[t].begin();
      for (int d = 0; d < num_dimensions; d++)
      {
        auto const &ops = pde.get_coefficients(used_terms[t], d);

        // the matrices of the kron products are organized into blocks
        // of a large matrix, the matrix is square with size num-ops by
        // kron-size rearrange in a sequential way (by columns) to avoid the lda
        for (int ocol = 0; ocol < num_1d_blocks; ocol++)
          for (int orow = 0; orow < num_1d_blocks; orow++)
            for (int i = 0; i < kron_size; i++)
              pA = std::copy_n(ops.data() + kron_size * orow +
                                   lda * (kron_size * ocol + i),
                               kron_size, pA);
      }
    }
#ifdef ASGARD_USE_CUDA
    std::vector<fk::vector<P, mem_type::owner, resource::device>> gpu_terms(
        num_terms);
    for (int t = 0; t < num_terms; t++)
      gpu_terms[t] = terms[t].clone_onto_device();
    mat.update_stored_coefficients(std::move(gpu_terms));
#else
    mat.update_stored_coefficients(std::move(terms));
#endif
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
    for (int row = 0; row < spcache.cells1d.num_rows(); row++)
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
#ifdef ASGARD_USE_CUDA
    mat.update_stored_coefficients(vA.clone_onto_device());
#else
    mat.update_stored_coefficients(std::move(vA));
#endif
  }
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

#ifdef KRON_MODE_GLOBAL
template<typename precision>
bool check_identity_term(PDE<precision> const &pde, int term_id, int dim)
{
  auto const &tterm = pde.get_terms()[term_id][dim];
  if (tterm.get_partial_terms().size() == 1)
    return (tterm.get_partial_terms()[0].g_func == nullptr and
            tterm.get_partial_terms()[0].coeff_type == coefficient_type::mass);
  else
    return false;
}

template<typename precision>
global_kron_matrix<precision>
make_global_kron_matrix(PDE<precision> const &pde,
                        adapt::distributed_grid<precision> const &dis_grid,
                        options const &program_options)
{
  auto const &grid                 = dis_grid.get_subgrid(get_rank());
  int const *const flattened_table = dis_grid.get_table().get_active_table().data();

  int const porder    = pde.get_dimensions()[0].get_degree() - 1;
  int const pterms    = porder + 1; // poly degrees of freedom
  int const max_level = (program_options.do_adapt_levels) ? program_options.max_level : pde.max_level;

  int const num_dimensions = pde.num_dims;
  int const num_terms      = pde.num_terms;

  connect_1d cell_pattern(max_level, asgard::connect_1d::level_edge_skip);
  connect_1d dof_pattern(cell_pattern, porder);
  //dof_pattern.dump();

  int const num_non_padded   = grid.col_stop - grid.col_start + 1;
  vector2d<int> active_cells = asg2tsg_convert(num_dimensions, num_non_padded, flattened_table);

  indexset pad_complete = compute_ancestry_completion(make_index_set(active_cells), cell_pattern);

  vector2d<int> ilist = complete_poly_order(active_cells, pad_complete, porder);
  dimension_sort dsort(ilist);

  int64_t num_all_dof    = ilist.num_strips();
  int64_t num_active_dof = num_non_padded * pterms;
  for (int d = 1; d < num_dimensions; d++)
    num_active_dof *= pterms;

  // form the 1D pattern for the matrices in each dimension
  std::vector<std::vector<int>> global_pntr(num_dimensions,
                                            std::vector<int>(num_all_dof + 1));
  std::vector<std::vector<int>> global_indx(num_dimensions);
  std::vector<std::vector<int>> global_diag(num_dimensions,
                                            std::vector<int>(num_all_dof));
  std::vector<std::vector<int>> global_ivals(num_dimensions);

  // figure out the global Kronecker patterns of non-zeros and the corresponding values
  std::vector<int> nz_count(num_all_dof);
  for (int dim = 0; dim < num_dimensions; dim++)
  {
    std::vector<int> &pntr  = global_pntr[dim];
    std::vector<int> &indx  = global_indx[dim];
    std::vector<int> &diag  = global_diag[dim];
    std::vector<int> &ivals = global_ivals[dim];

    // we parse the pattern twice, once to find the number of non-zeros and
    // and then to record the non-zeros in the indx structures
    for (int stage = 0; stage < 2; stage++)
    {
      if (stage == 1)
      {
        int64_t num_nz = 0;
        for (auto n : nz_count)
          num_nz += n;
        indx  = std::vector<int>(num_nz);
        ivals = std::vector<int>(2 * num_nz);
        for (size_t i = 1; i < pntr.size(); i++)
          pntr[i] = pntr[i - 1] + nz_count[i - 1];
      }
      std::fill(nz_count.begin(), nz_count.end(), 0);
      // sparse sub-vectors after the sort in the given dimension
      int const num_vecs = dsort.num_vecs(dim);
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
          int row       = dsort(ilist, dim, j); // 1D index of this output in y
          int mat_begin = dof_pattern.row_begin(row);
          int mat_end   = dof_pattern.row_end(row);

          // loop over the matrix row and the vector looking for matching non-zeros
          int mat_j = mat_begin;
          int vec_j = vec_begin;
          while (mat_j < mat_end and vec_j < vec_end)
          {
            int const vec_index = dsort(ilist, dim, vec_j); // pattern index 1d
            int const mat_index = dof_pattern[mat_j];
            // the sort helps here, since indexes are in order, it is easy to
            // match the index patterns
            if (vec_index < mat_index)
              vec_j += 1;
            else if (mat_index < vec_index)
              mat_j += 1;
            else // mat_index == vec_index, found matching entry, add to output
            {
              if (stage == 0)
                nz_count[dsort.map(dim, j)] += 1;
              else
              {
                int const g_row      = dsort.map(dim, j); // global row
                int const idx_in_row = pntr[g_row] + nz_count[g_row];

                indx[idx_in_row]          = dsort.map(dim, vec_j);
                ivals[2 * idx_in_row]     = row;
                ivals[2 * idx_in_row + 1] = mat_index;

                if (g_row == indx[idx_in_row])
                  diag[g_row] = idx_in_row;

                nz_count[g_row] += 1;
              }
              // entry match, increment both indexes for the pattern
              vec_j += 1;
              mat_j += 1;
            }
          }
        }
      }
    }
  }

  // figure out the permutations pattern
  std::vector<kronmult::permutes> permutations;
  permutations.reserve(num_terms);
  std::vector<int> active_dirs(num_dimensions);
  for (int t = 0; t < num_terms; t++)
  {
    active_dirs.clear();
    for (int d = 0; d < num_dimensions; d++)
      if (not check_identity_term(pde, t, d))
        active_dirs.push_back(d);

    int const num_active = static_cast<int>(active_dirs.size());
    permutations.push_back(kronmult::permutes(num_active));
    if (num_active != num_dimensions)
      permutations.back().remap_directions(active_dirs);
  }

  // make the pattern for the local contribution
  std::vector<int> lpntr(grid.row_stop - grid.row_start + 2);
#pragma omp parallel for
  for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
  {
    int const *const row_coords = flattened_table + 2 * num_dimensions * row;

    for (int64_t col = grid.col_start; col < grid.col_stop + 1; col++)
    {
      int const *const col_coords = flattened_table + 2 * num_dimensions * col;
      if (check_connected_edge(num_dimensions, row_coords, col_coords))
        lpntr[row - grid.row_start + 1]++;
    }
  }

  for (size_t i = 1; i < lpntr.size(); i++)
    lpntr[i] += lpntr[i - 1];

  int num_nonz = lpntr.back();

  std::vector<int> lindx(num_nonz); // compress all_indx

  for (int64_t row = grid.row_start; row < grid.row_stop + 1; row++)
  {
    int const *const row_coords = flattened_table + 2 * num_dimensions * row;

    int col = 0;
    for (int j = lpntr[row]; j < lpntr[row + 1]; j++)
    {
      int const *col_coords = flattened_table + 2 * num_dimensions * col;
      while (not check_connected_edge(num_dimensions, row_coords, col_coords))
      {
        col += 1;
        col_coords = flattened_table + 2 * num_dimensions * col;
      }
      lindx[j] = col;
      col += 1;
    }
  }

  return global_kron_matrix<precision>(
      std::move(dof_pattern), std::move(ilist), num_active_dof, std::move(permutations),
      std::move(global_pntr), std::move(global_indx), std::move(global_diag),
      std::move(global_ivals),
      porder, connect_1d(max_level), std::move(lpntr), std::move(lindx));
}

template<typename precision>
void set_specific_mode(PDE<precision> const &pde,
                       adapt::distributed_grid<precision> const &dis_grid,
                       options const &program_options, imex_flag const imex,
                       global_kron_matrix<precision> &mat)
{
  int const imex_indx = global_kron_matrix<precision>::flag2int(imex);

#ifdef ASGARD_USE_CUDA
  // gpu mode, load into std::vector and at the end send to the gpu
  std::vector<int> indx;
  std::vector<precision> vals;
#else
  // cpu mode, get refs to the std::vectors and load directly in
  std::vector<int> &indx = mat.local_opindex_[imex_indx];
  std::vector<precision> &vals = mat.local_opvalues_[imex_indx];
#endif

  mat.term_groups[imex_indx] = get_used_terms(pde, program_options, imex);

  std::vector<int> const &used_terms = mat.term_groups[imex_indx];

  auto const &grid                 = dis_grid.get_subgrid(get_rank());
  int const *const flattened_table = dis_grid.get_table().get_active_table().data();

  int const porder     = pde.get_dimensions()[0].get_degree() - 1;
  mat.porder_          = porder;
  int const pterms     = porder + 1; // poly degrees of freedom
  int const block_size = pterms * pterms;
  int const max_level  = (program_options.do_adapt_levels) ? program_options.max_level : pde.max_level;

  int const num_dimensions = pde.num_dims;

  // set the global pattern
  for (int t : used_terms)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      if (not check_identity_term(pde, t, d))
      {
        std::vector<precision> &gvals = mat.gvals_[t * num_dimensions + d];
        std::vector<int> &givals      = mat.givals_[d];

        int64_t num_entries = static_cast<int64_t>(mat.gindx_[d].size());

        gvals = std::vector<precision>(num_entries);

        fk::matrix<precision> const &ops = pde.get_coefficients(t, d);

#pragma omp parallel for
        for (int64_t i = 0; i < num_entries; i++)
          gvals[i] = ops(givals[2 * i], givals[2 * i + 1]);
      }
    }
  }

  int64_t lda = pterms * fm::two_raised_to(max_level);

  int const num_terms = static_cast<int>(used_terms.size());
  if (num_terms == 0)
    return;

  connect_1d const &edges = mat.edges_;
  int const dim_block     = edges.num_connections() * block_size;
  //edges.dump();

  // load the indexes
  indx.resize(mat.local_indx_.size() * num_terms * num_dimensions);
#pragma omp parallel
  {
    std::vector<int> offsets(num_dimensions);

#pragma omp for
    for (int64_t row = 0; row < grid.row_stop + 1; row++)
    {
      auto ia = indx.begin() + num_dimensions * num_terms * mat.local_pntr_[row];

      int const *const row_coords = flattened_table + 2 * num_dimensions * row;
      // (L, p) = (row_coords[i], row_coords[i + num_dimensions])
      for (int64_t j = mat.local_pntr_[row]; j < mat.local_pntr_[row + 1]; j++)
      {
        int const *const col_coords =
            flattened_table + 2 * num_dimensions * mat.local_indx_[j];
        compute_coefficient_offsets(edges, row_coords, col_coords, offsets);

        for (int t = 0; t < num_terms; t++)
          for (int d = 0; d < num_dimensions; d++)
            *ia++ = (t * num_dimensions + d) * dim_block + offsets[d] * block_size;
      }
    }
  }

  vals.resize(num_terms * num_dimensions * dim_block);
  for (int t = 0; t < num_terms; t++)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      auto iv = vals.begin() + (t * num_dimensions + d) * dim_block;

      precision const *const ops = pde.get_coefficients(used_terms[t], d).data();

      for (int cell = 0; cell < edges.num_rows(); cell++)
      {
        for (int j = edges.row_begin(cell); j < edges.row_end(cell); j++)
        {
          int const col = edges[j] * pterms;
          for (int p = 0; p < pterms; p++)
            iv = std::copy_n(ops + pterms * cell + lda * (col + p), pterms, iv);
        }
      }
    }
  }

#ifdef ASGARD_USE_CUDA
  // load to the gpu
  mat.local_opindex_[imex_indx]  = indx;
  mat.local_opvalues_[imex_indx] = vals;
#endif

  // compute the size of the in-cell degrees of freedom
  int64_t tensor_size = int_pow(pterms, num_dimensions);

  if (imex == imex_flag::imex_implicit or program_options.use_implicit_stepping)
  {
    // prepare a preconditioner
#ifdef ASGARD_USE_CUDA
    std::vector<precision> pc;
#else
    std::vector<precision> &pc = mat.pre_con_;
#endif
    int64_t num_entries = mat.num_active_;

    pc.resize(num_entries);
    std::vector<int> midx(num_dimensions); // multi-index for each row

    for (int64_t row = 0; row < grid.row_stop + 1; row++)
    {
      int const *const row_coords = flattened_table + 2 * num_dimensions * row;
      asg2tsg_convert(num_dimensions, row_coords, midx.data());

      for (int tentry = 0; tentry < tensor_size; tentry++)
      {
        for (int t : used_terms)
        {
          precision a = 1;

          int tt = tentry;
          for (int d = num_dimensions - 1; d >= 0; d--)
          {
            int const rc = midx[d] * pterms + tt % pterms;
            a *= pde.get_coefficients(t, d)(rc, rc);
            tt /= pterms;
          }
          pc[row * tensor_size + tentry] += a;
        }
      }
    }
#ifdef ASGARD_USE_CUDA
    mat.pre_con_ = pc;
#endif
  }

  int64_t gflops = 0; // flops for global component
  for (auto t : used_terms)
    for (int d = 0; d < num_dimensions; d++)
      gflops += mat.gvals_[t * num_dimensions + d].size();

  int64_t lflops = tensor_size * pterms * mat.local_opindex_[imex_indx].size();

  std::cout << "Kronmult using global algorithm:\n";
  std::cout << "    global: " << static_cast<double>(gflops) * 1.E-9 << " Gflops\n";
  std::cout << "     local: " << static_cast<double>(lflops) * 1.E-9 << " Gflops\n";
  std::cout << "     total: " << static_cast<double>(gflops + lflops) * 1.E-9 << " Gflops\n";
  mat.flops_[imex_indx] = gflops + lflops;

  int64_t num_cints = 0;
  int64_t num_cfps  = 0;
  for (int d = 0; d < num_dimensions; d++)
  {
    num_cints += mat.gpntr_[d].size();
    num_cints += mat.gindx_[d].size();
    num_cints += mat.gdiag_[d].size();
    num_cints += mat.givals_[d].size();
  }
  for (auto t : used_terms)
    for (int d = 0; d < num_dimensions; d++)
      num_cfps += mat.gvals_[t * num_dimensions + d].size();

  num_cfps += mat.expanded.size() + mat.workspace.size();
  int64_t num_lints = mat.local_pntr_.size() + mat.local_indx_.size() + mat.local_opindex_[imex_indx].size();
  int64_t num_lfps  = mat.local_opvalues_[imex_indx].size();
  std::cout << "  -- memory usage:\n";
  std::cout << "    common: " << get_MB<precision>(num_cfps) + get_MB<int>(num_cints) << "MB\n";
  std::cout << "   variant: " << get_MB<precision>(num_lfps) + get_MB<int>(num_lints) << "MB\n";
  int64_t total_mem = get_MB<precision>(num_lfps + num_cfps) + get_MB<int>(num_cints + num_lints);
  if (total_mem > 1024)
    std::cout << "     total: " << static_cast<double>(total_mem) / 1024.0 << "GB\n";
  else
    std::cout << "     total: " << total_mem << "MB\n";
}

template<typename precision>
void update_matrix_coefficients(PDE<precision> const &pde,
                                options const &program_options,
                                imex_flag const imex,
                                global_kron_matrix<precision> &mat)
{
  int const imex_indx = global_kron_matrix<precision>::flag2int(imex);

#ifdef ASGARD_USE_CUDA
  std::vector<precision> lvals;
#else
  std::vector<precision> &lvals = mat.local_opvalues_[imex_indx];
#endif

  std::vector<int> const &used_terms = mat.term_groups[imex_indx];

  int const pdegree        = pde.get_dimensions()[0].get_degree() - 1;
  int const kron_size      = pdegree + 1;
  int const num_dimensions = pde.num_dims;

  int64_t lda = kron_size * fm::two_raised_to((program_options.do_adapt_levels)
                                                  ? program_options.max_level
                                                  : pde.max_level);

  int const block_size = kron_size * kron_size;

  int const num_terms = static_cast<int>(used_terms.size());
  if (num_terms == 0)
    return;

  for (auto t : used_terms)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      std::vector<precision> &vals = mat.gvals_[t * num_dimensions + d];
      if (vals.empty()) // identity term, no values
        continue;
      std::vector<int> &ivals = mat.givals_[d];

      int64_t num_entries = static_cast<int64_t>(mat.gindx_[d].size());

      fk::matrix<precision> const &ops = pde.get_coefficients(t, d);

#pragma omp parallel for
      for (int64_t i = 0; i < num_entries; i++)
        vals[i] = ops(ivals[2 * i], ivals[2 * i + 1]);
    }
  }

  connect_1d const &edges = mat.edges_;
  int const dim_block     = edges.num_connections() * block_size;

  if (lvals.empty())
    lvals.resize(num_terms * num_dimensions * dim_block);

  for (int t = 0; t < num_terms; t++)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      auto iv = lvals.begin() + (t * num_dimensions + d) * dim_block;

      precision const *const ops = pde.get_coefficients(used_terms[t], d).data();

      for (int cell = 0; cell < edges.num_rows(); cell++)
      {
        for (int j = edges.row_begin(cell); j < edges.row_end(cell); j++)
        {
          int const col = edges[j] * kron_size;
          for (int p = 0; p < kron_size; p++)
            iv = std::copy_n(ops + kron_size * cell + lda * (col + p), kron_size, iv);
        }
      }
    }
  }
#ifdef ASGARD_USE_CUDA
  mat.local_opvalues_[imex_indx] = lvals;
#endif
}

template<typename precision>
template<resource rec>
void global_kron_matrix<precision>::apply(matrix_entry etype, precision alpha, precision const *x, precision beta, precision *y) const
{
  tools::time_event kron_time_("kronmult global");
  int const imex = flag2int(etype);

  std::vector<int> const &used_terms = term_groups[imex];
  if (used_terms.size() == 0)
  {
    if (beta != 0)
      lib_dispatch::scal<resource::host>(num_active_, beta, y, 1);
    return;
  }

  if constexpr (rec == resource::device)
  {
#ifdef ASGARD_USE_CUDA
#ifdef ASGARD_USE_PINNED_MEMORY
    cudaMemcpyAsync(pinned_mem, x, num_active_ * sizeof(precision), cudaMemcpyDeviceToHost, io_stream);
    kronmult::gpu_sparse(num_dimensions_, porder_ + 1, ydev_.size(), local_cols_.size(),
                         local_cols_.data(), local_rows_.data(), used_terms.size(),
                         local_opindex_[imex].data(), local_opvalues_[imex].data(),
                         alpha, x, beta, y);

    std::fill(expanded.begin() + num_active_, expanded.end(), precision{0});
    precision *yglobal = expanded.data() + ilist_.num_strips();

    cudaStreamSynchronize(io_stream);
    std::copy_n(pinned_mem, num_active_, expanded.begin());
    kronmult::global_cpu(num_dimensions_, perms_, gpntr_, gindx_, gdiag_, gvals_,
                         used_terms, expanded.data(), yglobal, workspace.data());

    std::copy_n(yglobal, num_active_, pinned_mem);
    cudaMemcpyAsync(ydev_.data(), pinned_mem, num_active_ * sizeof(precision), cudaMemcpyHostToDevice, io_stream);
    cudaStreamSynchronize(io_stream);
    lib_dispatch::axpy<resource::device>(num_active_, alpha, ydev_.data(), 1, y, 1);
#else
    fk::copy_to_host<precision>(expanded.data(), x, num_active_); // can do asynchronously

    kronmult::gpu_sparse(num_dimensions_, porder_ + 1, ydev_.size(), local_cols_.size(),
                         local_cols_.data(), local_rows_.data(), used_terms.size(),
                         local_opindex_[imex].data(), local_opvalues_[imex].data(),
                         alpha, x, beta, y);

    std::fill(expanded.begin() + num_active_, expanded.end(), precision{0});
    precision *yglobal = expanded.data() + ilist_.num_strips();

    kronmult::global_cpu(num_dimensions_, perms_, gpntr_, gindx_, gdiag_, gvals_,
                         used_terms, expanded.data(), yglobal, workspace.data());

    fk::copy_to_device<precision>(ydev_.data(), yglobal, num_active_);
    lib_dispatch::axpy<resource::device>(num_active_, alpha, ydev_.data(), 1, y, 1);
#endif
#endif
  }
  else
  {
#ifdef ASGARD_USE_CUDA
    // the local part can only be done on the GPU
    fk::copy_to_device<precision>(xdev_.data(), x, xdev_.size());
    fk::copy_to_device<precision>(ydev_.data(), y, ydev_.size());
    kronmult::gpu_sparse(num_dimensions_, porder_ + 1, ydev_.size(), local_cols_.size(),
                         local_cols_.data(), local_rows_.data(), used_terms.size(),
                         local_opindex_[imex].data(), local_opvalues_[imex].data(),
                         alpha, xdev_.data(), beta, ydev_.data());
    fk::copy_to_host<precision>(y, ydev_.data(), ydev_.size());
#else
    kronmult::cpu_sparse(num_dimensions_, porder_ + 1, local_pntr_.size() - 1,
                         local_pntr_.data(), local_indx_.data(), used_terms.size(),
                         local_opindex_[imex].data(), local_opvalues_[imex].data(),
                         alpha, x, beta, y);
#endif

    std::copy_n(x, num_active_, expanded.begin());
    std::fill(expanded.begin() + num_active_, expanded.end(), precision{0});
    precision *yglobal = expanded.data() + ilist_.num_strips();
    kronmult::global_cpu(num_dimensions_, perms_, gpntr_, gindx_, gdiag_, gvals_,
                         used_terms, expanded.data(), yglobal, workspace.data());

#pragma omp parallel for
    for (int64_t i = 0; i < num_active_; i++)
      y[i] += alpha * yglobal[i];
  }
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template std::vector<int> get_used_terms(PDE<double> const &pde, options const &opts,
                                         imex_flag const imex);

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
#ifdef KRON_MODE_GLOBAL
template global_kron_matrix<double>
make_global_kron_matrix(PDE<double> const &,
                        adapt::distributed_grid<double> const &,
                        options const &);
template void update_matrix_coefficients(PDE<double> const &,
                                         options const &,
                                         imex_flag const,
                                         global_kron_matrix<double> &);
template void set_specific_mode<double>(PDE<double> const &,
                                        adapt::distributed_grid<double> const &,
                                        options const &, imex_flag const,
                                        global_kron_matrix<double> &);
template class global_kron_matrix<double>;
template void global_kron_matrix<double>::apply<resource::host>(
    matrix_entry, double, double const *, double, double *) const;
#ifdef ASGARD_USE_CUDA
template void global_kron_matrix<double>::apply<resource::device>(
    matrix_entry, double, double const *, double, double *) const;
#endif
#endif
#endif

#ifdef ASGARD_ENABLE_FLOAT
template std::vector<int> get_used_terms(PDE<float> const &pde, options const &opts,
                                         imex_flag const imex);

template kronmult_matrix<float>
make_kronmult_matrix<float>(PDE<float> const &,
                            adapt::distributed_grid<float> const &,
                            options const &, memory_usage const &,
                            imex_flag const, kron_sparse_cache &, bool);
template void update_kronmult_coefficients<float>(PDE<float> const &, options const &,
                                                  imex_flag const, kron_sparse_cache &,
                                                  kronmult_matrix<float> &);
template memory_usage
compute_mem_usage<float>(PDE<float> const &,
                         adapt::distributed_grid<float> const &,
                         options const &, imex_flag const, kron_sparse_cache &,
                         int, int64_t, bool);
#ifdef KRON_MODE_GLOBAL
template global_kron_matrix<float>
make_global_kron_matrix(PDE<float> const &,
                        adapt::distributed_grid<float> const &,
                        options const &);
template void update_matrix_coefficients(PDE<float> const &,
                                         options const &,
                                         imex_flag const,
                                         global_kron_matrix<float> &);
template void set_specific_mode<float>(PDE<float> const &,
                                       adapt::distributed_grid<float> const &,
                                       options const &, imex_flag const,
                                       global_kron_matrix<float> &);
template class global_kron_matrix<float>;
template void global_kron_matrix<float>::apply<resource::host>(
    matrix_entry, float, float const *, float, float *) const;
#ifdef ASGARD_USE_CUDA
template void global_kron_matrix<float>::apply<resource::device>(
    matrix_entry, float, float const *, float, float *) const;
#endif
#endif
#endif

} // namespace asgard
