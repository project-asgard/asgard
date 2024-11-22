#include "asgard_kronmult_matrix.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace asgard
{
//! \brief Extract the actual set of terms based on the pde
template<typename precision>
std::vector<int> get_used_terms(PDE<precision> const &pde, imex_flag const imex)
{
  if (pde.use_imex()) // respect the imex flag
  {
    std::vector<int> terms;
    terms.reserve(pde.num_terms());
    for (int t = 0; t < pde.num_terms(); t++)
      if (pde.get_terms()[t][0].flag() == imex)
        terms.push_back(t);

    return terms;
  }
  else // take all terms
  {
    std::vector<int> terms(pde.num_terms());
    std::iota(terms.begin(), terms.end(), 0); // fills with 0, 1, 2, 3 ...
    return terms;
  }
}

template<typename precision>
vector2d<int> get_cells(int num_dimensions, adapt::distributed_grid<precision> const &dis_grid)
{
  auto const &grid         = dis_grid.get_subgrid(get_rank());
  int const *const asg_idx = dis_grid.get_table().get_active_table().data();
  int const num_cells      = grid.col_stop - grid.col_start + 1;
  return asg2tsg_convert(num_dimensions, num_cells, asg_idx);
}

//! Returns true if the current term is identity and can be omitted
template<typename precision>
bool check_identity_term(PDE<precision> const &pde, int term_id, int dim)
{
  // Check that the volumne jacobian in this dimension is not identity,
  if (pde.get_dimensions()[dim].volume_jacobian_dV != nullptr)
    return false;
  // Now check g_func, the lhs_func, and local surface jacobian are identity.
  // TODO: There is an edge case where the mass matrices with the same volume
  // jacobians can cancel out, but this requires us to know that both the
  // dimensions' and the local terms' volume jacobian are equal.
  // In the edge case, identity will be multiplied instead of ignored
  // resulting in extra work but correct output.
  for (auto const &pt : pde.get_terms()[term_id][dim].get_partial_terms())
    if (pt.coeff_type() != coefficient_type::mass or
        pt.g_func() != nullptr or
        pt.lhs_mass_func() != nullptr or
        pt.dv_func() != nullptr)
      return false;
  return true;
}

/*!
 * \brief Constructs a preconditioner
 *
 * The preconditioner should go into another file, but that will come with a
 * big cleanup of the kronmult logic (and the removal of the old code).
 */
template<typename precision>
void build_preconditioner(PDE<precision> const &pde,
                          coefficient_matrices<precision> const &cmats,
                          connection_patterns const &conns,
                          adapt::distributed_grid<precision> const &dis_grid,
                          std::vector<int> const &used_terms,
                          std::vector<precision> &pc)
{
  auto const &grid           = dis_grid.get_subgrid(get_rank());
  int const *const asg_table = dis_grid.get_table().get_active_table().data();

  int const num_rows = grid.row_stop - grid.row_start + 1;

  int const pdofs = pde.get_dimensions()[0].get_degree() + 1;

  int num_dimensions  = pde.num_dims();
  int64_t tensor_size = fm::ipow(pdofs, num_dimensions);

  if (pc.size() == 0)
    pc.resize(tensor_size * num_rows);
  else
  {
    pc.resize(tensor_size * num_rows);
    std::fill(pc.begin(), pc.end(), precision{0});
  }

#pragma omp parallel
  {
    std::array<int, max_num_dimensions> midx;
    std::array<precision const *, max_num_dimensions> amats;

#pragma omp for
    for (int64_t row = 0; row < num_rows; row++)
    {
      int const *const row_coords = asg_table + 2 * num_dimensions * (grid.row_start + row);
      asg2tsg_convert(num_dimensions, row_coords, midx.data());

      for (int tentry = 0; tentry < tensor_size; tentry++)
      {
        for (int t : used_terms)
        {
          for (int d : indexof<int>(num_dimensions))
          {
            if (check_identity_term(pde, t, d))
              amats[d] = nullptr;
            else
            {
              block_sparse_matrix<precision> const &mat = cmats.term_coeffs[t * num_dimensions + d];
              connect_1d const &conn = conns(mat);
              amats[d] = mat[conn.row_diag(midx[d])];
            }
          }

          precision a = 1;

          int tt = tentry;
          for (int d = num_dimensions - 1; d >= 0; d--)
          {
            if (amats[d] != nullptr)
            {
              int const rc = tt % pdofs;
              a *= amats[d][rc * pdofs + rc];
            }
            tt /= pdofs;
          }
          pc[row * tensor_size + tentry] += a;
        }
      }
    }
  }
}

#ifndef KRON_MODE_GLOBAL

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
local_kronmult_matrix<precision>
make_kronmult_dense(PDE<precision> const &pde,
                    coefficient_matrices<precision> &cmats,
                    connection_patterns const &conn,
                    adapt::distributed_grid<precision> const &discretization,
                    imex_flag const imex, verbosity_level verb)
{
  // convert pde to kronmult dense matrix
  auto const &grid         = discretization.get_subgrid(get_rank());
  int const num_dimensions = pde.num_dims();
  int const kron_size      = pde.get_dimensions()[0].get_degree() + 1;
  int const num_rows       = grid.row_stop - grid.row_start + 1;
  int const num_cols       = grid.col_stop - grid.col_start + 1;

  int64_t lda = kron_size * fm::ipow2(pde.max_level());

  // take into account the terms that will be skipped due to the imex_flag
  std::vector<int> const used_terms = get_used_terms(pde, imex);
  int const num_terms               = static_cast<int>(used_terms.size());

  if (used_terms.empty())
    return asgard::local_kronmult_matrix<precision>(num_dimensions, kron_size,
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
              : (fm::ipow2(idx[d] - 1) + idx[d + num_dimensions]);
    }
  }

  std::vector<precision> prec;
  if (imex == imex_flag::imex_implicit or pde.use_implicit())
    // prepare a preconditioner
    build_preconditioner(pde, cmats, conn, discretization, used_terms, prec);

  int64_t flps = local_kronmult_matrix<precision>::compute_flops(
      num_dimensions, kron_size, num_terms, int64_t{num_rows} * num_cols);

  if (verb == verbosity_level::high)
  {
    std::cout << "  kronmult dense matrix size: " << num_rows << " rows/cols\n";
    std::cout << "  -- work: " << flps * 1.E-9 << " Gflops\n";

    std::cout << "  -- memory usage: "
              << get_MB<precision>(terms.size()) + get_MB<int>(elem.size())
              << "MB\n";
  }

#ifdef ASGARD_USE_CUDA
  std::vector<fk::vector<precision, mem_type::owner, resource::device>>
      gpu_terms(num_terms);
  for (int t = 0; t < num_terms; t++)
    gpu_terms[t] = terms[t].clone_onto_device();

  auto gpu_elem = elem.clone_onto_device();

  return asgard::local_kronmult_matrix<precision>(
      num_dimensions, kron_size, num_rows, num_cols, num_terms,
      std::move(gpu_terms), std::move(gpu_elem), grid.row_start, grid.col_start,
      num_1d_blocks, std::move(prec));
#else
  return asgard::local_kronmult_matrix<precision>(
      num_dimensions, kron_size, num_rows, num_cols, num_terms,
      std::move(terms), std::move(elem), grid.row_start, grid.col_start,
      num_1d_blocks, std::move(prec));
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
          ++edge_conn; // periodic boundary
        else
        {
          if (std::abs(row[num_dimensions + j] - col[num_dimensions + j]) == 1)
            ++edge_conn; // adjacent elements
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
local_kronmult_matrix<precision>
make_kronmult_sparse(PDE<precision> const &pde,
                     coefficient_matrices<precision> &cmats,
                     connection_patterns const &conn,
                     adapt::distributed_grid<precision> const &discretization,
                     memory_usage const &mem_stats, imex_flag const imex,
                     kron_sparse_cache &spcache, verbosity_level verb)
{
  tools::time_event performance_("make-kronmult-sparse");
  // convert pde to kronmult dense matrix
  auto const &grid         = discretization.get_subgrid(get_rank());
  int const num_dimensions = pde.num_dims();
  int const kron_size      = pde.get_dimensions()[0].get_degree() + 1;
  int const num_rows       = grid.row_stop - grid.row_start + 1;
  int const num_cols       = grid.col_stop - grid.col_start + 1;

  int64_t lda = kron_size * fm::ipow2(pde.max_level());

  // take into account the terms that will be skipped due to the imex_flag
  std::vector<int> const used_terms = get_used_terms(pde, imex);

  int const num_terms = static_cast<int>(used_terms.size());

  if (used_terms.empty())
    return asgard::local_kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols);

  // size of the small kron matrices
  int const kron_squared = kron_size * kron_size;

  int const num_1d = spcache.cells1d.num_connections();

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

#ifdef ASGARD_USE_CUDA
  int const tensor_size = fm::ipow<int>(kron_size, num_dimensions);
#else
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

  std::vector<precision> prec;
  if (imex == imex_flag::imex_implicit or pde.use_implicit())
    // prepare a preconditioner
    build_preconditioner(pde, cmats, conn, discretization, used_terms, prec);

  if (verb == verbosity_level::high)
  {
    std::cout << "  kronmult local, sparse matrix fill: "
              << 100.0 * double(spcache.num_nonz) /
                     (double(num_rows) * double(num_cols))
              << "%\n";

    int64_t flops = local_kronmult_matrix<precision>::compute_flops(
        num_dimensions, kron_size, num_terms, spcache.num_nonz);
    std::cout << "  -- work: " << flops * 1.E-9 << " Gflops\n";
  }

#ifdef ASGARD_USE_CUDA
  if (mem_stats.kron_call == memory_usage::one_call)
  {
    if (verb == verbosity_level::high)
      std::cout << "  -- memory usage (unique): "
                << get_MB<int>(list_row_indx[0].size()) +
                       get_MB<int>(list_col_indx[0].size()) +
                       get_MB<int>(list_iA[0].size()) +
                       get_MB<precision>(vA.size())
                << "\n";
    return local_kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols, num_terms,
        list_row_indx[0].clone_onto_device(),
        list_col_indx[0].clone_onto_device(), list_iA[0].clone_onto_device(),
        vA.clone_onto_device(), std::move(prec));
  }
  else
  {
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    if (verb == verbosity_level::high)
    {
      std::cout << "        memory usage (unique): "
                << get_MB<precision>(vA.size()) << "\n";
      std::cout << "        memory usage (shared): "
                << 2 * get_MB<int>(mem_stats.work_size) +
                       4 * get_MB<int>(mem_stats.row_work_size)
                << "\n";
    }
    return local_kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols, num_terms,
        std::move(list_row_indx), std::move(list_col_indx), std::move(list_iA),
        vA.clone_onto_device(), std::move(prec));
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
    if (verb == verbosity_level::high)
      std::cout << "        memory usage (MB): "
                << get_MB<precision>(vA.size()) + get_MB<int>(num_ints) << "\n";
    return local_kronmult_matrix<precision>(
        num_dimensions, kron_size, num_rows, num_cols, num_terms,
        std::move(gpu_row), std::move(gpu_col), std::move(gpu_iA),
        vA.clone_onto_device(), std::move(prec));
#endif
  }
#else

  return local_kronmult_matrix<precision>(
      num_dimensions, kron_size, num_rows, num_cols, num_terms,
      std::move(list_row_indx), std::move(list_col_indx), std::move(list_iA),
      std::move(vA), std::move(prec));

#endif
}

template<typename P>
local_kronmult_matrix<P>
make_local_kronmult_matrix(
    PDE<P> const &pde, coefficient_matrices<P> &cmats,
    connection_patterns const &conn,
    adapt::distributed_grid<P> const &grid,
    memory_usage const &mem_stats, imex_flag const imex, kron_sparse_cache &spcache,
    verbosity_level verb, bool force_sparse)
{
  if (pde.kron_mod() == kronmult_mode::dense and not force_sparse)
  {
    return make_kronmult_dense<P>(pde, cmats, conn, grid, imex, verb);
  }
  else
  {
    return make_kronmult_sparse<P>(pde, cmats, conn, grid, mem_stats, imex, spcache, verb);
  }
}

template<typename P>
void update_kronmult_coefficients(PDE<P> const &pde,
                                  coefficient_matrices<P> &,
                                  connection_patterns const &,
                                  imex_flag const imex,
                                  kron_sparse_cache &spcache,
                                  local_kronmult_matrix<P> &mat)
{
  tools::time_event kron_time_("kronmult-update-coefficients");
  int const num_dimensions = pde.num_dims();
  int const kron_size      = pde.get_dimensions()[0].get_degree() + 1;

  int64_t lda = kron_size * fm::ipow2(pde.max_level());

  // take into account the terms that will be skipped due to the imex_flag
  std::vector<int> const used_terms = get_used_terms(pde, imex);

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
                  imex_flag const imex, kron_sparse_cache &spcache,
                  int memory_limit_MB, int64_t index_limit,
                  bool force_sparse)
{
  auto const &grid         = discretization.get_subgrid(get_rank());
  int const num_dimensions = pde.num_dims();
  int const kron_size      = pde.get_dimensions()[0].get_degree() + 1;
  int const num_rows       = grid.row_stop - grid.row_start + 1;
  int const num_cols       = grid.col_stop - grid.col_start + 1;

  memory_usage stats;

#ifdef ASGARD_USE_GPU_MEM_LIMIT
  if (memory_limit_MB == 0)
    memory_limit_MB = pde.memory_limit();
#else
  ignore(memory_limit_MB);
#endif

  // parameters common to the dense and sparse cases
  // matrices_per_prod is the number of matrices per Kronecker product
  int64_t matrices_per_prod = pde.num_terms() * num_dimensions;

  // base_line_entries are the entries that must always be loaded in GPU memory
  // first we compute the size of the state vectors (x and y) and then we
  // add the size of the coefficients (based on sparse/dense mode)
  int64_t base_line_entries =
      (num_rows + num_cols) * fm::ipow<int64_t>(kron_size, num_dimensions);

  if (pde.kron_mod() == kronmult_mode::dense and not force_sparse)
  {
    // assume all terms will be loaded into the GPU, as one IMEX flag or another
    for (int t = 0; t < pde.num_terms(); t++)
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
    if (pde.max_level() != spcache.cells1d.max_loaded_level())
      spcache.cells1d = connect_1d(pde.max_level());

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
                         pde.num_terms() * kron_size * kron_size;

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
      int min_terms = pde.num_terms();
      if (imex != imex_flag::unspecified)
      {
        std::vector<int> const explicit_terms =
            get_used_terms(pde, imex_flag::imex_explicit);
        std::vector<int> const implicit_terms =
            get_used_terms(pde, imex_flag::imex_implicit);
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

#endif // KRON_MODE_GLOBAL

#ifdef KRON_MODE_GLOBAL
//! Return the direction of the flux term or -1 for mass term
template<typename precision>
int get_flux_direction(PDE<precision> const &pde, int term_id)
{
  for (int d = 0; d < pde.num_dims(); d++)
    for (auto const &pt : pde.get_terms()[term_id][d].get_partial_terms())
      if (pt.coeff_type() == coefficient_type::div or
          pt.coeff_type() == coefficient_type::grad or
          pt.coeff_type() == coefficient_type::penalty)
        return d;
  return -1;
}

template<typename precision>
template<resource rec>
void block_global_kron_matrix<precision>::apply(
    imex_flag etype, precision alpha, precision *y) const
{
  int const imex = static_cast<int>(etype);

  std::vector<int> const &used_terms = term_groups_[imex];

  std::fill_n(workspace_->y.begin(), num_padded_, precision{0});

  kronmult::global_cpu(num_dimensions_, blockn_, block_size_, ilist_, dsort_,
                       perms_, flux_dir_, *conn_volumes_, *conn_full_,
                       gvals_, used_terms, workspace_->x.data(),
                       workspace_->y.data(), *workspace_);

  precision const *py = workspace_->y.data();
#pragma omp parallel for
  for (int64_t i = 0; i < num_active_; i++)
    y[i] += alpha * py[i];
}

template<typename precision>
block_global_kron_matrix<precision>
make_block_global_kron_matrix(PDE<precision> const &pde,
                              adapt::distributed_grid<precision> const &dis_grid,
                              connect_1d const *volumes, connect_1d const *fluxes,
                              kronmult::block_global_workspace<precision> *workspace,
                              verbosity_level verb)
{
  int const degree = pde.get_dimensions()[0].get_degree();

  int const num_dimensions = pde.num_dims();
  int const num_terms      = pde.num_terms();

  int64_t block_size = fm::ipow(degree + 1, num_dimensions);

  vector2d<int> cells = get_cells(num_dimensions, dis_grid);
  int const num_cells = cells.num_strips();

  indexset padded = compute_ancestry_completion(make_index_set(cells), *volumes);
  if (verb == verbosity_level::high)
    std::cout << " number of padding cells = " << padded.num_indexes() << '\n';

  if (padded.num_indexes() > 0)
    cells.append(padded[0], padded.num_indexes());

  dimension_sort dsort(cells);

  // figure out the permutation patterns
  std::vector<int> flux_dir(num_terms, -1);
  std::vector<kronmult::permutes> permutations;
  permutations.reserve(num_terms);
  std::vector<int> active_dirs(num_dimensions);
  for (int t = 0; t < num_terms; t++)
  {
    flux_dir[t] = get_flux_direction(pde, t);

    active_dirs.clear();
    // add only the dimensions that are not identity
    // make sure that the flux direction comes first
    for (int d = 0; d < num_dimensions; d++)
      if (not check_identity_term(pde, t, d))
      {
        active_dirs.push_back(d);
        if (d == flux_dir[t] and active_dirs.size() > 1)
          std::swap(active_dirs.front(), active_dirs.back());
      }

    permutations.emplace_back(active_dirs);
  }

  int64_t num_padded = cells.num_strips() * block_size;
  workspace->x.resize(num_padded);
  std::fill_n(workspace->x.begin(), num_padded, precision{0});
  workspace->y.resize(num_padded);
  workspace->w1.resize(num_padded);
  workspace->w2.resize(num_padded);

  return block_global_kron_matrix<precision>(
      num_cells * block_size, num_padded,
      num_dimensions, degree + 1, block_size,
      std::move(cells), std::move(dsort), std::move(permutations),
      std::move(flux_dir), volumes, fluxes,
      workspace, verb);
}

template<typename precision>
void set_specific_mode(PDE<precision> const &pde,
                       coefficient_matrices<precision> const &cmats,
                       connection_patterns const &conns,
                       adapt::distributed_grid<precision> const &dis_grid,
                       imex_flag const imex,
                       block_global_kron_matrix<precision> &mat)
{
  int const imex_indx = static_cast<int>(imex);

  mat.term_groups_[imex_indx] = get_used_terms(pde, imex);

  std::vector<int> const &used_terms = mat.term_groups_[imex_indx];

  int const num_dimensions = pde.num_dims();

  for (int const t : used_terms)
  {
    for (int d : indexof<int>(num_dimensions))
    {
      if (not check_identity_term(pde, t, d))
      {
        // This should be an alias and not a copy
        cmats.term_coeffs[t * num_dimensions + d].copy_out(mat.gvals_[t * num_dimensions + d]);
      }
    }
  }

  if (imex == imex_flag::imex_implicit or pde.use_implicit())
    // prepare a preconditioner
    build_preconditioner(pde, cmats, conns, dis_grid, used_terms, mat.pre_con_);
}

#endif // KRON_MODE_GLOBAL

#ifdef ASGARD_ENABLE_DOUBLE
template std::vector<int> get_used_terms(PDE<double> const &pde,
                                         imex_flag const imex);

template vector2d<int> get_cells(int, adapt::distributed_grid<double> const &);

#ifdef KRON_MODE_GLOBAL
template class block_global_kron_matrix<double>;
template void block_global_kron_matrix<double>::apply<resource::host>(
    imex_flag, double, double *) const;

template block_global_kron_matrix<double>
make_block_global_kron_matrix<double>(PDE<double> const &,
                                      adapt::distributed_grid<double> const &,
                                      connect_1d const *, connect_1d const *,
                                      kronmult::block_global_workspace<double> *,
                                      verbosity_level);
template void set_specific_mode<double>(PDE<double> const &,
                                        coefficient_matrices<double> const &,
                                        connection_patterns const &conns,
                                        adapt::distributed_grid<double> const &,
                                        imex_flag const,
                                        block_global_kron_matrix<double> &);

#else // KRON_MODE_GLOBAL

template local_kronmult_matrix<double>
make_local_kronmult_matrix<double>(
    PDE<double> const &, coefficient_matrices<double> &,
    connection_patterns const &, adapt::distributed_grid<double> const &,
    memory_usage const &, imex_flag const, kron_sparse_cache &, verbosity_level,
    bool);
template void
update_kronmult_coefficients<double>(
    PDE<double> const &, coefficient_matrices<double> &,
    connection_patterns const &, imex_flag const, kron_sparse_cache &,
    local_kronmult_matrix<double> &);

template memory_usage
compute_mem_usage<double>(PDE<double> const &,
                          adapt::distributed_grid<double> const &,
                          imex_flag const, kron_sparse_cache &,
                          int, int64_t, bool);
#endif // KRON_MODE_GLOBAL
#endif // ASGARD_ENABLE_DOUBLE

#ifdef ASGARD_ENABLE_FLOAT
template std::vector<int> get_used_terms(PDE<float> const &pde,
                                         imex_flag const imex);

template vector2d<int> get_cells(int, adapt::distributed_grid<float> const &);

#ifdef KRON_MODE_GLOBAL
template class block_global_kron_matrix<float>;

template void block_global_kron_matrix<float>::apply<resource::host>(
    imex_flag, float, float *) const;

template block_global_kron_matrix<float>
make_block_global_kron_matrix<float>(PDE<float> const &,
                                     adapt::distributed_grid<float> const &,
                                     connect_1d const *, connect_1d const *,
                                     kronmult::block_global_workspace<float> *,
                                     verbosity_level);
template void set_specific_mode<float>(PDE<float> const &,
                                       coefficient_matrices<float> const &,
                                       connection_patterns const &,
                                       adapt::distributed_grid<float> const &,
                                       imex_flag const,
                                       block_global_kron_matrix<float> &);

#else // KRON_MODE_GLOBAL

template local_kronmult_matrix<float>
make_local_kronmult_matrix<float>(
    PDE<float> const &, coefficient_matrices<float> &, connection_patterns const &,
    adapt::distributed_grid<float> const &, memory_usage const &, imex_flag const,
    kron_sparse_cache &, verbosity_level, bool);
template void update_kronmult_coefficients<float>(
    PDE<float> const &, coefficient_matrices<float> &,
    connection_patterns const &, imex_flag const, kron_sparse_cache &,
    local_kronmult_matrix<float> &);

template memory_usage
compute_mem_usage<float>(PDE<float> const &,
                         adapt::distributed_grid<float> const &,
                         imex_flag const, kron_sparse_cache &,
                         int, int64_t, bool);
#endif
#endif // ASGARD_ENABLE_FLOAT

} // namespace asgard
