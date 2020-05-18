#include "kronmult.hpp"
#include "kronmult_cuda.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include <mutex>
#include <vector>

#include "connectivity.hpp"
#include "lib_dispatch.hpp"
#include "timer.hpp"
#include <limits.h>

namespace kronmult
{
// calculate how much workspace we need on device to compute a single connected
// element
//
// *does not include operator matrices - working for now on assumption they'll
// all be resident*
template<typename P>
inline double get_element_size_MB(PDE<P> const &pde)
{
  auto const elem_size      = element_segment_size(pde);
  auto const num_workspaces = 2;
  // each element requires two workspaces, X and W in Ed's code
  auto const elem_workspace_MB =
      get_MB<P>(pde.num_terms * elem_size) * num_workspaces;
  return elem_workspace_MB;
}

// determine how many subgrids will be required to solve the problem
// each subgrids is a subset of the element subgrid assigned to this rank,
// whose total workspace requirement is less than the limit passed in
// rank_size_MB
template<typename P>
inline int get_num_subgrids(PDE<P> const &pde, element_subgrid const &grid,
                            int const rank_size_MB)
{
  assert(grid.size() > 0);

  // determine total problem size
  auto const num_elems        = grid.size();
  double const space_per_elem = get_element_size_MB(pde);

  // determine size of assigned x and y vectors
  auto const elem_size   = element_segment_size(pde);
  auto const num_x_elems = static_cast<int64_t>(grid.nrows()) * elem_size;
  assert(num_x_elems < INT_MAX);
  auto const num_y_elems = static_cast<int64_t>(grid.ncols()) * elem_size;
  assert(num_y_elems < INT_MAX);
  double const xy_space_MB = get_MB<P>(num_y_elems + num_x_elems);

  // make sure rank size is something reasonable
  assert(space_per_elem < (0.5 * rank_size_MB));

  double const problem_size_MB = space_per_elem * num_elems;

  // FIXME here we assume all coefficients are of equal size; if we shortcut
  // computation for identity coefficients later, we will need to do this more
  // carefully
  int const coefficients_size_MB = static_cast<int>(std::ceil(
      get_MB<P>(static_cast<int64_t>(pde.get_coefficients(0, 0).size()) *
                pde.num_terms * pde.num_dims)));

  // make sure the coefficient matrices/xy vectors aren't leaving us without
  // room for anything else in device workspace
  auto const remaining_rank_MB =
      rank_size_MB - coefficients_size_MB - xy_space_MB;
  assert(remaining_rank_MB > space_per_elem * 4);

  // determine number of subgrids
  return static_cast<int>(std::ceil(problem_size_MB / remaining_rank_MB));
}

// helper - break subgrid into smaller subgrids to fit into DRAM
template<typename P>
std::vector<element_subgrid>
decompose(PDE<P> const &pde, element_subgrid const &my_subgrid,
          int const workspace_size_MB)
{
  assert(workspace_size_MB > 0);

  // min number subgrids
  auto const num_subgrids =
      get_num_subgrids(pde, my_subgrid, workspace_size_MB);
  if (num_subgrids == 1)
  {
    return std::vector<element_subgrid>{my_subgrid};
  }

  auto const max_elements_per_subgrid = my_subgrid.size() / num_subgrids;

  // max subgrid dimension (r or c)
  auto const subgrid_length =
      static_cast<int>(std::floor(std::sqrt(max_elements_per_subgrid)));

  // square tile the assigned subgrid
  std::vector<element_subgrid> grids;
  auto const round_up = [](int const to_round, int const multiple) {
    assert(multiple);
    return ((to_round + multiple - 1) / multiple) * multiple;
  };

  // create square tile iteration space
  // iterate over these creating square subgrids.
  // shrink if necessary to fit original subgrid boundary.
  auto const explode_rows = round_up(my_subgrid.nrows(), subgrid_length);
  auto const explode_cols = round_up(my_subgrid.ncols(), subgrid_length);

  for (auto i = 0; i < explode_rows / subgrid_length; ++i)
  {
    for (auto j = 0; j < explode_cols / subgrid_length; ++j)
    {
      auto const row_start = i * subgrid_length;
      auto const row_end =
          std::min(my_subgrid.row_stop, (i + 1) * subgrid_length - 1);
      auto const col_start = j * subgrid_length;
      auto const col_end =
          std::min(my_subgrid.col_stop, (j + 1) * subgrid_length - 1);
      grids.push_back(element_subgrid(row_start, row_end, col_start, col_end));
    }
  }
  return grids;
}

// helper - calculate element coordinates -> operator matrix indices
inline void
get_indices(fk::vector<int> const &coords, int indices[], int const degree)
{
  assert(degree > 0);

  int const indices_size = coords.size() / 2;
  for (int i = 0; i < indices_size; ++i)
  {
    indices[i] = get_1d_index(coords(i), coords(i + indices_size)) * degree;
  }
}

// helper, allocate memory WITHOUT init
template<typename P>
inline void allocate_device(P *&ptr, int const num_elems)
{
#ifdef ASGARD_USE_CUDA
  auto success = cudaMalloc((void **)&ptr, num_elems * sizeof(P));
  assert(success == 0);
#else
  ptr = new P[num_elems];
#endif
}

template<typename P>
inline void free_device(P *&ptr)
{
#ifdef ASGARD_USE_CUDA
  cudaFree(ptr);
#else
  delete[] ptr;
#endif
}

// private, directly execute one subgrid
template<typename P>
fk::vector<P, mem_type::view, resource::device>
execute(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid,
        fk::vector<P, mem_type::const_view, resource::device> const &x,
        fk::vector<P, mem_type::view, resource::device> &fx)
{
  static std::once_flag print_flag;

  // FIXME code relies on uniform degree across dimensions
  auto const degree     = pde.get_dimensions()[0].get_degree();
  auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));

  auto const output_size = my_subgrid.nrows() * deg_to_dim;

  assert(output_size == fx.size());
  auto const input_size = my_subgrid.ncols() * deg_to_dim;
  assert(input_size == x.size());

  // size of input/working space for kronmults - need 2
  auto const workspace_size = my_subgrid.size() * deg_to_dim * pde.num_terms;

  std::call_once(print_flag, [workspace_size] {
    // FIXME assumes (with everything else) that coefficients are equally sized
    node_out() << "workspace allocation (MB): " << get_MB<P>(workspace_size * 2)
               << '\n';
  });

  P *element_x;
  P *element_work;
  allocate_device(element_x, workspace_size);
  allocate_device(element_work, workspace_size);

  // stage x vector in writable regions for each element
#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
  for (auto i = 0; i < my_subgrid.nrows() * pde.num_terms; ++i)
  {
    fk::copy_on_device(element_x + i * x.size(), x.data(), x.size());
  }

  // loop over assigned elements, staging inputs and operators
  auto const total_kronmults = my_subgrid.size() * pde.num_terms;

  P **input_ptrs    = new P *[total_kronmults];
  P **work_ptrs     = new P *[total_kronmults];
  P **output_ptrs   = new P *[total_kronmults];
  P **operator_ptrs = new P *[total_kronmults * pde.num_dims];

#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
  for (auto i = my_subgrid.row_start; i <= my_subgrid.row_stop; ++i)
  {
    // calculate and store operator row indices for this element
    static int constexpr max_dims = 6;
    assert(pde.num_dims <= max_dims);
    int operator_row[max_dims];
    fk::vector<int> const &row_coords = elem_table.get_coords(i);
    assert(row_coords.size() == pde.num_dims * 2);
    get_indices(row_coords, operator_row, degree);

    for (auto j = my_subgrid.col_start; j <= my_subgrid.col_stop; ++j)
    {
      // calculate and store operator col indices for this element
      int operator_col[max_dims];
      fk::vector<int> const &col_coords = elem_table.get_coords(j);
      assert(col_coords.size() == pde.num_dims * 2);
      get_indices(col_coords, operator_col, degree);

      auto const x_start =
          element_x + (my_subgrid.to_local_row(i) * pde.num_terms * x.size() +
                       my_subgrid.to_local_col(j) * deg_to_dim);

      for (auto t = 0; t < pde.num_terms; ++t)
      {
        // get preallocated vector positions for this kronmult
        auto const num_kron =
            my_subgrid.to_local_row(i) * my_subgrid.ncols() * pde.num_terms +
            my_subgrid.to_local_col(j) * pde.num_terms + t;

        // point to inputs
        input_ptrs[num_kron] = x_start + t * x.size();

        // point to work/output
        work_ptrs[num_kron] = element_work + num_kron * deg_to_dim;
        output_ptrs[num_kron] =
            fx.data(my_subgrid.to_local_row(i) * deg_to_dim);

        // point to operators
        auto const operator_start = num_kron * pde.num_dims;
        for (auto d = 0; d < pde.num_dims; ++d)
        {
          auto const &coeff = pde.get_coefficients(t, d);
          operator_ptrs[operator_start + d] =
              coeff.data(operator_row[d], operator_col[d]);
        }
      }
    }
  }

  // FIXME assume all operators same size
  auto const lda = pde.get_coefficients(0, 0)
                       .stride(); // leading dimension of coefficient matrices

  double const flops = pde.num_dims * 2.0 *
                       (std::pow(degree, pde.num_dims + 1)) * total_kronmults;

  timer::record.start("kronmult");
  call_kronmult(degree, input_ptrs, output_ptrs, work_ptrs, operator_ptrs, lda,
                total_kronmults, pde.num_dims);
  timer::record.stop("kronmult", flops);

  free_device(element_x);
  free_device(element_work);

  free(input_ptrs);
  free(operator_ptrs);
  free(work_ptrs);
  free(output_ptrs);

  return fx;
}

// public, execute a given subgrid by decomposing and running over sub-subgrids
template<typename P>
fk::vector<P, mem_type::owner, resource::host>
execute(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid, int const workspace_size_MB,
        fk::vector<P, mem_type::owner, resource::host> const &x)
{
  static std::once_flag print_flag;
  auto const grids = decompose(pde, my_subgrid, workspace_size_MB);

  auto const degree     = pde.get_dimensions()[0].get_degree();
  auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));

  auto const output_size = my_subgrid.nrows() * deg_to_dim;
  assert(output_size < INT_MAX);
  fk::vector<P, mem_type::owner, resource::device> fx_dev(output_size);

  std::call_once(print_flag, [&pde, output_size] {
    // FIXME assumes (with everything else) that coefficients are equally sized
    auto const coefficients_size_MB =
        get_MB<P>(static_cast<int64_t>(pde.get_coefficients(0, 0).size()) *
                  pde.num_terms * pde.num_dims);
    node_out() << "kron workspace size..." << '\n';
    node_out() << "coefficient size (MB, existing allocation): "
               << coefficients_size_MB << '\n';
    node_out() << "x/fx allocation (MB): " << get_MB<P>(output_size) << '\n';
  });

  fk::vector<P, mem_type::owner, resource::device> const x_dev(
      x.clone_onto_device());

  for (auto const grid : grids)
  {
    fk::vector<P, mem_type::const_view, resource::device> const x_dev_grid(
        x_dev, grid.col_start * deg_to_dim,
        (grid.col_stop + 1) * deg_to_dim - 1);
    fk::vector<P, mem_type::view, resource::device> fx_dev_grid(
        fx_dev, grid.row_start * deg_to_dim,
        (grid.row_stop + 1) * deg_to_dim - 1);
    fx_dev_grid =
        kronmult::execute(pde, elem_table, grid, x_dev_grid, fx_dev_grid);
  }
  return fx_dev.clone_onto_host();
}

template std::vector<element_subgrid>
decompose(PDE<float> const &pde, element_subgrid const &my_subgrid,
          int const workspace_size_MB);

template std::vector<element_subgrid>
decompose(PDE<double> const &pde, element_subgrid const &my_subgrid,
          int const workspace_size_MB);

template fk::vector<float, mem_type::owner, resource::host>
execute(PDE<float> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid, int const workspace_size_MB,
        fk::vector<float, mem_type::owner, resource::host> const &x);

template fk::vector<double, mem_type::owner, resource::host>
execute(PDE<double> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid, int const workspace_size_MB,
        fk::vector<double, mem_type::owner, resource::host> const &x);

} // namespace kronmult
