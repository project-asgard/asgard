#include "chunk.hpp"
#include "fast_math.hpp"
#include <limits.h>

int num_elements_in_chunk(element_chunk const &g)
{
  int num_elems = 0;
  for (auto const &[row, cols] : g)
  {
    ignore(row);
    num_elems += cols.stop - cols.start + 1;
  }
  return num_elems;
}
int max_connected_in_chunk(element_chunk const &g)
{
  int current_max = 0;
  for (auto const &[row, cols] : g)
  {
    ignore(row);
    current_max = std::max(current_max, cols.stop - cols.start + 1);
  }
  return current_max;
}

grid_limits columns_in_chunk(element_chunk const &g)
{
  assert(g.size() > 0);
  int const min_col =
      (*std::min_element(g.begin(), g.end(),
                         [](auto const &a, auto const &b) {
                           return a.second.start < b.second.start;
                         }))
          .second.start;

  int const max_col = (*std::max_element(g.begin(), g.end(),
                                         [](auto const &a, auto const &b) {
                                           return a.second.stop < b.second.stop;
                                         }))
                          .second.stop;
  return grid_limits(min_col, max_col);
}

grid_limits rows_in_chunk(element_chunk const &g)
{
  assert(g.size() > 0);
  return grid_limits(g.begin()->first, g.rbegin()->first);
}

template<typename P>
host_workspace<P>::host_workspace(PDE<P> const &pde,
                                  element_subgrid const &grid,
                                  int const memory_limit_MB)
{
  assert(memory_limit_MB > 0);
  int const elem_size    = element_segment_size(pde);
  int64_t const col_size = elem_size * static_cast<int64_t>(grid.ncols());
  int64_t const row_size = elem_size * static_cast<int64_t>(grid.nrows());
  assert(col_size < INT_MAX);
  assert(row_size < INT_MAX);
  x_orig.resize(col_size);
  x.resize(col_size);
  fx.resize(row_size);
  reduced_fx.resize(row_size);
  scaled_source.resize(row_size);
  result_1.resize(col_size);
  result_2.resize(col_size);
  result_3.resize(col_size);

  /* we eventually need to make these checks consistent across realspace
     transform and time advance */
  /* size_MB() calculation relies on above members already being initialized */
  assert(size_MB() <= static_cast<double>(memory_limit_MB));
}

// calculate how much workspace we need on device to compute a single connected
// element
//
// *does not include operator matrices - working for now on assumption they'll
// all be resident*
template<typename P>
static double get_element_size_MB(PDE<P> const &pde)
{
  int const elem_size = element_segment_size(pde);
  // number of intermediate workspaces for kron product.
  // FIXME this only applies to explicit
  int const num_workspaces = std::min(pde.num_dims - 1, 2);

  // calc size of reduction space for a single work item
  double const elem_reduction_space_MB = get_MB<P>(pde.num_terms * elem_size);
  // calc size of intermediate space for a single work item
  double const elem_intermediate_space_MB =
      num_workspaces == 0 ? 0.0
                          : get_MB<P>(static_cast<double>(num_workspaces) *
                                      pde.num_terms * elem_size);

  return elem_reduction_space_MB + elem_intermediate_space_MB;
}

// determine how many chunks will be required to solve the problem
// a chunk is a subset of the element subgrid whose total workspace requirement
// is less than the limit passed in rank_size_MB
template<typename P>
int get_num_chunks(element_subgrid const &grid, PDE<P> const &pde,
                   int const rank_size_MB)
{
  assert(grid.size() > 0);

  // determine total problem size
  auto const num_elems        = grid.size();
  double const space_per_elem = get_element_size_MB(pde);

  // determine size of assigned x and y vectors
  int const elem_size = element_segment_size(pde);
  auto const num_x_elems =
      static_cast<uint64_t>(grid.col_stop - grid.col_start + 1) * elem_size;
  assert(num_x_elems < INT_MAX);
  auto const num_y_elems =
      static_cast<uint64_t>(grid.row_stop - grid.row_start + 1) * elem_size;
  assert(num_y_elems < INT_MAX);
  double const xy_space_MB = get_MB<P>(num_y_elems + num_x_elems);

  // make sure rank size is something reasonable
  // a single element is the finest we can split the problem
  // if that requires a lot of space relative to rank size,
  // roundoff of elements over chunks will cause us to exceed the limit
  assert(space_per_elem < (0.5 * rank_size_MB));

  double const problem_size_MB = space_per_elem * num_elems;

  // FIXME here we assume all coefficients are of equal size; if we shortcut
  // computation for identity coefficients later, we will need to do this more
  // carefully
  int const coefficients_size_MB = static_cast<int>(std::ceil(
      get_MB<P>(static_cast<uint64_t>(pde.get_coefficients(0, 0).size()) *
                pde.num_terms * pde.num_dims)));

  // make sure the coefficient matrices/xy vectors aren't leaving us without
  // room for anything else in device workspace
  int const remaining_rank_MB =
      rank_size_MB - coefficients_size_MB - xy_space_MB;
  assert(remaining_rank_MB > space_per_elem * 2);

  // determine number of chunks
  return static_cast<int>(std::ceil(problem_size_MB / remaining_rank_MB));
}

// divide the problem given the previously computed number of chunks
// this function divides via a greedy, row-major split.
// i.e., consecutive elements are taken row-wise until the end of a
// row, and continuing as needed to the next row, beginning with the first
// element of the new row (typewriter style). this is done to minimize the
// portion of the y-vector written to by each task, and ultimately the size of
// communication between ranks.
std::vector<element_chunk>
assign_elements(element_subgrid const &grid, int const num_chunks)
{
  assert(num_chunks > 0);

  auto const num_elems = grid.size();

  int64_t const elems_left_over = num_elems % num_chunks;
  int64_t const elems_per_chunk =
      num_elems / num_chunks + elems_left_over / num_chunks;
  int64_t const still_left_over = elems_left_over % num_chunks;

  std::vector<element_chunk> chunks;
  int64_t assigned = 0;

  for (int i = 0; i < num_chunks; ++i)
  {
    std::map<int, std::vector<int>> chunk_map;
    auto const insert = [&chunk_map, &grid](int const key, int const col) {
      chunk_map.try_emplace(grid.to_global_row(key), std::vector<int>());
      chunk_map[grid.to_global_row(key)].push_back(grid.to_global_col(col));
    };
    int64_t const elems_this_chunk =
        i < still_left_over ? elems_per_chunk + 1 : elems_per_chunk;
    int64_t const chunk_end = assigned + elems_this_chunk - 1;

    int64_t const chunk_start_row = assigned / grid.ncols();
    int64_t const chunk_start_col = assigned % grid.ncols();
    int64_t const chunk_end_row   = chunk_end / grid.ncols();
    int64_t const chunk_end_col   = chunk_end % grid.ncols();

    assigned += elems_this_chunk;

    if (chunk_end_row > chunk_start_row)
    {
      for (int i = chunk_start_row + 1; i < chunk_end_row; ++i)
      {
        for (int j = 0; j < grid.ncols(); ++j)
        {
          insert(i, j);
        }
      }
      for (int j = chunk_start_col; j < grid.ncols(); ++j)
      {
        insert(chunk_start_row, j);
      }
      for (int j = 0; j <= chunk_end_col; ++j)
      {
        insert(chunk_end_row, j);
      }
    }
    else
    {
      for (int j = chunk_start_col; j <= chunk_end_col; ++j)
      {
        insert(chunk_start_row, j);
      }
    }

    element_chunk chunk;
    for (auto const &[row, cols] : chunk_map)
    {
      chunk.insert({row, grid_limits(cols[0], cols.back())});
    }
    chunks.push_back(chunk);
  }
  return chunks;
}

template<typename P, resource resrc>
fk::vector<P, mem_type::owner, resrc> const &
reduce_chunk(PDE<P> const &pde,
             fk::vector<P, mem_type::owner, resrc> const &reduction_space,
             fk::vector<P, mem_type::owner, resrc> &output,
             fk::vector<P, mem_type::owner, resrc> const &unit_vector,
             element_subgrid const &subgrid, element_chunk const &chunk)
{
  int const elem_size = element_segment_size(pde);

  for (auto const &[row, cols] : chunk)
  {
    // assert that chunk is within passed subgrid's range
    assert(row >= subgrid.row_start);
    assert(row <= subgrid.row_stop);
    assert(cols.start >= subgrid.col_start);
    assert(cols.stop <= subgrid.col_stop);

    int const prev_row_elems = [i = row, &chunk] {
      if (i == chunk.begin()->first)
      {
        return 0;
      }
      int prev_elems = 0;
      for (int r = chunk.begin()->first; r < i; ++r)
      {
        prev_elems += chunk.at(r).stop - chunk.at(r).start + 1;
      }
      return prev_elems;
    }();

    fk::matrix<P, mem_type::const_view, resrc> const reduction_matrix(
        reduction_space, elem_size, cols.size() * pde.num_terms,
        prev_row_elems * elem_size * pde.num_terms);

    int const reduction_row = subgrid.to_local_row(row);
    fk::vector<P, mem_type::view, resrc> output_view(
        output, reduction_row * elem_size,
        ((reduction_row + 1) * elem_size) - 1);

    fk::vector<P, mem_type::const_view, resrc> const unit_view(
        unit_vector, 0, cols.size() * pde.num_terms - 1);

    P const alpha     = 1.0;
    P const beta      = 1.0;
    bool const transA = false;
    fm::gemv(reduction_matrix, unit_view, output_view, transA, alpha, beta);
  }
  return output;
}

template class host_workspace<float>;
template class host_workspace<double>;

template int get_num_chunks(element_subgrid const &grid, PDE<float> const &pde,
                            int const rank_size_MB);
template int get_num_chunks(element_subgrid const &grid, PDE<double> const &pde,
                            int const rank_size_MB);

template fk::vector<float, mem_type::owner, resource::host> const &reduce_chunk(
    PDE<float> const &pde,
    fk::vector<float, mem_type::owner, resource::host> const &reduction_space,
    fk::vector<float, mem_type::owner, resource::host> &output,
    fk::vector<float, mem_type::owner, resource::host> const &unit_vector,
    element_subgrid const &subgrid, element_chunk const &chunk);

template fk::vector<double, mem_type::owner, resource::host> const &
reduce_chunk(
    PDE<double> const &pde,
    fk::vector<double, mem_type::owner, resource::host> const &reduction_space,
    fk::vector<double, mem_type::owner, resource::host> &output,
    fk::vector<double, mem_type::owner, resource::host> const &unit_vector,
    element_subgrid const &subgrid, element_chunk const &chunk);

template fk::vector<float, mem_type::owner, resource::device> const &
reduce_chunk(
    PDE<float> const &pde,
    fk::vector<float, mem_type::owner, resource::device> const &reduction_space,
    fk::vector<float, mem_type::owner, resource::device> &output,
    fk::vector<float, mem_type::owner, resource::device> const &unit_vector,
    element_subgrid const &subgrid, element_chunk const &chunk);

template fk::vector<double, mem_type::owner, resource::device> const &
reduce_chunk(
    PDE<double> const &pde,
    fk::vector<double, mem_type::owner, resource::device> const
        &reduction_space,
    fk::vector<double, mem_type::owner, resource::device> &output,
    fk::vector<double, mem_type::owner, resource::device> const &unit_vector,
    element_subgrid const &subgrid, element_chunk const &chunk);
