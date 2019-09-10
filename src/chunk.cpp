#include "chunk.hpp"
#include "fast_math.hpp"

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

limits columns_in_chunk(element_chunk const &g)
{
  assert(g.size() > 0);
  int const min_col =
      (*std::min_element(g.begin(), g.end(),
                         [](auto const &a, auto const &b) {
                           return a.second.start < b.second.stop;
                         }))
          .second.start;

  int const max_col = (*std::max_element(g.begin(), g.end(),
                                         [](auto const &a, auto const &b) {
                                           return a.second.stop < b.second.stop;
                                         }))
                          .second.stop;
  return limits(min_col, max_col);
}

limits rows_in_chunk(element_chunk const &g)
{
  assert(g.size() > 0);
  return limits(g.begin()->first, g.rbegin()->first);
}

template<typename P>
rank_workspace<P>::rank_workspace(PDE<P> const &pde,
                                  std::vector<element_chunk> const &chunks)
{
  int const elem_size = element_segment_size(pde);

  int const max_elems =
      (*std::max_element(chunks.begin(), chunks.end(),
                         [](const element_chunk &a, const element_chunk &b) {
                           return a.size() < b.size();
                         }))
          .size();

  int const max_conn = max_connected_in_chunk(*std::max_element(
      chunks.begin(), chunks.end(),
      [](const element_chunk &a, const element_chunk &b) {
        return max_connected_in_chunk(a) < max_connected_in_chunk(b);
      }));

  int const max_total = num_elements_in_chunk(*std::max_element(
      chunks.begin(), chunks.end(),
      [](const element_chunk &a, const element_chunk &b) {
        return num_elements_in_chunk(a) < num_elements_in_chunk(b);
      }));

  batch_input.resize(elem_size * max_conn);
  batch_output.resize(elem_size * max_elems);
  reduction_space.resize(elem_size * max_total * pde.num_terms);

  // intermediate workspaces for kron product.
  int const num_workspaces = std::min(pde.num_dims - 1, 2);
  batch_intermediate.resize(reduction_space.size() * num_workspaces);

  unit_vector_.resize(pde.num_terms * max_conn);
  fk::vector<P, mem_type::owner, resource::host> unit_vect(unit_vector_.size());
  std::fill(unit_vect.begin(), unit_vect.end(), 1.0);
  unit_vector_.transfer_from(unit_vect);
}

template<typename P>
fk::vector<P, mem_type::owner, resource::device> const &
rank_workspace<P>::get_unit_vector() const
{
  return unit_vector_;
}

template<typename P>
host_workspace<P>::host_workspace(PDE<P> const &pde, element_table const &table)
{
  int elem_size             = element_segment_size(pde);
  int64_t const vector_size = elem_size * static_cast<int64_t>(table.size());
  x_orig.resize(vector_size);
  x.resize(vector_size);
  fx.resize(vector_size);
  scaled_source.resize(vector_size);
  result_1.resize(vector_size);
  result_2.resize(vector_size);
  result_3.resize(vector_size);
}

// calculate how much workspace we need on device to compute a single connected
// element
//
// *does not include operator matrices - working for now on assumption they'll
// all be resident*

template<typename P>
static double get_element_size_MB(PDE<P> const &pde)
{
  auto const get_MB = [](auto const num_elems) -> double {
    assert(num_elems > 0);
    double const bytes     = num_elems * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

  int const elem_size = element_segment_size(pde);
  // number of intermediate workspaces for kron product.
  // FIXME this only applies to explicit
  int const num_workspaces = std::min(pde.num_dims - 1, 2);

  // calc size of reduction space for a single work item
  double const elem_reduction_space_MB = get_MB(pde.num_terms * elem_size);
  // calc size of intermediate space for a single work item
  double const elem_intermediate_space_MB =
      num_workspaces == 0 ? 0.0
                          : get_MB(static_cast<double>(num_workspaces) *
                                   pde.num_terms * elem_size);

  // calc in and out vector sizes for each elem
  // since we will scheme to have most elems require overlapping pieces of
  // x and y, we will never need 2 addtl xy space per elem
  double const elem_xy_space_MB = get_MB(elem_size * 1.2);
  return (elem_reduction_space_MB + elem_intermediate_space_MB +
          elem_xy_space_MB);
}

// determine how many chunks will be required to solve the problem
// a chunk is a subset of all elements whose total workspace requirement
// is less than the limit passed in rank_size_MB
template<typename P>
int get_num_chunks(element_table const &table, PDE<P> const &pde,
                   int const num_ranks, int const rank_size_MB)
{
  assert(num_ranks > 0);
  assert(rank_size_MB > 0);
  // determine total problem size
  double const num_elems = static_cast<double>(table.size()) * table.size();
  double const space_per_elem = get_element_size_MB(pde);

  // make sure rank size is something reasonable
  // a single element is the finest we can split the problem
  // if that requires a lot of space relative to rank size,
  // roundoff of elements over chunks will cause us to exceed the limit
  //
  // also not feasible to solve problem with a tiny workspace limit
  assert(space_per_elem < (0.5 * rank_size_MB));
  double const problem_size_MB = space_per_elem * num_elems;

  // determine number of chunks
  double const problem_size_per_rank = problem_size_MB / rank_size_MB;
  int const num_chunks               = [problem_size_per_rank, num_ranks] {
    int const chunks_per_rank =
        static_cast<int>(problem_size_per_rank / num_ranks + 1);
    return chunks_per_rank * num_ranks;
  }();

  return num_chunks;
}

// divide the problem given the previously computed number of chunks
// this function divides via a greedy, row-major split.
// i.e., consecutive elements are taken row-wise until the end of a
// row, and continuing as needed to the next row, beginning with the first
// element of the new row (typewriter style). this is done to minimize the
// portion of the y-vector written to by each task, and ultimately the size of
// communication between ranks.
std::vector<element_chunk>
assign_elements(element_table const &table, int const num_chunks)
{
  assert(num_chunks > 0);

  int64_t const num_elems = static_cast<int64_t>(table.size()) * table.size();

  int64_t const elems_left_over = num_elems % num_chunks;
  int64_t const elems_per_task =
      num_elems / num_chunks + elems_left_over / num_chunks;
  int64_t const still_left_over = elems_left_over % num_chunks;

  std::vector<element_chunk> chunks;
  int64_t assigned = 0;

  for (int i = 0; i < num_chunks; ++i)
  {
    std::map<int, std::vector<int>> chunk_map;

    auto const insert = [&chunk_map](int const key, int col) {
      chunk_map.try_emplace(key, std::vector<int>());
      chunk_map[key].push_back(col);
    };
    int64_t const elems_this_task =
        i < still_left_over ? elems_per_task + 1 : elems_per_task;
    int64_t const task_end = assigned + elems_this_task - 1;

    int64_t const chunk_start_row = assigned / table.size();
    int64_t const chunk_start_col = assigned % table.size();
    int64_t const chunk_end_row   = task_end / table.size();
    int64_t const chunk_end_col   = task_end % table.size();

    assigned += elems_this_task;

    if (chunk_end_row > chunk_start_row)
    {
      for (int i = chunk_start_row + 1; i < chunk_end_row; ++i)
      {
        for (int j = 0; j < table.size(); ++j)
        {
          insert(i, j);
        }
      }
      for (int j = chunk_start_col; j < table.size(); ++j)
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
      chunk.insert({row, limits(cols[0], cols.back())});
    }
    chunks.push_back(chunk);
  }
  return chunks;
}

template<typename P>
void copy_chunk_inputs(PDE<P> const &pde, rank_workspace<P> &rank_space,
                       host_workspace<P> const &host_space,
                       element_chunk const &chunk)
{
  int const elem_size = element_segment_size(pde);
  auto const x_range  = columns_in_chunk(chunk);
  fk::vector<P, mem_type::view> const x_view(
      host_space.x, x_range.start * elem_size,
      (x_range.stop + 1) * elem_size - 1);
  rank_space.batch_input.transfer_from(x_view);
}

template<typename P>
void copy_chunk_outputs(PDE<P> const &pde, rank_workspace<P> &rank_space,
                        host_workspace<P> const &host_space,
                        element_chunk const &chunk)
{
  int const elem_size = element_segment_size(pde);
  auto const y_range  = rows_in_chunk(chunk);

  fk::vector<P, mem_type::view> y_view(host_space.fx, y_range.start * elem_size,
                                       (y_range.stop + 1) * elem_size - 1);

  fk::vector<P, mem_type::view, resource::device> const out_view(
      rank_space.batch_output, 0,
      (y_range.stop - y_range.start + 1) * elem_size - 1);

  fk::vector<P, mem_type::owner> const out_view_h(out_view.clone_onto_host());
  y_view = fm::axpy(out_view_h, y_view);
}

template<typename P>
void reduce_chunk(PDE<P> const &pde, rank_workspace<P> &rank_space,
                  element_chunk const &chunk)
{
  int const elem_size = element_segment_size(pde);

  fm::scal(static_cast<P>(0.0), rank_space.batch_output);
  for (auto const &[row, cols] : chunk)
  {
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

    fk::matrix<P, mem_type::view, resource::device> const reduction_matrix(
        rank_space.reduction_space, elem_size,
        (cols.stop - cols.start + 1) * pde.num_terms,
        prev_row_elems * elem_size * pde.num_terms);

    int const reduction_row = row - chunk.begin()->first;
    fk::vector<P, mem_type::view, resource::device> output_view(
        rank_space.batch_output, reduction_row * elem_size,
        ((reduction_row + 1) * elem_size) - 1);

    fk::vector<P, mem_type::view, resource::device> const unit_view(
        rank_space.get_unit_vector(), 0,
        (cols.stop - cols.start + 1) * pde.num_terms - 1);

    P const alpha     = 1.0;
    P const beta      = 1.0;
    bool const transA = false;
    fm::gemv(reduction_matrix, unit_view, output_view, transA, alpha, beta);
  }
}
template class rank_workspace<float>;
template class rank_workspace<double>;

template class host_workspace<float>;
template class host_workspace<double>;

template int get_num_chunks(element_table const &table, PDE<float> const &pde,
                            int const num_ranks, int const rank_size_MB);
template int get_num_chunks(element_table const &table, PDE<double> const &pde,
                            int const num_ranks, int const rank_size_MB);

template void copy_chunk_inputs(PDE<float> const &pde,
                                rank_workspace<float> &rank_space,
                                host_workspace<float> const &host_space,
                                element_chunk const &chunk);

template void copy_chunk_inputs(PDE<double> const &pde,
                                rank_workspace<double> &rank_space,
                                host_workspace<double> const &host_space,
                                element_chunk const &chunk);

template void copy_chunk_outputs(PDE<float> const &pde,
                                 rank_workspace<float> &rank_space,
                                 host_workspace<float> const &host_space,
                                 element_chunk const &chunk);

template void copy_chunk_outputs(PDE<double> const &pde,
                                 rank_workspace<double> &rank_space,
                                 host_workspace<double> const &host_space,
                                 element_chunk const &chunk);

template void reduce_chunk(PDE<float> const &pde,
                           rank_workspace<float> &rank_space,
                           element_chunk const &chunk);

template void reduce_chunk(PDE<double> const &pde,
                           rank_workspace<double> &rank_space,
                           element_chunk const &chunk);
