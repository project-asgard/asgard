#include "plan.hpp"

std::vector<element_chunk>
get_my_chunks(std::vector<element_chunk> const &all_chunks, int const my_rank,
              int const num_ranks)
{
  assert(all_chunks.size() % num_ranks == 0);
  assert(my_rank < num_ranks);

  int const chunks_per_rank = all_chunks.size() / num_ranks;
  auto const my_first       = all_chunks.cbegin() + chunks_per_rank * my_rank;
  auto const my_last = all_chunks.cbegin() + chunks_per_rank * (my_rank + 1);
  return std::vector<element_chunk>(my_first, my_last);
}

static bool is_contiguous(std::vector<element_chunk> const &chunks)
{
  enum element_status
  {
    unassigned,
    assigned
  };

  int total_rows = 0;
  int total_cols = 0;
  for (auto const &chunk : chunks)
  {
    limits<> rows = rows_in_chunk(chunk);
    limits<> cols = columns_in_chunk(chunk);
    total_rows += rows.stop - rows.start + 1;
    total_cols += cols.stop - cols.start + 1;
  }

  fk::matrix<element_status> coverage(total_rows, total_cols);
  for (auto const &chunk : chunks)
  {
    for (auto const &[row, cols] : chunk)
    {
      for (int col = cols.start; col <= cols.stop; ++col)
      {
        if (coverage(row, col) != element_status::unassigned)
        {
          return false;
        }
        coverage(row, col) = element_status::assigned;
      }
    }
  }
  for (auto const &elem : coverage)
  {
    if (elem != element_status::assigned)
    {
      return false;
    }
  }
  return true;
}

rank_to_inputs get_my_inputs(std::vector<element_chunk> const &my_chunks,
                             int const my_rank, int const num_ranks)
{
  assert(my_rank < num_ranks);
  assert(is_contiguous(my_chunks));
}

template<typename P>
limits<int64_t>
get_input_range(std::vector<element_chunk> const &my_chunks, PDE<P> const &pde)
{
  assert(is_contiguous(my_chunks));

  int64_t const first_col =
      columns_in_chunk(
          *std::min_element(my_chunks.begin(), my_chunks.end(),
                            [](const element_chunk &a, const element_chunk &b) {
                              return columns_in_chunk(a).start <
                                     columns_in_chunk(b).start;
                            }))
          .start;

  int64_t const last_col =
      columns_in_chunk(
          *std::max_element(my_chunks.begin(), my_chunks.end(),
                            [](const element_chunk &a, const element_chunk &b) {
                              return columns_in_chunk(a).stop <
                                     columns_in_chunk(b).stop;
                            }))
          .stop;

  return limits<int64_t>(first_col * element_segment_size(pde),
                         last_col * element_segment_size(pde));
}

template<typename P>
limits<int64_t>
get_output_range(std::vector<element_chunk> const &my_chunks, PDE<P> const &pde)
{
  assert(is_contiguous(my_chunks));

  int64_t const first_row =
      rows_in_chunk(
          *std::min_element(my_chunks.begin(), my_chunks.end(),
                            [](const element_chunk &a, const element_chunk &b) {
                              return rows_in_chunk(a).start <
                                     rows_in_chunk(b).start;
                            }))
          .start;

  int64_t const last_row =
      columns_in_chunk(
          *std::max_element(my_chunks.begin(), my_chunks.end(),
                            [](const element_chunk &a, const element_chunk &b) {
                              return rows_in_chunk(a).stop <
                                     rows_in_chunk(b).stop;
                            }))
          .stop;

  return limits<int64_t>(first_row * element_segment_size(pde),
                         last_row * element_segment_size(pde));
}

template limits<int64_t>
get_input_range(std::vector<element_chunk> const &my_chunks,
                PDE<float> const &pde);
template limits<int64_t>
get_input_range(std::vector<element_chunk> const &my_chunks,
                PDE<double> const &pde);

template limits<int64_t>
get_output_range(std::vector<element_chunk> const &my_chunks,
                 PDE<float> const &pde);
template limits<int64_t>
get_output_range(std::vector<element_chunk> const &my_chunks,
                 PDE<double> const &pde);
