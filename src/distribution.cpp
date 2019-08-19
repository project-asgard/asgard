#include "distribution.hpp"
#include <cmath>

// divide element grid into rectangular sub-areas, which will be assigned to
// each rank require number of ranks to be a perfect square or an even number;
// otherwise, we will ignore (leave unused) the highest rank.
element_subgrid
get_subgrid(int const num_ranks, int const my_rank, element_table const &table)
{
  assert(num_ranks > 0);
  assert(num_ranks % 2 == 0 || num_ranks == 1 ||
         std::sqrt(num_ranks == std::floor(std::sqrt(num_ranks))));
  assert(my_rank > 0);
  assert(my_rank < num_ranks);

  if (num_ranks == 1)
  {
    return element_subgrid(0, table.size(), 0, table.size());
  }

  // determine the side lengths that will give us the "squarest" rectangles
  // possible
  int const horz_divisions = [num_ranks] {
    int trial_factor = static_cast<int>(std::floor(std::sqrt(num_ranks)));
    while (trial_factor > 0)
    {
      int const other_factor = num_ranks / trial_factor;
      if (trial_factor * other_factor == num_ranks)
      {
        return std::max(trial_factor, other_factor);
      }
      trial_factor++;
    }
    // I believe this is mathematically impossible...
    assert(false);
  }();

  int const vert_divisions = num_ranks / horz_divisions;

  // determine which subgrid of the element grid belongs to my rank
  int const grid_row_index = my_rank / horz_divisions;
  int const grid_col_index = my_rank % horz_divisions;

  // split the elements into subgrids
  int const left_over_cols = table.size() % horz_divisions;
  int const grid_cols      = table.size() / horz_divisions;
  int const left_over_rows = table.size() % vert_divisions;
  int const grid_rows      = table.size() / vert_divisions;

  // define the bounds of my subgrid
  int const start_col =
      grid_col_index * grid_cols + std::min(grid_col_index, left_over_cols);
  int const start_row =
      grid_row_index * grid_rows + std::min(grid_row_index, left_over_rows);
  int const stop_col =
      start_col + grid_cols + (left_over_cols > grid_col_index ? 1 : 0);
  int const stop_row =
      start_row + grid_rows + (left_over_rows > grid_row_index ? 1 : 0);

  return element_subgrid(start_row, stop_row, start_col, stop_col);
}

distribution_plan get_plan(int const num_ranks, element_table const &table)
{
  assert(num_ranks > 0);

  int const num_splits = [num_ranks] {
    if (std::sqrt(num_ranks) == std::floor(std::sqrt(num_ranks)) ||
        num_ranks % 2 == 0)
      return num_ranks;
    return num_ranks - 1;
  }();

  distribution_plan plan;
  for (int i = 0; i < num_splits; ++i)
  {
    plan[i] = get_subgrid(num_ranks, i, table);
  }
  return plan;
}
