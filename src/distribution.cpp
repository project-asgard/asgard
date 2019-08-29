#include "distribution.hpp"
#include "lib_dispatch.hpp"
#include <cmath>
#include <mpi.h>
#include <numeric>

std::array<int, 2> initialize_distribution()
{
#ifdef ASGARD_USE_MPI
  auto status = MPI_Init(NULL, NULL);
  assert(status == 0);
  int num_ranks;
  status = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  assert(status == 0);
  int my_rank;
  status = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  assert(status == 0);
  initialize_libraries(get_local_rank());
  return {my_rank, num_ranks};
#endif
  return {0, 1};
}

void finalize_distribution()
{
#ifdef ASGARD_USE_MPI
  auto const status = MPI_Finalize();
  assert(status == 0);
#endif
}

int get_local_rank()
{
  static int const rank = []() {
#ifdef ASGARD_USE_MPI
    MPI_Comm local_comm;
    auto success = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                       MPI_INFO_NULL, &local_comm);
    assert(success == 0);
    int local_rank;
    success = MPI_Comm_rank(local_comm, &local_rank);
    assert(success == 0);
    return local_rank;
#endif
    return 0;
  }();
  return rank;
}

// determine the side lengths that will give us the "squarest" rectangles
// possible
auto const get_num_subgrid_cols = [](int const num_ranks) {
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
};

// divide element grid into rectangular sub-areas, which will be assigned to
// each rank require number of ranks to be a perfect square or an even number;
// otherwise, we will ignore (leave unused) the highest rank.
element_subgrid
get_subgrid(int const num_ranks, int const my_rank, element_table const &table)
{
  assert(num_ranks > 0);

  assert(num_ranks % 2 == 0 || num_ranks == 1 ||
         std::sqrt(num_ranks) == std::floor(std::sqrt(num_ranks)));
  assert(my_rank >= 0);
  assert(my_rank < num_ranks);

  assert(static_cast<int64_t>(table.size()) * table.size() > num_ranks);

  if (num_ranks == 1)
  {
    return element_subgrid(0, table.size() - 1, 0, table.size() - 1);
  }

  int const num_subgrid_cols = get_num_subgrid_cols(num_ranks);
  int const num_subgrid_rows = num_ranks / num_subgrid_cols;

  // determine which subgrid of the element grid belongs to my rank
  int const grid_row_index = my_rank / num_subgrid_cols;
  int const grid_col_index = my_rank % num_subgrid_cols;

  // split the elements into subgrids
  int const left_over_cols = table.size() % num_subgrid_cols;
  int const subgrid_width  = table.size() / num_subgrid_cols;
  int const left_over_rows = table.size() % num_subgrid_rows;
  int const subgrid_height = table.size() / num_subgrid_rows;

  // define the bounds of my subgrid
  int const start_col =
      grid_col_index * subgrid_width + std::min(grid_col_index, left_over_cols);
  int const start_row = grid_row_index * subgrid_height +
                        std::min(grid_row_index, left_over_rows);
  int const stop_col =
      start_col + subgrid_width + (left_over_cols > grid_col_index ? 1 : 0) - 1;
  int const stop_row = start_row + subgrid_height +
                       (left_over_rows > grid_row_index ? 1 : 0) - 1;

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
    plan.emplace(i, get_subgrid(num_splits, i, table));
  }

  return plan;
}

// get the rank number of everyone on my row
// assumes row major ordering - the get_plan function
// is tested for this layout
fk::vector<int>
get_reduction_partners(distribution_plan const &plan, int const my_rank)
{
  int const num_partners = get_num_subgrid_cols(plan.size());
  fk::vector<int> partners(num_partners);
  std::iota(partners.begin(), partners.end(), my_rank - my_rank % num_partners);
  return partners;
}
