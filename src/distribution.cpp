#include "distribution.hpp"
#include "lib_dispatch.hpp"
#include <cmath>
#include <mpi.h>
#include <numeric>

#ifdef ASGARD_USE_MPI
struct distribution_handler
{
  distribution_handler() {}

  void set_global_comm(MPI_Comm const &comm)
  {
    auto const status = MPI_Comm_dup(comm, &global_comm);
    assert(status == 0);
  }
  MPI_Comm get_global_comm()
  {
    if (global_comm)
    {
      return global_comm;
    }
    return MPI_COMM_WORLD;
  }

private:
  MPI_Comm global_comm;
};
static distribution_handler distro_handle;
#endif

int get_local_rank()
{
#ifdef ASGARD_USE_MPI
  static int const rank = []() {
    MPI_Comm local_comm;
    auto success = MPI_Comm_split_type(distro_handle.get_global_comm(),
                                       MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                       &local_comm);
    assert(success == 0);
    int local_rank;
    success = MPI_Comm_rank(local_comm, &local_rank);
    assert(success == 0);
    return local_rank;
  }();
  return rank;
#endif
  return 0;
}

auto const num_effective_ranks = [](int const num_ranks) {
  if (std::sqrt(num_ranks) == std::floor(std::sqrt(num_ranks)) ||
      num_ranks % 2 == 0)
    return num_ranks;
  return num_ranks - 1;
};

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

  auto const num_participating = num_effective_ranks(num_ranks);
  bool const participating     = my_rank < num_participating;
  int const comm_color         = participating ? 1 : MPI_UNDEFINED;
  MPI_Comm effective_communicator;
  auto success = MPI_Comm_split(MPI_COMM_WORLD, comm_color, my_rank,
                                &effective_communicator);
  assert(success == 0);

  status = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  assert(status == 0);

  if (effective_communicator != MPI_COMM_NULL)
  {
    distro_handle.set_global_comm(effective_communicator);
    initialize_libraries(get_local_rank());
  }
  return {my_rank, num_participating};

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

// determine the side lengths that will give us the "squarest" rectangles
// possible
auto const get_num_subgrid_cols = [](int const num_ranks) {
  assert(num_ranks > 0);
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

  int const num_splits = num_effective_ranks(num_ranks);

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
  assert(my_rank >= 0);
  assert(my_rank < static_cast<int>(plan.size()));
  int const num_partners = get_num_subgrid_cols(plan.size());
  fk::vector<int> partners(num_partners);
  std::iota(partners.begin(), partners.end(), my_rank - my_rank % num_partners);
  return partners;
}

template<typename P>
void reduce_results(fk::vector<P> const &source, fk::vector<P> &dest,
                    distribution_plan const &plan, int const my_rank)
{
  assert(source.size() == dest.size());
  assert(my_rank >= 0);
  assert(my_rank < static_cast<int>(plan.size()));

#ifndef ASGARD_USE_MPI
  fm::copy(source, dest);
  return;
#endif

  if (plan.size() == 1)
  {
    fm::copy(source, dest);
    return;
  }

  fm::scal(static_cast<P>(0.0), dest);
  int const num_cols = get_num_subgrid_cols(plan.size());

  int const my_row = my_rank / num_cols;
  int const my_col = my_rank % num_cols;

  MPI_Comm row_communicator;
  MPI_Comm const global_communicator = distro_handle.get_global_comm();

  auto success =
      MPI_Comm_split(global_communicator, my_row, my_col, &row_communicator);
  assert(success == 0);

  MPI_Datatype const mpi_type =
      std::is_same<P, double>::value ? MPI::DOUBLE : MPI::FLOAT;
  success = MPI_Allreduce((void *)source.data(), (void *)dest.data(),
                          source.size(), mpi_type, MPI_SUM, row_communicator);
  assert(success == 0);

  success = MPI_Comm_free(&row_communicator);
  assert(success == 0);
}

template<typename P>
void prepare_inputs(fk::vector<P> const &source, fk::vector<P> &dest,
                    distribution_plan const &plan, int const my_rank)
{
  assert(dest.size() <= source.size());
  assert(my_rank >= 0);
  assert(my_rank < static_cast<int>(plan.size()));
#ifndef ASGARD_USE_MPI
  fm::copy(source, dest);
  return;
#endif
  if (plan.size() == 1)
  {
    fm::copy(source, dest);
    return;
  }

  // FIXME harry's code and send/recv calls invoked here
}

template void reduce_results(fk::vector<float> const &source,
                             fk::vector<float> &dest,
                             distribution_plan const &plan, int const my_rank);
template void reduce_results(fk::vector<double> const &source,
                             fk::vector<double> &dest,
                             distribution_plan const &plan, int const my_rank);

template void prepare_inputs(fk::vector<float> const &source,
                             fk::vector<float> &dest,
                             distribution_plan const &plan, int const my_rank);
template void prepare_inputs(fk::vector<double> const &source,
                             fk::vector<double> &dest,
                             distribution_plan const &plan, int const my_rank);
