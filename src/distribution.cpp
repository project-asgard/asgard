#include "distribution.hpp"
// FIXME
#include "chunk.hpp"
#include "lib_dispatch.hpp"
#include "mpi_instructions.hpp"
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
/*
struct rank_to_range
{
  int const rank;
  limits<int64_t> const range;
  rank_to_range(int const rank, limits<int64_t> const range)
      : rank(rank), range(range)
  {}
};*/
/*
static std::vector<rank_to_range> find_partners(int const my_rank,
                                                distribution_plan const &plan,
                                                fk::vector<int> &ranks_used)
{
  assert(my_rank > 0);
  assert(my_rank < static_cast<int>(plan.size()));

  auto const num_cols = get_num_subgrid_cols(plan.size());
  assert(plan.size() % num_cols == 0);
  auto const num_rows = static_cast<int64_t>(plan.size()) / num_cols;

  auto const is_partner = [](element_subgrid const &me,
                             element_subgrid const &them) {
    return them.row_start >= me.col_start && them.row_start <= me.col_stop;
  };

  auto const partner_range = [is_partner](element_subgrid const &me,
                                          element_subgrid const &them) {
    assert(is_partner(me, them));
    return limits<int64_t>{them.row_start,
                           std::min(me.col_stop, them.row_stop)};
  };

  auto const my_row      = my_rank / num_cols;
  auto const &my_subgrid = plan.at(my_rank);
  std::vector<rank_to_range> partners;
  for (int i = 0; i < num_rows; ++i)
  {
    element_subgrid const &start_of_row = plan.at(i * num_cols);
    // if I need something from this subgrid row, find a partner
    if (is_partner(my_subgrid, start_of_row))
    {
      if (i == my_row)
      {
        partners.push_back(
            rank_to_range(my_rank, partner_range(my_subgrid, my_subgrid)));
        continue;
      }
      // pick least used on this row
      auto const row_start     = ranks_used.begin() + i * num_cols;
      auto const row_end       = row_start + num_cols;
      auto const partner_index = std::distance(
          ranks_used.begin(), std::min_element(row_start, row_end));
      ranks_used(partner_index)++; // TODO
      auto const &partner_subgrid = plan.at(partner_index);
      partners.push_back(rank_to_range(
          partner_index, partner_range(my_subgrid, partner_subgrid)));
    }

    return partners;
  }
}
*/

// using partner_map = std::map<int, std::vector<rank_to_range>>;

// static helper for copying my own output to input
template<typename P>
static void copy_to_input(fk::vector<P> const &source, fk::vector<P> &dest,
                          element_subgrid const &my_grid,
                          mpi_message const &message, int const segment_size)
{
  assert(segment_size > 0);
  if (message.mpi_message_type == mpi_message_enum::send)
  {
    int64_t const output_start =
        static_cast<int64_t>(my_grid.to_local_row(message.nar.start)) *
        segment_size;
    int64_t const output_end =
        static_cast<int64_t>(my_grid.to_local_row(message.nar.stop) + 1) *
            segment_size -
        1;
    int64_t const input_start =
        static_cast<int64_t>(my_grid.to_local_col(message.nar.start)) *
        segment_size;
    int64_t const input_end =
        static_cast<int64_t>(my_grid.to_local_col(message.nar.stop) + 1) *
            segment_size -
        1;

    fk::vector<P, mem_type::view> output_window(source, output_start,
                                                output_end);
    fk::vector<P, mem_type::view> input_window(dest, input_start, input_end);

    fm::copy(output_window, input_window);
  }
  // else ignore the matching receive; I am copying locally
}

// static helper for sending/receiving output/input data using mpi
template<typename P>
static void dispatch_message(fk::vector<P> const &source, fk::vector<P> &dest,
                             element_subgrid const &my_grid,
                             mpi_message const &message, int const segment_size)
{
  assert(segment_size > 0);

  MPI_Datatype const mpi_type =
      std::is_same<P, double>::value ? MPI::DOUBLE : MPI::FLOAT;
  MPI_Comm const communicator = distro_handle.get_global_comm();

  if (message.mpi_message_type == mpi_message_enum::send)
  {
    auto const output_start =
        static_cast<int64_t>(my_grid.to_local_row(message.nar.start)) *
        segment_size;
    auto const output_end =
        static_cast<int64_t>(my_grid.to_local_row(message.nar.stop) + 1) *
            segment_size -
        1;

    fk::vector<P, mem_type::view> const window(source, output_start,
                                               output_end);

    auto const success =
        MPI_Send((void *)window.data(), window.size(), mpi_type,
                 message.nar.linear_index, MPI_ANY_TAG, communicator);
    assert(success == 0);
  }
  else
  {
    auto const input_start =
        static_cast<int64_t>(my_grid.to_local_col(message.nar.start)) *
        segment_size;
    auto const input_end =
        static_cast<int64_t>(my_grid.to_local_col(message.nar.stop) + 1) *
            segment_size -
        1;

    fk::vector<P, mem_type::view> window(dest, input_start, input_end);

    auto const success = MPI_Recv((void *)window.data(), window.size(),
                                  mpi_type, message.nar.linear_index,
                                  MPI_ANY_TAG, communicator, MPI_STATUS_IGNORE);
    assert(success == 0);
  }
}

template<typename P>
void prepare_inputs(fk::vector<P> const &source, fk::vector<P> &dest,
                    int const segment_size, distribution_plan const &plan,
                    int const my_rank)
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

  /*fk::vector<int> ranks_used(plan.size());
  partner_map rank_to_partners;

  // ranks used is loop carried dependency
  for (auto const &[rank, subgrid] : plan)
  {
    rank_to_partners.emplace(rank, find_partners(rank, plan, ranks_used));
  }*/

  // build communication plan

  std::vector<int> row_boundaries;
  std::vector<int> col_boundaries;

  auto const num_cols = get_num_subgrid_cols(plan.size());
  assert(plan.size() % num_cols == 0);
  auto const num_rows = static_cast<int>(plan.size()) / num_cols;

  for (int i = 0; i < num_rows; ++i)
  {
    element_subgrid const &grid = plan.at(i * num_cols);
    row_boundaries.push_back(grid.row_stop);
  }

  for (int i = 0; i < num_cols; ++i)
  {
    element_subgrid const &grid = plan.at(i);
    col_boundaries.push_back(grid.col_stop);
  }

  mpi_instructions const message_list(std::move(row_boundaries),
                                      std::move(col_boundaries));

  // call send/recv
  auto const &my_subgrid = plan.at(my_rank);
  auto const messages    = message_list.get_mpi_instructions(my_rank);

  /* if (my_rank == 1)
   {
     for (auto const &message : messages.mpi_messages_in_order())
     {
       std::cout << (message.mpi_message_type == mpi_message_enum::send ? "send"
                                                                        :
   "recv")
                 << " " << message.nar.linear_index << '\n';
       std::cout << message.nar.start << " : " << message.nar.stop << '\n';
     }
   }*/
  for (auto const &message : messages.mpi_messages_in_order())
  {
    if (message.nar.linear_index == my_rank)
    {
      copy_to_input(source, dest, my_subgrid, message, segment_size);
      continue;
    }

    dispatch_message(source, dest, my_subgrid, message, segment_size);
  }
}

template void reduce_results(fk::vector<float> const &source,
                             fk::vector<float> &dest,
                             distribution_plan const &plan, int const my_rank);
template void reduce_results(fk::vector<double> const &source,
                             fk::vector<double> &dest,
                             distribution_plan const &plan, int const my_rank);

template void prepare_inputs(fk::vector<float> const &source,
                             fk::vector<float> &dest, int const segment_size,
                             distribution_plan const &plan, int const my_rank);
template void prepare_inputs(fk::vector<double> const &source,
                             fk::vector<double> &dest, int const segment_size,
                             distribution_plan const &plan, int const my_rank);
