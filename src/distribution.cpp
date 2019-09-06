#include "distribution.hpp"

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
  MPI_Comm get_global_comm() { return global_comm; }

private:
  MPI_Comm global_comm = MPI_COMM_WORLD;
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
    success = MPI_Comm_free(&local_comm);
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

  int const mpi_tag = 0;
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
                 message.nar.linear_index, mpi_tag, communicator);
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

// gather errors from other local ranks
// returns {rmse errors, relative errors}
template<typename P>
std::array<fk::vector<P>, 2>
gather_errors(P const root_mean_squared, P const relative)
{
  std::array<P, 2> const error{root_mean_squared, relative};

#ifdef ASGARD_USE_MPI
  MPI_Comm local_comm;
  auto success =
      MPI_Comm_split_type(distro_handle.get_global_comm(), MPI_COMM_TYPE_SHARED,
                          0, MPI_INFO_NULL, &local_comm);
  assert(success == 0);

  MPI_Datatype const mpi_type =
      std::is_same<P, double>::value ? MPI::DOUBLE : MPI::FLOAT;

  int local_rank;
  success = MPI_Comm_rank(local_comm, &local_rank);

  int local_size;
  success = MPI_Comm_size(local_comm, &local_size);
  assert(success == 0);

  fk::vector<P> error_vect(local_size * 2);

  MPI_Gather((void *)&error[0], 2, mpi_type, (void *)error_vect.data(), 2,
             mpi_type, 0, local_comm);

  success = MPI_Comm_free(&local_comm);
  assert(success == 0);

  if (local_rank == 0)
  {
    bool odd = false;
    std::vector<P> rmse;
    std::vector<P> relative;
    // split the even and odd elements into seperate vectors -
    // unpackage from MPI call
    std::partition_copy(error_vect.begin(), error_vect.end(),
                        std::back_inserter(rmse), std::back_inserter(relative),
                        [&odd](P) { return odd = !odd; });
    return {fk::vector<P>(rmse), fk::vector<P>(relative)};
  }

  return {fk::vector<P>{root_mean_squared}, fk::vector<P>{relative}};
#else
  return {fk::vector<P>{root_mean_squared}, fk::vector<P>{relative}};
#endif
}

template<typename P>
std::vector<P>
gather_results(fk::vector<P> const &my_results, distribution_plan const &plan,
               int const my_rank, int const element_segment_size)
{
  assert(my_rank >= 0);
  assert(my_rank < static_cast<int>(plan.size()));

  auto const own_results = [&my_results]() {
    std::vector<P> own_results(my_results.size());
    std::copy(my_results.begin(), my_results.end(), own_results.begin());
    return own_results;
  };
#ifdef ASGARD_USE_MPI

  if (plan.size() == 1)
  {
    return own_results();
  }

  int const num_subgrid_cols = get_num_subgrid_cols(plan.size());

  // get the length and displacement of non-root, first row ranks

  // TODO fix to int64 w/ MPI3
  fk::vector<int> rank_lengths(num_subgrid_cols);
  for (int i = 1; i < static_cast<int>(rank_lengths.size()); ++i)
  {
    rank_lengths(i) = plan.at(i).ncols() * element_segment_size;
  }

  // TODO need MPI 3 to make these longer...
  fk::vector<int> rank_displacements(rank_lengths);
  // I guess I'll do the exclusive scan myself...compiler can't find it in std
  int64_t running_total = 0;
  for (int i = 0; i < rank_lengths.size(); ++i)
  {
    rank_displacements(i) = running_total;
    running_total += rank_lengths(i);
  }

  // split the communicator - only need the first row
  bool const participating            = my_rank < num_subgrid_cols;
  int const comm_color                = participating ? 1 : MPI_UNDEFINED;
  MPI_Comm const &global_communicator = distro_handle.get_global_comm();
  MPI_Comm first_row_communicator;
  auto success = MPI_Comm_split(global_communicator, comm_color, my_rank,
                                &first_row_communicator);
  assert(success == 0);

  // gather values
  if (first_row_communicator != MPI_COMM_NULL)
  {
    int64_t const vect_size =
        my_rank ? 0
                : std::accumulate(rank_lengths.begin(), rank_lengths.end(),
                                  my_results.size());
    std::vector<P> results(vect_size);
    int const send_size = my_rank ? my_results.size() : 0;

    MPI_Datatype const mpi_type =
        std::is_same<P, double>::value ? MPI::DOUBLE : MPI::FLOAT;
    success = MPI_Gatherv((void *)my_results.data(), send_size, mpi_type,
                          (void *)(results.data() + my_results.size()),
                          rank_lengths.data(), rank_displacements.data(),
                          mpi_type, 0, first_row_communicator);
    assert(success == 0);

    if (my_rank == 0)
    {
      std::copy(my_results.begin(), my_results.end(), results.begin());
    }
    return results;
  }

  return own_results();

#else
  return own_results();
#endif
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

template std::array<fk::vector<float>, 2>
gather_errors(float const root_mean_squared, float const relative);

template std::array<fk::vector<double>, 2>
gather_errors(double const root_mean_squared, double const relative);

template std::vector<float> gather_results(fk::vector<float> const &my_results,
                                           distribution_plan const &plan,
                                           int const my_rank,
                                           int const element_segment_size);
template std::vector<double>
gather_results(fk::vector<double> const &my_results,
               distribution_plan const &plan, int const my_rank,
               int const element_segment_size);
