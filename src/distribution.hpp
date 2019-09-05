#pragma once
#include "element_table.hpp"
#include "mpi.h"
#include <map>

struct element_subgrid
{
  element_subgrid(int const row_start, int const row_stop, int const col_start,
                  int const col_stop)
      : row_start(row_start), row_stop(row_stop), col_start(col_start),
        col_stop(col_stop)
  {
    assert(row_start >= 0);
    assert(row_stop >= row_start);
    assert(col_start >= 0);
    assert(col_stop >= col_start);
  };
  element_subgrid(element_subgrid const &e)
      : row_start(e.row_start), row_stop(e.row_stop), col_start(e.col_start),
        col_stop(e.col_stop){};
  element_subgrid(element_subgrid const &&e)
      : row_start(e.row_start), row_stop(e.row_stop), col_start(e.col_start),
        col_stop(e.col_stop){};
  bool operator==(const element_subgrid &rhs) const
  {
    return row_start == rhs.row_start && row_stop == rhs.row_stop &&
           col_start == rhs.col_start && col_stop == rhs.col_stop;
  }

  int64_t size() const { return static_cast<int64_t>(nrows()) * ncols(); };

  int nrows() const { return row_stop - row_start + 1; }
  int ncols() const { return col_stop - col_start + 1; }

  // translation from local/global x and y
  int to_global_row(int const local_row) const
  {
    assert(local_row >= 0);
    assert(local_row < nrows());
    return local_row + row_start;
  }
  int to_global_col(int const local_col) const
  {
    assert(local_col >= 0);
    assert(local_col < ncols());
    return local_col + col_start;
  }
  int to_local_row(int const global_row) const
  {
    assert(global_row >= 0);
    int const local = global_row - row_start;
    assert(local >= 0);
    assert(local < nrows());
    return local;
  };
  int to_local_col(int const global_col) const
  {
    assert(global_col >= 0);
    int const local = global_col - col_start;
    assert(local >= 0);
    assert(local < ncols());
    return local;
  };

  int const row_start;
  int const row_stop;
  int const col_start;
  int const col_stop;
};

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
std::array<int, 2> initialize_distribution();
void finalize_distribution();
int get_local_rank();

struct node_out
{
  template<typename T>
  node_out &operator<<(T val)
  {
#ifdef ASGARD_USE_MPI
    if (!get_local_rank())
      std::cout << val;
#endif
    std::cout << val;
    return *this;
  }
};

// given a rank, determine element subgrid
element_subgrid
get_subgrid(int const num_ranks, int const my_rank, element_table const &table);

using distribution_plan = std::map<int, element_subgrid>;
distribution_plan get_plan(int const num_ranks, element_table const &table);

template<typename P>
void reduce_results(fk::vector<P> const &source, fk::vector<P> &dest,
                    distribution_plan const &plan, int const my_rank);

// FIXME matching bi-directional
//
//  -- who is/are my match(es), how much do I need
//  -- inverse
//

template<typename P>
void prepare_inputs(fk::vector<P> const &source, fk::vector<P> &dest,
                    distribution_plan const &plan, int const my_rank);

extern template void
reduce_results(fk::vector<float> const &source, fk::vector<float> &dest,
               distribution_plan const &plan, int const my_rank);
extern template void
reduce_results(fk::vector<double> const &source, fk::vector<double> &dest,
               distribution_plan const &plan, int const my_rank);

extern template void
prepare_inputs(fk::vector<float> const &source, fk::vector<float> &dest,
               distribution_plan const &plan, int const my_rank);
extern template void
prepare_inputs(fk::vector<double> const &source, fk::vector<double> &dest,
               distribution_plan const &plan, int const my_rank);
