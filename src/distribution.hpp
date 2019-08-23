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
    int local = global_row - row_start;
    assert(local >= 0);
    assert(local < nrows());
    return local;
  };
  int to_local_col(int const global_col) const
  {
    assert(global_col >= 0);
    int local = global_col - col_start;
    assert(local >= 0);
    assert(local < ncols());
    return local;
  };

  int const row_start;
  int const row_stop;
  int const col_start;
  int const col_stop;
};

// helper function
inline int get_local_rank()
{
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  return local_rank;
}

// given a rank, determine element subgrid
element_subgrid
get_subgrid(int const num_ranks, int const my_rank, element_table const &table);

using distribution_plan = std::map<int, element_subgrid>;
distribution_plan get_plan(int const num_ranks, element_table const &table);

fk::vector<int>
get_reduction_partners(distribution_plan const &plan, int const my_rank);

// FIXME matching bi-directional
//
//  -- who is/are my match(es), how much do I need
//  -- inverse
