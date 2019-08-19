#pragma once
#include "table.hpp"
#include <map>

struct element_subgrid
{
  element_subgrid(int const row_start, int const row_stop, int const col_start,
                  int const col_stop)
      : row_start(row_start), row_stop(row_stop), col_start(col_start),
        col_stop(col_stop){};
  element_subgrid(element_subgrid const &e)
      : row_start(e.row_start), row_stop(e.row_stop), col_start(e.col_start),
        col_stop(e.col_stop){};
  element_subgrid(element_subgrid const &&e)
      : row_start(e.row_start), row_stop(e.row_stop), col_start(e.col_start),
        col_stop(e.col_stop){};
  bool operator==(const element_subgrid &rhs) const
  {
    return row_start == rhs.row_start && row_stop == rhs.row_stop &&
           col_start == rhs.col_start && stop == rhs.col_stop;
  }

  int64_t size() const { return static_cast<int64_t>(nrows()) * ncols(); };

  int nrows() const { return row_stop - row_start + 1; }
  int ncols() const { return col_stop - col_start + 1; }

// translation from local/global x and y
int to_global_row(int const local_row) const
{
  return local_row + grid.row_start;
}
int to_global_col(int const local_col) const
{
  return local_col + grid.col_start;
}
int to_local_row(int const global_row) const
{
  return global_row - grid.row_start;
};
int to_local_col(int const global_col) const
{
  return global_col - grid.col_start;
};



  int const row_start;
  int const row_stop;
  int const col_start;
  int const col_stop;
};

// FIXME maybe, or don't hold state - recompute
using distribution_plan = map<int, element_subgrid>;

// given a rank, determine element subgrid
element_subgrid
get_subgrid(int const num_ranks, int const my_rank, element_table const &table);

distribution_plan get_plan(int const num_ranks, element_table const &table);



// FIXME matching bi-directional
//
//  -- who is/are my match(es), how much do I need
//  -- inverse
