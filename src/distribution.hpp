#pragma once
#include "table.hpp"
#include <map>

template<typename P = int>
struct element_subgrid
{
  element_subgrid(P const row_start, P const row_stop, P const col_start,
                  P const col_stop)
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

  P size() const
  {
    return (row_stop - row_start + 1) * (col_stop - col_start + 1);
  };

  P nrows() const { return row_stop - row_start + 1; }

  P ncols() const { return col_stop - col_start + 1; }

  P const row_start;
  P const row_stop;
  P const col_start;
  P const col_stop;
};

// FIXME maybe, or don't hold state - recompute
using distribution_plan = map<int, element_subgrid>;

// given a rank, determine element subgrid
element_subgrid split_problem(int const num_ranks, int const my_rank,
                              element_table const &table);

// FIXME parterning bi-directional
//
//  -- who is my partner, how much do I need
//  -- whose partner am I, how much do they need
//
