#pragma once
#include "program_options.hpp"
#include "tensors.hpp"
#include <cassert>
#include <map>
#include <vector>

// -----------------------------------------------------------------------------
// element table
// this object's purpose is:
// - to provide a mapping from elements' coordinates to an assigned positive
//   integer index
// - to provide the reverse mapping from index to coordinates.
//
// about coordinates
// - coordinates are composed of a set of dimension-many pairs (level, cell).
//   (technically, an element's coordinates will also have a degree component,
//   but we omit this as we assume uniform degree).
// - in our table, coordinates are stored with all level components grouped,
//   followed by cell components so that a single coordinate will look like
//   e.g.: (lev_1, lev_2, ..., lev_dim, cell_1, cell_2, ..., cell_dim).
//
// re: full and sparse grid
// - to build a full grid, all potentional level combinations are included in
//   the table; that is, all dimension-length permutations of integers less
//   than or equal to the number of levels selected for the simulation are
//   valid level components.
// - to build a sparse grid, we apply some rule to omit some of these
//   permutations. currently, we cull level combinations whose sum is greater
//   than the number of levels selected for the simulation.
// -----------------------------------------------------------------------------

class element_table
{
public:
  element_table(Options const program_opts, int const num_dims);

  // forward lookup
  int get_index(fk::vector<int> const coords) const;

  // reverse lookup
  fk::vector<int> get_coords(int const index) const;

  // returns the number of elements in table
  int size() const
  {
    assert(forward_table.size() == reverse_table.size());
    return forward_table.size();
  }

  // Given a cell and level coordinate, return a 1-dimensional index
  int get_1d_index(int const level, int const cell) const;

  //
  // Static construction helpers
  // (these will likely become private at some point)
  //

  // Return the cell indices given a level tuple
  static fk::matrix<int> get_cell_index_set(fk::vector<int> const levels);

  // Permutations counters

  static int count_eq_permutations(int const num_dims, int const limit);

  static int count_leq_permutations(int const num_dims, int const limit);

  static int count_max_permutations(int const num_dims, int const limit);

  // Permutations builders

  static fk::matrix<int> get_eq_permutations(int const num_dims,
                                             int const limit,
                                             bool const order_by_n);

  static fk::matrix<int> get_leq_permutations(int const num_dims,
                                              int const limit,
                                              bool const order_by_n);

  static fk::matrix<int> get_max_permutations(int const num_dims,
                                              int const limit,
                                              bool const last_index_decreasing);

private:
  // a map keyed on the element coordinates
  std::map<fk::vector<int>, int> forward_table;
  // given an integer index, give me back the element coordinates
  std::vector<fk::vector<int>> reverse_table;
};
