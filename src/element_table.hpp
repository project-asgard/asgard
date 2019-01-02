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
  element_table(Options const opts, int const dims);

  // forward lookup - returns the non-negative index of an element's
  // coordinates, or -1 if not found. FIXME get rid of return code and assert
  // instead
  int get_index(fk::vector<int> const coords) const;

  // reverse lookup - returns coordinates at a certain index, or empty vector if
  // out of range. FIXME get rid of return code and assert
  fk::vector<int> get_coords(int const index) const;

  // returns number of elements in table
  int size() const
  {
    assert(forward_table.size() == reverse_table.size());
    return forward_table.size();
  }

  // Given a cell and level coordinate, return a 1-dimensional index
  int get_1d_index(int const level, int const cell) const;

  //
  // Static construction helpers
  //

  //
  // Level/cell indexing functions
  //

  // Return number of cells for each level in a level tuple
  static fk::vector<int> get_cell_nums(fk::vector<int> levels);

  // Return the cell indices given a level tuple
  static fk::matrix<int> get_index_set(fk::vector<int> const levels);

  //
  // Permutations counters
  //

  // Count the number of n-tuples (n == 'num_dims') whose non-negative elements'
  // sum == 'limit'
  static int count_eq_permutations(int const num_dims, int const limit);

  // Count the number of n-tuples (where n == 'num_dims') whose non-negative
  // elements' sum <= 'limit'
  static int count_leq_permutations(int const num_dims, int const limit);

  // Count the number of n-tuples (where n == 'num_dims') whose non-negative max
  // element <= 'limit' (for full grid only)
  static int count_max_permutations(int const num_dims, int const limit);

  //
  // Permutations builders
  //

  // Produce n-tuples (n == 'num_dims') whose elements' are non-negative and
  // their sum == 'limit'. Each tuple becomes a row of the output matrix
  static fk::matrix<int> get_eq_permutations(int const num_dims,
                                             int const limit,
                                             bool const order_by_n);

  // Produce n-tuples (n == 'num_dims') whose elements are non-negative and
  // their sum <= 'limit'. Each tuple becomes a row of the output matrix
  static fk::matrix<int> get_leq_permutations(int const num_dims,
                                              int const limit,
                                              bool const order_by_n);

  // Produce n-tuples (n == 'num_dims') whose elements are non-negative and the
  // max element <= 'limit' (for full grid only). Each tuple becomes a row of
  // the output matrix
  static fk::matrix<int> get_max_permutations(int const num_dims,
                                              int const limit,
                                              bool const last_index_decreasing);

private:
  // a map keyed on the element coordinates
  std::map<fk::vector<int>, int> forward_table;
  // given an integer index, give me back the element coordinates
  std::vector<fk::vector<int>> reverse_table;
};
