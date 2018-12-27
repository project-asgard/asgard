#pragma once
#include "tensors.hpp"
#include <cassert>
#include <map>
#include <vector>

// --------------------------------------------------------------------------
// element table
// this object's purpose is:
// - to provide a mapping from elements' coordinates to an assigned positive
// integer index
// - to provide the reverse mapping from index to coordinates.
//
// about coordinates
// - coordinates are composed of dimension-many (lev, cell) tuples.
// (technically, an element's coordinates will also have a degree level,
// but we omit this as we assume uniform degree).
// - in our table, coordinates are stored with all level components grouped,
// followed by cell components.
// ex: (lev_1, lev_2, ..., lev_dim, cell_1, cell_2, ..., cell_dim).
//
// re: full and sparse grid
// - to build a full grid, all potentional level combinations are included in
// the table; that is, all dimension-length permutations of integers less than
// or equal to the number of levels selected for the simulation are valid level
// components.
// - to build a sparse grid, we apply some rule to omit some of these
// permutations. currently, we cull level combinations whose sum is greater than
// the number of levels selected for the simulation.
// --------------------------------------------------------------------------

class element_table
{
public:
  element_table(int const dim, int const level, bool const full_grid = false);

  // forward lookup - returns the index of coordinates (positive), or -1 if not
  // found
  int get_index(fk::vector<int> const coords) const;

  // reverse lookup - returns coordinates at a certain index, or empty vector if
  // out of range
  fk::vector<int> get_coords(int const index) const;

  // returns number of elements in table
  int size() const
  {
    assert(forward_table.size() == reverse_table.size());
    return forward_table.size();
  }

  //
  // Static helpers for element table construction
  //

  //
  // Level/cell indexing functions
  //

  // Given a cell and level coordinate, return a 1-dimensional index
  static int get_1d_index(int const level, int const cell);

  // Return the cell indices given a level tuple
  static fk::matrix<int> get_index_set(fk::vector<int> levels);

  //
  // Permutations enumerators
  //

  // Calculate the number of dims-tuples whose sum == n
  static int permutations_eq_count(int const dims, int const n);

  // Calculate the number of dims-tuples whose sum <= n
  static int permutations_leq_count(int const dims, int const n);

  // Calculate the number of dims-tuples whose max element <= n (for full grid)
  static int permutations_max_count(int const dims, int const n);

  //
  // Permutations builders
  //

  // Produce dims-tuples whose sum == n
  static fk::matrix<int>
  permutations_eq(int const dims, int const n, bool const order_by_n);

  // Produce dims-tuples whose sum <= to n
  static fk::matrix<int>
  permutations_leq(int const dims, int const n, bool const order_by_n);

  // Produce dims-tuples whose max element <= n (for full grid)
  static fk::matrix<int> permutations_max(int const dims, int const n,
                                          bool const last_index_decreasing);

private:
  std::map<fk::vector<int>, int> forward_table;
  std::vector<fk::vector<int>> reverse_table;
};
