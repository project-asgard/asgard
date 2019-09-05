#pragma once
#include "permutations.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
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
//   valid.get_level() components.
// - to build a sparse grid, we apply some rule to omit some of these
//   permutations. currently, we cull level combinations whose sum is greater
//   than the number of levels selected for the simulation.
// -----------------------------------------------------------------------------

template<typename T = int64_t>
class element_table
{
public:
  element_table(options const program_opts, int const num_dims);

  // forward lookup
  int get_index(fk::vector<int> const coords) const;

  // reverse lookup
  fk::vector<int> get_coords(T const index) const;
  // fk::vector<int> get_coords_sparse(long int const index) const;

  // returns the number of elements in table
  int size() const
  {
    assert(forward_table.size() == reverse_table.size());
    return forward_table.size();
  }

  // returns the number of elements in table
  // int size_sparse() const
  //{
  //  assert(forward_table_sparse.size() > 0);
  //  return forward_table_sparse.size();
  //}

  // Static construction helper
  // Return the cell indices given a level tuple
  static fk::matrix<int> get_cell_index_set(fk::vector<int> const levels);

  int lev_cell_to_1D_index(int const level, int const cell);

  int lev_cell_to_element_index(fk::vector<int> const levels,
                                     fk::vector<int> const cells,
                                     int const max_levels);

private:
  // a map keyed on the element coordinates
  std::map<fk::vector<int>, T> forward_table;
  // a map keyed on the element index
  // std::map<fk::vector<int>,long int> forward_table_sparse;
  // given an integer index, give me back the element coordinates
  std::vector<fk::vector<int>> reverse_table;
};
//template class element_table<int>;
//template class element_table<int64_t>;
//extern template fk::vector<int> element_table<int>::get_coords(int const index) const;
//extern template fk::vector<int> element_table<int64_t>::get_coords(int64_t const index) const;
//template int element_table<int>::get_index(fk::vector<int> const coords);
//template int element_table<int64_t>::get_index(fk::vector<int> const coords);
