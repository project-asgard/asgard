#pragma once
#include "pde.hpp"
#include "permutations.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <map>
#include <vector>

// TODO namespace elements {

// yield single-d linear index for level/cell combo
int64_t get_1d_index(int const level, int const cell);

// yield level/cell for a single-d index
std::array<int64_t, 2> get_level_cell(int64_t const single_dim_id);

// return the linear index given element coordinates
template<typename P>
int64_t map_to_index(fk::vector<int> const &coords, options const &opts,
                     PDE<P> const &pde);

// return the element coordinates given linear index
template<typename P>
fk::vector<int>
map_to_coords(int64_t const id, options const &opts, PDE<P> const &pde);

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

class element_table
{
public:
  // TODO delete, old version
  element_table(options const program_opts, int const num_levels,
                int const num_dims);

  // new adaptivity version
  template<typename P>
  element_table(options const opts, PDE<P> const &pde);

  // FIXME the below are
  // forward lookup
  int get_index(fk::vector<int> const coords) const;

  // reverse lookup
  fk::vector<int> const &get_coords(int const index) const;

  // get ref to flattened reverse table
  fk::vector<int, mem_type::owner, resource::device> const &
  get_device_table() const
  {
    return reverse_table_d_;
  }

  // returns the number of elements in table
  int size() const
  {
    assert(forward_table_.size() == reverse_table_.size());
    return forward_table_.size();
  }

  // static construction helper
  // return the cell indices given a level tuple
  static fk::matrix<int> get_cell_index_set(fk::vector<int> const &levels);

private:
  // a map keyed on the element coordinates
  // TODO DELETE, using function above
  std::map<fk::vector<int>, int> forward_table_;
  // given an integer index, give me back the element coordinates
  // TODO DELETE
  std::vector<fk::vector<int>> reverse_table_;

  // TODO rename
  // table of active elements staged for on-device kron list building
  fk::vector<int, mem_type::owner, resource::device> reverse_table_d_;

  // --------------------------------------------------------------------------

  // FIXME change to fk vector if upgraded to 64 bit indexing
  // ordering of active elements
  std::vector<int64_t> active_element_ids_;

  // map from element id to coords
  std::unordered_map<int64_t, fk::vector<int>> ids_to_coords_;

  // table of active elements staged for on-device kron list building
  fk::vector<int, mem_type::owner, resource::device> active_table_d_;
};
