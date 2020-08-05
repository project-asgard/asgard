#pragma once
#include "connectivity.hpp"
#include "permutations.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <algorithm>
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

enum class element_type
{
  leaf,
  interior
};

class element_table
{
public:
  element_table(options const program_opts, int const num_levels,
                int const num_dims);

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

  // static helper
  // return the linear index given element coordinates
  // TODO is this needed outside class? private?
  static int64_t map_to_index(fk::vector<int> const &coords, int const num_dims,
                              int const max_levels)
  {
    assert(num_dims > 0);
    assert(coords.size() == num_dims);
    int64_t id     = 1;
    int64_t stride = 1;
    for (int i = 0; i < num_dims; ++i)
    {
      assert(coords(i) < max_levels);
      id += get_1d_index(coords(i), coords(i + num_dims)) * stride;
      stride += stride * fm::two_raised_to(max_levels);
    }
    assert(id >= 0);
    assert(id < fm::two_raised_to(static_cast<int64_t>(max_levels) * num_dims));
    return id;
  }

  // adaptivity functions
  void move_to_leaf(int64_t const element_id)
  {
    assert(interior_nodes.count(element_id) == 1);
    assert(leaf_nodes.count(element_id) == 0);
    interior_nodes.erase(element_id);
    leaf_nodes.insert(element_id);
  }
  void move_to_interior(int const element_id)
  {
    assert(interior_nodes.count(element_id) == 0);
    assert(leaf_nodes.count(element_id) == 1);
    interior_nodes.insert(element_id);
    leaf_nodes.erase(element_id);
  }

private:
  // a map keyed on the element coordinates
  // TODO DELETE, using function above
  std::map<fk::vector<int>, int> forward_table_;
  // given an integer index, give me back the element coordinates
  // TODO rename, only table
  std::vector<fk::vector<int>> reverse_table_;
  std::map <

      // a flattened reverse table staged for on-device kron list building
      // TODO must be updated at end of adapt step
      fk::vector<int, mem_type::owner, resource::device> reverse_table_d_;

  // adaptivity sets
  std::set<int64_t> leaf_nodes;
  std::set<int64_t> interior_nodes;
};
