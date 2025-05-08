#pragma once
#include "asgard_compute.hpp"

// -----------------------------------------------------------------------------
// permutations
// this components's purpose is to provide utilities used in
// construction of the element table
// -----------------------------------------------------------------------------

/*!
 * \defgroup permuts Permutations and index selection
 *
 * Collection of algorithms for generating multi-indexes
 * with different order and selection criteria.
 */

namespace asgard::permutations
{
//! \brief Builds a sorted multi-index set with
inline std::vector<int> generate_lower_index_set(
    size_t num_dims, std::function<bool(std::array<int, max_num_dimensions> const &index)> inside)
{
  size_t c   = 0;
  bool is_in = true;
  std::array<int, max_num_dimensions> root;
  std::fill_n(root.begin(), num_dims, 0);
  std::vector<int> indexes;
  while (is_in || (c > 0))
  {
    if (is_in)
    {
      indexes.insert(indexes.end(), root.begin(), root.begin() + num_dims);
      c = num_dims - 1;
      root[c]++;
    }
    else
    {
      //std::fill(root.begin(), root.begin() + c + 1, 0);
      std::fill(root.begin() + c, root.begin() + num_dims, 0);
      root[--c]++;
      // c++;
      // if (c == num_dims)
      //   break;
      // root[c]++;
    }
    is_in = inside(root);
  }
  return indexes;
}

} // namespace asgard::permutations
