#include "element_table.hpp"

#include "matlab_utilities.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

// Construct forward and reverse element tables
element_table::element_table(options const program_opts, int const num_dims)
{
  int const num_levels     = program_opts.get_level();
  bool const use_full_grid = program_opts.using_full_grid();

  assert(num_dims > 0);
  assert(num_levels > 0);

  // get permutation table for some num_dims, num_levels
  // each row of this table becomes a level tuple, and is the "level" component
  // of some number of elements' coordinates
  fk::matrix<int> const perm_table =
      use_full_grid ? get_max_permutations(num_dims, num_levels, false)
                    : get_leq_permutations(num_dims, num_levels, false);

  // in the matlab, the tables sizes are precomputed/preallocated.
  // I wrote the code to do that, however, this isn't a performance
  // critical area (constructed only once at startup) and we'd need
  // to explore the thread-safety of our tables / build a thread-safe
  // table to see any benefit. -TM

  // build the element tables (forward and reverse)
  int index = 0;
  for (int row = 0; row < perm_table.nrows(); ++row)
  {
    // get the level tuple to work on
    fk::vector<int> const level_tuple =
        perm_table.extract_submatrix(row, 0, 1, num_dims);
    // calculate all possible cell indices allowed by this level tuple
    fk::matrix<int> const index_set = get_cell_index_set(level_tuple);

    for (int cell_set = 0; cell_set < index_set.nrows(); ++cell_set)
    {
      fk::vector<int> cell_indices =
          index_set.extract_submatrix(cell_set, 0, 1, num_dims);

      // the element table key is the full element coordinate - (levels,cells)
      // (level-1, ..., level-d, cell-1, ... cell-d)
      fk::vector<int> key = level_tuple;
      key.concat(cell_indices);

      forward_table[key] = index++;
      // note the matlab code has an option to append 1d cell indices to the
      // reverse element table. //FIXME do we need to precompute or can we call
      // the 1d helper as needed?
      reverse_table.push_back(key);
    }
  }
  assert(forward_table.size() == reverse_table.size());
}

// forward lookup - returns the non-negative index of an element's
// coordinates
int element_table::get_index(fk::vector<int> const coords) const
{
  assert(coords.size() > 0);
  // purposely not catching std::out_of_range so that program will die
  return forward_table.at(coords);
}

// reverse lookup - returns coordinates at a certain index
fk::vector<int> element_table::get_coords(int const index) const
{
  assert(index >= 0);
  assert(static_cast<size_t>(index) < reverse_table.size());
  return reverse_table[index];
}

// Static construction helper
// Return the cell indices, given a level tuple
// Each row in the returned matrix is the cell portion of an element's
// coordinate
fk::matrix<int>
element_table::get_cell_index_set(fk::vector<int> const level_tuple)
{
  assert(level_tuple.size() > 0);
  for (auto const level : level_tuple)
  {
    assert(level >= 0);
  }
  int const num_dims = level_tuple.size();

  // get number of cells for each level coordinate in the input tuple
  // 2^(max(0, level-1))
  fk::vector<int> const cells_per_level = [level_tuple]() {
    fk::vector<int> v(level_tuple.size());
    std::transform(
        level_tuple.begin(), level_tuple.end(), v.begin(),
        [](int level) { return two_raised_to(std::max(0, level - 1)); });
    return v;
  }();

  // total number of cells for an entire level tuple will be all possible
  // combinations, so just a product of the number of cells per level in that
  // tuple
  int const total_cells = [cells_per_level]() {
    int total = 1;
    return std::accumulate(cells_per_level.begin(), cells_per_level.end(),
                           total, std::multiplies<int>());
  }();

  // allocate the returned
  fk::matrix<int> cell_index_set(total_cells, num_dims);

  // recursion base case
  if (num_dims == 1)
  {
    std::vector<int> entries(total_cells);
    std::iota(begin(entries), end(entries), 0);
    cell_index_set.update_col(0, entries);
    return cell_index_set;
  }

  // recursively build the cell index set
  int const cells_this_dim = cells_per_level(num_dims - 1);
  int const rows_per_iter  = total_cells / cells_this_dim;
  for (auto i = 0; i < cells_this_dim; ++i)
  {
    int const row_pos = i * rows_per_iter;
    fk::matrix<int> partial_result(rows_per_iter, num_dims);
    fk::vector<int> partial_level_tuple = level_tuple;
    partial_level_tuple.resize(num_dims - 1);
    partial_result.set_submatrix(0, 0, get_cell_index_set(partial_level_tuple));
    std::vector<int> last_col(rows_per_iter, i);
    partial_result.update_col(num_dims - 1, last_col);
    cell_index_set.set_submatrix(row_pos, 0, partial_result);
  }

  return cell_index_set;
}
