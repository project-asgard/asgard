#include "element_table.hpp"

#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

template<typename T>
// Construct forward and reverse element tables
element_table<T>::element_table(options const program_opts, int const num_dims)
{
  int const num_levels     = program_opts.get_level();
  bool const use_full_grid = program_opts.using_full_grid();
  int const max_levels     = program_opts.get_max_levels();
  std::cout << " max levels " << max_levels << std::endl;

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

      int64_t element_idx =
          lev_cell_to_element_index(level_tuple, cell_indices, max_levels);
      std::cout << "element_idx " << element_idx << std::endl;
      // the element table key is the full element coordinate - (levels,cells)
      // (level-1, ..., level-d, cell-1, ... cell-d)
      fk::vector<int> key = level_tuple;
      key.concat(cell_indices);

      forward_table[key] = index++;
      // forward_table_sparse[key] = element_idx;
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
template<typename T>
int element_table<T>::get_index(fk::vector<int> const coords) const
{
  assert(coords.size() > 0);
  // purposely not catching std::out_of_range so that program will die
  return forward_table.at(coords);
}

// reverse lookup - returns coordinates at a certain index
template<typename T>
fk::vector<int> element_table<T>::get_coords(T const index) const
{
  assert(index >= 0);
  assert(static_cast<size_t>(index) < reverse_table.size());
  return reverse_table[index];
}

// fk::vector<int> element_table::get_coords_sparse(long int const index) const
//{
//  assert(index >= 0);
//  assert(static_cast<size_t>(index) < reverse_table.size());
//  return reverse_table[index];
//}

// Static construction helper
// Return the cell indices, given a level tuple
// Each row in the returned matrix is the cell portion of an element's
// coordinate
template<typename T>
fk::matrix<int>
element_table<T>::get_cell_index_set(fk::vector<int> const level_tuple)
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
        [](int level) { return fm::two_raised_to(std::max(0, level - 1)); });
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

template<typename T>
int element_table<T>::lev_cell_to_1D_index(int const level, int const cell)
{
  std::cout << " here3 " << level << "level and cell " << cell << std::endl;

  int64_t index = 0;
  std::cout << " here3.5 " << std::endl;

  if (level > 0)
  {
    index = fm::two_raised_to(level - 1) + cell;
  }

  std::cout << " here4 1d index " << index << std::endl;
  return index;
}

template<typename T>
int element_table<T>::lev_cell_to_element_index(fk::vector<int> const levels,
                                            fk::vector<int> const cells,
                                            int const max_levels)
{
  std::cout << " here1 " << std::endl;
  int const num_dimensions = levels.size();
  assert(cells.size() == num_dimensions);

  int64_t eIdx   = 0;
  int64_t stride = 1;
  std::cout << " here2 " << std::endl;

  for (int d = 0; d < num_dimensions; d++)
  {
    assert(levels(d) <= max_levels);
    int64_t idx_1D = lev_cell_to_1D_index(levels(d), cells(d));
    eIdx            = eIdx + (idx_1D)*stride;
    stride          = stride * fm::two_raised_to(max_levels);
  }

  std::cout << " here5 eIdx" << eIdx << std::endl;
  assert(eIdx >= 0);
  assert(eIdx < pow(fm::two_raised_to(max_levels), num_dimensions));
  std::cout << " here6 " << std::endl;

  return eIdx;
}
template class element_table<int>;
template class element_table<int64_t>;

//template fk::vector<int> element_table<int>::get_coords(int const index) const;
//template fk::vector<int> element_table<int64_t>::get_coords(int64_t const index) const;
//template int element_table<int>::get_index(fk::vector<int> const coords);
//template int element_table<int64_t>::get_index(fk::vector<int> const coords);
