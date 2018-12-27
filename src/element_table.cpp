#include "element_table.hpp"

#include "tensors.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

// Construct forward and reverse element tables
element_table::element_table(int const dims, int const levels,
                             bool const full_grid)
{
  assert(dims > 0);
  assert(levels > 0);

  // get permutation table of level coordinates
  fk::matrix<int> perm_table = full_grid
                                   ? permutations_max(dims, levels, false)
                                   : permutations_leq(dims, levels, false);

  for (int row = 0; row < perm_table.nrows(); ++row)
  {
    // FIXME I need to be able to extract rows/cols from matrices

    // int total_size = 1;
    // total_size     = std::accumulate(sizes.begin(), sizes.end(), total_size,
    //                           std::multiplies<int>());
  }
}

// Forward lookup - returns the index of coordinates (positive), or -1 if not
// found
int element_table::get_index(fk::vector<int> const coords) const
{
  assert(coords.size() > 0);
  try
  {
    return forward_table.at(coords);
  }
  catch (std::out_of_range)
  {
    return -1;
  }
}

// Reverse lookup - returns coordinates at a certain index, or empty vector if
// out of range
fk::vector<int> element_table::get_coords(int const index) const
{
  assert(index > 0);
  if (static_cast<size_t>(index) < reverse_table.size())
  {
    return reverse_table[index];
  }
  else
  {
    return fk::vector<int>();
  }
}

//
// Static helpers for construction
//

//
// Indexing helpers
//

// Return number of cells for each level
fk::vector<int> element_table::get_cell_nums(fk::vector<int> levels)
{
  assert(levels.size() > 0);
  fk::vector<int> sizes(levels.size());
  std::transform(levels.begin(), levels.end(), sizes.begin(), [](int level) {
    return static_cast<int>(std::pow(2, std::max(0, level - 1)));
  });
  return sizes;
}

// Given a cell and level coordinate, return a 1-dimensional index
int element_table::get_1d_index(int const level, int const cell)
{
  assert(level >= 0);
  assert(cell >= 0);

  if (level == 0)
  {
    return 1;
  }
  return static_cast<int>(std::pow(2, level - 1)) + cell + 1;
}

fk::matrix<int> element_table::get_index_set(fk::vector<int> const levels)
{
  assert(levels.size() > 0);
  for (auto level : levels)
  {
    assert(level > 0);
  }
  int const dims = levels.size();

  // get number of cells for each level coord
  fk::vector<int> sizes = get_cell_nums(levels);

  // total number of cells will be product of these
  int total_size = 1;
  total_size     = std::accumulate(sizes.begin(), sizes.end(), total_size,
                               std::multiplies<int>());

  fk::matrix<int> index_set(total_size, dims);

  // base case
  if (dims == 1)
  {
    std::vector<int> entries(total_size);
    std::iota(begin(entries), end(entries), 0);
    index_set.update_col(0, entries);
    return index_set;
  }

  // recursively build index set
  int const cells_this_dim = sizes(dims - 1);
  int const rows_per_iter  = total_size / cells_this_dim;
  for (auto i = 0; i < cells_this_dim; ++i)
  {
    int const row_pos = i * rows_per_iter;
    fk::matrix<int> partial_result(rows_per_iter, dims);
    fk::vector<int> partial_levels = levels;
    partial_levels.resize(dims - 1);
    partial_result.set_submatrix(0, 0, get_index_set(partial_levels));
    std::vector<int> last_col(rows_per_iter, i);
    partial_result.update_col(dims - 1, last_col);
    index_set.set_submatrix(row_pos, 0, partial_result);
  }

  return index_set;
}

//
// Permutation enumerators
//

// Given dims and n, produce the number of dims-tuples whose sum == n
int element_table::permutations_eq_count(int const dims, int const n)
{
  assert(dims > 0);
  assert(n >= 0);
  if (dims == 1)
  {
    return 1;
  }

  if (dims == 2)
  {
    return n + 1;
  }

  int count = 0;
  for (auto i = 0; i <= n; ++i)
  {
    count += permutations_eq_count(dims - 1, i);
  }

  return count;
}

// Given dims and n, produce the number of dims-tuples whose sum <= n
int element_table::permutations_leq_count(int dims, int n)
{
  assert(dims > 0);
  assert(n >= 0);

  int count = 0;
  for (auto i = 0; i <= n; ++i)
  {
    count += permutations_eq_count(dims, i);
  }
  return count;
}

// Given dims and n, produce the number of dims-tuples whose max element <= n
int element_table::permutations_max_count(int const dims, int const n)
{
  assert(dims > 0);
  assert(n >= 0);

  return static_cast<int>(std::pow(n + 1, dims));
}

//
// Permutations builders
//

// Given dims and n, produce dims-tuples whose sum == n
fk::matrix<int> element_table::permutations_eq(int const dims, int const n,
                                               bool const order_by_n)
{
  assert(dims > 0);
  assert(n >= 0);

  int const num_tuples = permutations_eq_count(dims, n);
  fk::matrix<int> result(num_tuples, dims);

  if (dims == 1)
  {
    return fk::matrix<int>{{n}};
  }

  int counter = 0;
  for (auto i = 0; i <= n; ++i)
  {
    int partial_sum;
    int difference;

    if (order_by_n)
    {
      partial_sum = i;
      difference  = n - i;
    }
    else
    {
      partial_sum = n - i;
      difference  = i;
    }

    // build set of dims-1-tuples which sum to partial_sum,
    // then append a column with difference to make a dims-tuple.
    int const rows = permutations_eq_count(dims - 1, partial_sum);
    fk::matrix<int> partial_result(rows, dims);
    partial_result.set_submatrix(
        0, 0, permutations_eq(dims - 1, partial_sum, order_by_n));
    fk::vector<int> last_col = std::vector<int>(rows, difference);
    partial_result.update_col(dims - 1, last_col);
    result.set_submatrix(counter, 0, partial_result);

    counter += rows;
  }
  return result;
}

// Given dims and n, produce dims-tuples whose sum <= to n
fk::matrix<int> element_table::permutations_leq(int const dims, int const n,
                                                bool const order_by_n)
{
  assert(dims > 0);
  assert(n >= 0);

  int const num_tuples = permutations_leq_count(dims, n);
  fk::matrix<int> result(num_tuples, dims);

  if (order_by_n)
  {
    int counter = 0;
    for (auto i = 0; i <= n; ++i)
    {
      int const rows = permutations_eq_count(dims, i);
      result.set_submatrix(counter, 0, permutations_eq(dims, i, false));
      counter += rows;
    }
    return result;
  }

  if (dims == 1)
  {
    std::vector<int> entries(n + 1);
    std::iota(begin(entries), end(entries), 0);
    result.update_col(0, entries);
    return result;
  }

  int counter = 0;
  for (auto i = 0; i <= n; ++i)
  {
    int const rows = permutations_leq_count(dims - 1, n - i);
    fk::matrix<int> partial_result(rows, dims);
    partial_result.set_submatrix(0, 0,
                                 permutations_leq(dims - 1, n - i, order_by_n));
    fk::vector<int> last_col = std::vector<int>(rows, i);
    partial_result.update_col(dims - 1, last_col);
    result.set_submatrix(counter, 0, partial_result);

    counter += rows;
  }

  return result;
}

// Produce dims-tuples whose max element <= n (for full grid)
fk::matrix<int>
element_table::permutations_max(int const dims, int const n,
                                bool const last_index_decreasing)
{
  assert(dims > 0);
  assert(n >= 0);

  int const num_tuples = permutations_max_count(dims, n);
  fk::matrix<int> result(num_tuples, dims);

  if (dims == 1)
  {
    std::vector<int> entries(n + 1);
    std::iota(begin(entries), end(entries), 0);
    if (last_index_decreasing)
    {
      std::reverse(begin(entries), end(entries));
    }
    result.update_col(0, entries);
    return result;
  }

  // recursively build the lower dim tuples
  fk::matrix<int> lower_dims =
      permutations_max(dims - 1, n, last_index_decreasing);
  int const m = lower_dims.nrows();

  for (auto i = 0; i <= n; ++i)
  {
    int const row_position = i * m;
    int const rows         = m;
    int const last_entry   = last_index_decreasing ? (n - i) : i;

    fk::matrix<int> partial_result(rows, dims);
    fk::matrix<int> lower_dims_i =
        lower_dims.extract_submatrix(0, 0, m, dims - 1);
    partial_result.set_submatrix(0, 0, lower_dims_i);
    fk::vector<int> last_col = std::vector<int>(rows, last_entry);
    partial_result.update_col(dims - 1, last_col);
    result.set_submatrix(row_position, 0, partial_result);
  }

  return result;
}
