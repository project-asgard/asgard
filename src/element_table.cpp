#include "element_table.hpp"

#include "tensors.hpp"
#include <array>
#include <numeric>
#include <vector>

// TODO this constructor will invoke the
// static permutation/cell builder helpers to construct the table
element_table::element_table(int const dim, int const level,
                             bool const full_grid)
    : size_{0}
{}

// TODO forward lookup
int element_table::get_index(std::vector<int> const coords) const { return 0; }

// TODO reverse lookup
std::vector<int> element_table::get_coords(int const index) const
{
  return std::vector<int>(0);
}

//
// Static helpers for construction
//

//
// Indexing helpers
//

// Given a cell and level coordinate, return a 1-dimensional index
int element_table::get_1d_index(int const level, int const cell)
{
  assert(level >= 0);
  assert(cell >= 0);

  if (level == 0) { return 1; }
  return static_cast<int>(std::pow(2, level - 1)) + cell + 1;
}

//
// Permutation enumerators
//

// Given dims and n, produce the number of dims-tuples whose sum == n
int element_table::permutations_eq_count(int const dims, int const n)
{
  assert(dims > 0);
  assert(n >= 0);
  if (dims == 1) { return 1; }

  if (dims == 2) { return n + 1; }

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

  if (dims == 1) { return fk::matrix<int>{{n}}; }

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
    if (last_index_decreasing) { std::reverse(begin(entries), end(entries)); }
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
