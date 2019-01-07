#include "permutations.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include "tensors.hpp"
#include <vector>

//
// Permutation enumerators
//

// Given the number of dimensions and a limit, count the number of n-tuples
// (where n == 'num_dims') whose non-negative elements' sum == 'limit'
int count_eq_permutations(int const num_dims, int const limit)
{
  assert(num_dims > 0);
  assert(limit >= 0);
  if (num_dims == 1)
  {
    return 1;
  }

  if (num_dims == 2)
  {
    return limit + 1;
  }

  int count = 0;
  for (auto i = 0; i <= limit; ++i)
  {
    count += count_eq_permutations(num_dims - 1, i);
  }

  return count;
}

// Given the number of dimensions and a limit, count the number of n-tuples
// (where n == 'num_dims') whose non-negative elements' sum <= 'limit'
int count_leq_permutations(int num_dims, int limit)
{
  assert(num_dims > 0);
  assert(limit >= 0);

  int count = 0;
  for (auto i = 0; i <= limit; ++i)
  {
    count += count_eq_permutations(num_dims, i);
  }
  return count;
}

// Given the number of dimensions and a limit, count the number of n-tuples
// (where n == 'num_dims') whose non-negative max element <= 'limit' (for full
// grid only)
int count_max_permutations(int const num_dims, int const limit)
{
  assert(num_dims > 0);
  assert(limit >= 0);

  return static_cast<int>(std::pow(limit + 1, num_dims));
}

//
// Permutations builders
//

// Given the number of dimensions and a limit, produce n-tuples (where n ==
// 'num_dims') whose elements' are non-negative and their sum == 'limit'. Each
// tuple becomes a row of the output matrix
fk::matrix<int>
get_eq_permutations(int const num_dims, int const limit, bool const order_by_n)
{
  assert(num_dims > 0);
  assert(limit >= 0);

  int const num_tuples = count_eq_permutations(num_dims, limit);
  fk::matrix<int> result(num_tuples, num_dims);

  if (num_dims == 1)
  {
    return fk::matrix<int>{{limit}};
  }

  int counter = 0;
  for (auto i = 0; i <= limit; ++i)
  {
    int partial_sum;
    int difference;

    if (order_by_n)
    {
      partial_sum = i;
      difference  = limit - i;
    }
    else
    {
      partial_sum = limit - i;
      difference  = i;
    }

    // build set of num_dims-1 sized tuples which sum to partial_sum,
    // then append a column with difference to make a num_dims-tuple.
    int const rows = count_eq_permutations(num_dims - 1, partial_sum);
    fk::matrix<int> partial_result(rows, num_dims);
    partial_result.set_submatrix(
        0, 0, get_eq_permutations(num_dims - 1, partial_sum, order_by_n));
    fk::vector<int> last_col = std::vector<int>(rows, difference);
    partial_result.update_col(num_dims - 1, last_col);
    result.set_submatrix(counter, 0, partial_result);

    counter += rows;
  }
  return result;
}

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and their sum <= 'limit'. Each
// tuple becomes a row of the output matrix
fk::matrix<int>
get_leq_permutations(int const num_dims, int const limit, bool const order_by_n)
{
  assert(num_dims > 0);
  assert(limit >= 0);

  int const num_tuples = count_leq_permutations(num_dims, limit);
  fk::matrix<int> result(num_tuples, num_dims);

  if (order_by_n)
  {
    int counter = 0;
    for (auto i = 0; i <= limit; ++i)
    {
      int const rows = count_eq_permutations(num_dims, i);
      result.set_submatrix(counter, 0, get_eq_permutations(num_dims, i, false));
      counter += rows;
    }
    return result;
  }

  // the recursive base case
  if (num_dims == 1)
  {
    std::vector<int> entries(limit + 1);
    std::iota(begin(entries), end(entries), 0);
    result.update_col(0, entries);
    return result;
  }

  int counter = 0;
  for (auto i = 0; i <= limit; ++i)
  {
    int const rows = count_leq_permutations(num_dims - 1, limit - i);
    fk::matrix<int> partial_result(rows, num_dims);
    partial_result.set_submatrix(
        0, 0, get_leq_permutations(num_dims - 1, limit - i, order_by_n));
    fk::vector<int> last_col = std::vector<int>(rows, i);
    partial_result.update_col(num_dims - 1, last_col);
    result.set_submatrix(counter, 0, partial_result);

    counter += rows;
  }

  return result;
}

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and the max element <= 'limit'
// (for full grid only). Each tuple becomes a row of the output matrix
fk::matrix<int> get_max_permutations(int const num_dims, int const limit,
                                     bool const last_index_decreasing)
{
  assert(num_dims > 0);
  assert(limit >= 0);

  int const num_tuples = count_max_permutations(num_dims, limit);
  fk::matrix<int> result(num_tuples, num_dims);

  if (num_dims == 1)
  {
    std::vector<int> entries(limit + 1);
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
      get_max_permutations(num_dims - 1, limit, last_index_decreasing);
  int const m = lower_dims.nrows();

  for (auto i = 0; i <= limit; ++i)
  {
    int const row_position = i * m;
    int const rows         = m;
    int const last_entry   = last_index_decreasing ? (limit - i) : i;

    fk::matrix<int> partial_result(rows, num_dims);
    fk::matrix<int> lower_dims_i =
        lower_dims.extract_submatrix(0, 0, m, num_dims - 1);
    partial_result.set_submatrix(0, 0, lower_dims_i);
    fk::vector<int> last_col = std::vector<int>(rows, last_entry);
    partial_result.update_col(num_dims - 1, last_col);
    result.set_submatrix(row_position, 0, partial_result);
  }

  return result;
}
