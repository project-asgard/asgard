#include "permutations.hpp"

#include "matlab_utilities.hpp"
#include "tensors.hpp"
#include "tools.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace permutations
{
//
// Permutation enumerators
//

// Given the number of dimensions and a limit, count the number of n-tuples
// (where n == 'num_dims') whose non-negative elements' sum == 'limit'
int count_equal(int const num_dims, int const limit)
{
  tools::expect(num_dims > 0);
  tools::expect(limit >= 0);
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
    count += count_equal(num_dims - 1, i);
  }

  return count;
}

// Given the number of dimensions and a limit, count the number of n-tuples
// (where n == 'num_dims') whose non-negative elements' sum <= 'limit'
// this version handles the non-uniform level case
int count_equal_multi(fk::vector<int> const &levels, int const num_dims,
                      int const limit)
{
  tools::expect(num_dims > 0);
  tools::expect(levels.size() == num_dims);
  tools::expect(limit >= 0);

  if (num_dims == 1)
  {
    if (levels(0) >= limit)
    {
      return 1;
    }
    return 0;
  }

  auto count = 0;
  for (auto i = 0; i <= std::min(levels(num_dims - 1), limit); ++i)
  {
    auto const dim_limit = limit - i;
    count += count_equal_multi(levels.extract(0, num_dims - 2), num_dims - 1,
                               dim_limit);
  }

  return count;
}

// Given the number of dimensions and a limit, count the number of n-tuples
// (where n == 'num_dims') whose non-negative elements' sum <= 'limit'
int count_lequal(int const num_dims, int const limit)
{
  tools::expect(num_dims > 0);
  tools::expect(limit >= 0);

  int count = 0;
  for (auto i = 0; i <= limit; ++i)
  {
    count += count_equal(num_dims, i);
  }
  return count;
}

// Given the number of dimensions and a limit, count the number of n-tuples
// (where n == 'num_dims') whose non-negative elements' sum <= 'limit'
// this version handles the non-uniform level case
int count_lequal_multi(fk::vector<int> const &levels, int num_dims, int limit)
{
  tools::expect(num_dims > 0);
  tools::expect(levels.size() == num_dims);
  tools::expect(limit >= 0);

  auto count = 0;
  for (auto i = 0; i <= limit; ++i)
  {
    count += count_equal_multi(levels, num_dims, i);
  }

  return count;
}

// Given the number of dimensions and a limit, count the number of n-tuples
// (where n == 'num_dims') whose non-negative max element <= 'limit' (for full
// grid only)
int count_max(int const num_dims, int const limit)
{
  tools::expect(num_dims > 0);
  tools::expect(limit >= 0);

  return static_cast<int>(std::pow(limit + 1, num_dims));
}

//
// Permutations builders
//

// Given the number of dimensions and a limit, produce n-tuples (where n ==
// 'num_dims') whose elements' are non-negative and their sum == 'limit'. Each
// tuple becomes a row of the output matrix
fk::matrix<int>
get_equal(int const num_dims, int const limit, bool const order_by_n)
{
  tools::expect(num_dims > 0);
  tools::expect(limit >= 0);

  int const num_tuples = count_equal(num_dims, limit);
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
    int const rows = count_equal(num_dims - 1, partial_sum);
    fk::matrix<int> partial_result(rows, num_dims);
    partial_result.set_submatrix(
        0, 0, get_equal(num_dims - 1, partial_sum, order_by_n));
    fk::vector<int> last_col = std::vector<int>(rows, difference);
    partial_result.update_col(num_dims - 1, last_col);
    result.set_submatrix(counter, 0, partial_result);

    counter += rows;
  }
  return result;
}

// Given the number of dimensions and a limit, produce n-tuples (where n ==
// 'num_dims') whose elements' are non-negative and their sum == 'limit'. Each
// tuple becomes a row of the output matrix
// this version handles the non-uniform level case via a vector "levels"
// argument
fk::matrix<int> get_equal_multi(fk::vector<int> const &levels,
                                int const num_dims, int const limit,
                                bool const last_index_decreasing)
{
  tools::expect(num_dims > 0);
  tools::expect(levels.size() == num_dims);
  tools::expect(limit >= 0);

  if (num_dims == 1)
  {
    if (levels(0) >= limit)
    {
      return fk::matrix<int>{{limit}};
    }
    else
    {
      return fk::matrix<int>();
    }
  }

  auto const num_tuples = count_equal_multi(levels, num_dims, limit);
  fk::matrix<int> result(num_tuples, num_dims);
  auto row_counter = 0;
  auto const stop  = std::min(levels(num_dims - 1), limit);

  for (auto i = 0; i <= stop; ++i)
  {
    auto const last_elem     = last_index_decreasing ? stop - i : i;
    auto const partial_limit = limit - last_elem;
    auto const partial_size = count_equal_multi(levels.extract(0, num_dims - 2),
                                                num_dims - 1, partial_limit);

    if (partial_size < 1)
    {
      continue;
    }

    fk::matrix<int, mem_type::view> partial_result(
        result, row_counter, row_counter + partial_size - 1, 0, num_dims - 2);
    partial_result =
        get_equal_multi(levels.extract(0, num_dims - 2), num_dims - 1,
                        partial_limit, last_index_decreasing);

    fk::matrix<int, mem_type::view> partial_last_col(
        result, row_counter, row_counter + partial_size - 1, num_dims - 1,
        num_dims - 1);
    partial_last_col = fk::vector(std::vector<int>(partial_size, last_elem));
    row_counter += partial_size;
  }

  return result;
}

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and their sum <= 'limit'. Each
// tuple becomes a row of the output matrix
fk::matrix<int>
get_lequal(int const num_dims, int const limit, bool const order_by_n)
{
  tools::expect(num_dims > 0);
  tools::expect(limit >= 0);

  int const num_tuples = count_lequal(num_dims, limit);
  fk::matrix<int> result(num_tuples, num_dims);

  if (order_by_n)
  {
    int counter = 0;
    for (auto i = 0; i <= limit; ++i)
    {
      int const rows = count_equal(num_dims, i);
      result.set_submatrix(counter, 0, get_equal(num_dims, i, false));
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
    int const rows = count_lequal(num_dims - 1, limit - i);
    fk::matrix<int> partial_result(rows, num_dims);
    partial_result.set_submatrix(
        0, 0, get_lequal(num_dims - 1, limit - i, order_by_n));
    fk::vector<int> last_col = std::vector<int>(rows, i);
    partial_result.update_col(num_dims - 1, last_col);
    result.set_submatrix(counter, 0, partial_result);

    counter += rows;
  }

  return result;
}

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and their sum <= 'limit'. Each
// tuple becomes a row of the output matrix
// this version works with non-uniform levels across dimension
fk::matrix<int> get_lequal_multi(fk::vector<int> const &levels,
                                 int const num_dims, int const limit,
                                 bool const increasing_sum_order)
{
  tools::expect(num_dims > 0);
  tools::expect(levels.size() == num_dims);
  tools::expect(limit >= 0);

  auto const num_tuples = count_lequal_multi(levels, num_dims, limit);
  fk::matrix<int> result(num_tuples, num_dims);

  auto row_counter = 0;
  for (auto i = 0; i <= limit; ++i)
  {
    auto const partial_limit = increasing_sum_order ? i : limit - i;
    auto const partial_size =
        count_equal_multi(levels, num_dims, partial_limit);
    if (partial_size > 0)
    {
      fk::matrix<int, mem_type::view> partial_result(
          result, row_counter, row_counter + partial_size - 1, 0, num_dims - 1);
      partial_result = get_equal_multi(levels, num_dims, partial_limit, false);
      row_counter += partial_size;
    }
  }

  return result;
}

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and the max element <= 'limit'
// (for full grid only). Each tuple becomes a row of the output matrix
fk::matrix<int>
get_max(int const num_dims, int const limit, bool const last_index_decreasing)
{
  tools::expect(num_dims > 0);
  tools::expect(limit >= 0);

  int const num_tuples = count_max(num_dims, limit);
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
      get_max(num_dims - 1, limit, last_index_decreasing);
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

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and the max element <= 'limit'
// (for full grid only). Each tuple becomes a row of the output matrix
// this version handles non-uniform levels passed as vector argument
fk::matrix<int> get_max_multi(fk::vector<int> const &levels, int const num_dims,
                              bool const last_index_decreasing)
{
  tools::expect(num_dims > 0);
  tools::expect(levels.size() > 0);

  auto const limit = levels(num_dims - 1);

  if (num_dims == 1)
  {
    std::vector<int> entries(limit + 1);
    std::iota(begin(entries), end(entries), 0);
    if (last_index_decreasing)
    {
      std::reverse(begin(entries), end(entries));
    }
    return fk::matrix<int>(limit + 1, 1).update_col(0, entries);
  }

  auto const lower_perm =
      get_max_multi(levels, num_dims - 1, last_index_decreasing);
  fk::matrix<int> result(lower_perm.nrows() * (limit + 1), num_dims);

  for (auto i = 0; i <= limit; ++i)
  {
    auto const row_pos  = i * lower_perm.nrows();
    auto const row_stop = row_pos + lower_perm.nrows() - 1;

    fk::matrix<int, mem_type::view> partial_result(result, row_pos, row_stop, 0,
                                                   num_dims - 2);
    partial_result = fk::matrix<int, mem_type::const_view>(
        lower_perm, 0, lower_perm.nrows() - 1, 0, num_dims - 2);

    fk::vector<int, mem_type::view> last_col(result, num_dims - 1, row_pos,
                                             row_stop);
    auto const last_col_val = last_index_decreasing ? limit - i : i;
    last_col =
        fk::vector<int>(std::vector<int>(row_stop - row_pos + 1, last_col_val));
  }

  return result;
}

//
// Index finding functions
//

// count the number of rows in the matrix returned by the index finder
// get_leq_max_indices() below
int count_leq_max_indices(list_set const &lists, int const num_dims,
                          int const max_sum, int const max_val)
{
  tools::expect(lists.size() > 0);
  tools::expect(num_dims > 0);
  tools::expect(num_dims <= static_cast<int>(lists.size()));

  // base case
  if (num_dims == 1)
  {
    auto is_valid = [max_sum, max_val](int const &i) {
      return (i <= max_sum) && (i <= max_val);
    };
    return find(lists[0], is_valid).size();
  }

  // recursive count
  int count                  = 0;
  fk::vector<int> const list = lists[num_dims - 1];
  auto is_valid              = [max_val](int const &i) { return i <= max_val; };
  auto const valid_indices   = find(list, is_valid);
  for (auto i = 0; i < valid_indices.size(); ++i)
  {
    int const balance = max_sum - list(i);
    count += count_leq_max_indices(lists, num_dims - 1, balance, max_val);
  }
  return count;
}

// given a set of integer lists and a sum and value limit, build an n*num_lists
// matrix whose elements are indices into the lists (column x contains an index
// into list x). when elements are used to reference their corresponding list,
// each row will contain a tuple whose dereferenced (into the original lists)
// elements sum to less than max_sum and whose maximum value is less than
// max_val.
// num_dims is the initial number of lists at the start of the recursion
fk::matrix<int> get_leq_max_indices(list_set const &lists, int const num_dims,
                                    int const max_sum, int const max_val)
{
  tools::expect(lists.size() > 0);
  tools::expect(num_dims > 0);
  tools::expect(num_dims <= static_cast<int>(lists.size()));

  int const num_entries =
      count_leq_max_indices(lists, num_dims, max_sum, max_val);
  fk::matrix<int> result(num_entries, num_dims);

  // base case
  if (num_dims == 1)
  {
    auto is_valid = [max_sum, max_val](int const &i) {
      return (i <= max_sum) && (i <= max_val);
    };
    fk::vector<int> indices = find(lists[0], is_valid);
    result.update_col(0, indices);
    return result;
  }

  // recursive build
  int row_pos                = 0;
  fk::vector<int> const list = lists[num_dims - 1];
  auto is_valid              = [max_val](int const &i) { return i <= max_val; };
  auto const valid_indices   = find(list, is_valid);

  for (auto i = 0; i < valid_indices.size(); ++i)
  {
    int const balance = max_sum - list(i);
    int const num_rows =
        count_leq_max_indices(lists, num_dims - 1, balance, max_val);
    fk::matrix<int> const partial_result =
        get_leq_max_indices(lists, num_dims - 1, balance, max_val);
    result.set_submatrix(row_pos, 0, partial_result);
    fk::matrix<int> last_col(num_rows, 1);
    last_col = fk::vector<int>(std::vector<int>(num_rows, i));
    result.set_submatrix(row_pos, num_dims - 1, last_col);
    row_pos += num_rows;
  }

  return result;
}

} // end namespace permutations
