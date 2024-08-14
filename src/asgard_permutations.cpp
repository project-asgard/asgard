#include "asgard_permutations.hpp"

#include "asgard_matlab_utilities.hpp"

namespace asgard::permutations
{
// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and their sum <= 'limit'. Each
// tuple becomes a row of the output matrix
fk::matrix<int>
get_lequal(int const num_dims, int const limit, bool const order_by_n)
{
  expect(num_dims > 0);
  expect(limit >= 0);

  return select_indexex(num_dims, order_by_n, false,
                        [&](std::array<int, max_num_dimensions> const &index) -> bool {
                          int level = index[0];
                          for (int i = 1; i < num_dims; i++)
                            level += index[i];
                          return (level <= limit);
                        });
}

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and their sum <= 'limit'. Each
// tuple becomes a row of the output matrix
// this version works with non-uniform levels across dimension
fk::matrix<int> get_lequal_multi(fk::vector<int> const &levels,
                                 int const num_dims, int const limit,
                                 bool const increasing_sum_order)
{
  expect(num_dims > 0);
  expect(levels.size() == num_dims);
  expect(limit >= 0);

  return select_indexex(num_dims, true, not increasing_sum_order,
                        [&](std::array<int, max_num_dimensions> const &index) -> bool {
                          int level = 0;
                          for (int i = 0; i < num_dims; i++)
                          {
                            if (index[i] > levels(i))
                              return false;
                            level += index[i];
                          }
                          return (level <= limit);
                        });
}

fk::matrix<int>
get_mix_leqmax_multi(fk::vector<int> const &levels, int const num_dims,
                     fk::vector<int> const &mixed_max,
                     int const num_first_group, bool const increasing_sum_order)
{
  expect(num_dims > 0);
  expect(levels.size() == num_dims);
  expect(mixed_max.size() == 2);
  expect(mixed_max[0] >= 0);
  expect(mixed_max[1] >= 0);

  return select_indexex(num_dims, true, not increasing_sum_order,
                        [&](std::array<int, max_num_dimensions> const &index) -> bool {
                          // check first group
                          int level_g1 = 0;
                          for (int i = 0; i < num_first_group; i++)
                          {
                            if (index[i] > levels(i))
                              return false;
                            level_g1 += index[i];
                          }
                          if (level_g1 > mixed_max[0])
                            return false;
                          // check second group
                          int level_g2 = 0;
                          for (int i = num_first_group; i < num_dims; i++)
                          {
                            if (index[i] > levels(i))
                              return false;
                            level_g2 += index[i];
                          }
                          if (level_g2 > mixed_max[1])
                            return false;
                          // if both groups pass, return true
                          return true;
                        });
}

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and the max element <= 'limit'
// (for full grid only). Each tuple becomes a row of the output matrix
fk::matrix<int>
get_max(int const num_dims, int const limit, bool const last_index_decreasing)
{
  expect(num_dims > 0);
  expect(limit >= 0);

  return get_max_multi(std::vector<int>(num_dims, limit), num_dims,
                       last_index_decreasing);
}

// Given the number of dimensions and a limit, produce n-tuples (n ==
// 'num_dims') whose elements are non-negative and the max element <= 'limit'
// (for full grid only). Each tuple becomes a row of the output matrix
// this version handles non-uniform levels passed as vector argument
fk::matrix<int> get_max_multi(fk::vector<int> const &levels, int const num_dims,
                              bool const last_index_decreasing)
{
  expect(num_dims > 0);
  expect(levels.size() > 0);

  int num_indexes = levels(0) + 1;
  for (int j = 1; j < levels.size(); j++)
    num_indexes *= (levels(j) + 1);

  fk::matrix<int> result(num_indexes, num_dims);

  auto load_multiindex = [&](int row, int i) -> void {
    int t = i;
    for (int j = 0; j < num_dims; j++)
    {
      result(row, j) = t % (levels(j) + 1);
      t /= (levels(j) + 1);
    }
  };

  if (last_index_decreasing)
  {
    for (int i = num_indexes - 1; i >= 0; i--)
    {
      load_multiindex(num_indexes - i - 1, i);
    }
  }
  else
  {
    for (int i = 0; i < num_indexes; i++)
    {
      load_multiindex(i, i);
    }
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
  expect(lists.size() > 0);
  expect(num_dims > 0);
  expect(num_dims <= static_cast<int>(lists.size()));

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
  expect(lists.size() > 0);
  expect(num_dims > 0);
  expect(num_dims <= static_cast<int>(lists.size()));

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

} // end namespace asgard::permutations
