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

/*
//
// Permutations functions to build level combinations
// in prgs

static int permutations_eq_count(int dims, int n)
{
  if (dims == 1) { return 1; }

  if (dims == 2) { return n + 1; }

  int count = 0;
  for (auto i = 0; i <= n; ++i)
  {
    count += permutations_eq_count(dims - 1, i);
  }

  return count;
}

static int permutations_leq_count(int dims, int n)
{
  int count = 0;

  for (auto i = 0; i <= n; ++i)
  {
    count += permutations_eq_count(dims, i);
  }
  return count;
}


//
// Given dims and n, produce dims-tuples whose sum is equal to n
// TODO in prgs

static fk::matrix permutations_eq(int const dims, int const n)
{

int const num_tuples = permutations_eq_count(dims, n);
fk::matrix result(num_tuples, dims);

if (dims == 1) {
  return fk::matrix {{n}};
}


int counter = 0;
for(auto i = 0; i <= n; ++i) {

  int const rows = permutations_eq_count( dims-1, n-i);
  fk::matrix partial_result(rows, dims);
  fk::matrix partial_tuples = permutations_eq(dims - 1, n-i);
  partial_result.set_submatrix(counter, 0, partial_tuples);

  std::vector<int> last_col = std::vector<int>(rows, i);
  partial_result.update_col(dims-1, last_col);
  //result( ip:(ip+isize-1), 1:(idim-1)) =
perm_eq(idim-1,n-i,last_index_decreasing);
  //result( ip:(ip+isize-1), idim) = i;

  counter = counter + rows;
}

return result;
}

//
// Given dims and n, produce dims-tuples whose sum is less than or equal to n
//

static fk::matrix permutations_leq(int const dims, int const n, bool const
order_by_n)
{

  int const rows_result = permutations_leq_count(dims, n);
  fk::matrix result(rows_result, dims);

  if(order_by_n) {
    int row_pos = 0;
    for(auto i = 0; i<=n; ++i) {
      int const num_rows = permutations_eq_count(dims, i);
      fk::matrix partial = permutations_eq(dims, i);
      result.set_submatrix(count, 0, partial);
      row_pos += num_rows;
    }
    return result;
  }

  if (dims == 1) {
      std::vector<int> entries(n+1);
      std::iota(begin(entries), end(entries), 0);
      result.update_col(0, entries);
      //for (auto i = 0; i < entries.size(); ++i) {
      //    result[i] =i std::array<int,dims>{entries[i]};
      //}
      return result;
  }

  int row_pos = 0;
  for(auto i = 0; i <= n; ++i) {
      int const num_rows = permutations_leq_count(dims-1, n-i);
      fk::matrix partial = permutations_leq(dims, i);
      result.set_submatrix(count, 0, partial);
      //std::vector<double> partial_last_col(
      row_pos += num_rows;
  }
}

*/
