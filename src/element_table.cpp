#include "element_table.hpp"

#include <array>
#include <numeric>
#include "tensors.hpp"
#include <vector>

//
// Permutations functions to build level combinations
//

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
//

static fk::matrix permutations_eq(int dims, int n)
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
  //result( ip:(ip+isize-1), 1:(idim-1)) = perm_eq(idim-1,n-i,last_index_decreasing);
  //result( ip:(ip+isize-1), idim) = i;

  counter = counter + rows;
}

return result;
}

//
// Given dims and n, produce dims-tuples whose sum is less than or equal to n
//
static fk::matrix permutations_leq(int dims, int n)
{

  int const tuples_count = permutations_leq_count(dims, n);
  fk::matrix result(tuples_count, dims);
  
  if (dims == 1) {
      std::vector<int> entries(n+1);
      std::iota(begin(entries), end(entries), 0);
      for (auto i = 0u; i < entries.size(); ++i) {
          result(i,1) = entries[i];
      }
   return result;
  }
 
  int counter = 0;
  for(auto i = 0; i <= n; ++i) {
    int const size = permutations_leq_count(dims-1, n-1);
    fk::matrix partial  
  }
}
/*
  ip = 1;
    for
      i = 0 : n, isize = perm_leq_count(idim - 1, n - i);
    result(ip
           : (ip + isize - 1), 1
           : (idim - 1))                = perm_leq(idim - 1, n - i, order_by_n);
    result(ip : (ip + isize - 1), idim) = i;

    ip = ip + isize;
    end;

    return
    */
  
