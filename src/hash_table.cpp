#include "hash_table.hpp"

#include <array>
#include <numeric>
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

// given dims and n, produce dims-tuples whose sum is less than or equal to n
static fk::matrix permutations_leq(int n)
{

  int const count = permutations_leq_count(dims, n);
  fk::matrix result(count, dims);
  
  if (dims == 1) {
      std::vector<int> entries(n+1);
      std::iota(begin(entries), end(entries), 0);
      for (auto i = 0; i < entries.size(); ++i) {
          result[i] = std::array<int,dims>{entries[i]};
      }
   return result;
  }
 
  int count = 0;
  for(auto i = 0; i <= n; ++i) {
    int size = permutation_leq_count(dims-1, n-1);
    std::vector<std::array<int, dims>   
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
  
