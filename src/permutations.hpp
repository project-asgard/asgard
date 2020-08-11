#pragma once
#include "tensors.hpp"
#include <vector>

// -----------------------------------------------------------------------------
// permutations
// this components's purpose is to provide utilities used in
// construction of the element table
// -----------------------------------------------------------------------------

namespace permutations
{
// Permutations counters

int count_equal(int const num_dims, int const limit);

int count_equal_multi(fk::vector<int> const &levels, int const num_dims,
                      int const limit);

int count_lequal(int const num_dims, int const limit);

int count_lequal_multi(fk::vector<int> const &levels, int const num_dims,
                       int const limit);

int count_max(int const num_dims, int const limit);

// Permutations builders

fk::matrix<int>
get_equal(int const num_dims, int const limit, bool const order_by_n);

fk::matrix<int> get_equal_multi(fk::vector<int> const &levels,
                                int const num_dims, int const limit,
                                bool const last_index_decreasing);

fk::matrix<int>
get_lequal(int const num_dims, int const limit, bool const order_by_n);

fk::matrix<int> get_lequal_multi(fk::vector<int> const &levels,
                                 int const num_dims, int const limit,
                                 bool const increasing_sum_order);

fk::matrix<int>
get_max(int const num_dims, int const limit, bool const last_index_decreasing);

using list_set = std::vector<fk::vector<int>>;

// Index counter

int count_leq_max_indices(list_set const &lists, int const num_dims,
                          int const max_sum, int const max_val);

// Index finder

fk::matrix<int> get_leq_max_indices(list_set const &lists, int const num_dims,
                                    int const max_sum, int const max_val);

} // end namespace permutations
