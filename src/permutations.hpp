#pragma once
#include "tensors.hpp"

// -----------------------------------------------------------------------------
// permutations
// this components's purpose is to provide utilities used in 
// the construction of the element table and in connectivity
// functions
// -----------------------------------------------------------------------------

  // Permutations counters

  int count_eq_permutations(int const num_dims, int const limit);

  int count_leq_permutations(int const num_dims, int const limit);

  int count_max_permutations(int const num_dims, int const limit);

  // Permutations builders

  fk::matrix<int> get_eq_permutations(int const num_dims,
                                             int const limit,
                                             bool const order_by_n);

  fk::matrix<int> get_leq_permutations(int const num_dims,
                                              int const limit,
                                              bool const order_by_n);

  fk::matrix<int> get_max_permutations(int const num_dims,
                                              int const limit,
                                              bool const last_index_decreasing);
