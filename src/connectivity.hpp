#pragma once

#include "element_table.hpp"
#include "tensors.hpp"
#include <vector>

// -----------------------------------------------------------------------------
// connectivity
// this components's purpose is to define the connectivity between
// elements in the element_table
// -----------------------------------------------------------------------------

int get_1d_index(int const level, int const cell);

fk::matrix<int> make_1d_connectivity(int const num_levels);

using list_set = std::vector<fk::vector<int>>;
template<typename T>
list_set make_connectivity(element_table<T> table, int const num_dims,
                           int const max_level_sum, int const max_level_val,
                           bool const sort_connected = true);
extern template list_set
make_connectivity(element_table<int> table, int const num_dims,
                  int const max_level_sum, int const max_level_val,
                  bool const sort_connected = true);
extern template list_set
make_connectivity(element_table<int64_t> table, int const num_dims,
                  int const max_level_sum, int const max_level_val,
                  bool const sort_connected = true);
