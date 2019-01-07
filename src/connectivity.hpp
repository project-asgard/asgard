#pragma once

#include "program_options.hpp"
#include "tensors.hpp"
#include <cassert>
#include <map>
#include <vector>

// -----------------------------------------------------------------------------
// connectivity
// this components's purpose is to define the connectivity between
// elements in the element_table
// -----------------------------------------------------------------------------

// FIXME need to determine which of these need to be
int get_1d_index(int const level, int const cell);
fk::matrix<int> connect_1d(int const num_levels);
