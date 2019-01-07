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

//
// Connectivity functions
//

int get_1d_index(int const level, int const cell);
