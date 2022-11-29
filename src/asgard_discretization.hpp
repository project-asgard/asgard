#pragma once

#include "asgard_field.hpp"

namespace asgard
{

template<typename precision>
struct field_discretization {

    adapt::distributed_grid<precision> grid;

    std::vector<dimension<precision>> dims;

};


}
