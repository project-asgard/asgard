#pragma once

#include "asgard_field.hpp"

namespace asgard
{

template<typename precision>
struct field_discretization {

  field_discretization(parser const &cli_input,
                       dimension_set<precision> const &d_set,
                       std::vector<std::string> const &f_names)
  {
    dims.reserve(f_names.size());
    for(size_t i=0; i<dims.size(); i++)
      dims.push_back(dimension<precision>(d_set(f_names[i])));

    grid = adapt::distributed_grid<precision>(cli_input, dims);
  }

  adapt::distributed_grid<precision> grid;

  std::vector<dimension<precision>> dims;

};


}
