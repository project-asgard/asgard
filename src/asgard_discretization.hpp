#pragma once

#include "asgard_field.hpp"

namespace asgard
{

template<typename precision, resource resrc>
struct field_discretization {

  field_discretization(parser const &cli_input,
                       dimension_set<precision> const &d_set,
                       asgard::basis::wavelet_transform<precision, resrc> &wavelet_transformer,
                       std::vector<std::string> const &d_names)
    : cli(cli_input), transformer(wavelet_transformer)
  {
    dims.reserve(d_names.size());
    for(size_t i=0; i<d_names.size(); i++)
      dims.emplace_back(dimension<precision>(d_set(d_names[i])));

    grid = std::make_unique<adapt::distributed_grid<precision>>(cli_input, dims);
  }

  bool can_discretize(field_description<precision> const &field) {
    for(auto const &d_name : field.d_names) {
      if (not std::any_of(dims.begin(), dims.end(), [&](dimension<precision> const &x)->bool{ return (x.name == d_name); }))
      {
        return false;
      }
    }
    return true;
  }

  fk::vector<precision> get_initial_conditions(field_description<precision> const &field)
  {
    return grid->get_initial_condition(cli, dims, field.init_cond, 1.0, transformer);
  }

  parser const &cli;

  asgard::basis::wavelet_transform<precision, resrc> &transformer;

  std::unique_ptr<adapt::distributed_grid<precision>> grid;

  std::vector<dimension<precision>> dims;

};

}
