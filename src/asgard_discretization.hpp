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
    : cli(cli_input), transformer(wavelet_transformer), num_dof(0)
  {
    dims.reserve(d_names.size());
    for(size_t i=0; i<d_names.size(); i++)
      dims.emplace_back(dimension<precision>(d_set(d_names[i])));

    grid = std::make_unique<adapt::distributed_grid<precision>>(cli_input, dims);
    auto const subgrid = grid->get_subgrid(get_rank());
    num_dof = (subgrid.col_stop - subgrid.col_start + 1) * std::pow(dims.front().degree_, dims.size());
  }

  bool can_discretize(field<precision> const &field) {
    for(auto const &d_name : field.d_names) {
      if (not std::any_of(dims.begin(), dims.end(), [&](dimension<precision> const &x)->bool{ return (x.name == d_name); }))
      {
        return false;
      }
    }
    return true;
  }

  int64_t size() const { return num_dof; }

  void get_initial_conditions(field<precision> const &field, fk::vector<precision, mem_type::view> result)
  {
    expect(result.size() == size());
    grid->get_initial_condition(cli, dims, field.init_cond, 1.0, transformer, result);
  }
  fk::vector<precision> get_initial_conditions(field_description<precision> const &field)
  {
    fk::vector<precision> result(num_dof);
    get_initial_conditions(field, fk::vector<precision, mem_type::view>(result));
    return result;
  }

  parser const &cli;

  asgard::basis::wavelet_transform<precision, resrc> &transformer;

  std::unique_ptr<adapt::distributed_grid<precision>> grid;
  int64_t num_dof;

  std::vector<dimension<precision>> dims;

};

}
