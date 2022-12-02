#pragma once

#include "asgard_discretization.hpp"

namespace asgard
{

template<typename precision>
class pde_system {
public:
  pde_system(parser const &cli_input,
             std::vector<dimension_description<precision>> const &dimensions,
             std::vector<field_description<precision>> const &field_terms
             )
    : pde_system(cli_input, dimensions, std::vector<field_description<precision>>(field_terms))
  {}

  pde_system(parser const &cli_input,
             std::vector<dimension_description<precision>> const &dimensions,
             std::vector<field_description<precision>> &&field_terms
             )
    : cli(cli_input), dims(cli, dimensions), fields(std::move(field_terms))
  {
    static_assert(std::is_same<precision, float>::value
                  or std::is_same<precision, double>::value,
                  "ASGARD supports only float and double as template parameters for precision.");

    expect(dimensions.size() > 0);

    for(auto &f : fields) f.verify_dimensions(dims);

    std::vector<std::string> field_names(fields.size());
    for(size_t i=0; i<field_names.size(); i++) field_names[i] = fields[i].name;
    verify_unique_strings(field_names);

    finalizes_discretization();
  }

  void finalizes_discretization()
  {
    // TODO: move the automatic discretization into a method that finalizes discretizations
    field_2_grid.reserve(fields.size());
    for(size_t i=0; i<fields.size(); i++) {
      // for each field, see if we already have discretization for this field
      size_t grid_index = 0;
      while(grid_index < grids.size())
      {
        if (grids[grid_index].can_discretize(fields[i]))
        {
          field_2_grid[i] = grid_index;
          break;
        }
        grid_index++;
      }
      if (grid_index == grids.size())
      {
        grids.push_back(
            field_discretization<precision>(cli, dims, fields[i].d_names)
          );
        field_2_grid[i] = grid_index;
      }
    }
  }

  // TODO: will work with operators similar to field
  // there is not need to have a separate "field_set",
  // we can just cross-reference the names of fields in the operator with the names in the fields vector
  void add_operator(){}

private:
  parser const &cli;
  dimension_set<precision> dims;
  std::vector<field_description<precision>> fields;
  std::vector<field_discretization<precision>> grids;
  std::vector<size_t> field_2_grid;
};

}
