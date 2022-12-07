#pragma once

#include "asgard_discretization.hpp"

namespace asgard
{

template<typename precision>
std::vector<field<precision>>
make_fields(std::vector<field_description<precision>> const &descriptions)
{
  std::vector<field<precision>> result;
  result.reserve(descriptions.size());
  for(auto const &d : descriptions)
    result.push_back(field<precision>(d));
  return result;
}

template<typename precision, resource resrc = asgard::resource::host>
class pde_system {
public:
  pde_system(parser const &cli_input,
             std::vector<dimension_description<precision>> const &dimensions,
             std::vector<field_description<precision>> const &field_terms
             )
    : cli(cli_input), dims(cli, dimensions), fields(make_fields(field_terms)),
      transformer(cli_input, dimensions.front().degree, true)
  {
    static_assert(std::is_same<precision, float>::value
                  or std::is_same<precision, double>::value,
                  "ASGARD supports only float and double as template parameters for precision.");

    expect(dimensions.size() > 0);

    for(auto &f : field_terms)
      f.verify_dimensions(dims);

    std::vector<std::string> field_names(fields.size());
    for(size_t i=0; i<field_names.size(); i++) field_names[i] = fields[i].name;
    verify_unique_strings(field_names);

    finalizes_discretization();
  }

  void finalizes_discretization()
  {
    // TODO: move the automatic discretization into a method that finalizes discretizations
    for(size_t i=0; i<fields.size(); i++)
    {
      // for each field, see if we already have discretization for this field
      int64_t grid_index = 0;
      while(grid_index < static_cast<int64_t>(grids.size()))
      {
        if (grids[grid_index].can_discretize(fields[i]))
        {
          fields[i].grid_index = grid_index;
          break;
        }
        grid_index++;
      }
      if (grid_index == static_cast<int64_t>(grids.size()))
      {
        grids.push_back(
            field_discretization<precision, resrc>(cli, dims, transformer, fields[i].d_names)
          );
        fields[i].grid_index = grid_index;
      }
    }

    int64_t total_size = 0;
    for(auto &f : fields)
    {
      int64_t size = grids[f.grid_index].size();
      f.set_global_index(total_size,
                         total_size + size);
      total_size += size;
    }

    state = fk::vector<precision>(total_size);
  }

  fk::vector<precision, mem_type::const_view>
  get_field(std::string const &name)
  {
    size_t findex = 0;
    while(findex < fields.size() && fields[findex].name != name)
      findex++;
    return fk::vector<precision, mem_type::const_view>
      (state, fields[findex].global_begin, fields[findex].global_end-1);
  }

  // TODO: will work with operators similar to field
  // there is not need to have a separate "field_set",
  // we can just cross-reference the names of fields in the operator with the names in the fields vector
  void add_operator(){}

private:
  parser const &cli;
  dimension_set<precision> dims;
  std::vector<field<precision>> fields;

  asgard::basis::wavelet_transform<precision, resrc> transformer;
  std::vector<field_discretization<precision, resrc>> grids;

  fk::vector<precision> state;
};

}
