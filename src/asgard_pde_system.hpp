#pragma once

#include "asgard_field.hpp"

namespace asgard
{

template<typename precision>
class pde_system {
public:
  pde_system(parser const &cli_input, std::vector<dimension_description<precision>> const dimensions)
    : dims(cli_input, dimensions)
  {
    static_assert(std::is_same<precision, float>::value
                  or std::is_same<precision, double>::value,
                  "ASGARD supports only float and double as template parameters for precision.");

    expect(dimensions.size() > 0);
  }

  void add_field(field_description<precision> const description)
  {
    for(auto const &f : fields)
      if (f.name == description.name)
        throw std::runtime_error(description.name + " is an existing field name, add_field() already called for a field with that name.");
    fields.push_back(field<precision>(dims, description));
  }

  // TODO: will work with operators similar to field
  // there is not need to have a separate "field_set",
  // we can just cross-reference the names of fields in the operator with the names in the fields vector
  void add_operator(){}

private:
  dimension_set<precision> dims;
  std::vector<field<precision>> fields;
};

}
