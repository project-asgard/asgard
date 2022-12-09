#pragma once

#include "adapt.hpp"
#include "transformations.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits.h>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "pde/pde_base.hpp"

namespace asgard
{

/*!
 * \brief Copies the dimensions and modifies them based on the cli-parameters
 */
template<typename precision>
std::vector<dimension_description<precision>>
cli_apply_level_degree_correction(parser const &cli_input,
                                  std::vector<dimension_description<precision>> const &dimensions)
{
  size_t num_dims = dimensions.size();
  std::vector<int> levels(dimensions.size()), degrees(dimensions.size());
  for(size_t i=0; i<num_dims; i++)
  {
    levels[i] = dimensions[i].level;
    degrees[i] = dimensions[i].degree;
  }

  // modify for appropriate level/degree
  // if default lev/degree not used
  auto const user_levels = cli_input.get_starting_levels().size();
  if (user_levels != 0 && user_levels != static_cast<int>(num_dims))
  {
    throw std::runtime_error(
        std::string("failed to parse dimension-many starting levels - parsed ")
        + std::to_string(user_levels) + " levels");
  }
  if (user_levels == static_cast<int>(num_dims))
  {
    auto counter = 0;
    for (int &l : levels)
    {
      l = cli_input.get_starting_levels()(counter++);
      expect(l > 1);
    }
  }
  auto const cli_degree = cli_input.get_degree();
  if (cli_degree != parser::NO_USER_VALUE)
  {
    expect(cli_degree > 0);
    for (int &d : degrees) d = cli_degree;
  }

  // check all dimensions
  for(size_t i=0; i<dimensions.size(); i++)
  {
    expect(degrees[i] > 0);
    expect(levels[i] > 1);
  }

  std::vector<dimension_description<precision>> result;
  result.reserve(num_dims);
  for(size_t i=0; i<num_dims; i++)
  {
    result.push_back(
      dimension_description<precision>(dimensions[i].d_min, dimensions[i].d_max,
                                       levels[i], degrees[i],
                                       dimensions[i].name)
                     );
  }
  return result;
}

/*!
 * \brief Throws an exception if there are repeated entries among the names.
 */
inline bool check_unique_strings(std::vector<std::string> const &names) {
  size_t num_dims = names.size();
  for(size_t i=0; i<num_dims; i++)
  {
    for(size_t j=i+1; j<num_dims; j++)
    {
      if (names[i] == names[j])
        return false;
    }
  }
  return true;
}

template<typename precision>
struct dimension_set {
  dimension_set(parser const &cli_input, std::vector<dimension_description<precision>> const &dimensions)
    : list(cli_apply_level_degree_correction(cli_input, dimensions))
  {
    std::vector<std::string> names(list.size());
    for(size_t i=0; i<list.size(); i++)
      names[i] = list[i].name;

    if (not check_unique_strings(names))
      throw std::runtime_error("dimensions should have unique names");
  }

  dimension_description<precision> operator() (std::string const &name) const
  {
    for(size_t i=0; i<list.size(); i++)
    {
      if (list[i].name == name)
        return list[i];
    }
    throw std::runtime_error(std::string("invalid dimension name: '") + name + "', has not been defined.");
  }

  std::vector<dimension_description<precision>> const list;
};


enum class field_mode {
  evolution, closure
};

template<typename precision>
struct field_description
{
  field_description(field_mode fmode,
                    std::string const &dimension_name,
                    vector_func<precision> const initial_condition,
                    vector_func<precision> const exact_solution,
                    std::string const &field_name
                    )
    : field_description(fmode, std::vector<std::string>{dimension_name}, {initial_condition}, {exact_solution}, field_name)
    {}

  field_description(field_mode fmode,
                    std::string const &dimension_name,
                    vector_func<precision> const initial_condition,
                    std::string const &field_name
                    )
    : field_description(fmode, std::vector<std::string>{dimension_name}, {initial_condition}, field_name)
    {}

  field_description(field_mode fmode,
                    std::vector<std::string> const &dimension_names,
                    std::vector<vector_func<precision>> const &initial_conditions,
                    std::vector<vector_func<precision>> const &exact_solution,
                    std::string const &field_name
                    )
      : field_description(fmode,
                          std::vector<std::string>(dimension_names),
                          std::vector<vector_func<precision>>(initial_conditions),
                          std::vector<vector_func<precision>>(exact_solution),
                          std::string(field_name))
  {}
    field_description(field_mode fmode,
                      std::vector<std::string> const &dimension_names,
                      std::vector<vector_func<precision>> const &initial_conditions,
                      std::string const &field_name
                      )
      : field_description(fmode,
                          std::vector<std::string>(dimension_names),
                          std::vector<vector_func<precision>>(initial_conditions),
                          std::string(field_name))
  {}

  field_description(field_mode fmode,
                    std::vector<std::string> &&dimensions,
                    std::vector<vector_func<precision>> &&initial_conditions,
                    std::string &&field_name
                    )
      : field_description(fmode, std::move(dimensions), std::move(initial_conditions), {}, std::move(field_name))
  {}
  field_description(field_mode fmode,
                    std::vector<std::string> &&dimensions,
                    std::vector<vector_func<precision>> &&initial_conditions,
                    std::vector<vector_func<precision>> &&exact_solution,
                    std::string &&field_name
                    )
      : mode(fmode), d_names(std::move(dimensions)),
        init_cond(std::move(initial_conditions)), exact(std::move(exact_solution)),
        name(std::move(field_name))
  {
    static_assert(std::is_same<precision, float>::value
                  or std::is_same<precision, double>::value,
                  "ASGARD supports only float and double as template parameters for precision.");

    expect(d_names.size() > 0);
    expect(d_names.size() == init_cond.size());
    expect(exact.size() == 0 or d_names.size() == init_cond.size());
    if (not check_unique_strings(d_names))
      throw std::runtime_error("repeated dimensions in the field definition");
  }

  void verify_dimensions(dimension_set<precision> const &d_set) const
  {
    for(size_t i=0; i<d_names.size(); i++) {
      bool found = false;
      for(size_t j=0; j<d_set.list.size(); j++)
      {
        if (d_names[i] == d_set.list[j].name) {
          found = true;
          break;
        }
      }
      if (not found)
        throw std::runtime_error(std::string("invalid dimension name: '") + d_names[i] + "', has not been defined.");
    }
  }

  size_t num_dimensions() const { return d_names.size(); }
  bool has_exact_solution() const { return (exact.size() > 0); }

  field_mode mode;
  std::vector<std::string> const d_names;
  std::vector<vector_func<precision>> init_cond;
  std::vector<vector_func<precision>> exact;
  std::string const name;
};

template<typename precision>
struct field
{
  field(field_description<precision> const &description) :
    mode(description.mode), d_names(description.d_names),
    init_cond(description.init_cond), exact(description.exact),
    name(description.name),
    grid_index(-1), global_begin(-1), global_end(-1)
  {}

  void set_global_index(int64_t begin, int64_t end) {
    global_begin = begin;
    global_end   = end;
  }

  field_mode mode;
  std::vector<std::string> const d_names;
  std::vector<vector_func<precision>> init_cond;
  std::vector<vector_func<precision>> exact;
  std::string const name;

  int64_t grid_index, global_begin, global_end;
};

template<typename P>
static fk::vector<P>
eval_md_func(int const degree,
	     std::vector<dimension<P>> const &dims,
	     std::vector<std::vector<vector_func<P>>> const &md_funcs,
	     adapt::distributed_grid<P> const &grid,
	     basis::wavelet_transform<P,
	     resource::host> const &transformer,
	     P const time
	     )
{
  auto const my_subgrid = grid.get_subgrid(get_rank());
  // FIXME assume uniform degree
  auto const dof    = std::pow(degree, dims.size()) * my_subgrid.nrows();
  fk::vector<P> coeffs(dof);
  for (int i = 0; i<md_funcs.size(); ++i)
  {
    auto const coeff_vect = transform_and_combine_dimensions(
        dims, md_funcs[i], grid.get_table(), transformer,
        my_subgrid.row_start, my_subgrid.row_stop, degree, time,
        1.0); // TODO: Add time function to last argument
    fm::axpy(coeff_vect, coeffs);
  }
  return coeffs;
}

}
