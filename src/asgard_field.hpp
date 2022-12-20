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
 * \internal
 * \ingroup AsgardPDESystem
 * \brief Creates a copy of the dimensions and modifies them based on the cli-parameters
 *
 * \tparam precision is either \b float or \b double
 * \param cli_input is a parser with the current command line arguments
 * \param dimensions is a user-provided default values for the dimensions of the PDE system
 *
 * \returns a copy of the dimensions, but with the command line corrections applied
 *
 * \throws runtime_error if sanity check fails on the parser data or the values of the levels and order
 *
 * \endinternal
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
 * \internal
 * \ingroup AsgardPDESystem
 * \brief Throws an exception if there are repeated entries among the names.
 *
 * \param names is a list of strings that need a sanity check
 *
 * \return \b true if there are no unique entries among the strings, and \b false if repeated entries are found
 *
 * \endinternal
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

/*!
 * \internal
 * \ingroup AsgardPDESystem
 * \brief Wrapper for an arbitrary set of dimension_description objects.
 *
 * \tparam precision is either float or double
 *
 * Holds a vector of dimension_description objects and provides methods to extract a specific description from the list.
 *
 * \endinternal
 */
template<typename precision>
struct dimension_set {
  /*!
   * \brief Creates a new set of dimensions from the provided list modified by the command line arguments.
   *
   * \param cli_input is a parser of the command line arguments used to modify the default values in dimensions
   * \param dimensions is a list of dimensions with default data provided by the user
   *
   * \throws runtime_error if there are entries with the same name
   */
  dimension_set(parser const &cli_input, std::vector<dimension_description<precision>> const &dimensions)
    : list(cli_apply_level_degree_correction(cli_input, dimensions))
  {
    std::vector<std::string> names(list.size());
    for(size_t i=0; i<list.size(); i++)
      names[i] = list[i].name;

    if (not check_unique_strings(names))
      throw std::runtime_error("dimensions should have unique names");
  }

  /*!
   * \brief Returns the dimension_description for the dimension with the given name.
   *
   * \param name is the name to search among the dimensions in the set
   *
   * \returns const-reference to the dimension_description with the same name,
   *          the descriptions have already been updated with the command line arguments.
   *
   * \throws runtime_error if the name is not in the list of dimensions
   */
  dimension_description<precision> const& operator() (std::string const &name) const
  {
    for(size_t i=0; i<list.size(); i++)
    {
      if (list[i].name == name)
        return list[i];
    }
    throw std::runtime_error(std::string("invalid dimension name: '") + name + "', has not been defined.");
  }

  //! \brief Contains the vector of dimension_description that has been updated by the command line arguments.
  std::vector<dimension_description<precision>> const list;
};

/*!
 * \ingroup AsgardPDESystem
 * \brief Defines the type of each field, specifically whether it has a time-differential operator or not.
 *
 * TODO: See ... Will add a link to the math document here ...
 */
enum class field_mode {
  //! \brief Each evolution equation will have an implicit time derivative operator added its equation.
  evolution,
  //! \brief Fields with no time derivative, i.e., fields that depend explicitly on the field_mode::evolution fields.
  closure
};

/*!
 * \ingroup AsgardPDESystem
 * \brief Simple container holding the basic properties of a field.
 *
 * The user has to crate a description object for each field of the PDE system.
 * A vector holding all of these objects will be passed into the constructor of asgard::pde_system
 *
 * \tparam precision is either \b float or \b double and indicates whether to use 32-bit or 64-bit floating point precision
 */
template<typename precision>
struct field_description
{
  /*!
   * \brief Constructs a new 1D field with known exact solution and the following parameters.
   *
   * \param fmode identifies the field as either field_mode::evolution or field_mode::closure
   * \param dimension_name is the name of the single dimension associated with this field
   * \param initial_condition provides the initial condition for this field
   * \param exact_solution is an optional parameter that defines the exact solution for the field
   * \param field_name a unique identifier for this field
   */
  field_description(field_mode fmode,
                    std::string const &dimension_name,
                    vector_func<precision> const initial_condition,
                    vector_func<precision> const exact_solution,
                    std::string const &field_name
                    )
    : field_description(fmode, std::vector<std::string>{dimension_name}, {initial_condition}, {exact_solution}, field_name)
    {}

  /*!
   * \brief Constructs a new 1D field without a known exact solution and the following parameters.
   */
  field_description(field_mode fmode,
                    std::string const &dimension_name,
                    vector_func<precision> const initial_condition,
                    std::string const &field_name
                    )
    : field_description(fmode, std::vector<std::string>{dimension_name}, {initial_condition}, field_name)
    {}

  /*!
   * \brief Constructs a new multidimensional field with known exact solution and the following parameters.
   *
   * \param fmode identifies the field as either field_mode::evolution or field_mode::closure
   * \param dimension_names is a list of the dimensions associated with the field
   * \param initial_conditions provides initial condition for this field,
   *        note that the initial condition is a product of the 1D initial conditions
   *        as per the separability assumption in the code
   * \param exact_solution defines the exact solution for the field (assumed separable),
   *        if the exact solution is not knows, then this can be an empty vector
   * \param field_name a unique identifier for this field
   *
   * \throws runtime_error if there is a mismatch in the sizes of dimension_names,
   *         initial_conditions and exact_solution or if dimension_names contains repeated entries.
   */
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
  /*!
   * \brief Constructs a new multidimensional field without known exact solution.
   */
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
  //! \brief Overload for r-value inputs, used to avoid a redundant copy.
  field_description(field_mode fmode,
                    std::vector<std::string> &&dimensions,
                    std::vector<vector_func<precision>> &&initial_conditions,
                    std::string &&field_name
                    )
      : field_description(fmode, std::move(dimensions), std::move(initial_conditions), {}, std::move(field_name))
  {}
  //! \brief Overload for r-value inputs, used to avoid a redundant copy.
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

  /*!
   * \brief Throws an error if the provided set does not contain all the dimensions associated with this field.
   *
   * \param d_set is a wrapper around an existing set of dimensions
   *
   * \throws runtime_error if any of the dimensions associated with this field are not present in the set.
   */
  void verify_dimensions(dimension_set<precision> const &d_set) const
  {
    for(size_t i=0; i<d_names.size(); i++) {
      if (not std::any_of(d_set.list.begin(), d_set.list.end(),
        [&](dimension_description<precision> const &d_name)->bool{ return (d_names[i] == d_name.name); }))
        throw std::runtime_error(std::string("invalid dimension name: '") + d_names[i] + "', has not been defined.");
    }
  }

  //! \brief Returns the number of dimensions associated with this field.
  size_t num_dimensions() const { return d_names.size(); }
  //! \brief Returns \b true if this field has a user-provided exact solution.
  bool has_exact_solution() const { return (exact.size() > 0); }

  //! \brief Stores the mode of the field.
  field_mode const mode;
  //! \brief Stores the names of the associated dimensions.
  std::vector<std::string> const d_names;
  //! \brief Stores the initial conditions.
  std::vector<vector_func<precision>> init_cond;
  //! \brief Stores the exact solutions, if any have been provided.
  std::vector<vector_func<precision>> exact;
  //! \brief Stores the user-provided unique identifier.
  std::string const name;
};

/*!
 * \internal
 * \ingroup AsgardPDESystem
 * \brief Extension to the field_description that also stores internal asgard data.
 *
 * The class is very similar to asgard::field_description but it also contains internal data.
 *
 * - the index (withing the grid vector) of the grid that discretizes this field
 * - the start and end index of the field entries within the global state vector.
 *
 * \endinternal
 */
template<typename precision>
struct field
{
  //! \brief Create a filed from the given description.
  field(field_description<precision> const &description) :
    mode(description.mode), d_names(description.d_names),
    init_cond(description.init_cond), exact(description.exact),
    name(description.name),
    grid_index(-1), global_begin(-1), global_end(-1)
  {}

  /*!
   * \brief Set the begin and end offsets, end is the next offset similar to std::vector::end.
   *
   * \param being is the first entry associated with the field within the global state vector.
   * \param end is the first entry not-associated with this field
   *
   * Note that the indexing follows the conventions set by the C++ standard library,
   * which is different from MATLAB or FORTRAN but better suited for the zero-indexing of C++.
   */
  void set_global_index(int64_t begin, int64_t end) {
    expect(end >= begin);
    global_begin = begin;
    global_end   = end;
  }

  //! \brief Stores the mode of the field.
  field_mode const mode;
  //! \brief Stores the names of the associated dimensions.
  std::vector<std::string> const d_names;
  //! \brief Stores the initial conditions.
  std::vector<vector_func<precision>> init_cond;
  //! \brief Stores the exact solutions, if any have been provided.
  std::vector<vector_func<precision>> exact;
  //! \brief Stores the user-provided unique identifier.
  std::string const name;

  //! \brief The index of the grid that discretizes this field.
  int64_t grid_index;
  //! \brief The index of the first entry associated with the field.
  int64_t global_begin;
  //! \brief The index of the first entry not-associated with the field.
  int64_t global_end;
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
