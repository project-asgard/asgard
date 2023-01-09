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
 * \ingroup AsgardPDESystem
 * \brief Defines the type of each field, specifically whether it has a time-differential operator or not.
 *
 * TODO: See ... Will add a link to the math document here ...
 */
enum class field_mode
{
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
 * A vector holding all of these objects will be passed into the constructor of
 * asgard::pde_system
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
  field_description(field_mode fmode, std::string const &dimension_name,
                    vector_func<precision> const initial_condition,
                    vector_func<precision> const exact_solution,
                    std::string const &field_name)
      : field_description(fmode, std::vector<std::string>{dimension_name},
                          {initial_condition}, {exact_solution}, field_name)
  {}

  /*!
   * \brief Constructs a new 1D field without a known exact solution and the following parameters.
   */
  field_description(field_mode fmode, std::string const &dimension_name,
                    vector_func<precision> const initial_condition,
                    std::string const &field_name)
      : field_description(fmode, std::vector<std::string>{dimension_name},
                          {initial_condition}, field_name)
  {}

  /*!
   * \brief Constructs a new multidimensional field with known exact solution and the following parameters.
   *
   * \param fmode identifies the field as either field_mode::evolution or field_mode::closure
   * \param dimension_names is a list of the dimensions associated with the field
   * \param initial_conditions provides initial condition for this field,
   *        note that the initial condition is a product of the 1D initial
   * conditions as per the separability assumption in the code
   * \param exact_solution defines the exact solution for the field (assumed separable),
   *        if the exact solution is not knows, then this can be an empty vector
   * \param field_name a unique identifier for this field
   *
   * \throws runtime_error if there is a mismatch in the sizes of dimension_names,
   *         initial_conditions and exact_solution or if dimension_names
   * contains repeated entries.
   */
  field_description(field_mode fmode, std::vector<std::string> dimensions,
                    std::vector<vector_func<precision>> initial_conditions,
                    std::vector<vector_func<precision>> exact_solution,
                    std::string field_name)
      : mode(fmode), d_names(std::move(dimensions)),
        init_cond(std::move(initial_conditions)),
        exact(std::move(exact_solution)), name(std::move(field_name))
  {
    static_assert(std::is_same<precision, float>::value or
                      std::is_same<precision, double>::value,
                  "ASGARD supports only float and double as template "
                  "parameters for precision.");

    expect(d_names.size() > 0);
    expect(d_names.size() == init_cond.size());
    expect(exact.size() == 0 or d_names.size() == init_cond.size());
    if (not check_unique_strings(d_names))
      throw std::runtime_error("repeated dimensions in the field definition");
  }
  /*!
   * \brief Constructs a new multidimensional field without known exact solution.
   */
  field_description(field_mode fmode, std::vector<std::string> dimension_names,
                    std::vector<vector_func<precision>> initial_conditions,
                    std::string field_name)
      : field_description(
            fmode, std::move(dimension_names), std::move(initial_conditions),
            std::vector<vector_func<precision>>{}, std::move(field_name))
  {}

  /*!
   * \brief Throws an error if the provided set does not contain all the dimensions associated with this field.
   *
   * \param d_set is a wrapper around an existing set of dimensions
   *
   * \throws runtime_error if any of the dimensions associated with this field are not present in the set.
   */
  void verify_dimensions(dimension_set<precision> const &d_set) const
  {
    for (size_t i = 0; i < d_names.size(); i++)
    {
      if (not std::any_of(d_set.list.begin(), d_set.list.end(),
                          [&](dimension_description<precision> const &d_name)
                              -> bool { return (d_names[i] == d_name.name); }))
        throw std::runtime_error(std::string("invalid dimension name: '") +
                                 d_names[i] + "', has not been defined.");
    }
  }

  //! \brief Returns the number of dimensions associated with this field.
  size_t num_dimensions() const { return d_names.size(); }
  //! \brief Returns \b true if this field has a user-provided exact solution.
  bool has_exact_solution() const { return !exact.empty(); }

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
 * The class is very similar to asgard::field_description but it also contains
 * internal data.
 *
 * - the index (withing the grid vector) of the grid that discretizes this field
 * - the start and end index of the field entries within the global state
 * vector.
 *
 * \endinternal
 */
template<typename precision>
struct field
{
  //! \brief Create a filed from the given description.
  field(field_description<precision> const &description)
      : mode(description.mode), d_names(description.d_names),
        init_cond(description.init_cond), exact(description.exact),
        name(description.name), grid_index(-1), global_begin(-1), global_end(-1)
  {}

  /*!
   * \brief Set the begin and end offsets, end is the next offset similar to std::vector::end.
   *
   * \param being is the first entry associated with the field within the global state vector.
   * \param end is the first entry not-associated with this field
   *
   * Note that the indexing follows the conventions set by the C++ standard
   * library, which is different from MATLAB or FORTRAN but better suited for
   * the zero-indexing of C++.
   */
  void set_global_index(int64_t begin, int64_t end)
  {
    expect(begin >= 0);
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
eval_md_func(int const degree, std::vector<dimension<P>> const &dims,
             std::vector<std::vector<vector_func<P>>> const &md_funcs,
             adapt::distributed_grid<P> const &grid,
             basis::wavelet_transform<P, resource::host> const &transformer,
             P const time)
{
  auto const my_subgrid = grid.get_subgrid(get_rank());
  // FIXME assume uniform degree
  auto const dof = std::pow(degree, dims.size()) * my_subgrid.nrows();
  fk::vector<P> coeffs(dof);
  for (auto const &funcs : md_funcs)
  {
    auto const coeff_vect = transform_and_combine_dimensions(
        dims, funcs, grid.get_table(), transformer, my_subgrid.row_start,
        my_subgrid.row_stop, degree, time,
        1.0); // TODO: Add time function to last argument
    fm::axpy(coeff_vect, coeffs);
  }
  return coeffs;
}

} // namespace asgard
