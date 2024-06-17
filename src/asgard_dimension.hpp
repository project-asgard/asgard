#pragma once

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

#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "program_options.hpp"
#include "tools.hpp"

namespace asgard
{
template<typename P>
using vector_func = std::function<fk::vector<P>(fk::vector<P> const, P const)>;

template<typename P>
using md_func_type = std::vector<vector_func<P>>;

// same pi used by matlab
static constexpr double const PI = 3.141592653589793;

// for passing around vector/scalar-valued functions used by the PDE
template<typename P>
using scalar_func = std::function<P(P const)>;

template<typename P>
using g_func_type = std::function<P(P const, P const)>;

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
inline bool check_unique_strings(std::vector<std::string> const &names)
{
  size_t num_dims = names.size();
  for (size_t i = 0; i < num_dims; i++)
  {
    for (size_t j = i + 1; j < num_dims; j++)
    {
      if (names[i] == names[j])
        return false;
    }
  }
  return true;
}

/*!
 * \ingroup AsgardPDESystem
 * \brief Contains the user provided description of a dimension.
 *
 * The \b dimension_name is a user-provided unique identifier
 * that will help associate this dimension with various fields
 * and operators in the PDE system.
 */
template<typename precision>
struct dimension_description
{
  /*!
   * \brief Constructs a new dimension_description with the following parameters.
   *
   * \param domain_min is the lower bound for the domain in this dimension
   * \param domain_max is the upper bound for the domain in this dimension
   * \param grid_level is the default level of the sparse grid associated with this dimension
   * \param basis_degree is the default degree of the sparse grid basis associated with this dimension
   * \param dimension_name is a user-provided unique identifier for this dimension
   * \param volume_jacobian_dV is the Jacobian variable ... (TODO: I think this was supposed to change to an MD function)
   *
   * \par Several notes
   * - the level and degree can be modified using the command line
   * - the basis_degree uses dumb indexing, i.e., zeroth order is 1, first order
   * is 2 and so on (WILL BE FIXED!)
   */
  dimension_description(
      precision const domain_min, precision const domain_max,
      int const grid_level, int const basis_degree,
      std::string const dimension_name,
      g_func_type<precision> const volume_jacobian_dV = nullptr)
      : d_min(domain_min), d_max(domain_max), level(grid_level),
        degree(basis_degree), name(dimension_name), jacobian(volume_jacobian_dV)
  {
    // note that the constructor that is user facing will use the most
    // descriptive and clear names
    //      also the longest names for each variable/parameter
    //      the internal variables will use short-hand names
    expect(d_max > d_min);
    expect(level > 0);
    expect(degree > 1); // must fix this one, degree should count from zero
    expect(name.length() > 0);
  }
  //! \brief Dimension lower bound.
  precision const d_min;
  //! \brief Dimension upper bound.
  precision const d_max;
  //! \brief Dimension default level for the sparse grid.
  int const level;
  //! \brief Dimension default degree for the sparse grid basis.
  int const degree;
  //! \brief Unique user-provided identifier.
  std::string const name;
  //! \brief Volume Jacobian ...
  g_func_type<precision> const jacobian;
};

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
std::vector<dimension_description<precision>> cli_apply_level_degree_correction(
    parser const &cli_input,
    std::vector<dimension_description<precision>> const &dimensions)
{
  size_t num_dims = dimensions.size();
  std::vector<int> levels(dimensions.size()), degrees(dimensions.size());
  for (size_t i = 0; i < num_dims; i++)
  {
    levels[i]  = dimensions[i].level;
    degrees[i] = dimensions[i].degree;
  }

  // modify for appropriate level/degree
  // if default lev/degree not used
  auto const user_levels = cli_input.get_starting_levels().size();
  if (user_levels != 0 && user_levels != static_cast<int>(num_dims))
  {
    throw std::runtime_error(
        std::string(
            "failed to parse dimension-many starting levels - parsed ") +
        std::to_string(user_levels) + " levels");
  }
  if (user_levels == static_cast<int>(num_dims))
  {
    auto counter = 0;
    for (int &l : levels)
    {
      l = cli_input.get_starting_levels()(counter++);
      expect(l >= 0);
    }
  }
  auto const cli_degree = cli_input.get_degree();
  if (cli_degree != parser::NO_USER_VALUE)
  {
    expect(cli_degree > 0);
    for (int &d : degrees)
      d = cli_degree;
  }

  // check all dimensions
  for (size_t i = 0; i < dimensions.size(); i++)
  {
    expect(degrees[i] > 0);
    expect(levels[i] >= 0);
  }

  std::vector<dimension_description<precision>> result;
  result.reserve(num_dims);
  for (size_t i = 0; i < num_dims; i++)
  {
    result.push_back(dimension_description<precision>(
        dimensions[i].d_min, dimensions[i].d_max, levels[i], degrees[i],
        dimensions[i].name));
  }
  return result;
}

/*!
 * \internal
 * \ingroup AsgardPDESystem
 * \brief Wrapper for an arbitrary set of dimension_description objects.
 *
 * \tparam precision is either float or double
 *
 * Holds a vector of dimension_description objects and provides methods to
 * extract a specific description from the list.
 *
 * \endinternal
 */
template<typename precision>
struct dimension_set
{
  /*!
   * \brief Creates a new set of dimensions from the provided list modified by the command line arguments.
   *
   * \param cli_input is a parser of the command line arguments used to modify the default values in dimensions
   * \param dimensions is a list of dimensions with default data provided by the user
   *
   * \throws runtime_error if there are entries with the same name
   */
  dimension_set(parser const &cli_input,
                std::vector<dimension_description<precision>> const &dimensions)
      : list(cli_apply_level_degree_correction(cli_input, dimensions))
  {
    std::vector<std::string> names(list.size());
    for (size_t i = 0; i < list.size(); i++)
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
   *          the descriptions have already been updated with the command line
   * arguments.
   *
   * \throws runtime_error if the name is not in the list of dimensions
   */
  dimension_description<precision> const &
  operator()(std::string const &name) const
  {
    for (size_t i = 0; i < list.size(); i++)
    {
      if (list[i].name == name)
        return list[i];
    }
    throw std::runtime_error(std::string("invalid dimension name: '") + name +
                             "', has not been defined.");
  }

  //! \brief Contains the vector of dimension_description that has been updated by the command line arguments.
  std::vector<dimension_description<precision>> const list;
};

/*!
 * \internal
 * \ingroup AsgardPDESystem
 * \brief Extension to the dimension_description that contains internal Asgard data (also used in the old API).
 *
 * In addition to the parameters in the dimension_description,
 * this also holds the 1D mass matrix for this dimension.
 *
 * The class is also associated with the old/original API
 * and many of the internal operations work with a vector of dimensions.
 * For as long as the old API is maintained,
 * changes to this class should maintain compatibility.
 * The included \b initial_condition for example is used by the old API only,
 * and should not be used in the new one, since one dimension can be associated
 * with multiple fields each with different initial conditions.
 * \endinternal
 */
template<typename P>
struct dimension
{
  P domain_min;
  P domain_max;
  std::vector<vector_func<P>> initial_condition;
  g_func_type<P> volume_jacobian_dV;
  std::string name;
  dimension(P const d_min, P const d_max, int const level, int const degree,
            vector_func<P> const initial_condition_in,
            g_func_type<P> const volume_jacobian_dV_in,
            std::string const name_in)

      : dimension(d_min, d_max, level, degree,
                  std::vector<vector_func<P>>({initial_condition_in}),
                  volume_jacobian_dV_in, name_in)
  {}

  dimension(P const d_min, P const d_max, int const level, int const degree,
            std::vector<vector_func<P>> const initial_condition_in,
            g_func_type<P> const volume_jacobian_dV_in,
            std::string const name_in)

      : domain_min(d_min), domain_max(d_max),
        initial_condition(std::move(initial_condition_in)),
        volume_jacobian_dV(volume_jacobian_dV_in), name(name_in)
  {
    set_level(level);
    set_degree(degree);

    for (int i = 0; i <= level_; ++i)
    {
      auto const max_dof = fm::two_raised_to(i) * degree_;
      expect(max_dof < INT_MAX);
      this->mass_.push_back(eye<P>(max_dof));
    }
  }
  dimension(dimension_description<P> const desc)
      : domain_min(desc.d_min), domain_max(desc.d_max),
        initial_condition(std::vector<vector_func<P>>{
            [](fk::vector<P> const &, P const) -> fk::vector<P> {
              return fk::vector<P>();
            },
        }),
        volume_jacobian_dV(desc.jacobian), name(desc.name), level_(desc.level),
        degree_(desc.degree)
  {
    for (int i = 0; i <= level_; ++i)
    {
      auto const max_dof = fm::two_raised_to(i) * degree_;
      expect(max_dof < INT_MAX);
      this->mass_.push_back(eye<P>(max_dof));
    }
  }

  int get_level() const { return level_; }
  int get_degree() const { return degree_; }
  fk::matrix<P> const &get_mass_matrix() const
  {
    // default behavior is to get the mass matrix for the current level
    expect(level_ < static_cast<int>(this->mass_.size()));
    return mass_[level_];
  }
  fk::matrix<P> const &get_mass_matrix(int level) const
  {
    expect(level < static_cast<int>(this->mass_.size()));
    return mass_[level];
  }

  void set_level(int const level)
  {
    expect(level >= 0);
    level_ = level;
  }

  void set_degree(int const degree)
  {
    expect(degree > 0);
    degree_ = degree;
  }

  void set_mass_matrix(fk::matrix<P> &&new_mass, int level)
  {
    expect(level >= 0);

    if (level >= static_cast<int>(this->mass_.size()))
    {
      this->mass_.resize(level + 1);
    }
    this->mass_[level] = std::move(new_mass);
  }

  void set_mass_matrix(std::vector<fk::matrix<P>> const &new_mass)
  {
    this->mass_ = std::move(new_mass);
  }

  int level_;
  int degree_;
  std::vector<fk::matrix<P>> mass_;
};

} // namespace asgard
