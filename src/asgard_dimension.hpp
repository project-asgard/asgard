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

#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
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
 * \ingroup AsgardPDESystem
 * \brief Contains the user provided description of a dimension.
 *
 * The \b dimension_name is a user-provided unique identifier
 * that will help associate this dimension with various fields
 * and operators in the PDE system.
 */
template<typename precision>
struct dimension_description {
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
   * - the basis_degree uses dumb indexing, i.e., zeroth order is 1, first order is 2 and so on (WILL BE FIXED!)
   */
  dimension_description(precision const domain_min, precision const domain_max,
                        int const grid_level, int const basis_degree,
                        std::string const dimension_name,
                        g_func_type<precision> const volume_jacobian_dV =
                          [](precision const, precision const)->precision{ return 1.0; }) :
    d_min(domain_min), d_max(domain_max),
    level(grid_level), degree(basis_degree),
    name(dimension_name),
    jacobian(volume_jacobian_dV)
  {
    // note that the constructor that is user facing will use the most descriptive and clear names
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
  P const domain_min;
  P const domain_max;
  std::vector<vector_func<P>> const initial_condition;
  g_func_type<P> const volume_jacobian_dV;
  std::string const name;
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
  }
  dimension(dimension_description<P> const desc)
      : domain_min(desc.d_min), domain_max(desc.d_max),
        initial_condition(std::vector<vector_func<P>>{[](fk::vector<P> const&, P const)->fk::vector<float>{ return fk::vector<float>(); },}),
        volume_jacobian_dV(desc.jacobian), name(desc.name),
        level_(desc.level), degree_(desc.degree)
  {
    auto const max_dof =
        fm::two_raised_to(static_cast<int64_t>(level_)) *
        degree_;
    expect(max_dof < INT_MAX);
    this->mass_.clear_and_resize(max_dof, max_dof) = eye<P>(max_dof);
  }

  int get_level() const { return level_; }
  int get_degree() const { return degree_; }
  fk::matrix<P> const &get_mass_matrix() const { return mass_; }

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

  void set_mass_matrix(fk::matrix<P> const &new_mass)
  {
    this->mass_.clear_and_resize(new_mass.nrows(), new_mass.ncols()) = new_mass;
  }

  int level_;
  int degree_;
  fk::matrix<P> mass_;
};

}
