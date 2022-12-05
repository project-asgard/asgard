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
 * \brief Contains the user provided description of a dimension.
 *
 * The \b dimension_name will be used to identify when the dimension is used for each field.
 */
template<typename P>
struct dimension_description {
  dimension_description(P const domain_min, P const domain_max,
                        int const grid_level, int const basis_degree,
                        std::string const dimension_name,
                        g_func_type<P> const volume_jacobian_dV = [](P const, P const)->P{ return 1.0; }) :
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
  P const d_min;
  P const d_max;
  int const level;
  int const degree;
  std::string const name;
  g_func_type<P> const jacobian;
};

// Note (from Miro): there is nothing inside this class that requires internal consistency of the input data
// Why have private members when you can freely modify every one of them?
// Make a struct to keep related data together and have simple direct access to the data.
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
