#pragma once

#include "asgard_fast_math.hpp"
#include "asgard_matlab_utilities.hpp"
#include "asgard_program_options.hpp"

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
      auto const max_dof = fm::ipow2(i) * (degree + 1);
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
    expect(degree >= 0);
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
