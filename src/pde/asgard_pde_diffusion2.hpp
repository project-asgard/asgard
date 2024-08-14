#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// Example PDE using the 2D (1x-1y) Heat Equation. This example PDE is
// time dependent (although not all the terms are time dependent). This
// implies the need for an initial condition.
// PDE: df/dt = d^2 f/dx^2 + d^2 f/dy^2

template<typename P>
class PDE_diffusion_2d : public PDE<P>
{
public:
  PDE_diffusion_2d(prog_opts const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  static int constexpr num_dims_           = 2;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 2;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;
  static int constexpr default_level       = 3;
  static int constexpr default_degree      = 1;
  static int constexpr domain_min          = 0;
  static int constexpr domain_max          = 1;

  static fk::vector<P>
  initial_condition_dim(fk::vector<P> const &x, P const t = 0)
  {
    static double p     = -2.0 * PI * PI;
    P const coefficient = std::exp(p * t);

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [coefficient](P const x_value) -> P {
                     return coefficient * std::cos(PI * x_value);
                   });

    return fx;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 =
      dimension<P>(domain_min, domain_max, default_level, default_degree,
                   initial_condition_dim, nullptr, "x");

  inline static dimension<P> const dim_1 =
      dimension<P>(domain_min, domain_max, default_level, default_degree,
                   initial_condition_dim, nullptr, "y");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1};

  /* build the terms */
  inline static partial_term<P> const partial_term_I_ = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> partial_term_0 = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::downwind,
      boundary_condition::neumann, boundary_condition::neumann);

  static fk::vector<P> bc_func(fk::vector<P> const x, P const t)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const x_value) -> P { return std::cos(PI * x_value); });

    return fx;
  }

  static P bc_time_func(P const t)
  {
    /* e^(-2 * pi^2 * t )*/
    static double const p = -2.0 * PI * PI;
    return std::exp(p * t);
  }

  inline static const partial_term<P> partial_term_1 = partial_term<P>(
      coefficient_type::grad, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::inhomogeneous, homogeneity::inhomogeneous,
      {bc_func, bc_func}, bc_time_func, {bc_func, bc_func}, bc_time_func);

  inline static term<P> const term_0 =
      term<P>(false, // time-dependent
              "",    // name
              {partial_term_0, partial_term_1});

  inline static term<P> const term_1 =
      term<P>(false, // time-dependent
              "",    // name
              {partial_term_I_, partial_term_I_});

  inline static std::vector<term<P>> const terms_0 = {term_0, term_1};
  inline static std::vector<term<P>> const terms_1 = {term_1, term_0};

  inline static term_set<P> const terms_ = {terms_0, terms_1};

  /* exact solutions */
  static fk::vector<P> exact_solution(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(PI * x_v); });
    return fx;
  }

  static fk::vector<P> exact_time(fk::vector<P>, P const time)
  {
    constexpr P neg_two_pi_squared = static_cast<P>(-2.0 * PI * PI);
    return {
        std::exp(neg_two_pi_squared * time),
    };
  }

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution, exact_solution, exact_time};

  /* This is not used ever */
  static P exact_scalar_func(P const t)
  {
    static double neg_two_pi_squared = -2.0 * PI * PI;

    return std::exp(neg_two_pi_squared * t);
  }

  static P get_dt_(dimension<P> const &dim)
  {
    /* (1/2^level)^2 = 1/4^level */
    /* return dx; this will be scaled by CFL from command line */
    return std::pow(0.25, dim.get_level());
  }

  /* problem contains no sources */
  inline static std::vector<source<P>> const sources_ = {};
};
} // namespace asgard
