#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// Example PDE using the 1D Diffusion Equation. This example PDE is
// time dependent (although not all the terms are time dependent). This
// implies the need for an initial condition.
// PDE: df/dt = d^2 f/dx^2

template<typename P>
class PDE_diffusion_1d : public PDE<P>
{
public:
  PDE_diffusion_1d(prog_opts const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 1;
  static int constexpr num_terms_          = 3;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;

  static P constexpr nu = M_PI_2;

  static fk::vector<P>
  initial_condition_dim0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(nu * x_v); });
    return fx;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 =
      dimension<P>(0, 1, 3, 1, initial_condition_dim0, nullptr, "x");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0};

  /* Define terms */
  inline static const partial_term<P> partial_term_0 = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::neumann, boundary_condition::neumann);

  static fk::vector<P> bc_func(fk::vector<P> const x, P const t)
  {
    ignore(t);

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const x_v) -> P { return std::cos(nu * x_v); });

    return fx;
  }

  static P bc_time_func(P const t)
  {
    /* e^(-2 * nu^2 * t )*/
    static double const p = -2.0 * nu * nu;
    return std::exp(p * t);
  }

  // TODO: Add interior penalty terms?
  // TODO: update nu value, check initial conditions
  inline static const partial_term<P> partial_term_1 = partial_term<P>(
      coefficient_type::grad, nullptr, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::inhomogeneous, homogeneity::inhomogeneous, {bc_func},
      bc_time_func, {bc_func}, bc_time_func);

  inline static term<P> const term_0 =
      term<P>(true, // time-dependent
              "",   // name
              {partial_term_0, partial_term_1});

  // Interior penalty terms
  static P constexpr penalty = 0.0;
  static P g3(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return penalty;
  }
  static P g4(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return penalty;
  }
  inline static const partial_term<P> partial_term_2 = partial_term<P>(
      coefficient_type::div, g3, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, nullptr);

  inline static term<P> const term_1 = term<P>(false, // time-dependent
                                               "",    // name
                                               {partial_term_2});

  inline static const partial_term<P> partial_term_3 = partial_term<P>(
      coefficient_type::div, g4, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, nullptr);

  inline static term<P> const term_2 = term<P>(false, // time-dependent
                                               "",    // name
                                               {partial_term_3});

  inline static std::vector<term<P>> const terms_0 = {term_0};
  inline static std::vector<term<P>> const terms_1 = {term_1};
  inline static std::vector<term<P>> const terms_2 = {term_2};
  inline static term_set<P> const terms_ = {terms_0, terms_1, terms_2};

  /* Create sources */
  static fk::vector<P> source_0_x(fk::vector<P> const x, P const t)
  {
    ignore(t);
    static double const coefficient = -1.0 * nu * nu;

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return coefficient * std::cos(nu * x_v);
    });

    return fx;
  }

  static P source_0_t(P const t)
  {
    static double const coefficient = -2.0 * nu * nu;

    return std::exp(coefficient * t);
  }

  inline static source<P> const source_0 = source<P>({source_0_x}, source_0_t);

  inline static std::vector<source<P>> const sources_ = {source_0};

  /* exact solutions */
  static fk::vector<P> exact_solution_0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(nu * x_v); });
    return fx;
  }

  static fk::vector<P> exact_time(fk::vector<P>, P const time)
  {
    return {source_0_t(time),};
  }

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution_0, exact_time};

  static P get_dt_(dimension<P> const &dim)
  {
    /* (1/2^level)^2 = 1/4^level */
    /* return dx; this will be scaled by CFL from command line */
    return std::pow(0.25, dim.get_level());
  }
};
} // namespace asgard
