#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// 3D test case using relaxation problem
//
//  df/dt == div_v( (v-u(x))f + theta(x)\grad_v f)
//
//  where the domain is (x,v_1,v_2).  The moments of f are constant x.
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_relaxation_1x1v : public PDE<P>
{
public:
  PDE_relaxation_1x1v(prog_opts const &cli_input)
  {
    this->skip_old_moments = true; // temp-hack

    this->initialize(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
                     terms_, sources_, exact_vector_funcs_, get_dt_,
                     has_analytic_soln_, moment_funcs<P>{}, do_collision_operator_);
  }

private:
  static int constexpr num_dims_    = 2;
  static int constexpr num_sources_ = 0;
  static int constexpr num_terms_   = 4;
  // disable implicit steps in IMEX
  static bool constexpr do_collision_operator_ = true;
  static bool constexpr has_analytic_soln_     = true;
  static int constexpr default_degree          = 2;

  static P constexpr nu = 1e3;

  // IC 2 maxwellians
  static fk::vector<P>
  initial_condition_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::fill(fx.begin(), fx.end(), P{0.5});
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_x_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::fill(fx.begin(), fx.end(), P{0.5});
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_0_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    P constexpr theta   = 0.5;
    P constexpr ux      = -1.0;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      fx[i] =
          coefficient * std::exp(-(0.5 / theta) * (x[i] - ux) * (x[i] - ux));
    }
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_0_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    P constexpr theta   = 0.5;
    P constexpr ux      = 2.0;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      fx[i] =
          coefficient * std::exp(-(0.5 / theta) * (x[i] - ux) * (x[i] - ux));
    }
    return fx;
  }

  inline static dimension<P> const dim_0 = dimension<P>(
      -0.5, 0.5, 4, default_degree,
      {initial_condition_dim_x_0, initial_condition_dim_x_1}, nullptr, "x");

  inline static dimension<P> const dim_1 =
      dimension<P>(-8.0, 12.0, 3, default_degree,
                   {initial_condition_dim_v_0_0, initial_condition_dim_v_0_1},
                   nullptr, "v1");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1};

  // Term 3
  // v\cdot\grad_v f
  //
  static P const_nu(P const, P const = 0) { return nu; }

  static P get_v(P const v, P const = 0) { return v; }

  inline static const partial_term<P> pt_mass_nu = partial_term<P>(
      coefficient_type::mass, const_nu, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> pt_divv = partial_term<P>(
      coefficient_type::div, get_v, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const mass_nu =
      term<P>(false,  // time-dependent
              "I1_x", // name
              {pt_mass_nu}, imex_flag::imex_implicit);

  inline static term<P> const divv =
      term<P>(false,  // time-dependent
              "I1_v", // name
              {pt_divv}, imex_flag::imex_implicit);

  // moment components of the collision operator, split into 3 parts
  // see landau 1x-1v example

  inline static const partial_term<P> pt_mass_uf = partial_term<P>(
      coefficient_type::mass, pterm_dependence::moments_1by0);

  inline static const partial_term<P> pt_mass_uf_neg = partial_term<P>(
      coefficient_type::mass, pterm_dependence::moments_1by0_neg);

  inline static const partial_term<P> pt_mass_ef = partial_term<P>(
      coefficient_type::mass, pterm_dependence::moments_2by0);

  inline static const partial_term<P> pt_vdivf = partial_term<P>(
      coefficient_type::div, const_nu, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static const partial_term<P> pt_div_up = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static const partial_term<P> pt_nu_grad_down = partial_term<P>(
      coefficient_type::grad, const_nu, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const mass_uf_neg =
      term<P>(true,   // time-dependent
              "I",    // name
              {pt_mass_uf_neg, }, imex_flag::imex_implicit);

  inline static term<P> const mass_u2_neg =
      term<P>(true,  // time-dependent
              "I",   // name
              {pt_mass_uf, pt_mass_uf_neg}, imex_flag::imex_implicit);

  inline static term<P> const mass_ef =
      term<P>(true,   // time-dependent
              "I2_x", // name
              {/* identity, */ pt_mass_ef, }, imex_flag::imex_implicit);

  inline static term<P> const vdivf =
      term<P>(false,  // time-dependent
              "I2_v", // name
              {pt_vdivf,}, imex_flag::imex_implicit);

  inline static term<P> const nu_div_grad =
      term<P>(false,  // time-dependent
              "nu_div_grad", // name
              {pt_div_up, pt_nu_grad_down}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const term_dv = {mass_nu, divv};
  inline static std::vector<term<P>> const term_uf = {mass_uf_neg, vdivf};
  inline static std::vector<term<P>> const term_t1 = {mass_ef, nu_div_grad};
  inline static std::vector<term<P>> const term_t2 = {mass_u2_neg, nu_div_grad};

  inline static term_set<P> const terms_ = {term_dv, term_uf, term_t1, term_t2};

  static fk::vector<P> exact_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::fill(fx.begin(), fx.end(), P{1.0});
    return fx;
  }

  // Analytic solution
  static fk::vector<P> exact_dim_v_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    P constexpr theta   = 2.75;
    P constexpr ux      = 0.5;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      fx[i] =
          coefficient * std::exp(-(0.5 / theta) * (x[i] - ux) * (x[i] - ux));
    }
    return fx;
  }

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_dim_x_0, exact_dim_v_0};

  static P get_dt_(dimension<P> const &dim)
  {
    ignore(dim);
    /* return dx; this will be scaled by CFL from command line */
    // return std::pow(0.25, dim.get_level());

    // TODO: these are constants since we want dt always based on dim 2,
    //  but there is no way to force a different dim for this function!
    // (Lmax - Lmin) / 2 ^ LevX * CFL
    return (6.0 - (-6.0)) / fm::ipow2(3);
  }

  /* problem contains no sources */
  inline static std::vector<source<P>> const sources_ = {};
};

} // namespace asgard
