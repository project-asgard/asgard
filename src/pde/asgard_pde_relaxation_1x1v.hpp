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
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_, moments_,
               do_collision_operator_)
  {
    param_manager.add_parameter(parameter<P>{"n", n});
    param_manager.add_parameter(parameter<P>{"u", u});
    param_manager.add_parameter(parameter<P>{"theta", theta});
    param_manager.add_parameter(parameter<P>{"E", E});
    param_manager.add_parameter(parameter<P>{"S", S});
    param_manager.add_parameter(parameter<P>{"MaxAbsE", MaxAbsE});
  }

private:
  static int constexpr num_dims_          = 2;
  static int constexpr num_sources_       = 0;
  static int constexpr num_terms_         = 3;
  static bool constexpr do_poisson_solve_ = false;
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

  /* Define the moments */
  static fk::vector<P> moment0_f1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> f(x.size());
    std::fill(f.begin(), f.end(), 1.0);
    return f;
  }

  static fk::vector<P> moment1_f1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    return fk::vector<P>(x);
  }

  static fk::vector<P> moment2_f1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> f(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      f[i] = x[i] * x[i];
    }
    return f;
  }

  inline static moment_funcs<P> moments_ = {
      {{moment0_f1, moment0_f1, moment0_f1}},
      {{moment0_f1, moment1_f1, moment0_f1}},
      {{moment0_f1, moment2_f1, moment0_f1}}};

  /* Construct (n, u, theta) */
  static P n(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 1.0;
  }

  static P u(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  static P theta(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 1.0;
  }

  static P E(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  static P S(P const &y, P const t = 0)
  {
    ignore(t);
    // subtracts quadrature values by one
    return y - 1.0;
  }

  static P MaxAbsE(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  /* build the terms */

  // Constant Identity term

  inline static const partial_term<P> I_pterm = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const I_im =
      term<P>(false, // time-dependent
              "I",   // name
              {I_pterm}, imex_flag::imex_implicit);

  // Implcit Term 1
  // div_{v1} v_1 f
  //

  static P i1_g2(P const x, P const time = 0)
  {
    ignore(time);
    return nu * x;
  }

  inline static const partial_term<P> i1_pterm_v = partial_term<P>(
      coefficient_type::div, i1_g2, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i1v =
      term<P>(false,  // time-dependent
              "I1_v", // name
              {i1_pterm_v}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_im_1 = {I_im, term_i1v};

  // Implicit Term 2
  // d_{v1} -u_1 f
  //

  static P i2_g1(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("u");
    expect(param != nullptr);
    return -nu * param->value(x, time);
  }

  inline static const partial_term<P> i2_pterm_x = partial_term<P>(
      coefficient_type::mass, i2_g1, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i2_pterm_v = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i2x =
      term<P>(false,  // time-dependent
              "I2_x", // name
              {i2_pterm_x}, imex_flag::imex_implicit);

  inline static term<P> const term_i2v =
      term<P>(false,  // time-dependent
              "I2_v", // name
              {i2_pterm_v}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_im_2 = {term_i2x, term_i2v};

  // Implicit Term 3
  // d_{v1}(th q), q = d_{v1} f

  // Used in both terms 5 and 6
  static P i5_g2(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("theta");
    expect(param != nullptr);
    return param->value(x, time) * nu;
  }

  inline static const partial_term<P> i5_pterm_x1 = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i5_pterm_x2 = partial_term<P>(
      coefficient_type::mass, i5_g2, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term_i5x =
      term<P>(false,  // time-dependent
              "I3_x", // name
              {i5_pterm_x1, i5_pterm_x2}, imex_flag::imex_implicit);

  inline static const partial_term<P> i5_pterm_v1 = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static const partial_term<P> i5_pterm_v2 = partial_term<P>(
      coefficient_type::grad, nullptr, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i5v_diff =
      term<P>(false,  // time-dependent
              "I3_v", // name
              {i5_pterm_v1, i5_pterm_v2}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_im_3 = {term_i5x,
                                                         term_i5v_diff};

  inline static term_set<P> const terms_ = {terms_im_1, terms_im_2, terms_im_3};

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
