#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// 2D test case using continuity equation, i.e.,
//
//  df/dt == -v*\grad_x f + div_v( (v-u)f + theta\grad_v f)
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_vlasov_lb : public PDE<P>
{
public:
  PDE_vlasov_lb(prog_opts const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_, moments_)
  {
    param_manager.add_parameter(parameter<P>{"n", n});
    param_manager.add_parameter(parameter<P>{"u", u});
    param_manager.add_parameter(parameter<P>{"theta", theta});
  }

private:
  static int constexpr num_dims_           = 2;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 5;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = false;
  static int constexpr default_degree      = 3;

  static P constexpr nu = 1e3;

  static fk::vector<P>
  initial_condition_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return (std::abs(x_v) > 0.5) ? 1.0 : 0.0;
    });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_x_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return (std::abs(x_v) <= 0.5) ? 1.0 : 0.0;
    });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = 1.0 / std::sqrt(2.0 * PI);

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [coefficient](P const x_v) -> P {
                     return coefficient * std::exp(-std::pow(x_v, 2) / 2.0);
                   });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = (1.0 / 8.0) / std::sqrt(2.0 * PI * (4.0 / 5.0));

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [coefficient](P const x_v) -> P {
                     return coefficient *
                            std::exp(-std::pow(x_v, 2) / (2.0 * (4.0 / 5.0)));
                   });
    return fx;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 = dimension<P>(
      -1.0, 1.0, 4, default_degree,
      {initial_condition_dim_x_0, initial_condition_dim_x_1}, nullptr, "x");

  inline static dimension<P> const dim_1 = dimension<P>(
      -6.0, 6.0, 3, default_degree,
      {initial_condition_dim_v_0, initial_condition_dim_v_1}, nullptr, "v");

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
    std::transform(x.begin(), x.end(), f.begin(),
                   [](P const &x_v) -> P { return std::pow(x_v, 2); });
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

    P const first  = x < -0.5 ? 1.0 : 0.0;
    P const second = (x >= -0.5 && x <= 0.5) ? (1.0 / 8.0) : 0.0;
    P const third  = x > 0.5 ? 1.0 : 0.0;
    return first + second + third;
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

    P const first  = x < -0.5 ? 1.0 : 0.0;
    P const second = (x >= -0.5 && x <= 0.5) ? (4.0 / 5.0) : 0.0;
    P const third  = x > 0.5 ? 1.0 : 0.0;
    return first + second + third;
  }

  /* build the terms */

  // Term 1
  // -v\cdot\grad_x f for v > 0
  //
  static P e1_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }

  static P e1_g2(P const x, P const time = 0)
  {
    ignore(time);
    return (x > 0.0) ? x : 0.0;
  }

  inline static const partial_term<P> e1_pterm_x = partial_term<P>(
      coefficient_type::div, e1_g1, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e1_pterm_v = partial_term<P>(
      coefficient_type::mass, e1_g2, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term_e1x =
      term<P>(false,  // time-dependent
              "E1_x", // name
              {e1_pterm_x}, imex_flag::imex_explicit);

  inline static term<P> const term_e1v =
      term<P>(false,  // time-dependent
              "E1_v", // name
              {e1_pterm_v}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_1 = {term_e1x, term_e1v};

  // Term 2
  // -v\cdot\grad_x f for v < 0
  //
  static P e2_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }

  static P e2_g2(P const x, P const time = 0)
  {
    ignore(time);
    return (x < 0.0) ? x : 0.0;
  }

  inline static const partial_term<P> e2_pterm_x = partial_term<P>(
      coefficient_type::div, e2_g1, nullptr, flux_type::downwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e2_pterm_v = partial_term<P>(
      coefficient_type::mass, e2_g2, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term_e2x =
      term<P>(false,  // time-dependent
              "E2_x", // name
              {e2_pterm_x}, imex_flag::imex_explicit);

  inline static term<P> const term_e2v =
      term<P>(false,  // time-dependent
              "E2_v", // name
              {e2_pterm_v}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_2 = {term_e2x, term_e2v};

  // Term 3
  // v\cdot\grad_v f
  //
  static P i1_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return nu;
  }

  static P i1_g2(P const x, P const time = 0)
  {
    ignore(time);
    return x;
  }

  inline static const partial_term<P> i1_pterm_x = partial_term<P>(
      coefficient_type::mass, i1_g1, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i1_pterm_v = partial_term<P>(
      coefficient_type::div, i1_g2, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i1x =
      term<P>(false,  // time-dependent
              "I1_x", // name
              {i1_pterm_x}, imex_flag::imex_implicit);

  inline static term<P> const term_i1v =
      term<P>(false,  // time-dependent
              "I1_v", // name
              {i1_pterm_v}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_3 = {term_i1x, term_i1v};

  // Term 4
  // -u\cdot\grad_v f
  //
  static P i2_g1(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("u");
    expect(param != nullptr);
    return -param->value(x, time);
  }

  static P i2_g2(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return nu;
  }

  inline static const partial_term<P> i2_pterm_x = partial_term<P>(
      coefficient_type::mass, i2_g1, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i2_pterm_v = partial_term<P>(
      coefficient_type::div, i2_g2, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i2x =
      term<P>(false,  // time-dependent
              "I2_x", // name
              {i2_pterm_x}, imex_flag::imex_implicit);

  inline static term<P> const term_i2v =
      term<P>(false,  // time-dependent
              "I2_v", // name
              {i2_pterm_v}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_4 = {term_i2x, term_i2v};

  // Term 5
  // div_v(th\grad_v f)
  //
  // Split by LDG
  //
  // div_v(th q)
  // q = \grad_v f

  static P i3_g2(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("theta");
    expect(param != nullptr);
    return param->value(x, time) * nu;
  }

  inline static const partial_term<P> i3_pterm_x1 = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i3_pterm_x2 = partial_term<P>(
      coefficient_type::mass, i3_g2, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term_i3x =
      term<P>(false,  // time-dependent
              "I3_x", // name
              {i3_pterm_x1, i3_pterm_x2}, imex_flag::imex_implicit);

  inline static const partial_term<P> i3_pterm_v1 = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static const partial_term<P> i3_pterm_v2 = partial_term<P>(
      coefficient_type::grad, nullptr, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i3v =
      term<P>(false,  // time-dependent
              "I3_v", // name
              {i3_pterm_v1, i3_pterm_v2}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_5 = {term_i3x, term_i3v};

  inline static term_set<P> const terms_ = {terms_1, terms_2, terms_3, terms_4,
                                            terms_5};

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {};

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
