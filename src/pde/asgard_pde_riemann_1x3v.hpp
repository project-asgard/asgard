#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// 3D test case using relaxation problem
//
//  df/dt == div_v( (v-u(x))f + theta(x)\grad_v f)
//
//  where the domain is (x,v_x,v_y,v_z).  The moments of f are constant x.
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_riemann_1x3v : public PDE<P>
{
public:
  PDE_riemann_1x3v(prog_opts const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_, moments_,
               do_collision_operator_)
  {
    param_manager.add_parameter(parameter<P>{"n", n});
    param_manager.add_parameter(parameter<P>{"u", u});
    param_manager.add_parameter(parameter<P>{"u2", u2});
    param_manager.add_parameter(parameter<P>{"u3", u3});
    param_manager.add_parameter(parameter<P>{"theta", theta});
    param_manager.add_parameter(parameter<P>{"E", E});
    param_manager.add_parameter(parameter<P>{"S", S});
    param_manager.add_parameter(parameter<P>{"MaxAbsE", MaxAbsE});
  }

private:
  static int constexpr num_dims_          = 4;
  static int constexpr num_sources_       = 0;
  static int constexpr num_terms_         = 11;
  static bool constexpr do_poisson_solve_ = false;
  // disable implicit steps in IMEX
  static bool constexpr do_collision_operator_ = true;
  static bool constexpr has_analytic_soln_     = false;
  static int constexpr default_degree          = 2;

  static P constexpr nu = 1e3;

  static fk::vector<P>
  initial_condition_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return (std::abs(x_v) > 0.3) ? 1.0 : 0.0;
    });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_x_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return (std::abs(x_v) <= 0.3) ? (1.0 / 8.0) : 0.0;
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

    P const coefficient = 1.0 / std::sqrt(2.0 * PI * (4.0 / 5.0));

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [coefficient](P const x_v) -> P {
                     return coefficient *
                            std::exp(-std::pow(x_v, 2) / (2.0 * (4.0 / 5.0)));
                   });
    return fx;
  }

  inline static dimension<P> const dim_0 = dimension<P>(
      -0.6, 0.6, 7, default_degree,
      {initial_condition_dim_x_0, initial_condition_dim_x_1}, nullptr, "x");

  inline static dimension<P> const dim_1 = dimension<P>(
      -6.0, 6.0, 3, default_degree,
      {initial_condition_dim_v_0, initial_condition_dim_v_1}, nullptr, "v1");

  inline static dimension<P> const dim_2 = dimension<P>(
      -6.0, 6.0, 3, default_degree,
      {initial_condition_dim_v_0, initial_condition_dim_v_1}, nullptr, "v2");

  inline static dimension<P> const dim_3 = dimension<P>(
      -6.0, 6.0, 3, default_degree,
      {initial_condition_dim_v_0, initial_condition_dim_v_1}, nullptr, "v3");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1,
                                                               dim_2, dim_3};

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

  inline static moment_funcs<P> const moments_ = {
      {{moment0_f1, moment0_f1, moment0_f1, moment0_f1, moment0_f1}},
      {{moment0_f1, moment1_f1, moment0_f1, moment0_f1, moment0_f1}},
      {{moment0_f1, moment0_f1, moment1_f1, moment0_f1, moment0_f1}},
      {{moment0_f1, moment0_f1, moment0_f1, moment1_f1, moment0_f1}},
      {{moment0_f1, moment2_f1, moment0_f1, moment0_f1, moment0_f1}},
      {{moment0_f1, moment0_f1, moment2_f1, moment0_f1, moment0_f1}},
      {{moment0_f1, moment0_f1, moment0_f1, moment2_f1, moment0_f1}}};

  /* Construct (n, u, theta) */
  static P n(P const &x, P const t = 0)
  {
    ignore(t);
    P const first  = std::abs(x) > 0.3 ? 1.0 : 0.0;
    P const second = std::abs(x) < 0.3 ? (1.0 / 8.0) : 0.0;
    return first + second;
  }

  static P u(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  static P u2(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  static P u3(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  static P theta(P const &x, P const t = 0)
  {
    ignore(t);
    P const first  = std::abs(x) > 0.3 ? 1.0 : 0.0;
    P const second = std::abs(x) < 0.3 ? (4.0 / 5.0) : 0.0;
    return first + second;
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

  // ###############################
  // #### Explicit Terms ###########
  // ###############################

  // Constant Explicit Identity term

  inline static const partial_term<P> I_pterm_ex = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const I_ex =
      term<P>(false, // time-dependent
              "I",   // name
              {I_pterm_ex}, imex_flag::imex_explicit);

  // Explicit Term 1
  // -v_x\cdot\grad_x f for v_x > 0
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
    return std::max(P{0.0}, x);
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

  inline static std::vector<term<P>> const terms_ex_1 = {term_e1x, term_e1v,
                                                         I_ex, I_ex};

  // Explicit Term 2
  // -v_x\cdot\grad_x f for v_x < 0
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
    return std::min(P{0.0}, x);
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

  inline static std::vector<term<P>> const terms_ex_2 = {term_e2x, term_e2v,
                                                         I_ex, I_ex};

  // ###############################
  // #### Implicit Terms ###########
  // ###############################

  // Constant Implicit Identity term

  inline static const partial_term<P> I_pterm = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const I_im =
      term<P>(false, // time-dependent
              "I",   // name
              {I_pterm}, imex_flag::imex_implicit);

  // Implcit Term 1
  // div_{v_x} v_x f
  //

  static P nu_v_func(P const x, P const time = 0)
  {
    ignore(time);
    return nu * x;
  }

  inline static const partial_term<P> nu_v_pterm = partial_term<P>(
      coefficient_type::div, nu_v_func, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const nu_v_term =
      term<P>(false,  // time-dependent
              "I1_v", // name
              {nu_v_pterm}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_im_1 = {I_im, nu_v_term, I_im,
                                                         I_im};

  // Implicit Term 2
  // d_{v_x} -u_x f
  //

  static P nu_ux_func(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("u");
    expect(param != nullptr);
    return -nu * param->value(x, time);
  }

  inline static const partial_term<P> nu_ux_pterm = partial_term<P>(
      coefficient_type::mass, nu_ux_func, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> dv_pterm = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const nu_ux_term =
      term<P>(false,  // time-dependent
              "I2_x", // name
              {nu_ux_pterm}, imex_flag::imex_implicit);

  inline static term<P> const dv_term =
      term<P>(false,  // time-dependent
              "I2_v", // name
              {dv_pterm}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_im_2 = {nu_ux_term, dv_term,
                                                         I_im, I_im};

  // Implcit Term 3
  // div_{v_y} v_y f
  //

  // single dimension terms were already created in Implicit Term 1

  inline static std::vector<term<P>> const terms_im_3 = {I_im, I_im, nu_v_term,
                                                         I_im};

  // Implicit Term 4
  // d_{v_y} -u_y f
  //

  static P nu_uy_func(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("u2");
    expect(param != nullptr);
    return -nu * param->value(x, time);
  }

  inline static const partial_term<P> nu_uy_pterm = partial_term<P>(
      coefficient_type::mass, nu_uy_func, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const nu_uy_term =
      term<P>(false,  // time-dependent
              "I3_x", // name
              {nu_uy_pterm}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_im_4 = {nu_uy_term, I_im,
                                                         dv_term, I_im};

  // Implcit Term 5
  // div_{v_z} v_z f
  //

  // single dimension terms were already created in Implicit Term 1

  inline static std::vector<term<P>> const terms_im_5 = {I_im, I_im, I_im,
                                                         nu_v_term};

  // Implicit Term 6
  // d_{v_z} -u_z f
  //

  static P nu_uz_func(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("u3");
    expect(param != nullptr);
    return -nu * param->value(x, time);
  }

  inline static const partial_term<P> nu_uz_pterm = partial_term<P>(
      coefficient_type::mass, nu_uz_func, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const nu_uz_term =
      term<P>(false,  // time-dependent
              "I3_x", // name
              {nu_uz_pterm}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_im_6 = {nu_uz_term, I_im, I_im,
                                                         dv_term};

  // Term 7
  // d_{v_x}(th q), q = d_{v_x} f

  // Used in all 3 diffusion terms
  static P nu_theta(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("theta");
    expect(param != nullptr);
    return param->value(x, time) * nu;
  }

  inline static const partial_term<P> nu_theta_pterm = partial_term<P>(
      coefficient_type::mass, nu_theta, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const nu_theta_term =
      term<P>(false,  // time-dependent
              "I3_x", // name
              {I_pterm, nu_theta_pterm}, imex_flag::imex_implicit);

  inline static const partial_term<P> i5_pterm_v1 = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static const partial_term<P> i5_pterm_v2 = partial_term<P>(
      coefficient_type::grad, nullptr, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const diff_v_term =
      term<P>(false,  // time-dependent
              "I3_v", // name
              {i5_pterm_v1, i5_pterm_v2}, imex_flag::imex_implicit);

  inline static term<P> const I_diff_im =
      term<P>(false,  // time-dependent
              "I3_v", // name
              {I_pterm, I_pterm}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_im_7 = {
      nu_theta_term, diff_v_term, I_diff_im, I_diff_im};

  // Term 8
  // d_{v_y}(th q), q = d_{v_y} f
  // Same as term 7 but order if changed

  inline static std::vector<term<P>> const terms_im_8 = {
      nu_theta_term, I_diff_im, diff_v_term, I_diff_im};

  // Term 9
  // d_{v_z}(th q), q = d_{v_z} f
  // Same as term 7 but order if changed

  inline static std::vector<term<P>> const terms_im_9 = {
      nu_theta_term, I_diff_im, I_diff_im, diff_v_term};

  inline static term_set<P> const terms_ = {
      terms_ex_1, terms_ex_2, terms_im_1, terms_im_2, terms_im_3, terms_im_4,
      terms_im_5, terms_im_6, terms_im_7, terms_im_8, terms_im_9};

  static fk::vector<P> exact_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::fill(fx.begin(), fx.end(), 1.0);
    return fx;
  }

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {};

  static P get_dt_(dimension<P> const &dim)
  {
    ignore(dim);
    /* return dx; this will be scaled by CFL from command line */
    // return std::pow(0.25, dim.get_level());

    // TODO: these are constants since we want dt always based on dim 2,
    //  but there is no way to force a different dim for this function!
    // (Lmax - Lmin) / 2 ^ LevX * CFL
    return (6.0 - (-6.0)) / std::pow(2, 3);
  }

  /* problem contains no sources */
  inline static std::vector<source<P>> const sources_ = {};
};

} // namespace asgard
