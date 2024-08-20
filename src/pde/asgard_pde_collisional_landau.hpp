#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// 2D collisional landau, i.e.,
//
//  df/dt == -v*\grad_x f -E\grad_v f + div_v( (v-u)f + theta\grad_v f)
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_collisional_landau : public PDE<P>
{
public:
  PDE_collisional_landau(prog_opts const &cli_input)
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
  static int constexpr num_dims_               = 2;
  static int constexpr num_sources_            = 0;
  static int constexpr num_terms_              = 7;
  static bool constexpr do_poisson_solve_      = true;
  static bool constexpr do_collision_operator_ = true;
  static bool constexpr has_analytic_soln_     = false;
  static int constexpr default_degree          = 3;

  static P constexpr nu       = 1.0;    // collision frequency
  static P constexpr A        = 1.0e-4; // amplitude
  static P constexpr theta_in = 1.0;

  static fk::vector<P>
  initial_condition_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return 1.0 + A * std::cos(0.5 * x_v);
    });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta_in);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient](P const x_v) -> P {
          return coefficient *
                 std::exp(-0.5 * (1.0 / theta_in) * std::pow(x_v, 2));
        });
    return fx;
  }

  static P dV(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 =
      dimension<P>(-2.0 * PI, 2.0 * PI, 4, default_degree,
                   initial_condition_dim_x_0, dV, "x");

  inline static dimension<P> const dim_1 = dimension<P>(
      -6.0, 6.0, 3, default_degree, initial_condition_dim_v_0, dV, "v");

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

  inline static moment_funcs<P> const moments_ = {
      {{moment0_f1, moment0_f1, moment0_f1}},
      {{moment0_f1, moment1_f1, moment0_f1}},
      {{moment0_f1, moment2_f1, moment0_f1}}};

  /* Construct (n, u, theta) */
  // n = density
  // u = bulk velocity
  // theta = temperature
  static P n(P const &x, P const t = 0)
  {
    ignore(t);

    return (1.0 + A * std::cos(0.5 * x));
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

  // E = -d_x phi
  static P E(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  // source for poisson problem is -d_xx phi = n - 1 = S(n)
  static P S(P const &y, P const t = 0)
  {
    ignore(t);
    // subtracts quadrature values by one
    return y - 1.0;
  }

  // holds the maximum absolute value of E
  static P MaxAbsE(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
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

  inline static std::vector<term<P>> const terms_2 = {term_e2x, term_e2v};

  // Term 3
  // Central Part of E\cdot\grad_v f
  //

  static P E_func(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("E");
    expect(param != nullptr);
    return param->value(x, time);
  }

  static P negOne(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }

  inline static const partial_term<P> pterm_E_mass_x = partial_term<P>(
      coefficient_type::mass, E_func, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const E_mass_x =
      term<P>(true, // time-dependent
              "",   // name
              {pterm_E_mass_x}, imex_flag::imex_explicit);

  inline static const partial_term<P> pterm_div_v = partial_term<P>(
      coefficient_type::div, negOne, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const div_v =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_3 = {E_mass_x, div_v};

  // Term 4 + 5
  // Penalty Part of E\cdot\grad_v f
  //

  static P MaxAbsE_func(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("MaxAbsE");
    expect(param != nullptr);
    return param->value(x, time);
  }

  inline static const partial_term<P> pterm_MaxAbsE_mass_x = partial_term<P>(
      coefficient_type::mass, MaxAbsE_func, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const MaxAbsE_mass_x_1 =
      term<P>(true, // time-dependent
              "",   // name
              {pterm_MaxAbsE_mass_x}, imex_flag::imex_explicit);

  inline static term<P> const MaxAbsE_mass_x_2 =
      term<P>(true, // time-dependent
              "",   // name
              {pterm_MaxAbsE_mass_x}, imex_flag::imex_explicit);

  inline static const partial_term<P> pterm_div_v_downwind = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const div_v_downwind =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v_downwind}, imex_flag::imex_explicit);

  // Central Part Defined Above (div_v; can do this due to time independence)

  inline static std::vector<term<P>> const terms_4 = {MaxAbsE_mass_x_1,
                                                      div_v_downwind};

  inline static std::vector<term<P>> const terms_5 = {MaxAbsE_mass_x_2, div_v};

  // Terms 3 - 5 from vlasov_lb_full_f PDE:

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

  inline static std::vector<term<P>> const terms_6 = {term_i1x, term_i1v};

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

  inline static std::vector<term<P>> const terms_7 = {term_i2x, term_i2v};

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

  inline static std::vector<term<P>> const terms_8 = {term_i3x, term_i3v};

  // Term 9
  // Penalty Part of collision operator
  //
  inline static const partial_term<P> penalty_mass_x_pterm = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const penalty_mass_x =
      term<P>(true, // time-dependent
              "",   // name
              {penalty_mass_x_pterm}, imex_flag::imex_implicit);

  static P penalty_func(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    // hardcoded for level 4: (vmax - vmin) / (2^lev_v)
    return 10.0 / ((6.0 - (-6.0)) / (16.0));
  }

  inline static const partial_term<P> e_penalty_pterm = partial_term<P>(
      coefficient_type::penalty, penalty_func, nullptr, flux_type::upwind,
      boundary_condition::neumann, boundary_condition::neumann,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const e_penalty =
      term<P>(false, // time-dependent
              "",    // name
              {e_penalty_pterm}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_9 = {penalty_mass_x,
                                                      e_penalty};

  // terms 6, 7, 8 are terms 3,4,5 from vlasov_lb_full_f
  inline static term_set<P> const terms_ = {terms_1, terms_2, terms_3, terms_6,
                                            terms_7, terms_8, terms_9};

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {};

  static P get_dt_(dimension<P> const &dim)
  {
    ignore(dim);
    // TODO: these are constants since we want dt always based on dim 2,
    //  but there is no way to force a different dim for this function!
    // (Lmax - Lmin) / 2 ^ LevX * CFL, where 2 ^ LevX = 8 (LevX = 3)
    return static_cast<P>((6.0 - (-6.0)) / 8.0);
  }

  /* problem contains no sources */
  inline static std::vector<source<P>> const sources_ = {};
};

} // namespace asgard
