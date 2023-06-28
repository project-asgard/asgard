#pragma once
#include "pde_base.hpp"

namespace asgard
{
// 2D test case using continuity equation, i.e.,
//
//  df/dt == -v*\grad_x f + div_v( (v-u)f + theta\grad_v f)
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_vlasov_two_stream : public PDE<P>
{
public:
  PDE_vlasov_two_stream(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
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
  static int constexpr num_terms_         = 4;
  static bool constexpr do_poisson_solve_ = true;
  // disable implicit steps in IMEX
  static bool constexpr do_collision_operator_ = false;
  static bool constexpr has_analytic_soln_     = false;
  static int constexpr default_degree          = 3;

  static fk::vector<P>
  initial_condition_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return 1.0 - 0.5 * std::cos(0.5 * x_v);
    });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = 1.0 / std::sqrt(PI);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient](P const x_v) -> P {
          return coefficient * std::pow(x_v, 2) * std::exp(-std::pow(x_v, 2));
        });
    return fx;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 =
      dimension<P>(-2.0 * PI, 2.0 * PI, 4, default_degree,
                   initial_condition_dim_x_0, nullptr, "x");

  inline static dimension<P> const dim_1 =
      dimension<P>(-2.0 * PI, 2.0 * PI, 3, default_degree,
                   initial_condition_dim_v_0, nullptr, "v");

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

  inline static moment<P> const moment0 = moment<P>(
      std::vector<md_func_type<P>>({{moment0_f1, moment0_f1, moment0_f1}}));
  inline static moment<P> const moment1 = moment<P>(
      std::vector<md_func_type<P>>({{moment0_f1, moment1_f1, moment0_f1}}));
  inline static moment<P> const moment2 = moment<P>(
      std::vector<md_func_type<P>>({{moment0_f1, moment2_f1, moment0_f1}}));

  inline static std::vector<moment<P>> const moments_ = {moment0, moment1,
                                                         moment2};

  /* Construct (n, u, theta) */
  static P n(P const &x, P const t = 0)
  {
    ignore(t);

    return (1.0 - 0.5 * std::cos(0.5 * x)) * 0.5;
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
    return 1.5;
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
      coefficient_type::div, e1_g1, nullptr, flux_type::downwind,
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
      coefficient_type::div, e2_g1, nullptr, flux_type::upwind,
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
  // -E\cdot\grad_v f for E > 0
  //

  static P E_func_pos(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("E");
    expect(param != nullptr);
    return std::max(P{0.0}, param->value(x, time));
  }

  static P negOne(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }

  inline static const partial_term<P> pterm_E_mass_x_pos = partial_term<P>(
      coefficient_type::mass, E_func_pos, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const E_mass_x_pos =
      term<P>(true, // time-dependent
              "",   // name
              {pterm_E_mass_x_pos}, imex_flag::imex_explicit);

  inline static const partial_term<P> pterm_div_v_dn = partial_term<P>(
      coefficient_type::div, negOne, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const div_v_dn =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v_dn}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_3 = {E_mass_x_pos, div_v_dn};

  // Term 4
  // E\cdot\grad_v f for E < 0
  //

  static P E_func_neg(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("E");
    expect(param != nullptr);
    return std::min(P{0.0}, param->value(x, time));
  }

  inline static const partial_term<P> pterm_E_mass_x_neg = partial_term<P>(
      coefficient_type::mass, E_func_neg, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const E_mass_x_neg =
      term<P>(true, // time-dependent
              "",   // name
              {pterm_E_mass_x_neg}, imex_flag::imex_explicit);

  inline static const partial_term<P> pterm_div_v_up = partial_term<P>(
      coefficient_type::div, negOne, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const div_v_up =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v_up}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_4 = {E_mass_x_neg, div_v_up};

  inline static term_set<P> const terms_ = {terms_1, terms_2, terms_3, terms_4};

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {};
  inline static scalar_func<P> const exact_scalar_func_               = {};

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
