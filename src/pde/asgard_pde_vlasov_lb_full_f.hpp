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
  {
    this->skip_old_moments = true; // temp-hack

    term_set<P> terms = {terms_1, terms_2};

    add_lenard_bernstein_collisions_1x1v(nu, terms);

    this->initialize(cli_input, num_dims_, num_sources_, dimensions_,
                     terms, sources_, exact_vector_funcs_,
                     get_dt_, has_analytic_soln_, do_collision_operator_);
  }

private:
  static int constexpr num_dims_           = 2;
  static int constexpr num_sources_        = 0;
  static bool constexpr has_analytic_soln_ = false;
  static int constexpr default_degree      = 3;

  static bool constexpr do_collision_operator_ = true;

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
