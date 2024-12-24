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
class PDE_vlasov_two_stream : public PDE<P>
{
public:
  PDE_vlasov_two_stream(prog_opts const &cli_input)
  {
    int constexpr num_dims          = 2;
    int constexpr num_sources       = 0;
    int constexpr num_terms         = 4; // should be 4
    // disable implicit steps in IMEX
    bool constexpr do_collision_operator = false;
    bool constexpr has_analytic_soln     = false;
    int constexpr default_degree         = 2;

    // using empty instances for exact_vector_funcs and exact_time
    this->initialize(cli_input, num_dims, num_sources, num_terms,
                     // defining the dimensions
                     std::vector<dimension<P>>{
                         dimension<P>(-2.0 * PI, 2.0 * PI, 4, default_degree,
                                      initial_condition_dim_x_0, nullptr, "x"),
                         dimension<P>(-2.0 * PI, 2.0 * PI, 3, default_degree,
                                      initial_condition_dim_v_0, nullptr, "v")},
                     // defining the set of terms
                     term_set<P>{std::vector<term<P>>{term_e1x, term_e1v},
                                 std::vector<term<P>>{term_e2x, term_e2v},
                                 std::vector<term<P>>{Emass_pos, div_v_dn},
                                 std::vector<term<P>>{Emass_neg, div_v_up},
                                 },
                     std::vector<source<P>>{},       // no sources
                     std::vector<md_func_type<P>>{}, // no exact solution
                     get_dt_, has_analytic_soln, moment_funcs<P>{},
                     do_collision_operator);
  }

private:

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

  // Term 3
  // -E\cdot\grad_v f for E > 0
  //

  inline static const partial_term<P> pterm_div_v_dn = partial_term<P>(
      coefficient_type::div, PDE<P>::gfunc_neg1, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const div_v_dn =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v_dn}, imex_flag::imex_explicit);

  // Term 4
  // E\cdot\grad_v f for E < 0
  //

  inline static const partial_term<P> pterm_div_v_up = partial_term<P>(
      coefficient_type::div, PDE<P>::gfunc_neg1, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const div_v_up =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v_up}, imex_flag::imex_explicit);

  inline static const partial_term<P> ptEmass_pos = partial_term<P>(
      coefficient_type::mass, pterm_dependence::electric_field, PDE<P>::gfunc_f_positive);
  inline static const partial_term<P> ptEmass_neg = partial_term<P>(
      coefficient_type::mass, pterm_dependence::electric_field, PDE<P>::gfunc_f_negative);

  inline static term<P> const Emass_pos =
      term<P>(true, // time-dependent
              "",   // name
              {ptEmass_pos, }, imex_flag::imex_explicit);
  inline static term<P> const Emass_neg =
      term<P>(true, // time-dependent
              "",   // name
              {ptEmass_neg, }, imex_flag::imex_explicit);

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
};

} // namespace asgard
