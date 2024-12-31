#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// 4D collisional landau, i.e.,
//
//  df/dt == -v*\grad_x f -E\grad_v f + div_v( (v-u)f + theta\grad_v f)
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_collisional_landau_1x3v : public PDE<P>
{
public:
  PDE_collisional_landau_1x3v(prog_opts const &cli_input)
  {
    term_set<P> terms = {terms_ex_1, terms_ex_2, terms_ex_3, terms_ex_4};
    add_lenard_bernstein_collisions_1x3v(nu, terms);

    this->initialize(cli_input, num_dims_, num_sources_, dimensions_,
                     terms, sources_, exact_vector_funcs_,
                     get_dt_, has_analytic_soln_, do_collision_operator_);
  }

private:
  static int constexpr num_dims_          = 4;
  static int constexpr num_sources_       = 0;
  // disable implicit steps in IMEX
  static bool constexpr do_collision_operator_ = true;
  static bool constexpr has_analytic_soln_     = false;
  static int constexpr default_degree          = 2;

  static P constexpr nu       = 1.0;
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

  inline static dimension<P> const dim_0 =
      dimension<P>(-2.0 * PI, 2.0 * PI, 4, default_degree,
                   initial_condition_dim_x_0, nullptr, "x");

  inline static dimension<P> const dim_1 = dimension<P>(
      -6.0, 6.0, 3, default_degree, initial_condition_dim_v_0, nullptr, "v1");

  inline static dimension<P> const dim_2 = dimension<P>(
      -6.0, 6.0, 3, default_degree, initial_condition_dim_v_0, nullptr, "v2");

  inline static dimension<P> const dim_3 = dimension<P>(
      -6.0, 6.0, 3, default_degree, initial_condition_dim_v_0, nullptr, "v3");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1,
                                                               dim_2, dim_3};


  // ###############################
  // #### Explicit Terms ###########
  // ###############################

  // Constant Explicit Identity term

  inline static term<P> const I_ex{"I", pt_identity, imex_flag::imex_explicit};

  // Explicit Term 1
  // -v_x\cdot\grad_x f for v_x > 0
  //
  inline static const
  partial_term<P> e1_pterm_x{pt_div_periodic, flux_type::upwind, PDE<P>::gfunc_neg1};

  inline static const partial_term<P> e1_pterm_v{pt_mass, PDE<P>::gfunc_positive};

  inline static term<P> const term_e1x{"E1_x", e1_pterm_x, imex_flag::imex_explicit};
  inline static term<P> const term_e1v{"E1_v", e1_pterm_v, imex_flag::imex_explicit};

  inline static std::vector<term<P>> const terms_ex_1 = {term_e1x, term_e1v,
                                                         I_ex, I_ex};

  // Explicit Term 2
  // -v_x\cdot\grad_x f for v_x < 0
  //
  inline static const
  partial_term<P> e2_pterm_x{pt_div_periodic, flux_type::downwind, PDE<P>::gfunc_neg1};

  inline static const partial_term<P> e2_pterm_v{pt_mass, PDE<P>::gfunc_negative};

  inline static term<P> const term_e2x{"E2_x", e2_pterm_x, imex_flag::imex_explicit};
  inline static term<P> const term_e2v{"E2_v", e2_pterm_v, imex_flag::imex_explicit};

  inline static std::vector<term<P>> const terms_ex_2 = {term_e2x, term_e2v,
                                                         I_ex, I_ex};

  // Explicit Term 3
  // -E\cdot\grad_{v_x} f for E > 0
  //

  inline static const partial_term<P> ptEmass_pos{
      pterm_dependence::electric_field, PDE<P>::gfunc_f_positive};
  inline static const partial_term<P> ptEmass_neg{
      pterm_dependence::electric_field, PDE<P>::gfunc_f_negative};

  inline static term<P> const Emass_pos =
      term<P>(true,        // time-dependent
              "Emass_pos", // name
              ptEmass_pos, imex_flag::imex_explicit);
  inline static term<P> const Emass_neg =
      term<P>(true,        // time-dependent
              "Emass_neg", // name
              ptEmass_neg, imex_flag::imex_explicit);

  inline static const
  partial_term<P> pterm_div_v_dn{pt_div_dirichlet_zero, flux_type::upwind, PDE<P>::gfunc_neg1};

  inline static term<P> const div_v_dn{"", pterm_div_v_dn, imex_flag::imex_explicit};

  inline static std::vector<term<P>> const terms_ex_3 = {Emass_pos, div_v_dn,
                                                         I_ex, I_ex};

  // Explicit Term 4
  // -E\cdot\grad_{v_x} f for E < 0
  //

  inline static const
  partial_term<P> pterm_div_v_up{pt_div_dirichlet_zero, flux_type::downwind, PDE<P>::gfunc_neg1};

  inline static term<P> const div_v_up{"", pterm_div_v_up, imex_flag::imex_explicit};

  inline static std::vector<term<P>> const terms_ex_4 = {Emass_neg, div_v_up,
                                                         I_ex, I_ex};

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
