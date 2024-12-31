#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// TODO: fix this comment, this is the wrong PDE
// 3D test case using relaxation problem
//
//  df/dt == div_v( (v-u(x))f + theta(x)\grad_v f)
//
//  where the domain is (x,v_x,v_y,v_z).  The moments of f are constant x.
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_riemann_1x2v : public PDE<P>
{
public:
  PDE_riemann_1x2v(prog_opts const &cli_input)
  {
    term_set<P> terms = {terms_ex_1, terms_ex_2}; // adds the explicit terms
    add_lenard_bernstein_collisions_1x2v(nu, terms);

    this->initialize(cli_input, num_dims_, num_sources_, dimensions_,
                     terms, sources_, exact_vector_funcs_, get_dt_,
                     has_analytic_soln_, do_collision_operator_);
  }

private:
  static int constexpr num_dims_    = 3;
  static int constexpr num_sources_ = 0;
  static int constexpr num_terms_   = 8;
  // disable implicit steps in IMEX
  static bool constexpr do_collision_operator_ = true;
  static bool constexpr has_analytic_soln_     = false;
  static int constexpr default_degree          = 3;

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

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1,
                                                               dim_2};

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

  inline static std::vector<term<P>> const terms_ex_1{term_e1x, term_e1v, I_ex};

  // Explicit Term 2
  // -v_x\cdot\grad_x f for v_x < 0
  //
  inline static const
  partial_term<P> e2_pterm_x{pt_div_periodic, flux_type::downwind, PDE<P>::gfunc_neg1};

  inline static const partial_term<P> e2_pterm_v{pt_mass, PDE<P>::gfunc_negative};

  inline static term<P> const term_e2x{"E2_x", e2_pterm_x, imex_flag::imex_explicit};
  inline static term<P> const term_e2v{"E2_v", e2_pterm_v, imex_flag::imex_explicit};

  inline static std::vector<term<P>> const terms_ex_2{term_e2x, term_e2v, I_ex};

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
