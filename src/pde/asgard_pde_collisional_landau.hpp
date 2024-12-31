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
  {
    term_set<P> terms;

    add_vlassov_1x1v(terms);

    terms.push_back(terms_3);

    add_lenard_bernstein_collisions_1x1v(nu, terms);

    partial_term<P> ptI{pt_identity};

    term<P> termI("identity", ptI, imex_flag::imex_implicit);

    auto pen_func = [pde=this](P const, P const = 0)
    {
      // may need to improve the logic here
      // how often should we recompute this during adaptivity? on level change?
      return 10.0 / ((6.0 - (-6.0)) / fm::ipow2(pde->get_dimensions()[1].get_level()));
    };

    partial_term<P> pt_pen(
        coefficient_type::penalty, pen_func, nullptr, flux_type::upwind,
        boundary_condition::free, boundary_condition::free);

    bool constexpr time_depend = true;
    term<P> term_pen(time_depend, "penalty", pt_pen, imex_flag::imex_implicit);

    terms.push_back({termI, term_pen});

    this->initialize(cli_input, num_dims_, num_sources_, dimensions_,
                     terms, sources_, exact_vector_funcs_,
                     get_dt_, has_analytic_soln_, do_collision_operator_);
  }

private:
  static int constexpr num_dims_               = 2;
  static int constexpr num_sources_            = 0;
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

  inline static const partial_term<P> e1_pterm_x{pt_div_periodic, flux_type::upwind, e1_g1};

  inline static const partial_term<P> e1_pterm_v{pt_mass, e1_g2};

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

  inline static const partial_term<P> e2_pterm_v{pt_mass, e2_g2};

  inline static term<P> const term_e2x =
      term<P>(false,  // time-dependent
              "E2_x", // name
              {pt_div_periodic, flux_type::downwind, e2_g1},
              imex_flag::imex_explicit);

  inline static term<P> const term_e2v =
      term<P>(false,  // time-dependent
              "E2_v", // name
              {e2_pterm_v}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_2 = {term_e2x, term_e2v};

  // Term 3
  // Central Part of E\cdot\grad_v f
  //

  inline static const partial_term<P> ptEmass{
      pterm_dependence::electric_field, PDE<P>::gfunc_f_field};
  inline static term<P> const Emass =
      term<P>(true,    // time-dependent
              "Emass", // name
              ptEmass, imex_flag::imex_explicit);

  inline static term<P> const div_v =
      term<P>("div_v",
              {pt_div_dirichlet_zero, flux_type::central, PDE<P>::gfunc_neg1},
              imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_3 = {Emass, div_v};

  // Term 4 + 5
  // Penalty Part of E\cdot\grad_v f
  //

  inline static const partial_term<P> ptEmassMaxAbsE = partial_term<P>(
      coefficient_type::mass, pterm_dependence::electric_field_infnrm,
      PDE<P>::gfunc_f_field);

  inline static term<P> const EmassMaxAbsE =
      term<P>(true, // time-dependent
              "",   // name
              {ptEmassMaxAbsE}, imex_flag::imex_explicit);

  inline static term<P> const div_v_downwind =
      term<P>(false, // time-dependent
              "",    // name
              {pt_div_dirichlet_zero, flux_type::upwind}, imex_flag::imex_explicit);

  // Central Part Defined Above (div_v; can do this due to time independence)

  inline static std::vector<term<P>> const terms_4 = {EmassMaxAbsE,
                                                      div_v_downwind};

  inline static std::vector<term<P>> const terms_5 = {EmassMaxAbsE, div_v};

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
