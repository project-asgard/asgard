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
    this->skip_old_moments = true; // temp-hack

    this->initialize(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
                     terms_, sources_, exact_vector_funcs_,
                     get_dt_, has_analytic_soln_, moment_funcs<P>{}, do_collision_operator_);
  }

private:
  static int constexpr num_dims_               = 2;
  static int constexpr num_sources_            = 0;
  static int constexpr num_terms_              = 8;
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

  inline static const partial_term<P> ptEmass = partial_term<P>(
      coefficient_type::mass, pterm_dependence::electric_field, PDE<P>::gfunc_f_field);
  inline static term<P> const Emass =
      term<P>(true, // time-dependent
              "",   // name
              {ptEmass, }, imex_flag::imex_explicit);

  inline static const partial_term<P> pterm_div_v = partial_term<P>(
      coefficient_type::div, PDE<P>::gfunc_neg1, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const div_v =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v}, imex_flag::imex_explicit);

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

  inline static const partial_term<P> pterm_div_v_downwind = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const div_v_downwind =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v_downwind}, imex_flag::imex_explicit);

  // Central Part Defined Above (div_v; can do this due to time independence)

  inline static std::vector<term<P>> const terms_4 = {EmassMaxAbsE,
                                                      div_v_downwind};

  inline static std::vector<term<P>> const terms_5 = {EmassMaxAbsE, div_v};

  // Terms 3 - 5 from vlasov_lb_full_f PDE:

  // Term 3
  // v\cdot\grad_v f

  // new collision operators
  static P const_nu(P const, P const = 0) { return nu; }

  static P get_v(P const v, P const = 0)
  {
    return v;
  }
  inline static const partial_term<P> i1_pterm_x = partial_term<P>(
      coefficient_type::mass, const_nu, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i1_pterm_v = partial_term<P>(
      coefficient_type::div, get_v, nullptr, flux_type::upwind,
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

  // Term 5
  // div_v(th\grad_v f)
  //
  // Split by LDG
  //
  // div_v(th q)
  // q = \grad_v f

  // Term 10
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

  inline static std::vector<term<P>> const terms_10 = {penalty_mass_x,
                                                       e_penalty};


  // moment components of the collision operator, split into 3 parts
  // (-u_f, nu * div_v) -> (pt_mass_uf_neg, pt_vdivf)
  // (-mom2/mom0, nu * div * grad) -> (pt_mass_ef, {pt_div_up, pt_nu_grad_down})
  // (-u_f^2, nu * div * grad) -> ({pt_mass_uf, pt_mass_uf_neg}, {pt_div_up, pt_nu_grad_down})

  inline static const partial_term<P> pt_mass_uf = partial_term<P>(
      coefficient_type::mass, pterm_dependence::moments_1by0);

  inline static const partial_term<P> pt_mass_uf_neg = partial_term<P>(
      coefficient_type::mass, pterm_dependence::moments_1by0_neg);

  inline static const partial_term<P> pt_mass_ef = partial_term<P>(
      coefficient_type::mass, pterm_dependence::moments_2by0);

  inline static const partial_term<P> pt_vdivf = partial_term<P>(
      coefficient_type::div, const_nu, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static const partial_term<P> pt_div_up = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static const partial_term<P> pt_nu_grad_down = partial_term<P>(
      coefficient_type::grad, const_nu, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const mass_uf_neg =
      term<P>(true,   // time-dependent
              "I",    // name
              {pt_mass_uf_neg, }, imex_flag::imex_implicit);

  inline static term<P> const mass_u2_neg =
      term<P>(true,  // time-dependent
              "I",   // name
              {pt_mass_uf, pt_mass_uf_neg}, imex_flag::imex_implicit);

  inline static term<P> const mass_ef =
      term<P>(true,   // time-dependent
              "I2_x", // name
              {/* identity, */ pt_mass_ef, }, imex_flag::imex_implicit);

  inline static term<P> const vdivf =
      term<P>(false,  // time-dependent
              "I2_v", // name
              {pt_vdivf,}, imex_flag::imex_implicit);

  inline static term<P> const nu_div_grad =
      term<P>(false,  // time-dependent
              "nu_div_grad", // name
              {pt_div_up, pt_nu_grad_down}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_7 = {mass_uf_neg, vdivf};
  inline static std::vector<term<P>> const terms_8 = {mass_ef, nu_div_grad};
  inline static std::vector<term<P>> const terms_9 = {mass_u2_neg, nu_div_grad};


  // terms 6, 7, 8 are terms 3,4,5 from vlasov_lb_full_f
  inline static term_set<P> const terms_ = {terms_1, terms_2, terms_3, terms_6,
                                            terms_7, terms_8, terms_9, terms_10};

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
