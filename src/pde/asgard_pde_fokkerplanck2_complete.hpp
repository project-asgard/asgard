#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// ---------------------------------------------------------------------------
// Full PDE from the 2D runaway electron paper
//
// d/dt f(p,z) == -div(flux_C + flux_E + flux_R)
//
// where
//
// flux_C is flux due to electron-ion collisions
// flux_E is the flux due to E accleration
// flux_R is the flux due to radiation damping
//
// -div(flux_C) == termC1 + termC2 + termC3
//
// termC1 == 1/p^2*d/dp*p^2*Ca*df/dp
// termC2 == 1/p^2*d/dp*p^2*Cf*f
// termC3 == termC3 == Cb(p)/p^4 * d/dz( (1-z^2) * df/dz )
//
// -div(flux_E) == termE1 + termE2
//
// termE1 == -E*z*f(z) * 1/p^2 (d/dp p^2 f(p))
// termE2 == -E*p*f(p) * d/dz (1-z^2) f(z)
//
// -div(flux_R) == termR1 + termR2
//
// termR1 == 1/p^2 d/dp p^2 gamma(p) p / tau f(p) * (1-z^2) * f(z)
// termR2 == -1/(tau*gam(p)) f(p) * d/dz z(1-z^2) f(z)
// ---------------------------------------------------------------------------

template<typename P, PDE_case_opts user_case = PDE_case_opts::case1>
class PDE_fokkerplanck_2d_complete : public PDE<P>
{
public:
  PDE_fokkerplanck_2d_complete(prog_opts const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 2;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 6;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = false;

  // ------------------------------------------------
  //
  // define constants / data / functions for this PDE
  //
  // ------------------------------------------------

  static auto constexpr phi = [](P x) { return std::erf(x); };

  static auto constexpr psi = [](P x) {
    auto const dphi_dx = 2.0 / std::sqrt(M_PI) * std::exp(-std::pow(x, 2));
    auto ret           = 1.0 / (2 * std::pow(x, 2)) * (phi(x) - x * dphi_dx);
    if (std::abs(x) < 1e-5)
      ret = 0;
    return ret;
  };

  static P constexpr nuEE  = 1;
  static P constexpr vT    = 1;
  static P constexpr delta = []() {
    if constexpr (user_case == PDE_case_opts::case4)
    {
      return 0.3;
    }
    else
    {
      return 0.042;
    }
  }();
  static P constexpr Z = []() {
    if constexpr (user_case == PDE_case_opts::case4)
    {
      return 5.0;
    }
    else
    {
      return 1.0;
    }
  }();
  static P constexpr E = []() {
    if constexpr (user_case == PDE_case_opts::case1)
    {
      return 0.0025;
    }
    else if constexpr (user_case == PDE_case_opts::case2)
    {
      return 0.25;
    }
    else if constexpr (user_case == PDE_case_opts::case3)
    {
      return 0.0025;
    }
    else if constexpr (user_case == PDE_case_opts::case4)
    {
      return 0.4;
    }
  }();
  static P constexpr tau      = 1e5;
  static auto constexpr gamma = [](P p) {
    return std::sqrt(1 + std::pow(delta * p, 2));
  };
  static auto constexpr vx = [](P p) { return 1.0 / vT * (p / gamma(p)); };

  static auto constexpr Ca = [](P p) {
    return nuEE * std::pow(vT, 2) * (psi(vx(p)) / vx(p));
  };

  static auto constexpr Cb = [](P p) {
    return 1.0 / 2.0 * nuEE * std::pow(vT, 2) * 1.0 / vx(p) *
           (Z + phi(vx(p)) - psi(vx(p)) +
            std::pow(delta, 4) * std::pow(vx(p), 2) / 2.0);
  };

  static auto constexpr Cf = [](P p) { return 2.0 * nuEE * vT * psi(vx(p)); };

  // -----------------
  //
  // define dimensions
  //
  // -----------------

  // specify initial conditions for each dim
  // p dimension

  // initial condition in p
  static fk::vector<P>
  initial_condition_p_case1(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> transformed(x);
    std::transform(x.begin(), x.end(), transformed.begin(),
                   [](P const x_elem) -> P {
                     if (x_elem <= 5.0)
                     {
                       return 3.0 / (2.0 * std::pow(5, 3));
                     }
                     else
                     {
                       return 0.0;
                     }
                   });
    return transformed;
  }
  static fk::vector<P>
  initial_condition_p_case2(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> transformed(x);
    std::transform(
        x.begin(), x.end(), transformed.begin(), [](P const x_elem) -> P {
          return std::exp(-std::pow(x_elem, 2) / fm::ipow2(3));
        });
    transformed.scale(2.0 / (std::sqrt(M_PI) * fm::ipow2(3)));
    return transformed;
  }
  static fk::vector<P>
  initial_condition_p_case3(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> transformed(x);
    std::transform(
        x.begin(), x.end(), transformed.begin(),
        [](P const x_elem) -> P { return exp(-std::pow(x_elem, 2)); });
    transformed.scale(2.0 / (3.0 * std::sqrt(M_PI)));
    return transformed;
  }
  static fk::vector<P>
  initial_condition_p_case4(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);

    P N = 1000.0;
    P h = 20.0 / N;
    P Q = 0;

    auto const function = [](fk::vector<P> const &p) -> fk::vector<P> {
      fk::vector<P> transformed(p);
      std::transform(p.begin(), p.end(), transformed.begin(),
                     [](P const p_elem) -> P {
                       return std::exp(-2 / std::pow(delta, 2) *
                                       std::sqrt(1 + std::pow(delta, 2) *
                                                         std::pow(p_elem, 2)));
                     });

      return transformed;
    };

    for (int i = 0; i < N; ++i)
    {
      P const x_0 = i * h;
      P const x_1 = (i + 1) * h;

      /* Matlab uses 20 points - even though this is probably not the degree,
         putting 20 for the degree argument and "true" for use_degree_points
         forces consistency */
      std::array<fk::vector<P>, 2> rw =
          legendre_weights<P>(20, x_0, x_1, quadrature_mode::use_degree);

      fk::vector<P> transformed = function(rw[0]);

      std::transform(rw[0].begin(), rw[0].end(), transformed.begin(),
                     transformed.begin(),
                     [](P const root, P const t_elem) -> P {
                       return std::pow(root, 2) * t_elem;
                     });

      std::transform(
          rw[1].begin(), rw[1].end(), transformed.begin(), transformed.begin(),
          [](P const weight, P const t_elem) -> P { return weight * t_elem; });

      Q += std::accumulate(transformed.begin(), transformed.end(), 0.0);
    }

    fk::vector<P> ret = function(x);

    ret.scale(1 / (2 * Q));

    return ret;
  }

  inline static vector_func<P> const initial_condition_p = []() {
    if constexpr (user_case == PDE_case_opts::case1)
    {
      return initial_condition_p_case1;
    }
    else if constexpr (user_case == PDE_case_opts::case2)
    {
      return initial_condition_p_case2;
    }
    else if constexpr (user_case == PDE_case_opts::case3)
    {
      return initial_condition_p_case3;
    }
    else if constexpr (user_case == PDE_case_opts::case4)
    {
      return initial_condition_p_case4;
    }
  }();

  static P volume_jacobian_dV_p(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(time);
    return std::pow(x, 2);
  }

  // initial conditionn in z
  static fk::vector<P>
  initial_condition_z_case1(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> ret(x);
    std::transform(x.begin(), x.end(), ret.begin(), [](P const x_elem) -> P {
      if (x_elem <= 0.0)
      {
        return 3.0 / (2.0 * std::pow(5, 3));
      }
      else
      {
        return 0.0;
      }
    });
    return ret;
  }
  static fk::vector<P>
  initial_condition_z_case2(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> ret(x.size());
    std::fill(ret.begin(), ret.end(), 1.0);
    return ret;
  }
  static fk::vector<P>
  initial_condition_z_case3(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> f(x.size());
    std::vector<P> const legendre_coeffs = {3, 0.5, 1, 0.7, 3, 0, 3};

    auto const [P_m, dP_m] =
        legendre(x, legendre_coeffs.size(), legendre_normalization::matlab);
    ignore(dP_m);

    for (int i = 0; i < static_cast<int>(legendre_coeffs.size()); ++i)
    {
      fk::vector<P> const P_0 = P_m.extract_submatrix(0, i, x.size(), 1);
      f                       = f + (P_0 * legendre_coeffs[i]);
    }

    return f;
  }
  static fk::vector<P>
  initial_condition_z_case4(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> ret(x.size());
    std::fill(ret.begin(), ret.end(), 1.0);
    return ret;
  }

  inline static vector_func<P> const initial_condition_z = []() {
    if constexpr (user_case == PDE_case_opts::case1)
    {
      return initial_condition_z_case1;
    }
    else if constexpr (user_case == PDE_case_opts::case2)
    {
      return initial_condition_z_case2;
    }
    else if constexpr (user_case == PDE_case_opts::case3)
    {
      return initial_condition_z_case3;
    }
    else if constexpr (user_case == PDE_case_opts::case4)
    {
      return initial_condition_z_case4;
    }
  }();

  // p dimension
  // FIXME matlab value is 0.1 - 10, but this produces ill-conditioned matrices
  // the math wizards will conjure us a new pde with a better behaved domain
  // soon
  inline static P const p_domain_min = 0.0;
  inline static P const p_domain_max = 10.0;
  inline static dimension<P> const dim_p =
      dimension<P>(p_domain_min,        // domain min
                   p_domain_max,        // domain max
                   2,                   // levels
                   1,                   // degree
                   initial_condition_p, // initial condition
                   volume_jacobian_dV_p,
                   "p"); // name

  // z dimension
  inline static P const z_domain_min = -1;
  inline static P const z_domain_max = 1;
  inline static dimension<P> const dim_z =
      dimension<P>(z_domain_min,        // domain min
                   z_domain_max,        // domain max
                   2,                   // levels
                   1,                   // degree
                   initial_condition_z, // initial condition
                   nullptr,
                   "z"); // name

  // assemble dimensions

  inline static std::vector<dimension<P>> const dimensions_ = {dim_p, dim_z};

  // ----------------------------------------
  //
  // Setup the terms of the PDE
  //
  // -div(flux_C) == termC1 + termC2 + termC3
  //
  // ----------------------------------------

  static P dV_p(P const x, P const time)
  {
    ignore(time);
    return std::pow(x, 2);
  }

  static P dV_p3(P const x, P const time)
  {
    ignore(time);
    return x;
  }

  static P dV_z3(P const x, P const time)
  {
    ignore(time);
    return sqrt(1.0 - std::pow(x, 2));
  }

  inline static partial_term<P> const pterm_I =
      partial_term<P>(coefficient_type::mass, nullptr);
  inline static term<P> const I_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "massY",         // name
              {pterm_I});

  // termC1 == 1/p^2*d/dp*p^2*Ca*df/dp
  //
  //
  // becomes
  //
  // termC1 == g1(p) q(p)        [mass, g1(p) = 1/p^2,  BC N/A]
  //   q(p) == d/dp g2(p) r(p)   [grad, g2(p) = p^2*Ca, BCL=D,BCR=N]
  //   r(p) == d/dp g3(p) f(p)   [grad, g3(p) = 1,      BCL=N,BCR=D]

  static P c1_g1(P const x, P const time = 0)
  {
    ignore(time);
    return sqrt(Ca(x));
  }

  // 1. create partial_terms
  inline static partial_term<P> const c1_pterm1 =
      partial_term<P>(coefficient_type::div, c1_g1, nullptr, flux_type::downwind,
                      boundary_condition::dirichlet,
                      boundary_condition::neumann, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr, dV_p);
  inline static partial_term<P> const c1_pterm2 =
      partial_term<P>(coefficient_type::grad, c1_g1, nullptr,
                      flux_type::upwind, boundary_condition::neumann,
                      boundary_condition::dirichlet, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr, dV_p);
  inline static partial_term<P> const c1_pterm3 =
      partial_term<P>(coefficient_type::mass, nullptr, nullptr,
                      flux_type::central, boundary_condition::neumann,
                      boundary_condition::neumann, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr);

  // 2. combine partial terms into single dimension term
  inline static term<P> const c1_term_p = term<P>(false,  // time-dependent
                                                  "C1_p", // name
                                                  {c1_pterm1, c1_pterm2});
  inline static term<P> const c1_term_z = term<P>(false,  // time-dependent
                                                  "C1_z", // name
                                                  {c1_pterm3, c1_pterm3});

  // 3. combine single dimension terms into multi dimension term
  inline static std::vector<term<P>> const termC1 = {c1_term_p, c1_term_z};

  // termC2 == 1/p^2*d/dp*p^2*Cf*f
  //
  // becomes
  //
  // termC2 == g1(p) q(p)       [mass, g1(p)=1/p^2,  BC N/A]
  //   q(p) == d/dp g2(p) f(p)  [grad, g2(p)=p^2*Cf, BCL=N,BCR=D]

  static P c2_g1(P const x, P const time = 0)
  {
    ignore(time);
    return Cf(x);
  }

  // 1. create partial_terms
  inline static partial_term<P> const c2_pterm1 =
      partial_term<P>(coefficient_type::div, c2_g1, nullptr,
                      flux_type::upwind, boundary_condition::neumann,
                      boundary_condition::dirichlet, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr, dV_p);
  inline static partial_term<P> const c2_pterm2 =
      partial_term<P>(coefficient_type::mass, nullptr, nullptr,
                      flux_type::central, boundary_condition::neumann,
                      boundary_condition::neumann, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr);

  // 2. combine partial terms into single dimension term
  inline static term<P> const c2_term_p = term<P>(false,  // time-dependent
                                                  "C2_p", // name
                                                  {c2_pterm1});
  inline static term<P> const c2_term_z = term<P>(false,  // time-dependent
                                                  "C2_z", // name
                                                  {c2_pterm2});

  // 3. combine single dimension terms into multi dimension term
  inline static std::vector<term<P>> const termC2 = {c2_term_p, c2_term_z};

  // termC3 == Cb(p)/p^4 * d/dz( (1-z^2) * df/dz )
  //
  // becomes
  //
  // termC3 == q(p) r(z)
  // q(p) == g1(p)            [mass, g1(p) = Cb(p)/p^4, BC N/A]
  // r(z) == d/dz g2(z) s(z)  [grad, g2(z) = 1-z^2,     BCL=D,BCR=D]
  // s(z) == d/dz g3(z) f(z)  [grad, g3(z) = 1,         BCL=N,BCR=N]

  static P c3_g1(P const x, P const time = 0)
  {
    ignore(time);
    return sqrt(Cb(x));
  }

  // 1. create partial_terms
  inline static partial_term<P> const c3_pterm1 = partial_term<P>(
      coefficient_type::mass, c3_g1, nullptr, flux_type::central,
      boundary_condition::neumann, boundary_condition::neumann,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, dV_p3);

  inline static partial_term<P> const c3_pterm2 = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, dV_z3);

  inline static partial_term<P> const c3_pterm3 = partial_term<P>(
      coefficient_type::grad, nullptr, nullptr, flux_type::upwind,
      boundary_condition::neumann, boundary_condition::neumann,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, dV_z3);

  // 2. combine partial terms into single dimension term
  inline static term<P> const c3_term_p = term<P>(false,  // time-dependent
                                                  "C3_p", // name
                                                  {c3_pterm1, c3_pterm1});
  inline static term<P> const c3_term_z = term<P>(false,  // time-dependent
                                                  "C3_z", // name
                                                  {c3_pterm2, c3_pterm3});

  // 3. combine single dimension terms into multi dimension term
  inline static std::vector<term<P>> const termC3 = {c3_term_p, c3_term_z};

  // -div(flux_E) == termE1 + termE2

  // termE1 == -E*z*f(z) * 1/p^2 (d/dp p^2 f(p))
  //        == r(z) * q(p)
  //   r(z) == g1(z) f(z)       [mass, g1(z) = -E*z,  BC N/A]
  //   q(p) == g2(p) u(p)       [mass, g2(p) = 1/p^2, BC N/A]
  //   u(p) == d/dp g3(p) f(p)  [grad, g3(p) = p^2,   BCL=N,BCR=D]

  static P e1_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -E;
  }
  static P e1_g2(P const x, P const time = 0)
  {
    ignore(time);
    if (x > 0)
    {
      return x;
    }
    return 0.0;
  }

  // 1. create partial_terms
  inline static partial_term<P> const e1_pterm1 =
      partial_term<P>(coefficient_type::div, e1_g1, nullptr,
                      flux_type::upwind, boundary_condition::dirichlet,
                      boundary_condition::neumann, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr, dV_p);
  inline static partial_term<P> const e1_pterm2 =
      partial_term<P>(coefficient_type::mass, e1_g2, nullptr,
                      flux_type::central, boundary_condition::neumann,
                      boundary_condition::neumann, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr);

  // 2. combine partial terms into single dimension term
  inline static term<P> const e1_term_p = term<P>(false,  // time-dependent
                                                  "E1_p", // name
                                                  {e1_pterm1});
  inline static term<P> const e1_term_z = term<P>(false,  // time-dependent
                                                  "E1_z", // name
                                                  {e1_pterm2});

  // 3. combine single dimension terms into multi dimension term
  inline static std::vector<term<P>> const termE1 = {e1_term_p, e1_term_z};

  // termE2 == -E*p*f(p) * d/dz (1-z^2) f(z)
  //        == q(p) * r(z)
  //   q(p) == g1(p) f(p)       [mass, g1(p) = -E*p,  BC N/A]
  //   r(z) == d/dz g2(z) f(z)  [grad, g2(z) = 1-z^2, BCL=N,BCR=N]

  static P e2_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -E;
  }
  static P e2_g2(P const x, P const time = 0)
  {
    ignore(time);
    if (x < 0)
    {
      return x;
    }
    return 0.0;
  }

  // 1. create partial_terms
  inline static partial_term<P> const e2_pterm1 =
      partial_term<P>(coefficient_type::div, e2_g1, nullptr, flux_type::downwind,
                      boundary_condition::neumann,
                      boundary_condition::dirichlet, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr, dV_p);

  inline static partial_term<P> const e2_pterm2 =
      partial_term<P>(coefficient_type::mass, e2_g2, nullptr,
                      flux_type::central, boundary_condition::neumann,
                      boundary_condition::neumann, homogeneity::homogeneous,
                      homogeneity::homogeneous, {}, nullptr, {}, nullptr);

  // 2. combine partial terms into single dimension term
  inline static term<P> const e2_term_p = term<P>(false,  // time-dependent
                                                  "E2_p", // name
                                                  {e2_pterm1});
  inline static term<P> const e2_term_z = term<P>(false,  // time-dependent
                                                  "E2_z", // name
                                                  {e2_pterm2});

  // 3. combine single dimension terms into multi dimension term
  inline static std::vector<term<P>> const termE2 = {e2_term_p, e2_term_z};

  static P e3_g2(P const x, P const time = 0)
  {
    ignore(time);
    return -E * sqrt(1.0 - std::pow(x, 2));
  }

  // 1. create partial_terms
  inline static partial_term<P> const e3_pterm1 = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::neumann, boundary_condition::neumann,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, dV_p3);
  inline static partial_term<P> const e3_pterm2 = partial_term<P>(
      coefficient_type::div, e3_g2, nullptr, flux_type::upwind,
      boundary_condition::neumann, boundary_condition::neumann,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, dV_z3);

  // 2. combine partial terms into single dimension term
  inline static term<P> const e3_term_p = term<P>(false,  // time-dependent
                                                  "E3_p", // name
                                                  {e3_pterm1});
  inline static term<P> const e3_term_z = term<P>(false,  // time-dependent
                                                  "E3_z", // name
                                                  {e3_pterm2});

  // 3. combine single dimension terms into multi dimension term
  inline static std::vector<term<P>> const termE3 = {e3_term_p, e3_term_z};

  // -div(flux_R) == termR1 + termR2

  // clang-format off
  //
  // termR1 == 1/p^2 d/dp p^2 gamma(p) p / tau f(p) * (1-z^2) * f(z)
  //        == q(p) * r(z)
  //   q(p) == g1(p) u(p)       [mass, g1(p) = 1/p^2,                BC N/A]
  //   u(p) == d/dp g2(p) f(p)  [grad, g2(p) = p^3 * gamma(p) / tau, BCL=N,BCR=D]
  //   r(z) == g3(z) f(z)       [mass, g3(z) = 1-z^2,                BC N/A]
  //
  // clang-format on
  /* clang-format off */
  /*
  static P r1_g1(P const x, P const time = 0)
  {
    ignore(time);
    return x * gamma(x) / tau;
  }
  static P r1_g2(P const x, P const time = 0)
  {
    ignore(time);
    return 1.0 - std::pow(x, 2);
  }

  // 1. create partial_terms
  inline static partial_term<P> const r1_pterm1 = partial_term<P>(
      coefficient_type::div, r1_g1, nullptr,
      flux_type::downwind, boundary_condition::neumann,
      boundary_condition::dirichlet, homogeneity::homogeneous,
      homogeneity::homogeneous, {}, nullptr, {},
      nullptr, dV_p);
  inline static partial_term<P> const r1_pterm2 = partial_term<P>(
      coefficient_type::mass, r1_g1, nullptr,
      flux_type::central, boundary_condition::neumann,
      boundary_condition::neumann, homogeneity::homogeneous,
      homogeneity::homogeneous, {}, nullptr, {},
      nullptr);

  // 2. combine partial terms into single dimension term
  inline static term<P> const r1_term_p =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "R1_p",          // name
              {r1_pterm1});
  inline static term<P> const r1_term_z =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "R1_z",          // name
              {r1_pterm2});

  // 3. combine single dimension terms into multi dimension term
  inline static std::vector<term<P>> const termR1 = {r1_term_p, r1_term_z};

  // clang-format off
  //
  // termR2 == -1/(tau*gam(p)) f(p) * d/dz z(1-z^2) f(z)
  //        == q(p) * r(z)
  //   q(p) == g1(p) f(p)       [mass, g1(p) = -1/(tau*gamma(p)),    BC N/A]
  //   r(z) == d/dz g2(z) f(z)  [grad, g2(z) = z(1-z^2),             BCL=N,BCR=N]
  //
  // clang-format on

  static P r2_g1(P const x, P const time = 0)
  {
    ignore(time);
    return -x / (tau * gamma(x));
  }
  static P r2_g2(P const x, P const time = 0)
  {
    ignore(time);
    return -x * sqrt(1.0 - std::pow(x, 2));
  }

  // 1. create partial_terms
  inline static partial_term<P> const r2_pterm1 = partial_term<P>(
      coefficient_type::mass, r2_g1, nullptr,
      flux_type::central, boundary_condition::neumann,
      boundary_condition::neumann, homogeneity::homogeneous,
      homogeneity::homogeneous, {}, nullptr, {},
      nullptr, dV_p3);
  inline static partial_term<P> const r2_pterm2 = partial_term<P>(
      coefficient_type::div, r2_g2, nullptr,
      flux_type::downwind, boundary_condition::neumann,
      boundary_condition::neumann, homogeneity::homogeneous,
      homogeneity::homogeneous, {}, nullptr, {},
      nullptr, dV_z3);

  // 2. combine partial terms into single dimension term
  inline static term<P> const r2_term_p =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "R2_p",          // name
              {r2_pterm1});
  inline static term<P> const r2_term_z =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "R2_z",          // name
              {r2_pterm2});

  // 3. combine single dimension terms into multi dimension term
  inline static std::vector<term<P>> const termR2 = {r2_term_p, r2_term_z};

  // clang-format on
  */
  // collect all the terms
  // inline static term_set<P> const terms_ = {termC1, termC2, termC3, termE1,
  //                                          termE2, termE3, termR1, termR2};
  inline static term_set<P> const terms_ = {termC1, termC2, termC3,
                                            termE1, termE2, termE3};
  // --------------
  //
  // define sources
  //
  // --------------

  inline static std::vector<source<P>> const sources_ = {};

  // ------------------
  //
  // get time step (dt)
  //
  // ------------------

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dx      = x_range / fm::ipow2(dim.get_level());
    P const dt      = dx;
    // this will be scaled by CFL from command line
    return dt;
  }

  // -------------------------------------------------
  //
  // define exact soln functions (unused for this PDE)
  //
  // -------------------------------------------------

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {};
};
} // namespace asgard
