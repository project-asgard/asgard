#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "../tensors.hpp"
#include "pde_base.hpp"

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

template<typename P>
class PDE_fokkerplanck_2d_complete : public PDE<P>
{
public:
  PDE_fokkerplanck_2d_complete(int const num_levels = -1, int const degree = -1)
      : PDE<P>(num_levels, degree, num_dims_, num_sources_, num_terms_,
               dimensions_, terms_, sources_, exact_vector_funcs_,
               exact_scalar_func_, get_dt_, do_poisson_solve_,
               has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 2;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 1;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = false;

  //
  // define constants / data / functions for this PDE
  //

  static auto constexpr phi = [](P x) { return std::erf(x); };

  static auto constexpr psi = [](P x) {
    auto dphi_dx = 2.0 / std::sqrt(M_PI) * std::exp(-std::pow(x, 2));
    auto ret     = 1.0 / (2 * std::pow(x, 2)) * (phi(x) - x * dphi_dx);
    if (ret < 1e-5)
      ret = 0;
    return ret;
  };

  static P constexpr nuEE     = 1;
  static P constexpr vT       = 1;
  static P constexpr delta    = 0.042;
  static P constexpr Z        = 1;
  static P constexpr E        = 0.0025;
  static P constexpr tau      = 1e5;
  static auto constexpr gamma = [](P p) {
    return std::sqrt(1 + std::pow(delta * p, 2));
  };
  static auto constexpr vx = [](P p) { return 1.0 / vT * (p / gamma(p)); };

  static auto constexpr Ca = [](P p) {
    return nuEE * std::pow(vT, 2) * (psi(vx(p)) / vx(p));
  };

  //
  // define dimensions
  //

  // specify initial conditions for each dim
  // p dimension

  static int constexpr test = 1;

  // initial condition in p
  static fk::vector<P> initial_condition_p(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> ret(x.size());
    switch (test)
    {
    case 1:
      std::transform(x.begin(), x.end(), ret.begin(), [](auto elem) {
        return (elem <= 5) ? 3.0 / (2.0 * std::pow(5.0, 3)) : 0;
      });
    case 2:
      std::transform(x.begin(), x.end(), ret.begin(), [](auto elem) {
        int const a = 2;
        return 2.0 / (std::sqrt(M_PI) * std::pow(a, 3)) *
               std::exp(-std::pow(elem, 2) / std::pow(a, 2));
      });
    case 3:
      std::transform(x.begin(), x.end(), ret.begin(), [](auto elem) {
        return 2.0 / (3.0 * std::sqrt(M_PI)) * std::exp(-std::pow(elem, 2));
      });
    }
    return ret;
  }

  // initial conditionn in z
  static fk::vector<P> initial_condition_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> ret(x.size());
    switch (test)
    {
    case 1:
      std::fill(ret.begin(), ret.end(), 1.0);
    case 2:
      std::fill(ret.begin(), ret.end(), 1.0);
    case 3:
      fk::vector<P> ret(x.size());

      std::vector<P> const legendre_coeffs = {3, 0.5, 1, 0.7, 3, 0, 3};

      auto const [P_m, dP_m] =
          legendre(x, legendre_coeffs.size(), legendre_normalization::matlab);
      ignore(dP_m);

      // let's see you C++ nerds do this without a raw loop
      for (int i = 0; i < static_cast<int>(legendre_coeffs.size()); ++i)
      {
        fk::vector<P> const P_0 = P_m.extract_submatrix(0, i, x.size(), 1);
        ret = ret + (P_0 * legendre_coeffs[i] * std::exp(-i * (i + 1) * t));
      }
    }
    return ret;
  }

  // p dimension
  inline static dimension<P> const dim_p =
      dimension<P>(0.0,                 // domain min
                   10.0,                // domain max
                   2,                   // levels
                   2,                   // degree
                   initial_condition_p, // initial condition
                   "p");                // name

  // z dimension
  inline static dimension<P> const dim_z =
      dimension<P>(-1.0,                // domain min
                   +1.0,                // domain max
                   2,                   // levels
                   2,                   // degree
                   initial_condition_z, // initial condition
                   "z");                // name

  // assemble dimensions

  inline static std::vector<dimension<P>> const dimensions_ = {dim_p, dim_z};

  //
  // Setup the terms of the PDE
  //
  // -div(flux_C) == termC1 + termC2 + termC3
  //
  //
  // termC1 == 1/p^2*d/dp*p^2*Ca*df/dp
  //
  // becomes
  //
  // termC1 == g1(p) q(p)        [mass, g1(p) = 1/p^2,  BC N/A]
  //   q(p) == d/dp g2(p) r(p)   [grad, g2(p) = p^2*Ca, BCL=D,BCR=N]
  //   r(p) == d/dp g3(p) f(p)   [grad, g3(p) = 1,      BCL=N,BCR=D]

  // create a default mass matrix
  static P gI(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  inline static partial_term<P> pterm_I =
      partial_term<P>(coefficient_type::mass, gI);
  inline static term<P> const I_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "massY",         // name
              dim_p,           // owning dim
              {pterm_I});

  static P g1(P const x, P const time = 0)
  {
    ignore(time);
    return 1.0 / std::pow(x, 2);
  }
  static P g2(P const x, P const time = 0)
  {
    ignore(time);
    return std::pow(x, 2) * Ca(x);
  }
  static P g3(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  // create partial_terms

  inline static class partial_term<P> pterm1 =
      partial_term<P>(coefficient_type::mass, g1);

  inline static class partial_term<P> pterm2 = partial_term<P>(
      coefficient_type::grad, g2, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::neumann);

  inline static class partial_term<P> pterm3 = partial_term<P>(
      coefficient_type::grad, g3, flux_type::downwind,
      boundary_condition::neumann, boundary_condition::dirichlet);

  // combine partial terms into single dimension term

  inline static term<P> const term1_p =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "d_dx",          // name
              dim_p,           // owning dim
              {pterm1, pterm2, pterm3});

  // combine single dimension terms into multi dimension term

  inline static const std::vector<term<P>> termC1 = {term1_p, I_};

  // collect all the terms

  inline static term_set<P> const terms_ = {termC1};

  // define sources

  inline static std::vector<source<P>> const sources_ = {};

  // get time step (dt)

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dx      = x_range / std::pow(2, dim.get_level());
    P const dt      = std::pow(dx, 2);
    // this will be scaled by CFL
    // from command line
    return dt;
  }

  // define exact soln functions (unused for this PDE)

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {};
  inline static scalar_func<P> const exact_scalar_func_               = {};
};
