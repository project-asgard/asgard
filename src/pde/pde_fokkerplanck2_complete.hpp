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
  static int constexpr num_terms_          = 7;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = false;

  // define dimensions

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

  // g-funcs
  static P g_func_0(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -1.0;
  }
  static P g_func_1(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(time);
    return 1 - std::pow(x, 2);
  }
  static P g_func_2(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

  // define terms (1 in this case)
  // d/dz( (1-z^2) df/dz )
  //
  //    termC_z.type = 'diff';
  // eq1 : 1 * d/dx (1-z^2) q
  //    termC_z.G1 = @(z,p,t,dat) 1-z.^2;
  //    termC_z.LF1 = -1; % upwind left
  //    termC_z.BCL1 = 'D';
  //    termC_z.BCR1 = 'D';
  // eq2 : q = df/dx
  //    termC_z.G2 = @(z,p,t,dat) z*0+1;
  //    termC_z.LF2 = +1; % upwind right
  //    termC_z.BCL2 = 'N';
  //    termC_z.BCR2 = 'N';

  inline static class partial_term<P> partial_term_0 = partial_term<P>(
      coefficient_type::grad, g_func_1, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static class partial_term<P> partial_term_1 =
      partial_term<P>(coefficient_type::grad, g_func_2, flux_type::upwind,
                      boundary_condition::neumann, boundary_condition::neumann);

  inline static term<P> const term0_dim0_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "d_dx",          // name
              dim_p,           // owning dim
              {partial_term_0, partial_term_1});

  inline static const std::vector<term<P>> terms0_ = {term0_dim0_};

  inline static term_set<P> const terms_ = {terms0_};

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
