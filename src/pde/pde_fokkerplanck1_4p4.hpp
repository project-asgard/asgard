#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "../tensors.hpp"
#include "pde_base.hpp"

// ---------------------------------------------------------------------------
//
// the "fokkerplanck 1d - problem 4.4" pde
//
// Problem 4.4 from the RE paper - evolution of the pitch angle dependence
// of f in the presence of electric field acceleration and collisions
//
// df/dt == -E d/dz((1-z^2) f) + C d/dz((1-z^2) df/dz)
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_fokkerplanck_1d_4p4 : public PDE<P>
{
public:
  PDE_fokkerplanck_1d_4p4(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 2;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;

  //
  // function definitions needed to build up the "dimension", "term", and
  // "source" member objects below for this PDE
  //

  // specify initial condition vector functions...
  static fk::vector<P>
  initial_condition_dim0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    return analytic_solution_dim0(x, 0);
  }

  static P constexpr sig = 0.1;
  static P constexpr E   = 4.0;
  static P constexpr C   = 1.0;

  // analytic solution

  static P phi(P const z, P const t)
  {
    return z * std::exp(-t) /
           std::sqrt(1 - (std::exp(-2 * t) - 1) * std::pow(z, 2));
  }
  static P f0(P const z)
  {
    auto const shift = 0.36;
    return std::exp(-std::pow(z + shift, 2) / std::pow(sig, 2));
  }

  static fk::vector<P> f0_vec(fk::vector<P> const z, P const t = 0)
  {
    ignore(t);
    fk::vector<P> f(z.size());
    for (int i = 0; i < z.size(); ++i)
    {
      f(i) = f0(z(i));
    }
    return f;
  }

  static fk::vector<P>
  analytic_solution_dim0(fk::vector<P> const z, P const t = 0)
  {
    ignore(t);
    fk::vector<P> f(z.size());
    for (int i = 0; i < z.size(); ++i)
    {
      auto const A = E / C;
      f(i)         = A / (2 * std::sinh(A) * std::exp(A * z(i)));
    }
    return f;
  }

  static P analytic_solution_time(P const time)
  {
    ignore(time);
    return 1.0;
  }

  // specify source functions...

  // N/A

  // get time step (dt)

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dx      = x_range / std::pow(2, dim.get_level());
    P const dt      = dx;
    // this will be scaled by CFL
    // from command line
    return dt;
  }

  // g-funcs
  static P g_func_0(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }
  static P g_func_t1_z(P const x, P const time)
  {
    ignore(time);
    return -E * (1 - std::pow(x, 2));
  }
  static P g_func_t2_z1(P const x, P const time)
  {
    ignore(time);
    return 1 - std::pow(x, 2);
  }
  static P g_func_t2_z2(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  // define dimensions
  inline static dimension<P> const dim0_ =
      dimension<P>(-1.0,   // domain min
                   1.0,    // domain max
                   2,      // levels
                   2,      // degree
                   f0_vec, // initial condition
                   "x");   // name

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_};

  // define terms

  // term 1
  //
  // -E d/dz((1-z^2) f)

  inline static partial_term<P> const partial_term_0 = partial_term<P>(
      coefficient_type::grad, g_func_t1_z, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const termE_z =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "d_dx",          // name
              {partial_term_0});

  inline static std::vector<term<P>> const termE = {termE_z};

  // term 2
  //
  // +C * d/dz( (1-z^2) df/dz )
  inline static partial_term<P> const partial_term_1 = partial_term<P>(
      coefficient_type::grad, g_func_t2_z1, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static partial_term<P> const partial_term_2 =
      partial_term<P>(coefficient_type::grad, g_func_t2_z2, flux_type::upwind,
                      boundary_condition::neumann, boundary_condition::neumann);

  inline static term<P> const termC_z =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "d_dx",          // name
              {partial_term_1, partial_term_2});

  inline static std::vector<term<P>> const termC = {termC_z};

  inline static term_set<P> const terms_ = {termE, termC};

  // define sources
  inline static std::vector<source<P>> const sources_ = {};

  // define exact soln functions
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      analytic_solution_dim0};

  inline static scalar_func<P> const exact_scalar_func_ =
      analytic_solution_time;
};
