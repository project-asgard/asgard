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
// the "fokkerplanck 1d - problem 4.3" pde
//
// Problem 4.3 from the RE paper - radiation damping term
// df/dt == -d/dz ( z(1-z^2)f )
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_fokkerplanck_1d_4p3 : public PDE<P>
{
public:
  PDE_fokkerplanck_1d_4p3(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 1;
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

  // analytic solution

  static P phi(P const z, P const t)
  {
    return z * std::exp(-t) /
           std::sqrt(1 + (std::exp(-2 * t) - 1) * std::pow(z, 2));
  }

  static P f0(P const z)
  {
    static P const sig = 0.1;

    return std::exp(-std::pow(z, 2) / std::pow(sig, 2));
  }

  static fk::vector<P>
  analytic_solution_dim0(fk::vector<P> const z, P const t = 0)
  {
    fk::vector<P> f(z.size());
    for (int i = 0; i < z.size(); ++i)
    {
      auto const p  = phi(z(i), t);
      auto const t1 = p * (1 - std::pow(p, 2));
      auto const t2 = z(i) * (1 - std::pow(z(i), 2));
      auto const t3 = f0(p);
      f(i)          = t1 / t2 * t3;
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

  // define dimensions
  inline static dimension<P> const dim0_ =
      dimension<P>(-1.0,                   // domain min
                   1.0,                    // domain max
                   2,                      // levels
                   2,                      // degree
                   initial_condition_dim0, // initial condition
                   "x");                   // name

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_};

  // define terms (1 in this case)
  //
  //  -d/dz ( (1-z^2)*f )

  // g-funcs
  static P g_func_1(P const x, P const time)
  {
    ignore(time);
    return -x * (1 - std::pow(x, 2));
  }

  inline static partial_term<P> const partial_term_0 = partial_term<P>(
      coefficient_type::grad, g_func_1, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term0_dim0_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "d_dx",          // name
              {partial_term_0});

  inline static std::vector<term<P>> const terms0_ = {term0_dim0_};

  inline static term_set<P> const terms_ = {terms0_};

  // define sources
  inline static std::vector<source<P>> const sources_ = {};

  // define exact soln functions
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      analytic_solution_dim0};

  inline static scalar_func<P> const exact_scalar_func_ =
      analytic_solution_time;
};
