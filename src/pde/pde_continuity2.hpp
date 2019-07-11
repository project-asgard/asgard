#pragma once
#include <algorithm>
#include <cassert>
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
// the "continuity 2d" pde
//
// 2D test case using continuity equation, i.e.,
// df/dt + v_x * df/dx + v_y * df/dy == 0
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_continuity_2d : public PDE<P>
{
public:
  PDE_continuity_2d(int const num_levels = -1, int const degree = -1)
      : PDE<P>(num_levels, degree, num_dims_, num_sources_, num_terms_,
               dimensions_, terms_, sources_, exact_vector_funcs_,
               exact_scalar_func_, get_dt_, do_poisson_solve_,
               has_analytic_soln_)
  {}

private:
  // these fields used to check correctness of specification
  static int constexpr num_dims_           = 2;
  static int constexpr num_sources_        = 3;
  static int constexpr num_terms_          = 2;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;

  //
  // function definitions needed to build up the "dimension", "term", and
  // "source" member objects below for this PDE
  //

  // specify initial condition vector functions...
  static fk::vector<P> initial_condition_dim0(fk::vector<P> const x)
  {
    return fk::vector<P>(std::vector<P>(x.size(), 0.0));
  }
  static fk::vector<P> initial_condition_dim1(fk::vector<P> const x)
  {
    return fk::vector<P>(std::vector<P>(x.size(), 0.0));
  }

  // specify exact solution vectors/time function...
  static fk::vector<P> exact_solution_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(PI * x); });
    return fx;
  }
  static fk::vector<P> exact_solution_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * PI * x); });
    return fx;
  }

  static P exact_time(P const time) { return std::sin(2.0 * time); }

  // specify source functions...

  // source 0
  static fk::vector<P> source_0_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(PI * x); });
    return fx;
  }

  static fk::vector<P> source_0_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * PI * x); });
    return fx;
  }

  static P source_0_time(P const time) { return 2.0 * std::cos(2.0 * time); }

  // source 1
  static fk::vector<P> source_1_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(PI * x); });
    return fx;
  }

  static fk::vector<P> source_1_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * PI * x); });
    return fx;
  }

  static P source_1_time(P const time)
  {
    return 2.0 * PI * std::sin(2.0 * time);
  }

  // source 2
  static fk::vector<P> source_2_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(PI * x); });
    return fx;
  }

  static fk::vector<P> source_2_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * PI * x); });
    return fx;
  }

  static P source_2_time(P const time) { return -PI * std::sin(2.0 * time); }

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dx      = x_range / std::pow(2, dim.get_level());
    // return dx; this will be scaled by CFL
    // from command line
    return dx;
  }

  // g-funcs for terms (optional)
  static P g_func_identity(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

  // define dimensions
  inline static dimension<P> const dim0_ =
      dimension<P>(boundary_condition::periodic, // left boundary condition
                   boundary_condition::periodic, // right boundary condition
                   -1.0,                         // domain min
                   1.0,                          // domain max
                   2,                            // levels
                   2,                            // degree
                   initial_condition_dim0,       // initial condition
                   "x");                         // name

  inline static dimension<P> const dim1_ =
      dimension<P>(boundary_condition::periodic, // left boundary condition
                   boundary_condition::periodic, // right boundary condition
                   -2.0,                         // domain min
                   2.0,                          // domain max
                   2,                            // levels
                   2,                            // degree
                   initial_condition_dim1,       // initial condition
                   "y");                         // name

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_, dim1_};

  // define terms
  // term 0
  inline static term<P> const term0_dim0_ =
      term<P>(coefficient_type::grad, // operator type
              g_func_identity,        // construction function
              false,                  // time-dependent
              flux_type::central,     // flux type
              fk::vector<P>(),        // additional data vector
              "v_x.d_dx",             // name
              dim0_);                 // owning dim

  inline static term<P> const term0_dim1_ =
      term<P>(coefficient_type::mass, // operator type
              g_func_identity,        // construction function
              false,                  // time-dependent
              flux_type::central,     // flux type
              fk::vector<P>(),        // additional data vector
              "massY",                // name
              dim1_);                 // owning dim

  inline static const std::vector<term<P>> terms0_ = {term0_dim0_, term0_dim1_};

  // term 1
  inline static term<P> const term1_dim0_ =
      term<P>(coefficient_type::mass, // operator type
              g_func_identity,        // construction function
              false,                  // time-dependent
              flux_type::central,     // flux type
              fk::vector<P>(),        // additional data vector
              "massX",                // name
              dim0_);                 // owning dim

  inline static term<P> const term1_dim1_ =
      term<P>(coefficient_type::grad, // operator type
              g_func_identity,        // construction function
              false,                  // time-dependent
              flux_type::central,     // flux type
              fk::vector<P>(),        // additional data vector
              "massY",                // name
              dim1_);                 // owning dim

  inline static const std::vector<term<P>> terms1_ = {term1_dim0_, term1_dim1_};

  inline static term_set<P> const terms_ = {terms0_, terms1_};

  // define sources
  inline static source<P> const source0_ =
      source<P>({source_0_dim0, source_0_dim1}, source_0_time);
  inline static source<P> const source1_ =
      source<P>({source_1_dim0, source_1_dim1}, source_1_time);
  inline static source<P> const source2_ =
      source<P>({source_2_dim0, source_2_dim1}, source_2_time);
  inline static std::vector<source<P>> const sources_ = {source0_, source1_,
                                                         source2_};

  // define exact soln
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution_dim0, exact_solution_dim1};

  inline static scalar_func<P> const exact_scalar_func_ = exact_time;
};
