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

#include "pde_base.hpp"
#include "tensors.hpp"

// ---------------------------------------------------------------------------
//
// the "continuity 1" pde
//
// 1D test case using continuity equation, i.e.,
// df/dt + df/dx = 0
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_continuity_1 : public PDE<P>
{
public:
  PDE_continuity_1()
      : PDE<P>(_num_dims, _num_sources, _num_terms, dimensions, terms, sources,
               _exact_vector_funcs, _exact_scalar_func, _do_poisson_solve,
               _has_analytic_soln)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr _num_dims           = 1;
  static int constexpr _num_sources        = 2;
  static int constexpr _num_terms          = 1;
  static bool constexpr _do_poisson_solve  = false;
  static bool constexpr _has_analytic_soln = true;

  // function definitions...

  // specify initial condition vector functions...
  static fk::vector<P> initial_condition_dim0(fk::vector<P> const x)
  {
    return fk::vector<P>(std::vector<P>(x.size(), 0.0));
  }

  // specify exact solution vectors/time function...
  static fk::vector<P> exact_solution_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * PI * x); });
    return fx;
  }

  static P exact_time(P const time) { return std::sin(time); }

  // specify source functions...

  // source 0
  static fk::vector<P> source_0_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * PI * x); });
    return fx;
  }

  static P source_0_time(P const time) { return std::cos(time); }

  // source 1
  static fk::vector<P> source_1_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * PI * x); });
    return fx;
  }

  static P source_1_time(P const time) { return -2.0 * PI * std::sin(time); }

  // g-funcs for terms (optional)
  static P g_func_0(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

  // define dimensions
  static boundary_condition constexpr dim0_BCL = boundary_condition::periodic;
  static boundary_condition constexpr dim0_BCR = boundary_condition::periodic;
  static P constexpr dim0_min                  = -1.0;
  static P constexpr dim0_max                  = 1.0;
  static int constexpr dim0_level              = 2;
  static int constexpr dim0_degree             = 2;
  inline static vector_func<P> const dim0_initial_condition =
      initial_condition_dim0;
  static auto constexpr dim0_name = "x";
  inline static dimension<P> const dim0 =
      dimension<P>(dim0_BCL, dim0_BCR, dim0_min, dim0_max, dim0_level,
                   dim0_degree, dim0_initial_condition, dim0_name);

  inline static std::vector<dimension<P>> const dimensions = {dim0};

  // define terms (1 in this case)
  static coefficient_type constexpr term0_dim0_type  = coefficient_type::grad;
  inline static g_func_type<P> const term0_dim0_func = g_func_0;
  static bool constexpr term0_dim0_time_dependent    = false;
  inline static fk::vector<P> const term0_dim0_data; // empty in this case
  static flux_type constexpr term0_dim0_flux = flux_type::central;
  static auto constexpr term0_dim0_name      = "d_dx";

  inline static term<P> const term0_dim0 =
      term<P>(term0_dim0_type, term0_dim0_func, term0_dim0_time_dependent,
              term0_dim0_flux, term0_dim0_data, term0_dim0_name, dim0);
  inline static const std::vector<term<P>> terms0 = {term0_dim0};

  inline static term_set<P> const terms = {terms0};

  // define sources
  inline static std::vector<vector_func<P>> const source0_funcs = {
      source_0_dim0};
  inline static scalar_func<P> const source0_time = source_0_time;
  inline static source<P> const source0 =
      source<P>(source0_funcs, source0_time);

  inline static std::vector<vector_func<P>> const source1_funcs = {
      source_1_dim0};
  inline static scalar_func<P> const source1_time = source_1_time;
  inline static source<P> const source1 =
      source<P>(source1_funcs, source1_time);

  inline static std::vector<source<P>> const sources = {source0, source1};

  // define exact soln functions
  inline static std::vector<vector_func<P>> const _exact_vector_funcs = {
      exact_solution_dim0};

  inline static scalar_func<P> const _exact_scalar_func = exact_time;
};
