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
// the "continuity 3d" pde
//
// 2D test case using continuity equation, i.e.,
// df/dt + v.grad(f) == 0 where v = {1,1,1}
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_continuity_3d : public PDE<P>
{
public:
  PDE_continuity_3d()
      : PDE<P>(num_dims_, num_sources_, num_terms_, dimensions, terms, sources,
               _exact_vector_funcs, _exact_scalar_func, do_poisson_solve_,
               has_analytic_soln_)
  {}

private:
  // these fields used to check correctness of specification
  static int constexpr num_dims_           = 3;
  static int constexpr num_sources_        = 4;
  static int constexpr num_terms_          = 3;
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
  static fk::vector<P> initial_condition_dim2(fk::vector<P> const x)
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

  static fk::vector<P> exact_solution_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * PI * x / 3.0); });
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

  static fk::vector<P> source_0_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * PI * x / 3.0); });
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

  static fk::vector<P> source_1_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * PI * x / 3.0); });
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

  static fk::vector<P> source_2_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * PI * x / 3.0); });
    return fx;
  }

  static P source_2_time(P const time) { return -PI * std::sin(2.0 * time); }

  // source 3
  static fk::vector<P> source_3_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(PI * x); });
    return fx;
  }

  static fk::vector<P> source_3_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * PI * x); });
    return fx;
  }

  static fk::vector<P> source_3_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * PI * x / 3.0); });
    return fx;
  }

  static P source_3_time(P const time)
  {
    return -2.0 / 3.0 * PI * std::sin(2.0 * time);
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

  static boundary_condition constexpr dim1_BCL = boundary_condition::periodic;
  static boundary_condition constexpr dim1_BCR = boundary_condition::periodic;
  static P constexpr dim1_min                  = -2.0;
  static P constexpr dim1_max                  = 2.0;
  static int constexpr dim1_level              = 2;
  static int constexpr dim1_degree             = 2;
  inline static vector_func<P> const dim1_initial_condition =
      initial_condition_dim1;
  static auto constexpr dim1_name = "y";
  inline static dimension<P> const dim1 =
      dimension<P>(dim1_BCL, dim1_BCR, dim1_min, dim1_max, dim1_level,
                   dim1_degree, dim1_initial_condition, dim1_name);

  static boundary_condition constexpr dim2_BCL = boundary_condition::periodic;
  static boundary_condition constexpr dim2_BCR = boundary_condition::periodic;
  static P constexpr dim2_min                  = -3.0;
  static P constexpr dim2_max                  = 3.0;
  static int constexpr dim2_level              = 2;
  static int constexpr dim2_degree             = 2;
  inline static vector_func<P> const dim2_initial_condition =
      initial_condition_dim2;
  static auto constexpr dim2_name = "z";
  inline static dimension<P> const dim2 =
      dimension<P>(dim2_BCL, dim2_BCR, dim2_min, dim2_max, dim2_level,
                   dim2_degree, dim2_initial_condition, dim2_name);

  inline static std::vector<dimension<P>> const dimensions = {dim0, dim1, dim2};

  // define terms

  // term 0
  static coefficient_type constexpr term0_dim0_type  = coefficient_type::grad;
  inline static g_func_type<P> const term0_dim0_func = g_func_identity;
  static bool constexpr term0_dim0_time_dependent    = false;
  inline static fk::vector<P> const term0_dim0_data; // empty in this case
  static flux_type constexpr term0_dim0_flux = flux_type::central;
  static auto constexpr term0_dim0_name      = "v_x.d_dx";

  inline static term<P> const term0_dim0 =
      term<P>(term0_dim0_type, term0_dim0_func, term0_dim0_time_dependent,
              term0_dim0_flux, term0_dim0_data, term0_dim0_name, dim0);

  static coefficient_type constexpr term0_dim1_type  = coefficient_type::mass;
  inline static g_func_type<P> const term0_dim1_func = g_func_identity;
  static bool constexpr term0_dim1_time_dependent    = false;
  inline static fk::vector<P> const term0_dim1_data; // empty in this case
  static flux_type constexpr term0_dim1_flux = flux_type::central;
  static auto constexpr term0_dim1_name      = "massY";

  inline static term<P> const term0_dim1 =
      term<P>(term0_dim1_type, term0_dim1_func, term0_dim1_time_dependent,
              term0_dim1_flux, term0_dim1_data, term0_dim1_name, dim1);

  static coefficient_type constexpr term0_dim2_type  = coefficient_type::mass;
  inline static g_func_type<P> const term0_dim2_func = g_func_identity;
  static bool constexpr term0_dim2_time_dependent    = false;
  inline static fk::vector<P> const term0_dim2_data; // empty in this case
  static flux_type constexpr term0_dim2_flux = flux_type::central;
  static auto constexpr term0_dim2_name      = "massZ";

  inline static term<P> const term0_dim2 =
      term<P>(term0_dim2_type, term0_dim2_func, term0_dim2_time_dependent,
              term0_dim2_flux, term0_dim2_data, term0_dim2_name, dim2);

  inline static const std::vector<term<P>> terms0 = {term0_dim0, term0_dim1,
                                                     term0_dim2};

  // term 1
  static coefficient_type constexpr term1_dim0_type  = coefficient_type::mass;
  inline static g_func_type<P> const term1_dim0_func = g_func_identity;
  static bool constexpr term1_dim0_time_dependent    = false;
  inline static fk::vector<P> const term1_dim0_data; // empty in this case
  static flux_type constexpr term1_dim0_flux = flux_type::central;
  static auto constexpr term1_dim0_name      = "massX";

  inline static term<P> const term1_dim0 =
      term<P>(term1_dim0_type, term1_dim0_func, term1_dim0_time_dependent,
              term1_dim0_flux, term1_dim0_data, term1_dim0_name, dim0);

  static coefficient_type constexpr term1_dim1_type  = coefficient_type::grad;
  inline static g_func_type<P> const term1_dim1_func = g_func_identity;
  static bool constexpr term1_dim1_time_dependent    = false;
  inline static fk::vector<P> const term1_dim1_data; // empty in this case
  static flux_type constexpr term1_dim1_flux = flux_type::central;
  static auto constexpr term1_dim1_name      = "v_y.d_dy";

  inline static term<P> const term1_dim1 =
      term<P>(term1_dim1_type, term1_dim1_func, term1_dim1_time_dependent,
              term1_dim1_flux, term1_dim1_data, term1_dim1_name, dim1);

  static coefficient_type constexpr term1_dim2_type  = coefficient_type::mass;
  inline static g_func_type<P> const term1_dim2_func = g_func_identity;
  static bool constexpr term1_dim2_time_dependent    = false;
  inline static fk::vector<P> const term1_dim2_data; // empty in this case
  static flux_type constexpr term1_dim2_flux = flux_type::central;
  static auto constexpr term1_dim2_name      = "massZ";

  inline static term<P> const term1_dim2 =
      term<P>(term1_dim2_type, term1_dim2_func, term1_dim2_time_dependent,
              term1_dim2_flux, term1_dim2_data, term1_dim2_name, dim2);

  inline static const std::vector<term<P>> terms1 = {term0_dim0, term0_dim1,
                                                     term0_dim2};

  // term 2
  static coefficient_type constexpr term2_dim0_type  = coefficient_type::mass;
  inline static g_func_type<P> const term2_dim0_func = g_func_identity;
  static bool constexpr term2_dim0_time_dependent    = false;
  inline static fk::vector<P> const term2_dim0_data; // empty in this case
  static flux_type constexpr term2_dim0_flux = flux_type::central;
  static auto constexpr term2_dim0_name      = "massX";

  inline static term<P> const term2_dim0 =
      term<P>(term2_dim0_type, term2_dim0_func, term2_dim0_time_dependent,
              term2_dim0_flux, term2_dim0_data, term2_dim0_name, dim0);

  static coefficient_type constexpr term2_dim1_type  = coefficient_type::mass;
  inline static g_func_type<P> const term2_dim1_func = g_func_identity;
  static bool constexpr term2_dim1_time_dependent    = false;
  inline static fk::vector<P> const term2_dim1_data; // empty in this case
  static flux_type constexpr term2_dim1_flux = flux_type::central;
  static auto constexpr term2_dim1_name      = "massY";

  inline static term<P> const term2_dim1 =
      term<P>(term2_dim1_type, term2_dim1_func, term2_dim1_time_dependent,
              term2_dim1_flux, term2_dim1_data, term2_dim1_name, dim1);

  static coefficient_type constexpr term2_dim2_type  = coefficient_type::grad;
  inline static g_func_type<P> const term2_dim2_func = g_func_identity;
  static bool constexpr term2_dim2_time_dependent    = false;
  inline static fk::vector<P> const term2_dim2_data; // empty in this case
  static flux_type constexpr term2_dim2_flux = flux_type::central;
  static auto constexpr term2_dim2_name      = "v_z.d_dz";

  inline static term<P> const term2_dim2 =
      term<P>(term2_dim2_type, term2_dim2_func, term2_dim2_time_dependent,
              term2_dim2_flux, term2_dim2_data, term2_dim2_name, dim2);

  inline static const std::vector<term<P>> terms2 = {term2_dim0, term2_dim1,
                                                     term2_dim2};

  inline static term_set<P> const terms = {terms0, terms1, terms2};

  // define sources
  inline static std::vector<vector_func<P>> const source0_funcs = {
      source_0_dim0, source_0_dim1, source_0_dim2};
  inline static scalar_func<P> const source0_time = source_0_time;
  inline static source<P> const source0 =
      source<P>(source0_funcs, source0_time);

  inline static std::vector<vector_func<P>> const source1_funcs = {
      source_1_dim0, source_1_dim1, source_1_dim2};
  inline static scalar_func<P> const source1_time = source_1_time;
  inline static source<P> const source1 =
      source<P>(source1_funcs, source1_time);

  inline static std::vector<vector_func<P>> const source2_funcs = {
      source_2_dim0, source_2_dim1, source_2_dim2};
  inline static scalar_func<P> const source2_time = source_2_time;
  inline static source<P> const source2 =
      source<P>(source2_funcs, source2_time);

  inline static std::vector<vector_func<P>> const source3_funcs = {
      source_3_dim0, source_3_dim1, source_3_dim2};
  inline static scalar_func<P> const source3_time = source_3_time;
  inline static source<P> const source3 =
      source<P>(source3_funcs, source3_time);

  inline static std::vector<source<P>> const sources = {source0, source1,
                                                        source2, source3};

  // define exact soln
  inline static std::vector<vector_func<P>> const _exact_vector_funcs = {
      exact_solution_dim0, exact_solution_dim1, exact_solution_dim2};

  inline static scalar_func<P> const _exact_scalar_func = exact_time;
};
