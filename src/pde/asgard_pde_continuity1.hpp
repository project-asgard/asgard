#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// ---------------------------------------------------------------------------
//
// The "continuity 1d" PDE
//
// 1D test case using the Continuity Equation of simple transport, i.e.,
//
//                          df/dt + df/dx = S
//
// where S is an external source so that we may manufacture a solution
//
// The analytic solution is given by
//
//                      f(x,t) = cos(2*pi*x)*sin(t)
//
// The corresponding source is
//
//         S(x,t) = cos(2*pi*x)*cos(t) - 2*pi*sin(2*pi*x)*sin(t)
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_continuity_1d : public PDE<P>
{
public:
  PDE_continuity_1d(prog_opts const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 2;
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
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(2.0 * PI * x_v); });
    return fx;
  }

  // specify exact solution vectors/time function...
  static fk::vector<P> exact_solution_dim0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(2.0 * PI * x_v); });
    return fx;
  }

  static fk::vector<P> exact_time(fk::vector<P>, P const time)
  {
    return {
        std::sin(time),
    };
  }

  // specify source functions...

  // source 0
  static fk::vector<P> source_0_dim0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(2.0 * PI * x_v); });
    return fx;
  }

  static P source_0_time(P const time) { return std::cos(time); }

  // source 1
  static fk::vector<P> source_1_dim0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(2.0 * PI * x_v); });
    return fx;
  }

  static P source_1_time(P const time) { return -2.0 * PI * std::sin(time); }

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dx      = x_range / fm::ipow2(dim.get_level());
    // return dx; this will be scaled by CFL
    // from command line
    return dx;
  }
  // g-funcs for terms (optional)
  static P g_func_0(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -1.0;
  }

  // define dimensions
  inline static dimension<P> const dim0_ =
      dimension<P>(-1.0,                   // domain min
                   1.0,                    // domain max
                   2,                      // levels
                   1,                      // degree
                   initial_condition_dim0, // initial condition
                   nullptr,                // volume portion
                   "x");                   // name

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_};

  inline static const partial_term<P> partial_term_0 = partial_term<P>(
      coefficient_type::div, g_func_0, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  // define terms (1 in this case)
  inline static term<P> term0_dim0_ = term<P>(false,  // time-dependent
                                              "d_dx", // name
                                              {partial_term_0});

  inline static std::vector<term<P>> const terms0_ = {term0_dim0_};

  inline static term_set<P> const terms_ = {terms0_};

  // define sources
  inline static source<P> const source0_ =
      source<P>({source_0_dim0}, source_0_time);

  inline static source<P> const source1_ =
      source<P>({source_1_dim0}, source_1_time);

  inline static std::vector<source<P>> const sources_ = {source0_, source1_};

  // define exact soln functions
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution_dim0, exact_time};
};
} // namespace asgard
