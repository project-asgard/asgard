#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
template<typename P>
class PDE_advection_1d : public PDE<P>
{
public:
  PDE_advection_1d(prog_opts const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 1;
  static int constexpr num_terms_          = 1;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;

  //
  // function definitions needed to build up the "dimension", "term", and
  // "source" member objects below for this PDE
  //

  // Initial Conditions ======================================================
  static fk::vector<P>
  initial_condition_dim0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      fx(i) = std::cos(x(i));
    }
    return fx;
  }
  // =========================================================================

  // Analytical Solutions ====================================================
  static fk::vector<P>
  exact_solution_dim0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      fx(i) = std::cos(x(i));
    }
    return fx;
  }

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution_dim0};

  // =========================================================================

  // Sources =================================================================
  static fk::vector<P> source_0_dim0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      fx(i) = -2.0 * std::sin(x(i));
    }
    return fx;
  }

  static P source_0_time(P const time)
  {
    ignore(time);
    return 1.0;
  }

  inline static source<P> const source0_ =
      source<P>({source_0_dim0}, source_0_time);
  inline static std::vector<source<P>> const sources_ = {source0_};
  // =========================================================================

  // Dimensions ==============================================================

  inline static dimension<P> const dim0_ = dimension<P>(
      0.0,  // domain min
      M_PI, // domain max
      4,    // levels - default (changed on command line with option -l)
      2,    // degree - default (changed on command line with option -d)
      initial_condition_dim0, // initial condition function
      nullptr,                // volume
      "x"                     // name of dimension
  );

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_};
  // =========================================================================

  // Terms ===================================================================
  static P g_func_0(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return -2.0;
  }

  static fk::vector<P> bc_func(fk::vector<P> const &x, P const time)
  {
    ignore(time);
    fk::vector<P> fx(x.size());
    std::fill(fx.begin(), fx.end(), 1.0);
    return fx;
  }

  static P bc_time_func(P const x)
  {
    ignore(x);
    return 1.0;
  }

  inline static const partial_term<P> partial_term_0 = partial_term<P>(
      coefficient_type::div, // type
      g_func_0,              // g func
      nullptr,               // lhs
      flux_type::upwind,     // flux = "-1" (upwind), "0" (central), "+1"

      boundary_condition::dirichlet, // left boundary condition type ("D", "N",
                                     // "P")
      boundary_condition::neumann,   // right boundary condition type,
      homogeneity::inhomogeneous,    // left homogeneity
      homogeneity::homogeneous,      // right homogeneity
      {bc_func},                     // left boundary condition function list
      bc_time_func,                  // left boundary time function
      {},                            // right boundary condition function list
      nullptr,                       // right boundary time function
      nullptr                        // surface jacobian
  );

  inline static term<P> term0_dim0_ =
      term<P>(false,           // time-dependency
              "term0",         // term name
              {partial_term_0} // list of partial terms to build the term
      );

  inline static std::vector<term<P>> const terms0_ = {term0_dim0_};
  inline static term_set<P> const terms_           = {terms0_};
  // =========================================================================

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dx      = x_range / fm::ipow2(dim.get_level());
    // return dx; this will be scaled by CFL
    // from command line
    return dx;
  }
};
} // namespace asgard
