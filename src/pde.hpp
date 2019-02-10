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

#include "tensors.hpp"

// used to suppress warnings in unused variables
auto const ignore = [](auto ignored) { (void)ignored; };

//
// the choices for supported PDE types
//
enum class PDE_opts
{
  continuity_1,
  continuity_2,
  continuity_3,
  // FIXME the below have not been implemented according to the
  // new specification. david is working on that in the matlab
  vlasov4,  // PDE corresponding to Fig. 4 in FIXME
  vlasov43, // PDE corresponding to Fig. 4.3 in FIXME
  vlasov5,  // PDE corresponding to Fig. 5 in FIXME
  vlasov7,  // PDE corresponding to Fig. 7 in FIXME
  vlasov8,  // PDE corresponding to Fig. 8 in FIXME
  pde_user  // FIXME will need to add the user supplied PDE choice
};

//
// map those choices to selection strings
//
using pde_map_t                    = std::map<std::string, PDE_opts>;
static pde_map_t const pde_mapping = {{"continuity_1", PDE_opts::continuity_1},
                                      {"continuity_2", PDE_opts::continuity_2},
                                      {"continuity_3", PDE_opts::continuity_3},
                                      {"pde_user", PDE_opts::pde_user},
                                      {"vlasov4", PDE_opts::vlasov4},
                                      {"vlasov7", PDE_opts::vlasov7},
                                      {"vlasov8", PDE_opts::vlasov8},
                                      {"vlasov5", PDE_opts::vlasov5},
                                      {"vlasov43", PDE_opts::vlasov43}};

// for passing around vector/scalar-valued functions used by the PDE
template<typename P>
using vector_func = std::function<fk::vector<P>(fk::vector<P> const)>;
template<typename P>
using scalar_func = std::function<P(P const)>;

//----------------------------------------------------------------------------
//
// Define class members of PDE
//
//----------------------------------------------------------------------------

enum class boundary_condition
{
  periodic,
  dirichlet,
  neumann
};

// ---------------------------------------------------------------------------
//
// Dimension: holds all information for a single dimension in the pde
//
// ---------------------------------------------------------------------------

template<typename P>
class dimension
{
public:
  boundary_condition const left;
  boundary_condition const right;
  P const domain_min;
  P const domain_max;
  int const level;
  int const degree;
  vector_func<P> const initial_condition;
  std::string const name;
  dimension(boundary_condition const left, boundary_condition const right,
            P const domain_min, P const domain_max, int const level,
            int const degree, vector_func<P> const initial_condition,
            std::string const name)

      : left(left), right(right), domain_min(domain_min),
        domain_max(domain_max), level(level), degree(degree),
        initial_condition(initial_condition), name(name)
  {}
};

enum class coefficient_type
{
  grad,
  mass,
  stiffness
};

enum class flux_type
{
  central,
  upwind,
  lax_friedrich
};

// ---------------------------------------------------------------------------
//
// Term: describes a single term in the pde for operator matrix
// construction
//
// ---------------------------------------------------------------------------

// FIXME need to work on relationship with dimension
// do dimensions own terms? need dimension info in
// term construction...

template<typename P>
using g_func_type = std::function<P(P const, P const)>;

template<typename P>
class term
{
  // this is to hold data that may
  // change over the course of the
  // simulation, from any source,
  // that is used in operator construction.
  //
  // initialized to one if not provided at instantiation,
  // which performs an identity operation where this is used,
  // until set by outside source.
  fk::vector<P> data;

public:
  coefficient_type const coeff;
  g_func_type<P> const g_func;
  bool const time_dependent;
  flux_type const flux;
  std::string const name;
  term(coefficient_type const coeff, g_func_type<P> const g_func,
       bool const time_dependent, flux_type const flux,
       fk::vector<P> const data, std::string const name,
       dimension<P> const owning_dim)
      : data(data), coeff(coeff), g_func(g_func),
        time_dependent(time_dependent), flux(flux), name(name)
  {
    int const degrees_freedom_1d =
        owning_dim.degree * static_cast<int>(std::pow(2, owning_dim.level));
    if (data.size() != 0)
    {
      assert(data.size() == degrees_freedom_1d);
    }
    else
    {
      this->data.resize(degrees_freedom_1d);
      this->data = fk::vector<P>(std::vector<P>(
          owning_dim.degree * static_cast<int>(std::pow(2, owning_dim.level)),
          1.0));
    }
  }

  void set_data(fk::vector<P> const new_data)
  {
    assert(new_data.size() == data.size());
    data = new_data;
  };
  fk::vector<P> get_data() const { return data; };
};

// ---------------------------------------------------------------------------
//
// Source: a pde can have arbitrarily many, given that each has dimension-many
// vector valued functions and one scalar valued function (for time)
//
// ---------------------------------------------------------------------------

template<typename P>
class source
{
public:
  std::vector<vector_func<P>> const source_funcs;
  scalar_func<P> const time_func;

  source(std::vector<vector_func<P>> const source_funcs,
         scalar_func<P> const time_func)

      : source_funcs(source_funcs), time_func(time_func)
  {}
};

// ---------------------------------------------------------------------------
//
// abstract base class defining interface for PDEs
//
// ----------------------------------------------------------------------------
template<typename P>
using term_set = std::vector<std::vector<term<P>>>;

template<typename P>
class PDE
{
public:
  // clang-format off
  PDE(int const num_dims,
      int const num_sources,
      int const num_terms,
      std::vector<dimension<P>> const dimensions, 
      term_set<P> const terms,
      std::vector<source<P>> const sources,
      std::vector<vector_func<P>> const exact_vector_funcs,
      scalar_func<P> const exact_time,
      bool const do_poisson_solve = false,
      bool const has_analytic_soln = false)
      : num_dims(num_dims),
        num_sources(num_sources),
        num_terms(num_terms),
	dimensions(dimensions),
	terms(terms),
	sources(sources),
        exact_vector_funcs(exact_vector_funcs),
	exact_time(exact_time),
	do_poisson_solve(do_poisson_solve),
        has_analytic_soln(has_analytic_soln)
  // clang-format on
  {
    assert(num_dims > 0);
    assert(num_sources >= 0);
    assert(num_terms > 0);

    assert(dimensions.size() == static_cast<unsigned>(num_dims));
    assert(terms.size() == static_cast<unsigned>(num_terms));
    assert(sources.size() == static_cast<unsigned>(num_sources));

    // ensure analytic solution functions
    // were provided if this flag is set
    if (has_analytic_soln)
    {
      assert(exact_vector_funcs.size() == static_cast<unsigned>(num_dims));
    }

    // check all dimensions
    for (dimension<P> const d : dimensions)
    {
      assert(d.degree > 0);
      assert(d.level > 0);
      assert(d.domain_max > d.domain_min);
    }

    // check all sources
    for (source<P> const s : sources)
    {
      assert(s.source_funcs.size() == static_cast<unsigned>(num_dims));
    }

    // check all terms
    for (std::vector<term<P>> const term_list : terms)
    {
      assert(term_list.size() == static_cast<unsigned>(num_dims));
    }
  }

  // public but const data. no getters
  int const num_dims;
  int const num_sources;
  int const num_terms;
  std::vector<dimension<P>> const dimensions;
  term_set<P> const terms;
  std::vector<source<P>> const sources;
  std::vector<vector_func<P>> const exact_vector_funcs;
  scalar_func<P> const exact_time;
  bool const do_poisson_solve;
  bool const has_analytic_soln;

  virtual ~PDE() {}
};

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

  //
  // specify initial condition vector functions...
  //
  static fk::vector<P> initial_condition_dim0(fk::vector<P> const x)
  {
    return fk::vector<P>(std::vector<P>(x.size(), 0.0));
  }

  //
  // specify exact solution vectors/time function...
  //
  static fk::vector<P> exact_solution_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::cos(2.0 * M_PI * x);
                   });
    return fx;
  }

  static P exact_time(P const time) { return std::sin(time); }

  //
  // specify source functions...
  //

  // source 0
  static fk::vector<P> source_0_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::cos(2.0 * M_PI * x);
                   });
    return fx;
  }

  static P source_0_time(P const time) { return std::sin(time); }

  // source 1
  static fk::vector<P> source_1_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::sin(2.0 * M_PI * x);
                   });
    return fx;
  }

  static P source_1_time(P const time) { return -2.0 * M_PI * std::sin(time); }

  // g-funcs for terms (optional)
  static P g_func_0(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

private:
  static int constexpr _num_dims           = 1;
  static int constexpr _num_sources        = 2;
  static int constexpr _num_terms          = 1;
  static bool constexpr _do_poisson_solve  = false;
  static bool constexpr _has_analytic_soln = true;

  // define dimensions
  static boundary_condition constexpr dim0_BCL = boundary_condition::periodic;
  static boundary_condition constexpr dim0_BCR = boundary_condition::periodic;
  static P constexpr dim0_min                  = -1.0;
  static P constexpr dim0_max                  = 1.0;
  static int constexpr dim0_level              = 2;
  static int constexpr dim0_degree             = 2;
  inline static vector_func<P> const dim0_initial_condition =
      &PDE_continuity_1::initial_condition_dim0;
  static auto constexpr dim0_name = "x";
  inline static dimension<P> const dim0 =
      dimension<P>(dim0_BCL, dim0_BCR, dim0_min, dim0_max, dim0_level,
                   dim0_degree, dim0_initial_condition, dim0_name);

  // define terms (1 in this case)
  static coefficient_type constexpr term0_dim0_type = coefficient_type::grad;
  inline static g_func_type<P> const term0_dim0_func =
      &PDE_continuity_1::g_func_0;
  static bool constexpr term0_dim0_time_dependent = false;
  inline static fk::vector<P> const term0_dim0_data; // empty in this case
  static flux_type constexpr term0_dim0_flux = flux_type::central;
  static auto constexpr term0_dim0_name      = "d_dx";

  // TODO need to work on relationship with dimension....?
  inline static term<P> const term0_dim0 =
      term<P>(term0_dim0_type, term0_dim0_func, term0_dim0_time_dependent,
              term0_dim0_flux, term0_dim0_data, term0_dim0_name, dim0);
  inline static const std::vector<term<P>> terms0 = {term0_dim0};

  // define sources
  inline static std::vector<vector_func<P>> const source0_funcs = {
      &PDE_continuity_1::source_0_dim0};
  inline static scalar_func<P> const source0_time =
      &PDE_continuity_1::source_0_time;
  inline static source<P> const source0 =
      source<P>(source0_funcs, source0_time);

  inline static std::vector<vector_func<P>> const source1_funcs = {
      &PDE_continuity_1::source_1_dim0};
  inline static scalar_func<P> const source1_time =
      &PDE_continuity_1::source_1_time;
  inline static source<P> const source1 =
      source<P>(source1_funcs, source1_time);

  // store PDE functions/information in members
  inline static std::vector<dimension<P>> const dimensions = {dim0};
  inline static term_set<P> const terms                    = {terms0};
  inline static std::vector<source<P>> const sources       = {source0, source1};

  inline static std::vector<vector_func<P>> const _exact_vector_funcs = {
      &PDE_continuity_1::exact_solution_dim0};

  inline static scalar_func<P> const _exact_scalar_func =
      &PDE_continuity_1::exact_time;
};

// ---------------------------------------------------------------------------
//
// the "continuity 2" pde
//
// 2D test case using continuity equation, i.e.,
// df/dt + v_x * df/dx + v_y * df/dy == 0
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_continuity_2 : public PDE<P>
{
public:
  PDE_continuity_2()
      : PDE<P>(_num_dims, _num_sources, _num_terms, dimensions, terms, sources,
               _exact_vector_funcs, _exact_scalar_func, _do_poisson_solve,
               _has_analytic_soln)
  {}

  //
  // specify initial condition vector functions...
  //
  static fk::vector<P> initial_condition_dim0(fk::vector<P> const x)
  {
    return fk::vector<P>(std::vector<P>(x.size(), 0.0));
  }
  static fk::vector<P> initial_condition_dim1(fk::vector<P> const x)
  {
    return fk::vector<P>(std::vector<P>(x.size(), 0.0));
  }

  //
  // specify exact solution vectors/time function...
  //
  static fk::vector<P> exact_solution_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::cos(M_PI * x);
                   });
    return fx;
  }
  static fk::vector<P> exact_solution_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::sin(2.0 * M_PI * x);
                   });
    return fx;
  }

  static P exact_time(P const time) { return std::sin(2.0 * time); }

  //
  // specify source functions...
  //

  // source 0
  static fk::vector<P> source_0_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::cos(M_PI * x);
                   });
    return fx;
  }

  static fk::vector<P> source_0_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::sin(2.0 * M_PI * x);
                   });
    return fx;
  }

  static P source_0_time(P const time) { return 2.0 * std::cos(2.0 * time); }

  // source 1
  static fk::vector<P> source_1_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::cos(M_PI * x);
                   });
    return fx;
  }

  static fk::vector<P> source_1_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::cos(2.0 * M_PI * x);
                   });
    return fx;
  }

  static P source_1_time(P const time)
  {
    return 2.0 * M_PI * std::sin(2.0 * time);
  }

  // source 2
  static fk::vector<P> source_2_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::sin(M_PI * x);
                   });
    return fx;
  }

  static fk::vector<P> source_2_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                   [](P const &x, P const &fx) {
                     ignore(fx);
                     return std::sin(2.0 * M_PI * x);
                   });
    return fx;
  }

  static P source_2_time(P const time) { return -M_PI * std::sin(2.0 * time); }

  // g-funcs for terms (optional)
  static P g_func_identity(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

private:
  static int constexpr _num_dims           = 2;
  static int constexpr _num_sources        = 3;
  static int constexpr _num_terms          = 2;
  static bool constexpr _do_poisson_solve  = false;
  static bool constexpr _has_analytic_soln = true;

  // define dimensions
  static boundary_condition constexpr dim0_BCL = boundary_condition::periodic;
  static boundary_condition constexpr dim0_BCR = boundary_condition::periodic;
  static P constexpr dim0_min                  = -1.0;
  static P constexpr dim0_max                  = 1.0;
  static int constexpr dim0_level              = 2;
  static int constexpr dim0_degree             = 2;
  inline static vector_func<P> const dim0_initial_condition =
      &PDE_continuity_2::initial_condition_dim0;
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
      &PDE_continuity_2::initial_condition_dim1;
  static auto constexpr dim1_name = "y";
  inline static dimension<P> const dim1 =
      dimension<P>(dim1_BCL, dim1_BCR, dim1_min, dim1_max, dim1_level,
                   dim1_degree, dim1_initial_condition, dim1_name);

  // define terms

  // term 0
  static coefficient_type constexpr term0_dim0_type = coefficient_type::grad;
  inline static g_func_type<P> const term0_dim0_func =
      &PDE_continuity_2::g_func_identity;
  static bool constexpr term0_dim0_time_dependent = false;
  inline static fk::vector<P> const term0_dim0_data; // empty in this case
  static flux_type constexpr term0_dim0_flux = flux_type::central;
  static auto constexpr term0_dim0_name      = "v_x.d_dx";

  inline static term<P> const term0_dim0 =
      term<P>(term0_dim0_type, term0_dim0_func, term0_dim0_time_dependent,
              term0_dim0_flux, term0_dim0_data, term0_dim0_name, dim0);

  static coefficient_type constexpr term0_dim1_type = coefficient_type::mass;
  inline static g_func_type<P> const term0_dim1_func =
      &PDE_continuity_2::g_func_identity;
  static bool constexpr term0_dim1_time_dependent = false;
  inline static fk::vector<P> const term0_dim1_data; // empty in this case
  static flux_type constexpr term0_dim1_flux = flux_type::central;
  static auto constexpr term0_dim1_name      = "massY";

  inline static term<P> const term0_dim1 =
      term<P>(term0_dim1_type, term0_dim1_func, term0_dim1_time_dependent,
              term0_dim1_flux, term0_dim1_data, term0_dim1_name, dim1);

  inline static const std::vector<term<P>> terms0 = {term0_dim0, term0_dim1};

  // term 1
  static coefficient_type constexpr term1_dim0_type = coefficient_type::mass;
  inline static g_func_type<P> const term1_dim0_func =
      &PDE_continuity_2::g_func_identity;
  static bool constexpr term1_dim0_time_dependent = false;
  inline static fk::vector<P> const term1_dim0_data; // empty in this case
  static flux_type constexpr term1_dim0_flux = flux_type::central;
  static auto constexpr term1_dim0_name      = "massX";

  inline static term<P> const term1_dim0 =
      term<P>(term1_dim0_type, term1_dim0_func, term1_dim0_time_dependent,
              term1_dim0_flux, term1_dim0_data, term1_dim0_name, dim0);

  static coefficient_type constexpr term1_dim1_type = coefficient_type::grad;
  inline static g_func_type<P> const term1_dim1_func =
      &PDE_continuity_2::g_func_identity;
  static bool constexpr term1_dim1_time_dependent = false;
  inline static fk::vector<P> const term1_dim1_data; // empty in this case
  static flux_type constexpr term1_dim1_flux = flux_type::central;
  static auto constexpr term1_dim1_name      = "massY";

  inline static term<P> const term1_dim1 =
      term<P>(term1_dim1_type, term1_dim1_func, term1_dim1_time_dependent,
              term1_dim1_flux, term1_dim1_data, term1_dim1_name, dim1);

  inline static const std::vector<term<P>> terms1 = {term0_dim0, term0_dim1};

  // define sources
  inline static std::vector<vector_func<P>> const source0_funcs = {
      &PDE_continuity_2::source_0_dim0, &PDE_continuity_2::source_0_dim1};
  inline static scalar_func<P> const source0_time =
      &PDE_continuity_2::source_0_time;
  inline static source<P> const source0 =
      source<P>(source0_funcs, source0_time);

  inline static std::vector<vector_func<P>> const source1_funcs = {
      &PDE_continuity_2::source_1_dim0, &PDE_continuity_2::source_1_dim1};
  inline static scalar_func<P> const source1_time =
      &PDE_continuity_2::source_1_time;
  inline static source<P> const source1 =
      source<P>(source1_funcs, source1_time);

  inline static std::vector<vector_func<P>> const source2_funcs = {
      &PDE_continuity_2::source_2_dim0, &PDE_continuity_2::source_2_dim1};
  inline static scalar_func<P> const source2_time =
      &PDE_continuity_2::source_2_time;
  inline static source<P> const source2 =
      source<P>(source2_funcs, source2_time);

  // store PDE functions/information in members
  inline static std::vector<dimension<P>> const dimensions = {dim0, dim1};
  inline static term_set<P> const terms                    = {terms0, terms1};
  inline static std::vector<source<P>> const sources       = {source0, source1,
                                                        source2};

  inline static std::vector<vector_func<P>> const _exact_vector_funcs = {
      &PDE_continuity_2::exact_solution_dim0,
      &PDE_continuity_2::exact_solution_dim1};

  inline static scalar_func<P> const _exact_scalar_func =
      &PDE_continuity_2::exact_time;
};

// ---------------------------------------------------------------------------
//
// the "continuity 3" pde
//
// 2D test case using continuity equation, i.e.,
// df/dt + v.grad(f) == 0 where v = {1,1,1}
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_continuity_3 : public PDE<P>
{
public:
  PDE_continuity_3()
      : PDE<P>(_num_dims, _num_sources, _num_terms, dimensions, terms, sources,
               _exact_vector_funcs, _exact_scalar_func, _do_poisson_solve,
               _has_analytic_soln)
  {}

  //
  // specify initial condition vector functions...
  //
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
  //
  // specify exact solution vectors/time function...
  //
  static fk::vector<P> exact_solution_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(M_PI * x); });
    return fx;
  }
  static fk::vector<P> exact_solution_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * M_PI * x); });
    return fx;
  }

  static fk::vector<P> exact_solution_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * M_PI * x / 3.0); });
    return fx;
  }

  static P exact_time(P const time) { return std::sin(2.0 * time); }

  //
  // specify source functions...
  //

  // source 0
  static fk::vector<P> source_0_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(M_PI * x); });
    return fx;
  }

  static fk::vector<P> source_0_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * M_PI * x); });
    return fx;
  }

  static fk::vector<P> source_0_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * M_PI * x / 3.0); });
    return fx;
  }

  static P source_0_time(P const time) { return 2.0 * std::cos(2.0 * time); }

  // source 1
  static fk::vector<P> source_1_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(M_PI * x); });
    return fx;
  }

  static fk::vector<P> source_1_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * M_PI * x); });
    return fx;
  }

  static fk::vector<P> source_1_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * M_PI * x / 3.0); });
    return fx;
  }

  static P source_1_time(P const time)
  {
    return 2.0 * M_PI * std::sin(2.0 * time);
  }

  // source 2
  static fk::vector<P> source_2_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(M_PI * x); });
    return fx;
  }

  static fk::vector<P> source_2_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * M_PI * x); });
    return fx;
  }

  static fk::vector<P> source_2_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * M_PI * x / 3.0); });
    return fx;
  }

  static P source_2_time(P const time) { return -M_PI * std::sin(2.0 * time); }

  // source 3
  static fk::vector<P> source_3_dim0(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(M_PI * x); });
    return fx;
  }

  static fk::vector<P> source_3_dim1(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::sin(2.0 * M_PI * x); });
    return fx;
  }

  static fk::vector<P> source_3_dim2(fk::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(2.0 * M_PI * x / 3.0); });
    return fx;
  }

  static P source_3_time(P const time)
  {
    return -2.0 / 3.0 * M_PI * std::sin(2.0 * time);
  }

  // g-funcs for terms (optional)
  static P g_func_identity(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

private:
  static int constexpr _num_dims           = 3;
  static int constexpr _num_sources        = 4;
  static int constexpr _num_terms          = 3;
  static bool constexpr _do_poisson_solve  = false;
  static bool constexpr _has_analytic_soln = true;

  // define dimensions
  static boundary_condition constexpr dim0_BCL = boundary_condition::periodic;
  static boundary_condition constexpr dim0_BCR = boundary_condition::periodic;
  static P constexpr dim0_min                  = -1.0;
  static P constexpr dim0_max                  = 1.0;
  static int constexpr dim0_level              = 2;
  static int constexpr dim0_degree             = 2;
  inline static vector_func<P> const dim0_initial_condition =
      &PDE_continuity_3::initial_condition_dim0;
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
      &PDE_continuity_3::initial_condition_dim1;
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
      &PDE_continuity_3::initial_condition_dim2;
  static auto constexpr dim2_name = "z";
  inline static dimension<P> const dim2 =
      dimension<P>(dim2_BCL, dim2_BCR, dim2_min, dim2_max, dim2_level,
                   dim2_degree, dim2_initial_condition, dim2_name);

  // define terms

  // term 0
  static coefficient_type constexpr term0_dim0_type = coefficient_type::grad;
  inline static g_func_type<P> const term0_dim0_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term0_dim0_time_dependent = false;
  inline static fk::vector<P> const term0_dim0_data; // empty in this case
  static flux_type constexpr term0_dim0_flux = flux_type::central;
  static auto constexpr term0_dim0_name      = "v_x.d_dx";

  inline static term<P> const term0_dim0 =
      term<P>(term0_dim0_type, term0_dim0_func, term0_dim0_time_dependent,
              term0_dim0_flux, term0_dim0_data, term0_dim0_name, dim0);

  static coefficient_type constexpr term0_dim1_type = coefficient_type::mass;
  inline static g_func_type<P> const term0_dim1_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term0_dim1_time_dependent = false;
  inline static fk::vector<P> const term0_dim1_data; // empty in this case
  static flux_type constexpr term0_dim1_flux = flux_type::central;
  static auto constexpr term0_dim1_name      = "massY";

  inline static term<P> const term0_dim1 =
      term<P>(term0_dim1_type, term0_dim1_func, term0_dim1_time_dependent,
              term0_dim1_flux, term0_dim1_data, term0_dim1_name, dim1);

  static coefficient_type constexpr term0_dim2_type = coefficient_type::mass;
  inline static g_func_type<P> const term0_dim2_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term0_dim2_time_dependent = false;
  inline static fk::vector<P> const term0_dim2_data; // empty in this case
  static flux_type constexpr term0_dim2_flux = flux_type::central;
  static auto constexpr term0_dim2_name      = "massZ";

  inline static term<P> const term0_dim2 =
      term<P>(term0_dim2_type, term0_dim2_func, term0_dim2_time_dependent,
              term0_dim2_flux, term0_dim2_data, term0_dim2_name, dim2);

  inline static const std::vector<term<P>> terms0 = {term0_dim0, term0_dim1,
                                                     term0_dim2};

  // term 1
  static coefficient_type constexpr term1_dim0_type = coefficient_type::mass;
  inline static g_func_type<P> const term1_dim0_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term1_dim0_time_dependent = false;
  inline static fk::vector<P> const term1_dim0_data; // empty in this case
  static flux_type constexpr term1_dim0_flux = flux_type::central;
  static auto constexpr term1_dim0_name      = "massX";

  inline static term<P> const term1_dim0 =
      term<P>(term1_dim0_type, term1_dim0_func, term1_dim0_time_dependent,
              term1_dim0_flux, term1_dim0_data, term1_dim0_name, dim0);

  static coefficient_type constexpr term1_dim1_type = coefficient_type::grad;
  inline static g_func_type<P> const term1_dim1_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term1_dim1_time_dependent = false;
  inline static fk::vector<P> const term1_dim1_data; // empty in this case
  static flux_type constexpr term1_dim1_flux = flux_type::central;
  static auto constexpr term1_dim1_name      = "v_y.d_dy";

  inline static term<P> const term1_dim1 =
      term<P>(term1_dim1_type, term1_dim1_func, term1_dim1_time_dependent,
              term1_dim1_flux, term1_dim1_data, term1_dim1_name, dim1);

  static coefficient_type constexpr term1_dim2_type = coefficient_type::mass;
  inline static g_func_type<P> const term1_dim2_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term1_dim2_time_dependent = false;
  inline static fk::vector<P> const term1_dim2_data; // empty in this case
  static flux_type constexpr term1_dim2_flux = flux_type::central;
  static auto constexpr term1_dim2_name      = "massZ";

  inline static term<P> const term1_dim2 =
      term<P>(term1_dim2_type, term1_dim2_func, term1_dim2_time_dependent,
              term1_dim2_flux, term1_dim2_data, term1_dim2_name, dim2);

  inline static const std::vector<term<P>> terms1 = {term0_dim0, term0_dim1,
                                                     term0_dim2};

  // term 2
  static coefficient_type constexpr term2_dim0_type = coefficient_type::mass;
  inline static g_func_type<P> const term2_dim0_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term2_dim0_time_dependent = false;
  inline static fk::vector<P> const term2_dim0_data; // empty in this case
  static flux_type constexpr term2_dim0_flux = flux_type::central;
  static auto constexpr term2_dim0_name      = "massX";

  inline static term<P> const term2_dim0 =
      term<P>(term2_dim0_type, term2_dim0_func, term2_dim0_time_dependent,
              term2_dim0_flux, term2_dim0_data, term2_dim0_name, dim0);

  static coefficient_type constexpr term2_dim1_type = coefficient_type::mass;
  inline static g_func_type<P> const term2_dim1_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term2_dim1_time_dependent = false;
  inline static fk::vector<P> const term2_dim1_data; // empty in this case
  static flux_type constexpr term2_dim1_flux = flux_type::central;
  static auto constexpr term2_dim1_name      = "massY";

  inline static term<P> const term2_dim1 =
      term<P>(term2_dim1_type, term2_dim1_func, term2_dim1_time_dependent,
              term2_dim1_flux, term2_dim1_data, term2_dim1_name, dim1);

  static coefficient_type constexpr term2_dim2_type = coefficient_type::grad;
  inline static g_func_type<P> const term2_dim2_func =
      &PDE_continuity_3::g_func_identity;
  static bool constexpr term2_dim2_time_dependent = false;
  inline static fk::vector<P> const term2_dim2_data; // empty in this case
  static flux_type constexpr term2_dim2_flux = flux_type::central;
  static auto constexpr term2_dim2_name      = "v_z.d_dz";

  inline static term<P> const term2_dim2 =
      term<P>(term2_dim2_type, term2_dim2_func, term2_dim2_time_dependent,
              term2_dim2_flux, term2_dim2_data, term2_dim2_name, dim2);

  inline static const std::vector<term<P>> terms2 = {term2_dim0, term2_dim1,
                                                     term2_dim2};

  // define sources
  inline static std::vector<vector_func<P>> const source0_funcs = {
      &PDE_continuity_3::source_0_dim0, &PDE_continuity_3::source_0_dim1,
      &PDE_continuity_3::source_0_dim2};
  inline static scalar_func<P> const source0_time =
      &PDE_continuity_3::source_0_time;
  inline static source<P> const source0 =
      source<P>(source0_funcs, source0_time);

  inline static std::vector<vector_func<P>> const source1_funcs = {
      &PDE_continuity_3::source_1_dim0, &PDE_continuity_3::source_1_dim1,
      &PDE_continuity_3::source_1_dim2};
  inline static scalar_func<P> const source1_time =
      &PDE_continuity_3::source_1_time;
  inline static source<P> const source1 =
      source<P>(source1_funcs, source1_time);

  inline static std::vector<vector_func<P>> const source2_funcs = {
      &PDE_continuity_3::source_2_dim0, &PDE_continuity_3::source_2_dim1,
      &PDE_continuity_3::source_2_dim2};
  inline static scalar_func<P> const source2_time =
      &PDE_continuity_3::source_2_time;
  inline static source<P> const source2 =
      source<P>(source1_funcs, source2_time);

  inline static std::vector<vector_func<P>> const source3_funcs = {
      &PDE_continuity_3::source_3_dim0, &PDE_continuity_3::source_3_dim1,
      &PDE_continuity_3::source_3_dim2};
  inline static scalar_func<P> const source3_time =
      &PDE_continuity_3::source_3_time;
  inline static source<P> const source3 =
      source<P>(source3_funcs, source3_time);

  // store PDE functions/information in members
  inline static std::vector<dimension<P>> const dimensions = {dim0, dim1, dim2};
  inline static term_set<P> const terms              = {terms0, terms1, terms2};
  inline static std::vector<source<P>> const sources = {source0, source1,
                                                        source2, source3};

  inline static std::vector<vector_func<P>> const _exact_vector_funcs = {
      &PDE_continuity_3::exact_solution_dim0,
      &PDE_continuity_3::exact_solution_dim1,
      &PDE_continuity_3::exact_solution_dim2};

  inline static scalar_func<P> const _exact_scalar_func =
      &PDE_continuity_3::exact_time;
};

// ---------------------------------------------------------------------------
//
// A free function factory for making pdes. eventually will want to change the
// return for some of these once we implement them...
//
// ---------------------------------------------------------------------------
template<typename P>
std::unique_ptr<PDE<P>> make_PDE(PDE_opts choice)
{
  switch (choice)
  {
  case PDE_opts::continuity_1:
    return std::make_unique<PDE_continuity_1<P>>();
  case PDE_opts::continuity_2:
    return std::make_unique<PDE_continuity_2<P>>();

  // TODO not yet implemented, replace return with appropriate types
  case PDE_opts::continuity_3:
    return std::make_unique<PDE_continuity_1<P>>();
  case PDE_opts::vlasov4:
    return std::make_unique<PDE_continuity_1<P>>();
  case PDE_opts::vlasov43:
    return std::make_unique<PDE_continuity_1<P>>();
  case PDE_opts::vlasov5:
    return std::make_unique<PDE_continuity_1<P>>();
  case PDE_opts::vlasov7:
    return std::make_unique<PDE_continuity_1<P>>();
  case PDE_opts::vlasov8:
    return std::make_unique<PDE_continuity_1<P>>();
  case PDE_opts::pde_user:
    return std::make_unique<PDE_continuity_1<P>>();
  default:
    std::cout << "Invalid pde choice" << std::endl;
    exit(-1);
  }
}
