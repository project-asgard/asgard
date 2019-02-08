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

//
// The choices for supported PDE types
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

//
// for passing around vector/scalar-valued functions used by the PDE
//
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
            int const degree, vector_func<P> const initial_condition)

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
  scalar_func<P> const g_func;
  bool const time_dependent;
  flux_type const flux;
  std::string const name;
  term(coefficient_type const coeff, scalar_func<P> const g_func,
       bool const time_dependent, flux_type const flux, dimension<P> const dim,
       fk::vector<P> const data)
      : coeff(coeff), g_func(g_func), time_dependent(time_dependent),
        data(data), name(name)
  {
    int const degrees_freedom_1d =
        dim.degree * static_cast<int>(std::pow(2, dim.level));
    if (data.size() != 0)
    {
      assert(data.size() == degrees_freedom_1d);
    }
    else
    {
      data.resize(degrees_freedom_1d);
      data = fk::vector<P>(std::vector<P>(
          dim.degree * static_cast<int>(std::pow(2, dim.level)), 1.0));
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
  std::vector<scalar_func<P>> const time_func;

  source(std::vector<vector_func<P>> const source_funcs,
         std::vector<scalar_func<P>> const time_func)

      : source_funcs(source_funcs), time_func(time_func)
  {}
};

// ---------------------------------------------------------------------------
//
// abstract base class defining interface for PDEs
//
// ----------------------------------------------------------------------------
template<typename P>
class PDE
{
public:
  // clang-format off
  PDE(int const num_dims,
      int const num_sources,
      int const num_terms,
      std::vector<dimension<P>> const dimensions, 
      std::vector<term<P>> const terms,
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
      assert(exact_vector_funcs().size() == static_cast<unsigned>(num_dims));
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
      for (auto const &funcs : s.source_funcs)
      {
        assert(funcs.size() == static_cast<unsigned>(num_dims));
      }

      assert(s.time_funcs.size() == static_cast<unsigned>(num_dims));
    }
  }

  /*
    // getters for PDE info provided by the derived implementation
    virtual std::vector<dimension<P>> get_dimensions() const = 0;
    virtual std::vector<term<P>> get_terms() const           = 0;
    virtual std::vector<source<P>> get_sources() const       = 0;

    // optional, only for pdes with analytic solution
    virtual std::vector<vector_func<P>> exact_vector_funcs() const
    {
      if (has_analytic_soln)
        std::cout << "exact functions expected, but not provided. exiting."
                  << std::endl;
      else
        std::cout << "PDE has not analytic solutions. exiting." << std::endl;

      exit(1);
    }
    virtual scalar_func<P> exact_scalar_func() const
    {
      if (has_analytic_soln)
        std::cout << "exact functions expected, but not provided. exiting."
                  << std::endl;
      else
        std::cout << "PDE has not analytic solutions. exiting." << std::endl;

      exit(1);
    }
  */
  // public but const data. no getters
  int const num_dims;
  int const num_sources;
  int const num_terms;
  std::vector<dimension<P>> const dimensions;
  std::vector<term<P>> const terms;
  std::vector<source<P>> const sources;
  std::vector<vector_func<P>> const exact_solution_funcs;
  scalar_func<P> const exact_time;
  bool const do_poisson_solve;
  bool const has_analytic_soln;

  virtual ~PDE() {}

  /*protected:
    void verify() const
    {
      assert(get_dimensions().size() == static_cast<unsigned>(num_dims));
      assert(get_terms().size() == static_cast<unsigned>(num_terms));
      assert(get_sources().size() == static_cast<unsigned>(num_sources));

      // ensure analytic solution functions
      // were provided if this flag is set
      if (has_analytic_soln)
      {
        assert(exact_vector_funcs().size() == static_cast<unsigned>(num_dims));
        assert(exact_scalar_func());
      }

      // check all dimensions
      for (dimension<P> const d : get_dimensions())
      {
        assert(d.degree > 0);
        assert(d.level > 0);
        assert(d.domain_max > d.domain_min);
      }

      // check all sources
      for (source<P> const s : get_sources())
      {
        for (auto const &funcs : s.source_funcs)
        {
          assert(funcs.size() == static_cast<unsigned>(num_dims));
        }

        assert(s.time_funcs.size() == static_cast<unsigned>(num_dims));
      }
    }*/
};

// ---------------------------------------------------------------------------
//
// the "continuity 1" pde
//
//
// It ... (FIXME explain properties like having an analytic solution, etc.)
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
  {
    // PDE<P>::verify();
  }

  /*  std::vector<dimension<P>> get_dimensions() const override {
          return dimensions;
    }

    std::vector<term<P>> get_terms() const override {
          return terms;
    }

    std::vector<source<P>> get_sources() const override {
          return sources;
    }
  */
  //
  // specify initial condition vector functions...
  //
  static fk::vector<P> initial_condition_0(fk::vector<P> const x)
  {
    return fk::vector<P>(std::vector<P>(x.size(), 0.0));
  }

  //
  // specify exact solution vectors/time function...
  //
  static std::vector<P> exact_solution_0(std::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(
        fx.begin(), fx.end(), x.begin(), fx.begin(),
        [](P const &x, P const &fx) { return std::cos(2.0 * M_PI * x); });
    return fx;
  }

  static P exact_time(P const time) { return std::sin(time); }

  //
  // specify source functions...
  //

  // source 0
  static std::vector<P> source_dim0_0(std::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(
        fx.begin(), fx.end(), x.begin(), fx.begin(),
        [](P const &x, P const &fx) { return std::cos(2.0 * M_PI * x); });
    return fx;
  }

  static P source_dim0_0_time(P const time) { return std::sin(time); }

  // source 1
  static std::vector<P> source_dim0_1(std::vector<P> const x)
  {
    fk::vector<P> fx(x.size());
    std::transform(
        fx.begin(), fx.end(), x.begin(), fx.begin(),
        [](P const &x, P const &fx) { return std::sin(2.0 * M_PI * x); });
    return fx;
  }

  static P source_dim0_1_time(P const time)
  {
    return -2.0 * M_PI * std::sin(time);
  }

  // g-funcs for terms (optional)
  static scalar_func<P> g_func_0(P const x, P const time) { return 1.0; }

  //
  // implement the virtual accessors expected by the PDE<P> base
  //

  std::vector<vector_func<P>> exact_vector_funcs() const override
  {
    return _exact_vector_funcs;
  }
  scalar_func<P> exact_scalar_func() const override
  {
    return _exact_scalar_func;
  }

private:
  static int constexpr _num_dims           = 1;
  static int constexpr _num_sources        = 3;
  static int constexpr _num_terms          = 2;
  static bool constexpr _do_poisson_solve  = false;
  static bool constexpr _has_analytic_soln = true;

  // define dimensions
  static bool constexpr dim0_BCL   = false;
  static bool constexpr dim0_BCR   = false;
  static P constexpr dim0_min      = -1.0;
  static P constexpr dim0_max      = 1.0;
  static int constexpr dim0_level  = 2;
  static int constexpr dim0_degree = 2;
  static vector_func<P> constexpr dim0_initial_condition =
      &PDE_continuity_1::initial_condition_0;
  static auto constexpr dim0_name = "x";
  static dimension<P> constexpr dim0 =
      dimension<P>(dim0_BCL, dim0_BCR, dim0_min, dim0_max, dim0_level,
                   dim0_degree, dim0_initial_condition, dim0_name);

  // define terms (1 in this case)
  static coefficient_type constexpr term0_type = coefficient_type::grad;
  static scalar_func<P> constexpr term0_func   = &PDE_continuity_1::g_func_0;
  static bool constexpr term0_time_dependent   = false;
  static fk::vector<P> term0_data; // empty in this case
  static flux_type constexpr term0_flux = flux_type::central;
  static auto constexpr term0_name      = "d_dx";
  static term<P> constexpr term0 =
      term<P>(term0_type, term0_func, term0_time_dependent, term0_data,
              term0_flux, term0_name);

  // define sources
  static source<P> constexpr source0 = source<P>(
      &PDE_continuity_1::source_dim0_0, &PDE_continuity_1::source_dim0_0_time);
  static source<P> constexpr source1 = source<P>(
      &PDE_continuity_1::source_dim0_0, &PDE_continuity_1::source_dim0_0_time);

  //
  // store PDE functions/information in members
  //

  std::vector<dimension<P>> const dimensions = {dim0};
  std::vector<term<P>> const terms           = {term0};
  std::vector<source<P>> const sources       = {source0, source1};

  std::vector<vector_func<P>> const _exact_vector_funcs = {
      &PDE_continuity_1::exact_solution_0};

  scalar_func<P> const _exact_scalar_func = &PDE_continuity_1::exact_time;
};

// ---------------------------------------------------------------------------
//
// a user-defined PDE
//
// It requires ... FIXME
//
// ---------------------------------------------------------------------------
// template<typename P>
// class PDE_user : public PDE<P>
//{

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
    return std::make_unique<PDE_continuity_1>;
    // TODO not yet implemented, replace return with 2 and 3
  case PDE_opts::continuity_2:
    return std::make_unique<PDE_continuity_1>;
  case PDE_opts::continuity_3:
    return std::make_unique<PDE_continuity_1>;

  case PDE_opts::vlasov4:

    return std::make_unique<PDE_continuity_1>;

  case PDE_opts::vlasov43:

    return std::make_unique<PDE_continuity_1>;

  case PDE_opts::vlasov5:

    return std::make_unique<PDE_continuity_1>;

  case PDE_opts::vlasov7:

    return std::make_unique<PDE_continuity_1>;

  case PDE_opts::vlasov8:

    return std::make_unique<PDE_continuity_1>;

  case PDE_opts::pde_user:

    return std::make_unique<PDE_continuity_1>;

  default:
    std::cout << "Invalid pde choice" << std::endl;
    exit(-1);
  }
}
