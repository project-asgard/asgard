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

//
// This file contains all of the interface and object definitions for our
// representation of a PDE
//

// same pi used by matlab
static double const PI = 3.141592653589793;

// used to suppress warnings in unused variables
auto const ignore = [](auto ignored) { (void)ignored; };

// for passing around vector/scalar-valued functions used by the PDE
template<typename P>
using vector_func = std::function<fk::vector<P>(fk::vector<P> const)>;
template<typename P>
using scalar_func = std::function<P(P const)>;

//----------------------------------------------------------------------------
//
// Define member classes of the PDE type: dimension, term, source
//
//----------------------------------------------------------------------------

// just a small enumeration of the possibly boundary condition types needed in
// the following 'dimension' member class
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
public:
  term(coefficient_type const coeff, g_func_type<P> const g_func,
       bool const time_dependent, flux_type const flux,
       fk::vector<P> const data, std::string const name,
       dimension<P> const owning_dim)
      : coeff(coeff), g_func(g_func), time_dependent(time_dependent),
        flux(flux), name(name), data_(data)
  {
    int const degrees_freedom_1d =
        owning_dim.degree * static_cast<int>(std::pow(2, owning_dim.level));
    if (data.size() != 0)
    {
      assert(data.size() == degrees_freedom_1d);
    }
    else
    {
      this->data_.resize(degrees_freedom_1d);
      this->data_ = fk::vector<P>(std::vector<P>(
          owning_dim.degree * static_cast<int>(std::pow(2, owning_dim.level)),
          1.0));
    }
  }

  void set_data(fk::vector<P> const new_data)
  {
    assert(new_data.size() == data_.size());
    data_ = new_data;
  };
  fk::vector<P> get_data() const { return data_; };

  // public but const data. no getters
  coefficient_type const coeff;
  g_func_type<P> const g_func;
  bool const time_dependent;
  flux_type const flux;
  std::string const name;

private:
  // this is to hold data that may change over the course of the simulation,
  // from any source, that is used in operator construction.
  //
  // initialized to one if not provided at instantiation, which performs an
  // identity operation where this is used, until set by outside source.
  fk::vector<P> data_;
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
  source(std::vector<vector_func<P>> const source_funcs,
         scalar_func<P> const time_func)

      : source_funcs(source_funcs), time_func(time_func)
  {}

  // public but const data. no getters
  std::vector<vector_func<P>> const source_funcs;
  scalar_func<P> const time_func;
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

    // ensure analytic solution functions were provided if this flag is set
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
