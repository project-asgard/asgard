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

#include "../basis.hpp"
#include "../matlab_utilities.hpp"
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
// forward dec
template<typename P>
class PDE;

template<typename P>
class dimension
{
public:
  boundary_condition const left;
  boundary_condition const right;
  P const domain_min;
  P const domain_max;
  vector_func<P> const initial_condition;
  std::string const name;
  dimension(boundary_condition const left, boundary_condition const right,
            P const domain_min, P const domain_max, int const level,
            int const degree, vector_func<P> const initial_condition,
            std::string const name)

      : left(left), right(right), domain_min(domain_min),
        domain_max(domain_max), initial_condition(initial_condition),
        name(name), level_(level), degree_(degree)
  {
    int const dofs = degree_ * two_raised_to(level_);
    to_basis_operator_.clear_and_resize(dofs, dofs) =
        operator_two_scale<double>(degree_, level_);
    from_basis_operator_.clear_and_resize(dofs, dofs) =
        fk::matrix<double>(to_basis_operator_).transpose();
  }

  int get_level() const { return level_; }
  int get_degree() const { return degree_; }
  fk::matrix<double> const &get_to_basis_operator() const
  {
    return to_basis_operator_;
  }
  fk::matrix<double> const &get_from_basis_operator() const
  {
    return from_basis_operator_;
  }

private:
  void set_level(int level)
  {
    assert(level > 0);
    level_         = level;
    int const dofs = degree_ * two_raised_to(level_);
    to_basis_operator_.clear_and_resize(dofs, dofs) =
        operator_two_scale<double>(degree_, level_);
    from_basis_operator_.clear_and_resize(dofs, dofs) =
        fk::matrix<double>(to_basis_operator_).transpose();
  }

  void set_degree(int degree)
  {
    assert(degree > 0);
    degree_        = degree;
    int const dofs = degree_ * two_raised_to(level_);
    to_basis_operator_.clear_and_resize(dofs, dofs) =
        operator_two_scale<double>(degree_, level_);
    from_basis_operator_.clear_and_resize(dofs, dofs) =
        fk::matrix<double>(to_basis_operator_).transpose();
  }

  int level_;
  int degree_;
  fk::matrix<double> to_basis_operator_;
  fk::matrix<double> from_basis_operator_;

  friend class PDE<P>;
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
    set_data(owning_dim, data);
    set_coefficients(owning_dim, eye<P>(degrees_freedom(owning_dim)));
  }

  void set_data(dimension<P> const owning_dim, fk::vector<P> const data)
  {
    int const degrees_freedom_1d = degrees_freedom(owning_dim);
    if (data.size() != 0)
    {
      assert(data.size() == degrees_freedom_1d);
      data_ = data;
    }
    else
    {
      this->data_.resize(degrees_freedom_1d);
      this->data_ = fk::vector<P>(std::vector<P>(degrees_freedom_1d, 1.0));
    }
    if (flux == flux_type::central)
    {
      flux_scale_ = 0.0;
    }
    else if (flux == flux_type::upwind)
    {
      flux_scale_ = 1.0;
    }
    else
    {
      flux_scale_ = 0.0;
    }
  }

  fk::vector<P> get_data() const { return data_; };

  void set_flux_scale(P const dfdu)
  {
    assert(flux == flux_type::lax_friedrich);
    flux_scale_ = dfdu;
  };
  P get_flux_scale() const { return flux_scale_; };

  void set_coefficients(dimension<P> const owning_dim,
                        fk::matrix<P> const new_coefficients)
  {
    int const degrees_freedom_1d = degrees_freedom(owning_dim);
    assert(degrees_freedom_1d == new_coefficients.nrows());
    assert(degrees_freedom_1d == new_coefficients.ncols());
    this->coefficients_.clear_and_resize(degrees_freedom_1d,
                                         degrees_freedom_1d) = new_coefficients;
  }
  fk::matrix<P> const &get_coefficients() const { return coefficients_; }

  // small helper to return degrees of freedom given dimension
  int degrees_freedom(dimension<P> const d) const
  {
    return d.get_degree() * static_cast<int>(std::pow(2, d.get_level()));
  };

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

  // scale the jump operator in coefficient construction by this amount,
  // determined by flux type. 0 or 1 for central or upwind, respectively,
  // and df/du for lax freidrich. should not be set after construction for
  // central or upwind.
  P flux_scale_;

  // operator matrix for this term at a single dimension
  fk::matrix<P> coefficients_;
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
using dt_func = std::function<P(dimension<P> const &dim)>;

template<typename P>
class PDE
{
public:
  // clang-format off
  PDE(int const num_levels,
      int const degree,
      int const num_dims,
      int const num_sources,
      int const num_terms,
      std::vector<dimension<P>> const dimensions,
      term_set<P> const terms,
      std::vector<source<P>> const sources,
      std::vector<vector_func<P>> const exact_vector_funcs,
      scalar_func<P> const exact_time,
      dt_func<P> const get_dt,
      bool const do_poisson_solve = false,
      bool const has_analytic_soln = false)
      : num_dims(num_dims),
        num_sources(num_sources),
        num_terms(num_terms),
	sources(sources),
        exact_vector_funcs(exact_vector_funcs),
	exact_time(exact_time),
	do_poisson_solve(do_poisson_solve),
        has_analytic_soln(has_analytic_soln),
	dimensions_(dimensions),
	terms_(terms)
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

    // modify for appropriate level/degree
    // if default lev/degree not used
    if (num_levels > 0 || degree > 0)
    {
      // FIXME -- temp -- eventually independent levels for each dim will be

      for (dimension<P> &d : dimensions_)
      {
        if (num_levels > 0)
          d.set_level(num_levels);
        if (degree > 0)
          d.set_degree(degree);
      }

      for (std::vector<term<P>> &term_list : terms_)
      {
        // positive, bounded size - safe compare
        for (int i = 0; i < static_cast<int>(term_list.size()); ++i)
        {
          term_list[i].set_data(dimensions_[i], fk::vector<P>());
          term_list[i].set_coefficients(
              dimensions_[i],
              eye<P>(term_list[i].degrees_freedom(dimensions_[i])));
        }
      }
    }
    // check all dimensions
    for (dimension<P> const d : dimensions_)
    {
      assert(d.get_degree() > 0);
      assert(d.get_level() > 0);
      assert(d.domain_max > d.domain_min);
    }

    // check all sources
    for (source<P> const s : this->sources)
    {
      assert(s.source_funcs.size() == static_cast<unsigned>(num_dims));
    }

    // check all terms
    for (std::vector<term<P>> const term_list : terms_)
    {
      assert(term_list.size() == static_cast<unsigned>(num_dims));
    }

    // set the dt
    dt_ = get_dt(dimensions_[0]);
  }

  // public but const data.
  int const num_dims;
  int const num_sources;
  int const num_terms;

  std::vector<source<P>> const sources;
  std::vector<vector_func<P>> const exact_vector_funcs;
  scalar_func<P> const exact_time;
  bool const do_poisson_solve;
  bool const has_analytic_soln;

  virtual ~PDE() {}

  std::vector<dimension<P>> const &get_dimensions() const
  {
    return dimensions_;
  }
  term_set<P> const &get_terms() const { return terms_; }

  fk::matrix<P> const &get_coefficients(int const term, int const dim) const
  {
    return terms_[term][dim].get_coefficients();
  }
  void
  set_coefficients(fk::matrix<P> const coeffs, int const term, int const dim)
  {
    terms_[term][dim].set_coefficients(dimensions_[dim], coeffs);
  }

  P get_dt() { return dt_; };

private:
  std::vector<dimension<P>> dimensions_;
  term_set<P> terms_;
  P dt_;
};
