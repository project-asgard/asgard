#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits.h>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "../fast_math.hpp"
#include "../matlab_utilities.hpp"
#include "../program_options.hpp"
#include "../tensors.hpp"
#include "../tools.hpp"

//
// This file contains all of the interface and object definitions for our
// representation of a PDE
//
// FIXME we plan a major rework of this component in the future
// for RAII compliance and readability

// same pi used by matlab
static double const PI = 3.141592653589793;

// for passing around vector/scalar-valued functions used by the PDE
template<typename P>
using vector_func = std::function<fk::vector<P>(fk::vector<P> const, P const)>;
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

enum class homogeneity
{
  homogeneous,
  inhomogeneous
};

// helper - single element size
auto const element_segment_size = [](auto const &pde) {
  int const degree = pde.get_dimensions()[0].get_degree();
  return static_cast<int>(std::pow(degree, pde.num_dims));
};

// ---------------------------------------------------------------------------
//
// Dimension: holds all information for a single dimension in the pde
//
// ---------------------------------------------------------------------------
// FIXME are degree and level defined per dimension? in rest of the code
// FIXME they are defined for all dimensions the same. Not using this class.
// forward dec
template<typename P>
class PDE;

template<typename P>
class dimension
{
public:
  P const domain_min;
  P const domain_max;
  vector_func<P> const initial_condition;
  std::string const name;
  dimension(P const domain_min, P const domain_max, int const level,
            int const degree, vector_func<P> const initial_condition,
            std::string const name)

      : domain_min(domain_min), domain_max(domain_max),
        initial_condition(initial_condition), name(name)
  {
    set_level(level);
    set_degree(degree);
  }

  int get_level() const { return level_; }
  int get_degree() const { return degree_; }

private:
  void set_level(int const level)
  {
    expect(level >= 0);
    level_ = level;
  }

  void set_degree(int const degree)
  {
    expect(degree > 0);
    degree_ = degree;
  }

  int level_;
  int degree_;

  friend class PDE<P>;
};

enum class coefficient_type
{
  grad,
  mass,
  diff
};

enum class flux_type
{

  downwind      = -1,
  central       = 0,
  upwind        = 1,
  lax_friedrich = 0
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

using g_func_type = std::function<double(double const, double const)>;

template<typename P>
class partial_term
{
public:
  static double null_gfunc(double const x, double const t)
  {
    ignore(x);
    ignore(t);
    return 1.0;
  }

  static P null_scalar_func(P const p) { return p; }

  partial_term(coefficient_type const coeff_type,
               g_func_type const g_func       = null_gfunc,
               flux_type const flux           = flux_type::central,
               boundary_condition const left  = boundary_condition::neumann,
               boundary_condition const right = boundary_condition::neumann,
               homogeneity const left_homo    = homogeneity::homogeneous,
               homogeneity const right_homo   = homogeneity::homogeneous,
               std::vector<vector_func<P>> const left_bc_funcs = {},
               scalar_func<P> const left_bc_time_func = null_scalar_func,
               std::vector<vector_func<P>> const right_bc_funcs = {},
               scalar_func<P> const right_bc_time_func = null_scalar_func)

      : coeff_type(coeff_type), g_func(g_func), flux(flux), left(left),
        right(right), left_homo(left_homo), right_homo(right_homo),
        left_bc_funcs(left_bc_funcs), right_bc_funcs(right_bc_funcs),
        left_bc_time_func(left_bc_time_func),
        right_bc_time_func(right_bc_time_func)
  {}

  P get_flux_scale() const { return static_cast<P>(flux); };

  coefficient_type const coeff_type;

  g_func_type const g_func;

  flux_type const flux;

  boundary_condition const left;

  boundary_condition const right;

  homogeneity const left_homo;
  homogeneity const right_homo;
  std::vector<vector_func<P>> const left_bc_funcs;
  std::vector<vector_func<P>> const right_bc_funcs;
  scalar_func<P> const left_bc_time_func;
  scalar_func<P> const right_bc_time_func;

  fk::matrix<P> const &get_coefficients() const { return coefficients_; }

  void set_coefficients(fk::matrix<P> const &new_coefficients)
  {
    this->coefficients_.clear_and_resize(
        new_coefficients.nrows(), new_coefficients.ncols()) = new_coefficients;
  }

private:
  fk::matrix<P> coefficients_;
};

template<typename P>
class term
{
  static P g_func_default(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

public:
  term(bool const time_dependent, fk::vector<P> const data,
       std::string const name,
       std::initializer_list<partial_term<P>> const partial_terms)
      : time_dependent(time_dependent), name(name),
        partial_terms_(partial_terms), data_(data)

  {}

  void set_data(fk::vector<P> const &data)
  {
    this->data_.resize(data.size()) = data;
  }

  fk::vector<P> get_data() const & { return data_; };

  void set_coefficients(fk::matrix<P> const &new_coefficients)
  {
    this->coefficients_.clear_and_resize(new_coefficients.nrows(),
                                         new_coefficients.ncols()) =
        new_coefficients.clone_onto_device();
  }

  void set_partial_coefficients(fk::matrix<P> const &coeffs, int const pterm)
  {
    expect(pterm >= 0);
    expect(pterm < static_cast<int>(partial_terms_.size()));
    partial_terms_[pterm].set_coefficients(coeffs);
  }

  fk::matrix<P, mem_type::owner, resource::device> const &
  get_coefficients() const
  {
    return coefficients_;
  }

  std::vector<partial_term<P>> const &get_partial_terms() const
  {
    return partial_terms_;
  }

  // after adapting to a new number of hierarchical basis levels,
  // recombine partial terms to form new coefficient matrices
  void rechain_coefficients(dimension<P> const &adapted_dim)
  {
    auto const new_dof =
        adapted_dim.get_degree() * fm::two_raised_to(adapted_dim.get_level());
    expect(coefficients_.nrows() == coefficients_.ncols());
    auto new_coeffs = eye<P>(new_dof);

    for (auto const &pterm : partial_terms_)
    {
      auto const &partial_coeff = pterm.get_coefficients();
      expect(partial_coeff.size() >
             new_dof); // make sure we built the partial terms to support
                       // new level/degree
      new_coeffs = new_coeffs *
                   fk::matrix<P, mem_type::const_view>(
                       partial_coeff, 0, new_dof - 1, 0,
                       new_dof - 1); // at some point, we could consider storing
                                     // these device-side after construction.
    }
    fk::matrix<P, mem_type::view, resource::device>(coefficients_, 0,
                                                    new_dof - 1, 0, new_dof - 1)
        .transfer_from(new_coeffs);
  }

  // public but const data. no getters
  bool const time_dependent;
  std::string const name;

private:
  std::vector<partial_term<P>> partial_terms_;

  // this is to hold data that may change over the course of the simulation,
  // from any source, that is used in operator construction.
  //
  // initialized to one if not provided at instantiation, which performs an
  // identity operation where this is used, until set by outside source.
  fk::vector<P> data_;

  // operator matrix for this term at a single dimension
  fk::matrix<P, mem_type::owner, resource::device> coefficients_;
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
  PDE(parser const &cli_input, int const num_dims, int const num_sources,
      int const num_terms, std::vector<dimension<P>> const dimensions,
      term_set<P> const terms, std::vector<source<P>> const sources,
      std::vector<vector_func<P>> const exact_vector_funcs,
      scalar_func<P> const exact_time, dt_func<P> const get_dt,
      bool const do_poisson_solve = false, bool const has_analytic_soln = false)
      : num_dims(num_dims), num_sources(num_sources), num_terms(num_terms),
        sources(sources), exact_vector_funcs(exact_vector_funcs),
        exact_time(exact_time), do_poisson_solve(do_poisson_solve),
        has_analytic_soln(has_analytic_soln), dimensions_(dimensions),
        terms_(terms)
  {
    expect(num_dims > 0);
    expect(num_sources >= 0);
    expect(num_terms > 0);

    expect(dimensions.size() == static_cast<unsigned>(num_dims));
    expect(terms.size() == static_cast<unsigned>(num_terms));
    expect(sources.size() == static_cast<unsigned>(num_sources));

    // ensure analytic solution functions were provided if this flag is set
    if (has_analytic_soln)
    {
      expect(exact_vector_funcs.size() == static_cast<unsigned>(num_dims));
    }

    // modify for appropriate level/degree
    // if default lev/degree not used
    auto const user_levels = cli_input.get_starting_levels().size();
    if (user_levels != 0 && user_levels != num_dims)
    {
      std::cerr << "failed to parse dimension-many starting levels - parsed "
                << user_levels << " levels\n";
      exit(1);
    }
    if (user_levels == num_dims)
    {
      auto counter = 0;
      for (dimension<P> &d : dimensions_)
      {
        auto const num_levels = cli_input.get_starting_levels()(counter++);
        expect(num_levels > 1);
        d.set_level(num_levels);
      }
    }

    auto const cli_degree = cli_input.get_degree();
    if (cli_degree != parser::NO_USER_VALUE)
    {
      expect(cli_degree > 0);
      for (dimension<P> &d : dimensions_)
      {
        d.set_degree(cli_degree);
      }
    }
    // assume uniform degree
    auto const degree = dimensions_[0].get_degree();

    // check all terms
    for (auto &term_list : terms_)
    {
      expect(term_list.size() == static_cast<unsigned>(num_dims));
      for (auto &term_1D : term_list)
      {
        expect(term_1D.get_partial_terms().size() > 0);

        auto const max_dof =
            fm::two_raised_to(static_cast<int64_t>(cli_input.get_max_level())) *
            degree;
        expect(max_dof < INT_MAX);

        term_1D.set_data(fk::vector<P>(std::vector<P>(max_dof, 1.0)));
        term_1D.set_coefficients(eye<P>(max_dof));

        for (auto &p : term_1D.get_partial_terms())
        {
          if (p.left_homo == homogeneity::homogeneous)
            expect(static_cast<int>(p.left_bc_funcs.size()) == 0);
          else if (p.left_homo == homogeneity::inhomogeneous)
            expect(static_cast<int>(p.left_bc_funcs.size()) == num_dims);

          if (p.right_homo == homogeneity::homogeneous)
            expect(static_cast<int>(p.right_bc_funcs.size()) == 0);
          else if (p.right_homo == homogeneity::inhomogeneous)
            expect(static_cast<int>(p.right_bc_funcs.size()) == num_dims);
        }
      }
    }

    // check all dimensions
    for (auto const &d : dimensions_)
    {
      expect(d.get_degree() > 0);
      expect(d.get_level() > 1);
      expect(d.domain_max > d.domain_min);
    }

    // check all sources
    for (auto const &s : sources)
    {
      expect(s.source_funcs.size() == static_cast<unsigned>(num_dims));
    }

    // set the dt
    if (cli_input.get_dt() == parser::NO_USER_VALUE_FP)
    {
      dt_ = get_dt(dimensions_[0]) * cli_input.get_cfl();
    }
    else
    {
      dt_ = cli_input.get_dt();
    }
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

  fk::matrix<P, mem_type::owner, resource::device> const &
  get_coefficients(int const term, int const dim) const
  {
    expect(term >= 0);
    expect(term < num_terms);
    expect(dim >= 0);
    expect(dim < num_dims);
    return terms_[term][dim].get_coefficients();
  }

  /* gives a vector of partial_term matrices to the term object so it can
     construct the full operator matrix */
  void
  set_coefficients(fk::matrix<P> const &coeffs, int const term, int const dim)
  {
    expect(term >= 0);
    expect(term < num_terms);
    expect(dim >= 0);
    expect(dim < num_dims);
    terms_[term][dim].set_coefficients(coeffs);
  }

  void set_partial_coefficients(int const term, int const dim, int const pterm,
                                fk::matrix<P> const &coeffs)
  {
    expect(term >= 0);
    expect(term < num_terms);
    expect(dim >= 0);
    expect(dim < num_dims);
    terms_[term][dim].set_partial_coefficients(coeffs, pterm);
  }

  void update_dimension(int const dim_index, int const new_level)
  {
    assert(dim_index >= 0);
    assert(dim_index < num_dims);
    assert(new_level >= 0);

    dimensions_[dim_index].set_level(new_level);
  }

  void rechain_dimension(int const dim_index)
  {
    expect(dim_index >= 0);
    expect(dim_index < num_dims);
    for (auto i = 0; i < num_terms; ++i)
    {
      terms_[i][dim_index].rechain_coefficients(dimensions_[dim_index]);
    }
  }

  P get_dt() const { return dt_; };

  void set_dt(P const dt)
  {
    expect(dt > 0.0);
    dt_ = dt;
  }

private:
  std::vector<dimension<P>> dimensions_;
  term_set<P> terms_;
  P dt_;
};
