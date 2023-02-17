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

#include "../asgard_dimension.hpp"
#include "../fast_math.hpp"
#include "../matlab_utilities.hpp"
#include "../moment.hpp"
#include "../program_options.hpp"
#include "../tensors.hpp"
#include "../tools.hpp"

namespace asgard
{
//
// This file contains all of the interface and object definitions for our
// representation of a PDE
//
// FIXME we plan a major rework of this component in the future
// for RAII compliance and readability

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
// forward dec
template<typename P>
class PDE;

template<typename P>
class moment;

enum class coefficient_type
{
  grad,
  mass,
  div,
  penalty
};

enum class flux_type
{

  downwind      = -1,
  central       = 0,
  upwind        = 1,
  lax_friedrich = 0
};

enum class imex_flag
{
  unspecified,
  imex_explicit,
  imex_implicit,
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
class partial_term
{
public:
  static fk::vector<P> null_vector_func(fk::vector<P> x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::fill(fx.begin(), fx.end(), 1.0);
    return fx;
  }

  partial_term(coefficient_type const coeff_type_in,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               flux_type const flux_in               = flux_type::central,
               boundary_condition const left_in  = boundary_condition::neumann,
               boundary_condition const right_in = boundary_condition::neumann,
               homogeneity const left_homo_in    = homogeneity::homogeneous,
               homogeneity const right_homo_in   = homogeneity::homogeneous,
               std::vector<vector_func<P>> const left_bc_funcs_in  = {},
               scalar_func<P> const left_bc_time_func_in           = nullptr,
               std::vector<vector_func<P>> const right_bc_funcs_in = {},
               scalar_func<P> const right_bc_time_func_in          = nullptr,
               g_func_type<P> const dv_func_in                     = nullptr)

      : coeff_type(coeff_type_in), g_func(g_func_in),
        lhs_mass_func(lhs_mass_func_in), flux(set_flux(flux_in)), left(left_in),
        right(right_in), ileft(set_bilinear_boundary(left_in)),
        iright(set_bilinear_boundary(right_in)), left_homo(left_homo_in),
        right_homo(right_homo_in), left_bc_funcs(left_bc_funcs_in),
        right_bc_funcs(right_bc_funcs_in),
        left_bc_time_func(left_bc_time_func_in),
        right_bc_time_func(right_bc_time_func_in), dv_func(dv_func_in)
  {}

  P get_flux_scale() const { return static_cast<P>(flux); };

  coefficient_type const coeff_type;

  g_func_type<P> const g_func;
  g_func_type<P> const lhs_mass_func;

  flux_type const flux;

  boundary_condition const left;

  boundary_condition const right;

  boundary_condition const ileft;
  boundary_condition const iright;

  homogeneity const left_homo;
  homogeneity const right_homo;
  std::vector<vector_func<P>> const left_bc_funcs;
  std::vector<vector_func<P>> const right_bc_funcs;
  scalar_func<P> const left_bc_time_func;
  scalar_func<P> const right_bc_time_func;

  g_func_type<P> const dv_func;

  fk::matrix<P> const get_coefficients(int const level) const
  {
    // returns precomputed inv(mass) * coeff for this level
    expect(static_cast<int>(coefficients_.size()) >= level);
    expect(level >= 0);
    return coefficients_[level];
  }

  fk::matrix<P> const &get_lhs_mass() const { return mass_; }

  void set_coefficients(fk::matrix<P> const &new_coefficients, int const deg,
                        int const max_level)
  {
    coefficients_.clear();

    // precompute inv(mass) * coeff for each level up to max level
    std::vector<int> ipiv(deg * fm::two_raised_to(max_level));
    for (int level = 0; level <= max_level; ++level)
    {
      auto const dof = deg * fm::two_raised_to(level);
      fk::matrix<P> result(new_coefficients, 0, dof - 1, 0, dof - 1);
      auto mass_tmp = mass_.extract_submatrix(0, 0, dof, dof);
      fm::gesv(mass_tmp, result, ipiv);
      coefficients_.push_back(std::move(result));
    }
  }

  void set_coefficients(std::vector<fk::matrix<P>> const &new_coefficients)
  {
    expect(new_coefficients.size() > 0);
    coefficients_.clear();
    coefficients_ = new_coefficients;
  }

  void set_coefficients(std::vector<fk::matrix<P>> &&new_coefficients)
  {
    expect(new_coefficients.size() > 0);
    coefficients_ = std::move(new_coefficients);
  }

  void set_mass(fk::matrix<P> const &new_mass)
  {
    this->mass_.clear_and_resize(new_mass.nrows(), new_mass.ncols()) = new_mass;
  }

  void set_mass(fk::matrix<P> &&new_mass) { this->mass_ = std::move(new_mass); }

  boundary_condition set_bilinear_boundary(boundary_condition const bc)
  {
    // Since we want the grad matrix to be a negative transpose of a
    // DIV matrix, we need to swap the wind direction as well as swap
    // the BCs N<=>D.  However, this swap will affect the BC call.
    // Instead we have another BC flag IBCL/IBCR which will build the
    // bilinear form with respect to Dirichlet/Free boundary
    // conditions while leaving the BC routine unaffected.
    if (coeff_type == coefficient_type::grad)
    {
      if (bc == boundary_condition::dirichlet)
      {
        return boundary_condition::neumann;
      }
      else if (bc == boundary_condition::neumann)
      {
        return boundary_condition::dirichlet;
      }
    }
    return bc;
  }

  flux_type set_flux(flux_type const flux_in)
  {
    if (coeff_type == coefficient_type::grad)
    {
      // Switch the upwinding direction
      return static_cast<flux_type>(-static_cast<P>(flux_in));
    }
    return flux_in;
  }

private:
  std::vector<fk::matrix<P>> coefficients_;
  fk::matrix<P> mass_;
};

template<typename P>
class term
{
public:
  term(bool const time_dependent_in, std::string const name_in,
       std::initializer_list<partial_term<P>> const partial_terms,
       imex_flag const flag_in = imex_flag::unspecified)
      : time_dependent(time_dependent_in), name(name_in), flag(flag_in),
        partial_terms_(partial_terms)
  {}

  void set_coefficients(fk::matrix<P> const &new_coefficients)
  {
    this->coefficients_.clear_and_resize(new_coefficients.nrows(),
                                         new_coefficients.ncols()) =
        new_coefficients.clone_onto_device();
  }

  void set_partial_coefficients(fk::matrix<P> const &coeffs, int const pterm,
                                int const deg, int const max_lev)
  {
    expect(pterm >= 0);
    expect(pterm < static_cast<int>(partial_terms_.size()));
    partial_terms_[pterm].set_coefficients(coeffs, deg, max_lev);
  }

  void set_partial_coefficients(std::vector<fk::matrix<P>> const &coeffs,
                                int const pterm)
  {
    expect(pterm >= 0);
    expect(pterm < static_cast<int>(partial_terms_.size()));
    partial_terms_[pterm].set_coefficients(coeffs);
  }

  void
  set_partial_coefficients(std::vector<fk::matrix<P>> &&coeffs, int const pterm)
  {
    expect(pterm >= 0);
    expect(pterm < static_cast<int>(partial_terms_.size()));
    partial_terms_[pterm].set_coefficients(std::move(coeffs));
  }

  void set_lhs_mass(fk::matrix<P> const &mass, int const pterm)
  {
    expect(pterm >= 0);
    expect(pterm < static_cast<int>(partial_terms_.size()));
    partial_terms_[pterm].set_mass(mass);
  }

  void set_lhs_mass(fk::matrix<P> &&mass, int const pterm)
  {
    expect(pterm >= 0);
    expect(pterm < static_cast<int>(partial_terms_.size()));
    partial_terms_[pterm].set_mass(std::move(mass));
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
      auto const &partial_coeff =
          pterm.get_coefficients(adapted_dim.get_level());
      expect(partial_coeff.ncols() ==
             new_dof); // make sure we built the partial terms to support
                       // new level/degree

      new_coeffs = new_coeffs *
                   partial_coeff; // at some point, we could consider storing
                                  // these device-side after construction.
    }

    fk::matrix<P, mem_type::view, resource::device>(coefficients_, 0,
                                                    new_dof - 1, 0, new_dof - 1)
        .transfer_from(new_coeffs);
  }

  // public but const data. no getters
  bool const time_dependent;
  std::string const name;

  imex_flag const flag;

private:
  std::vector<partial_term<P>> partial_terms_;

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
  source(std::vector<vector_func<P>> const source_funcs_in,
         scalar_func<P> const time_func_in)

      : source_funcs(source_funcs_in), time_func(time_func_in)
  {}

  // public but const data. no getters
  std::vector<vector_func<P>> const source_funcs;
  scalar_func<P> const time_func;
};

template<typename P>
struct parameter
{
  std::string const name;
  g_func_type<P> value;
};

// Singleton class for storing and receiving PDE parameters
template<typename P>
#define param_manager parameter_manager<P>::get_instance()
class parameter_manager
{
public:
  static parameter_manager<P> &get_instance()
  {
    static parameter_manager<P> instance;
    return instance;
  }

  // prevent potential copies from being created
  parameter_manager(parameter_manager<P> const &) = delete;
  void operator=(parameter_manager<P> const &) = delete;

  std::shared_ptr<parameter<P>> get_parameter(std::string const name)
  {
    auto p = check_param(name);
    if (p == nullptr)
    {
      throw std::runtime_error(
          std::string(" could not find parameter with name '" + name + "'\n"));
    }
    return p;
  }

  void add_parameter(parameter<P> const &param)
  {
    if (check_param(param.name) == nullptr)
    {
      parameters.push_back(std::make_shared<parameter<P>>(param));
    }
    else
    {
      throw std::runtime_error(std::string(
          "already have a parameter with name '" + param.name + "'\n"));
    }
  }

  size_t get_num_parameters() { return parameters.size(); }

private:
  parameter_manager() {}

  std::shared_ptr<parameter<P>> check_param(std::string const name)
  {
    for (auto p : parameters)
    {
      if (p->name == name)
      {
        return p;
      }
    }
    return nullptr;
  }

  std::vector<std::shared_ptr<parameter<P>>> parameters;
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
  PDE(parser const &cli_input, int const num_dims_in, int const num_sources_in,
      int const max_num_terms, std::vector<dimension<P>> const dimensions,
      term_set<P> const terms, std::vector<source<P>> const sources_in,
      std::vector<vector_func<P>> const exact_vector_funcs_in,
      scalar_func<P> const exact_time_in, dt_func<P> const get_dt,
      bool const do_poisson_solve_in          = false,
      bool const has_analytic_soln_in         = false,
      std::vector<moment<P>> const moments_in = {},
      bool const do_collision_operator_in     = true)
      : PDE(cli_input, num_dims_in, num_sources_in, max_num_terms, dimensions,
            terms, sources_in,
            std::vector<md_func_type<P>>({exact_vector_funcs_in}),
            exact_time_in, get_dt, do_poisson_solve_in, has_analytic_soln_in,
            moments_in, do_collision_operator_in)
  {}
  PDE(parser const &cli_input, int const num_dims_in, int const num_sources_in,
      int const max_num_terms, std::vector<dimension<P>> const dimensions,
      term_set<P> const terms, std::vector<source<P>> const sources_in,
      std::vector<md_func_type<P>> const exact_vector_funcs_in,
      scalar_func<P> const exact_time_in, dt_func<P> const get_dt,
      bool const do_poisson_solve_in          = false,
      bool const has_analytic_soln_in         = false,
      std::vector<moment<P>> const moments_in = {},
      bool const do_collision_operator_in     = true)
      : num_dims(num_dims_in), num_sources(num_sources_in),
        num_terms(get_num_terms(cli_input, max_num_terms)),
        max_level(get_max_level(cli_input, dimensions)), sources(sources_in),
        exact_vector_funcs(exact_vector_funcs_in), moments(moments_in),
        exact_time(exact_time_in), do_poisson_solve(do_poisson_solve_in),
        do_collision_operator(do_collision_operator_in),
        has_analytic_soln(has_analytic_soln_in), dimensions_(dimensions),
        terms_(terms)
  {
    expect(num_dims > 0);
    expect(num_sources >= 0);
    expect(num_terms > 0);

    expect(dimensions.size() == static_cast<unsigned>(num_dims));
    expect(terms.size() == static_cast<unsigned>(max_num_terms));
    expect(sources.size() == static_cast<unsigned>(num_sources));

    // ensure analytic solution functions were provided if this flag is set
    if (has_analytic_soln)
    {
      // each set of analytical solution functions must have num_dim functions
      for (const auto &md_func : exact_vector_funcs)
      {
        expect(md_func.size() == static_cast<unsigned>(num_dims));
      }
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

    auto const num_active_terms = cli_input.get_active_terms().size();
    if (num_active_terms != 0)
    {
      auto const active_terms = cli_input.get_active_terms();
      for (auto i = max_num_terms - 1; i >= 0; --i)
      {
        if (active_terms(i) == 0)
        {
          terms_.erase(terms_.begin() + i);
        }
      }
      expect(terms_.size() == static_cast<unsigned>(num_terms));
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

    // initialize mass matrices to a default value
    for (auto i = 0; i < num_dims; ++i)
    {
      auto const max_dof =
          fm::two_raised_to(static_cast<int64_t>(cli_input.get_max_level())) *
          degree;
      expect(max_dof < INT_MAX);
      update_dimension_mass_mat(i, eye<P>(max_dof));
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

    // check the moments
    for (auto const &m : moments)
    {
      // each moment should have ndim + 1 functions
      auto md_funcs = m.get_md_funcs();
      for (auto md_func : md_funcs)
      {
        expect(md_func.size() == static_cast<unsigned>(num_dims) + 1);
      }
    }

    if (cli_input.using_imex() && moments.size() == 0)
    {
      throw std::runtime_error(
          "Invalid PDE choice for IMEX time advance. PDE must have "
          "moments defined to use -x\n");
    }
  }

  constexpr static int extract_dim0 = 1;
  // copy constructor to create a 1D version of the PDE
  // this is used in the IMEX time advance to help define 1D mapping from
  // wavelet to realspace
  // TODO: there is likely a better way to do this. Another option is to flatten
  // element table to 1D (see hash_table_2D_to_1D.m)
  PDE(const PDE &pde, int)
      : num_dims(1), num_sources(pde.sources.size()),
        num_terms(pde.get_terms().size()), max_level(pde.max_level),
        sources(pde.sources), exact_vector_funcs(pde.exact_vector_funcs),
        moments(pde.moments), exact_time(pde.exact_time),
        do_poisson_solve(pde.do_poisson_solve),
        do_collision_operator(pde.do_collision_operator),
        has_analytic_soln(pde.has_analytic_soln),
        dimensions_({pde.get_dimensions()[0]}), terms_(pde.get_terms())
  {}

  // public but const data.
  int const num_dims;
  int const num_sources;
  int const num_terms;
  int const max_level;

  std::vector<source<P>> const sources;
  std::vector<md_func_type<P>> const exact_vector_funcs;
  std::vector<moment<P>> moments;
  scalar_func<P> const exact_time;
  bool const do_poisson_solve;
  bool const do_collision_operator;
  bool const has_analytic_soln;
  // data for poisson solver
  fk::vector<P> poisson_diag;
  fk::vector<P> poisson_off_diag;

  fk::vector<P> E_field;

  virtual ~PDE() {}

  std::vector<dimension<P>> const &get_dimensions() const
  {
    return dimensions_;
  }
  std::vector<dimension<P>> &get_dimensions() { return dimensions_; }

  term_set<P> const &get_terms() const { return terms_; }
  term_set<P> &get_terms() { return terms_; }

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
    terms_[term][dim].set_partial_coefficients(
        coeffs, pterm, dimensions_[dim].get_degree(), max_level);
  }

  void set_partial_coefficients(int const term, int const dim, int const pterm,
                                std::vector<fk::matrix<P>> const &coeffs)
  {
    expect(term >= 0);
    expect(term < num_terms);
    expect(dim >= 0);
    expect(dim < num_dims);
    terms_[term][dim].set_partial_coefficients(coeffs, pterm);
  }

  void set_lhs_mass(int const term, int const dim, int const pterm,
                    fk::matrix<P> const &mass)
  {
    expect(term >= 0);
    expect(term < num_terms);
    expect(dim >= 0);
    expect(dim < num_dims);
    terms_[term][dim].set_lhs_mass(mass, pterm);
  }

  void set_lhs_mass(int const term, int const dim, int const pterm,
                    fk::matrix<P> &&mass)
  {
    expect(term >= 0);
    expect(term < num_terms);
    expect(dim >= 0);
    expect(dim < num_dims);
    terms_[term][dim].set_lhs_mass(mass, std::move(pterm));
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

  void update_dimension_mass_mat(int const dim_index, fk::matrix<P> const &mass)
  {
    assert(dim_index >= 0);
    assert(dim_index < num_dims);

    dimensions_[dim_index].set_mass_matrix(mass);
  }

  P get_dt() const { return dt_; };

  void set_dt(P const dt)
  {
    expect(dt > 0.0);
    dt_ = dt;
  }

private:
  int get_num_terms(parser const &cli_input, int const max_num_terms) const
  {
    // returns either the number of terms set in the PDE specification, or the
    // number of terms toggled on by the user
    auto const num_active_terms = cli_input.get_active_terms().size();

    // verify that the CLI input matches the spec before altering the num_terms
    // we have
    if (num_active_terms != 0 && num_active_terms != max_num_terms)
    {
      throw std::runtime_error(
          std::string("failed to parse dimension-many active terms - parsed ") +
          std::to_string(num_active_terms) + " terms, expected " +
          std::to_string(max_num_terms));
    }
    // if nothing specified in the cli, use the default max_num_terms
    if (num_active_terms == 0)
      return max_num_terms;
    // terms specified in the cli, parse the new number of terms
    auto const active_terms = cli_input.get_active_terms();
    int new_num_terms =
        std::accumulate(active_terms.begin(), active_terms.end(), 0);
    if (new_num_terms == 0)
    {
      throw std::runtime_error("must have at least one term enabled");
    }
    return new_num_terms;
  }

  int get_max_level(parser const &cli_input,
                    std::vector<dimension<P>> const &dims) const
  {
    // set maximum level to generate term coefficients
    if (cli_input.do_adapt_levels())
    {
      return cli_input.get_max_level();
    }
    else
    {
      // if adaptivity is not used, only generate to the highest dim level
      auto const levels = cli_input.get_starting_levels();
      return levels.size() > 0
                 ? *std::max_element(levels.begin(), levels.end())
                 : std::max_element(
                       dims.begin(), dims.end(),
                       [](dimension<P> const &a, dimension<P> const &b) {
                         return a.get_level() < b.get_level();
                       })
                       ->get_level();
    }
  }

  std::vector<dimension<P>> dimensions_;
  term_set<P> terms_;
  P dt_;
};
} // namespace asgard
