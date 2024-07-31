#pragma once

#include "../asgard_dimension.hpp"
#include "../fast_math.hpp"
#include "../matlab_utilities.hpp"
#include "../moment.hpp"
#include "../program_options.hpp"
#include "../asgard_indexset.hpp"

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
  return fm::ipow(degree + 1, pde.num_dims());
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
  unspecified = 0,
  imex_explicit = 1,
  imex_implicit = 2,
};
int constexpr num_imex_variants = 3;

template<typename P>
struct gmres_info
{
  P error;
  int iterations;
};

template<typename P>
struct adaptive_info
{
  // Holds the DOF count for each coarsen and refine step for the current time
  // step
  int initial_dof;
  int coarsen_dof;
  std::vector<int> refine_dofs;
  // Hold a vector of the GMRES stats for each adapt step
  std::vector<std::vector<gmres_info<P>>> gmres_stats;
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

      : coeff_type_(coeff_type_in), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(set_flux(flux_in)), left_(left_in),
        right_(right_in), ileft_(set_bilinear_boundary(left_in)),
        iright_(set_bilinear_boundary(right_in)), left_homo_(left_homo_in),
        right_homo_(right_homo_in), left_bc_funcs_(left_bc_funcs_in),
        right_bc_funcs_(right_bc_funcs_in),
        left_bc_time_func_(left_bc_time_func_in),
        right_bc_time_func_(right_bc_time_func_in), dv_func_(dv_func_in)
  {}

  P get_flux_scale() const { return static_cast<P>(flux_); };

  fk::matrix<P>  get_coefficients(int const level) const
  {
    // returns precomputed inv(mass) * coeff for this level
    expect(static_cast<int>(coefficients_.size()) > level);
    expect(level >= 0);
    return coefficients_[level];
  }

  fk::matrix<P> const &get_lhs_mass() const { return mass_; }

  void set_coefficients(fk::matrix<P> const &new_coefficients, int const degree,
                        int const max_level)
  {
    coefficients_.clear();

    // precompute inv(mass) * coeff for each level up to max level
    std::vector<int> ipiv((degree + 1) * fm::two_raised_to(max_level));
    for (int level = 0; level <= max_level; ++level)
    {
      auto const dof = (degree + 1) * fm::two_raised_to(level);
      fk::matrix<P> result(new_coefficients, 0, dof - 1, 0, dof - 1);
      auto mass_tmp = mass_.extract_submatrix(0, 0, dof, dof);
      fm::gesv(mass_tmp, result, ipiv);
      coefficients_.push_back(std::move(result));
    }
  }

  void set_coefficients(fk::matrix<P> const &&new_coefficients, int const level)
  {
    // set the coefficients at the given level
    expect(coefficients_.size() > static_cast<size_t>(level));
    this->coefficients_[level].clear_and_resize(new_coefficients.nrows(),
                                                new_coefficients.ncols()) =
        std::move(new_coefficients);
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
    if (coeff_type_ == coefficient_type::grad)
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
    if (coeff_type_ == coefficient_type::grad)
    {
      // Switch the upwinding direction
      return static_cast<flux_type>(-static_cast<P>(flux_in));
    }
    return flux_in;
  }

  coefficient_type coeff_type() const { return coeff_type_; }

  g_func_type<P> const &g_func() const { return g_func_; }
  g_func_type<P> const &lhs_mass_func() const { return lhs_mass_func_; }

  flux_type flux() const { return flux_; }

  boundary_condition left() const { return left_; }

  boundary_condition right() const { return right_; }

  boundary_condition ileft() const { return ileft_; }
  boundary_condition iright() const { return iright_; }

  homogeneity left_homo() const { return left_homo_; };
  homogeneity right_homo() const { return right_homo_; };

  std::vector<vector_func<P>> const &left_bc_funcs() const
  {
    return left_bc_funcs_;
  };
  std::vector<vector_func<P>> const &right_bc_funcs() const
  {
    return right_bc_funcs_;
  };

  scalar_func<P> const &left_bc_time_func() const
  {
    return left_bc_time_func_;
  }

  scalar_func<P> const &right_bc_time_func() const
  {
    return right_bc_time_func_;
  }

  g_func_type<P> const &dv_func() const
  {
    return dv_func_;
  }

private:
  coefficient_type coeff_type_;

  g_func_type<P> g_func_;
  g_func_type<P> lhs_mass_func_;

  flux_type flux_;

  boundary_condition left_;

  boundary_condition right_;

  boundary_condition ileft_;
  boundary_condition iright_;

  homogeneity left_homo_;
  homogeneity right_homo_;

  std::vector<vector_func<P>> left_bc_funcs_;
  std::vector<vector_func<P>> right_bc_funcs_;

  scalar_func<P> left_bc_time_func_;
  scalar_func<P> right_bc_time_func_;
  g_func_type<P> dv_func_;

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
      : time_dependent_(time_dependent_in), name_(name_in), flag_(flag_in),
        partial_terms_(partial_terms)
  {}

  void set_coefficients(fk::matrix<P> const &new_coefficients)
  {
    this->coefficients_.clear_and_resize(
        new_coefficients.nrows(), new_coefficients.ncols()) = new_coefficients;
  }

  void set_partial_coefficients(fk::matrix<P> const &&coeffs, int const pterm,
                                int const deg, int const max_lev)
  {
    expect(pterm >= 0);
    expect(pterm < static_cast<int>(partial_terms_.size()));
    ignore(deg);
    partial_terms_[pterm].set_coefficients(std::move(coeffs), max_lev);
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

  fk::matrix<P> const &get_coefficients() const { return coefficients_; }

  std::vector<partial_term<P>> const &get_partial_terms() const
  {
    return partial_terms_;
  }

  // after adapting to a new number of hierarchical basis levels,
  // recombine partial terms to form new coefficient matrices
  void rechain_coefficients(dimension<P> const &adapted_dim)
  {
    int const level = adapted_dim.get_level();

    auto const new_dof =
        (adapted_dim.get_degree() + 1) * fm::two_raised_to(level);
    expect(coefficients_.nrows() == coefficients_.ncols());

    if (partial_terms_.empty())
    {
      // no partial_terms? don't know if this can happen
      fk::matrix<P, mem_type::view>(coefficients_, 0, new_dof - 1, 0,
                                    new_dof - 1) = eye<P>(new_dof);
    }
    else if (partial_terms_.size() == 1)
    {
      // there's only one coefficient, just copy
      // probably wasteful too
      auto const &new_mat = partial_terms_[0].get_coefficients(level);
      fk::matrix<P, mem_type::view>(coefficients_, 0, new_dof - 1, 0,
                                    new_dof - 1) = new_mat;
    }
    else
    {
      // multiplying the matrices, we need two matrices
      // one keeping the cumulative matrix and one storing the next matrix
      fk::matrix<P> temp1 = partial_terms_[0].get_coefficients(level);
      fk::matrix<P> temp2(temp1.nrows(), temp1.ncols());

      // make sure the partial term has been build large enough
      expect(temp1.ncols() == new_dof);

      for (size_t i = 1; i < partial_terms_.size(); i++)
      {
        auto const &pmat = partial_terms_[i].get_coefficients(level);
        fm::gemm(temp1, pmat, temp2);
        std::swap(temp1, temp2);
      }

      fk::matrix<P, mem_type::view>(coefficients_, 0, new_dof - 1, 0,
                                    new_dof - 1) = temp1;
    }
  }

  bool time_dependent() const { return time_dependent_; }
  std::string const &name() const { return name_; }

  imex_flag flag() const { return flag_; }

private:
  bool time_dependent_;
  std::string name_;

  imex_flag flag_;

  std::vector<partial_term<P>> partial_terms_;

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
  source(std::vector<vector_func<P>> const source_funcs_in,
         scalar_func<P> const time_func_in)

      : source_funcs_(source_funcs_in), time_func_(time_func_in)
  {}

  std::vector<vector_func<P>> const &source_funcs() const { return source_funcs_; }
  scalar_func<P> const &time_func() const { return time_func_; }

private:
  std::vector<vector_func<P>> source_funcs_;
  scalar_func<P> time_func_;
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

  void reset() { parameters = std::vector<std::shared_ptr<parameter<P>>>(); }

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
  // used for sanity/error checking
  using precision_mode = P;

  PDE() : num_dims_(0), num_sources_(0), num_terms_(0), max_level_(0) {}
  PDE(parser const &cli_input, int const num_dims_in, int const num_sources_in,
      int const max_num_terms, std::vector<dimension<P>> const dimensions,
      term_set<P> const terms, std::vector<source<P>> const sources_in,
      std::vector<vector_func<P>> const exact_vector_funcs_in,
      dt_func<P> const get_dt,
      bool const do_poisson_solve_in          = false,
      bool const has_analytic_soln_in         = false,
      std::vector<moment<P>> const moments_in = {},
      bool const do_collision_operator_in     = true)
      : PDE(cli_input, num_dims_in, num_sources_in, max_num_terms, dimensions,
            terms, sources_in,
            std::vector<md_func_type<P>>({exact_vector_funcs_in}),
            get_dt, do_poisson_solve_in, has_analytic_soln_in,
            moments_in, do_collision_operator_in)
  {}
  PDE(parser const &cli_input, int const num_dims_in, int const num_sources_in,
      int const max_num_terms, std::vector<dimension<P>> dimensions,
      term_set<P> terms, std::vector<source<P>> sources_in,
      std::vector<md_func_type<P>> exact_vector_funcs_in,
      dt_func<P> get_dt,
      bool const do_poisson_solve_in      = false,
      bool const has_analytic_soln_in     = false,
      std::vector<moment<P>> moments_in   = {},
      bool const do_collision_operator_in = true)
  {
    initialize(cli_input, num_dims_in, num_sources_in,
      max_num_terms, std::move(dimensions), std::move(terms), std::move(sources_in),
      std::move(exact_vector_funcs_in),
      std::move(get_dt),
      do_poisson_solve_in,
      has_analytic_soln_in,
      std::move(moments_in),
      do_collision_operator_in);
  }

  void initialize(parser const &cli_input, int const num_dims_in, int const num_sources_in,
      int const max_num_terms, std::vector<dimension<P>> const &dimensions,
      term_set<P> const &terms, std::vector<source<P>> const &sources_in,
      std::vector<vector_func<P>> const &exact_vector_funcs_in,
      dt_func<P> const &get_dt,
      bool const do_poisson_solve_in           = false,
      bool const has_analytic_soln_in          = false,
      std::vector<moment<P>> const &moments_in = {},
      bool const do_collision_operator_in      = true)
  {
    this->initialize(cli_input, num_dims_in, num_sources_in, max_num_terms, dimensions,
                     terms, sources_in, std::vector<md_func_type<P>>({exact_vector_funcs_in}),
                     get_dt, do_poisson_solve_in, has_analytic_soln_in, moments_in,
                     do_collision_operator_in);
  }
  void initialize(parser const &cli_input, int const num_dims_in, int const num_sources_in,
      int const max_num_terms, std::vector<dimension<P>> const &dimensions,
      term_set<P> const &terms, std::vector<source<P>> const &sources_in,
      std::vector<md_func_type<P>> const &exact_vector_funcs_in,
      dt_func<P> const &get_dt,
      bool const do_poisson_solve_in           = false,
      bool const has_analytic_soln_in          = false,
      std::vector<moment<P>> const &moments_in = {},
      bool const do_collision_operator_in      = true)
  {
    this->initialize(cli_input, num_dims_in, num_sources_in, max_num_terms,
                     std::vector<dimension<P>>(dimensions), term_set<P>(terms),
                     std::vector<source<P>>(sources_in),
                     std::vector<md_func_type<P>>(exact_vector_funcs_in),
                     dt_func<P>(get_dt), do_poisson_solve_in,
                     has_analytic_soln_in, std::vector<moment<P>>(moments_in),
                     do_collision_operator_in);
  }

  void initialize(parser const &cli_input, int const num_dims_in, int const num_sources_in,
      int const max_num_terms, std::vector<dimension<P>> &&dimensions,
      term_set<P> &&terms, std::vector<source<P>> &&sources_in,
      std::vector<md_func_type<P>> &&exact_vector_funcs_in,
      dt_func<P> &&get_dt,
      bool const do_poisson_solve_in      = false,
      bool const has_analytic_soln_in     = false,
      std::vector<moment<P>> &&moments_in = {},
      bool const do_collision_operator_in = true)
  {
    static_assert(std::is_same_v<P, float> or std::is_same_v<P, double>,
                  "incorrect precision_mode, asgard can only work with PDE<float> or PDE<double>");
#ifndef ASGARD_ENABLE_DOUBLE
    static_assert(std::is_same_v<P, float>,
                  "double precision is not available, recompile with -DASGARD_PRECISIONS=\"float;double\"");
#endif
#ifndef ASGARD_ENABLE_FLOAT
    static_assert(std::is_same_v<P, double>,
                  "single precision is not available, recompile with -DASGARD_PRECISIONS=\"float;double\"");
#endif

    num_dims_    = num_dims_in;
    num_sources_ = num_sources_in;
    num_terms_   = get_num_terms(cli_input, max_num_terms);
    max_level_   = get_max_level(cli_input, dimensions);

    sources_            = std::move(sources_in);
    exact_vector_funcs_ = std::move(exact_vector_funcs_in);
    moments             = std::move(moments_in);

    do_poisson_solve_      = do_poisson_solve_in;
    do_collision_operator_ = do_collision_operator_in;
    has_analytic_soln_     = has_analytic_soln_in;
    dimensions_            = std::move(dimensions);
    terms_                 = std::move(terms);

    expect(num_dims_ > 0 and num_dims_ <= max_num_dimensions);
    expect(num_sources_ >= 0);
    expect(num_terms_ > 0 or (num_terms_ == 0 and has_interp()));

    expect(dimensions_.size() == static_cast<unsigned>(num_dims_));
    expect(terms_.size() == static_cast<unsigned>(max_num_terms));
    expect(sources_.size() == static_cast<unsigned>(num_sources_));

    // ensure analytic solution functions were provided if this flag is set
    if (has_analytic_soln_)
    {
      // each set of analytical solution functions must have num_dim functions
      for (const auto &md_func : exact_vector_funcs_)
      {
        expect(md_func.size() == static_cast<size_t>(num_dims_) or md_func.size() == static_cast<size_t>(num_dims_ + 1));
      }
    }

    // modify for appropriate level/degree
    // if default lev/degree not used
    auto const user_levels = cli_input.get_starting_levels().size();
    if (user_levels >= 2 && user_levels != num_dims_)
    {
      std::cerr << "failed to parse dimension-many starting levels - parsed "
                << user_levels << " levels\n";
      exit(1);
    }
    if (user_levels == num_dims_)
    {
      auto counter = 0;
      for (dimension<P> &d : dimensions_)
      {
        auto const num_levels = cli_input.get_starting_levels()(counter++);
        expect(num_levels >= 0);
        d.set_level(num_levels);
      }
    }
    else if (user_levels == 1)
    {
      auto const num_levels = cli_input.get_starting_levels()[0];
      for (dimension<P> &d : dimensions_)
        d.set_level(num_levels);
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
      expect(terms_.size() == static_cast<unsigned>(num_terms_));
    }

    auto const cli_degree = cli_input.get_degree();
    if (cli_degree != parser::NO_USER_VALUE)
    {
      expect(cli_degree >= 0);
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
      expect(term_list.size() == static_cast<unsigned>(num_dims_));
      for (auto &term_1D : term_list)
      {
        expect(term_1D.get_partial_terms().size() > 0);

        auto const max_dof =
            fm::two_raised_to(static_cast<int64_t>(max_level_)) * (degree + 1);
        expect(max_dof < INT_MAX);

        term_1D.set_coefficients(eye<P>(max_dof));

        for (auto &p : term_1D.get_partial_terms())
        {
          if (p.left_homo() == homogeneity::homogeneous)
            expect(static_cast<int>(p.left_bc_funcs().size()) == 0);
          else if (p.left_homo() == homogeneity::inhomogeneous)
            expect(static_cast<int>(p.left_bc_funcs().size()) == num_dims_);

          if (p.right_homo() == homogeneity::homogeneous)
            expect(static_cast<int>(p.right_bc_funcs().size()) == 0);
          else if (p.right_homo() == homogeneity::inhomogeneous)
            expect(static_cast<int>(p.right_bc_funcs().size()) == num_dims_);
        }
      }
    }

    // check all dimensions
    for (auto const &d : dimensions_)
    {
      expect(d.get_degree() >= 0);
      expect(d.get_level() >= 0);
      expect(d.domain_max > d.domain_min);
    }

    // initialize mass matrices to a default value
    for (auto i = 0; i < num_dims_; ++i)
    {
      for (int level = 0; level <= max_level_; ++level)
      {
        auto const dof = fm::two_raised_to(level) * (degree + 1);
        expect(dof < INT_MAX);
        update_dimension_mass_mat(i, std::move(eye<P>(dof)), level);
      }
    }

    // check all sources
    for (auto const &s : sources_)
    {
      expect(s.source_funcs().size() == static_cast<unsigned>(num_dims_));
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
        expect(md_func.size() == static_cast<unsigned>(num_dims_) + 1);
      }
    }

    if (cli_input.using_imex() && moments.empty())
    {
      throw std::runtime_error(
          "Invalid PDE choice for IMEX time advance. PDE must have "
          "moments defined to use -x\n");
    }

    gmres_outputs.resize(cli_input.using_imex() ? 2 : 1);

    // hack to preallocate empty matrix for pterm coefficients for adapt
    for (auto i = 0; i < num_dims_; ++i)
    {
      auto const &dim = this->get_dimensions()[i];
      for (auto j = 0; j < num_terms_; ++j)
      {
        auto const &term_1D       = this->get_terms()[j][i];
        auto const &partial_terms = term_1D.get_partial_terms();
        for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
        {
          std::vector<fk::matrix<P>> pterm_coeffs;
          for (int level = 0; level <= max_level_; ++level)
          {
            auto const dof = (dim.get_degree() + 1) * fm::two_raised_to(level);
            fk::matrix<P> result_tmp = eye<P>(dof);
            pterm_coeffs.emplace_back(std::move(result_tmp));
          }
          this->set_partial_coefficients(j, i, k, std::move(pterm_coeffs));
        }
      }
    }

    expect(not (!!interp_nox_ and !!interp_x_));
  }

  constexpr static int extract_dim0 = 1;
  // copy constructor to create a 1D version of the PDE
  // this is used in the IMEX time advance to help define 1D mapping from
  // wavelet to realspace
  // TODO: there is likely a better way to do this. Another option is to flatten
  // element table to 1D (see hash_table_2D_to_1D.m)
  PDE(const PDE &pde, int)
      : moments(pde.moments), num_dims_(1), num_sources_(pde.sources_.size()),
        num_terms_(pde.get_terms().size()), max_level_(pde.max_level_),
        sources_(pde.sources_), exact_vector_funcs_(pde.exact_vector_funcs_),
        do_poisson_solve_(pde.do_poisson_solve()),
        do_collision_operator_(pde.do_collision_operator()),
        has_analytic_soln_(pde.has_analytic_soln()),
        dimensions_({pde.get_dimensions()[0]}), terms_(pde.get_terms())
  {}

  // public but const data.
  int num_dims() const { return num_dims_; }
  int num_sources() const { return num_sources_; }
  int num_terms() const { return num_terms_; }
  int max_level() const { return max_level_; }

  std::vector<source<P>> const &sources() const { return sources_; };
  std::vector<md_func_type<P>> const &exact_vector_funcs() const
  {
    return exact_vector_funcs_;
  }
  bool has_exact_time() const
  {
    return has_analytic_soln_ and
        static_cast<int>(exact_vector_funcs_.back().size()) == num_dims_ + 1;
  }
  P exact_time(P time) const
  {
    fk::vector<P> dummy;
    return exact_vector_funcs_.back().back()(dummy, time)[0];
  }

  std::vector<moment<P>> moments;
  bool do_poisson_solve() const { return do_poisson_solve_; }
  bool do_collision_operator() const { return do_collision_operator_; }
  bool has_analytic_soln() const { return has_analytic_soln_; }

  // data for poisson solver
  fk::vector<P> poisson_diag;
  fk::vector<P> poisson_off_diag;

  fk::vector<P> E_field;
  fk::vector<P> phi;
  fk::vector<P> E_source;
  // holds gmres error and iteration counts for writing to output file
  std::vector<gmres_info<P>> gmres_outputs;
  adaptive_info<P> adapt_info;

  virtual ~PDE() = default;

  std::vector<dimension<P>> const &get_dimensions() const
  {
    return dimensions_;
  }
  std::vector<dimension<P>> &get_dimensions() { return dimensions_; }

  term_set<P> const &get_terms() const { return terms_; }
  term_set<P> &get_terms() { return terms_; }

  fk::matrix<P> const &get_coefficients(int const term, int const dim) const
  {
    expect(term >= 0);
    expect(term < num_terms_);
    expect(dim >= 0);
    expect(dim < num_dims_);
    return terms_[term][dim].get_coefficients();
  }

  /* gives a vector of partial_term matrices to the term object so it can
     construct the full operator matrix */
  void
  set_coefficients(fk::matrix<P> const &coeffs, int const term, int const dim)
  {
    expect(term >= 0);
    expect(term < num_terms_);
    expect(dim >= 0);
    expect(dim < num_dims_);
    terms_[term][dim].set_coefficients(coeffs);
  }

  void set_partial_coefficients(int const term, int const dim, int const pterm,
                                fk::matrix<P> const &&coeffs)
  {
    expect(term >= 0);
    expect(term < num_terms_);
    expect(dim >= 0);
    expect(dim < num_dims_);
    terms_[term][dim].set_partial_coefficients(std::move(coeffs), pterm,
                                               dimensions_[dim].get_degree(),
                                               dimensions_[dim].get_level());
  }

  void set_partial_coefficients(int const term, int const dim, int const pterm,
                                std::vector<fk::matrix<P>> const &coeffs)
  {
    expect(term >= 0);
    expect(term < num_terms_);
    expect(dim >= 0);
    expect(dim < num_dims_);
    terms_[term][dim].set_partial_coefficients(coeffs, pterm);
  }

  void set_lhs_mass(int const term, int const dim, int const pterm,
                    fk::matrix<P> const &mass)
  {
    expect(term >= 0);
    expect(term < num_terms_);
    expect(dim >= 0);
    expect(dim < num_dims_);
    terms_[term][dim].set_lhs_mass(mass, pterm);
  }

  void set_lhs_mass(int const term, int const dim, int const pterm,
                    fk::matrix<P> &&mass)
  {
    expect(term >= 0);
    expect(term < num_terms_);
    expect(dim >= 0);
    expect(dim < num_dims_);
    terms_[term][dim].set_lhs_mass(mass, std::move(pterm));
  }

  void update_dimension(int const dim_index, int const new_level)
  {
    assert(dim_index >= 0);
    assert(dim_index < num_dims_);
    assert(new_level >= 0);

    dimensions_[dim_index].set_level(new_level);
  }

  void rechain_dimension(int const dim_index)
  {
    expect(dim_index >= 0);
    expect(dim_index < num_dims_);
    for (auto i = 0; i < num_terms_; ++i)
    {
      terms_[i][dim_index].rechain_coefficients(dimensions_[dim_index]);
    }
  }

  void update_dimension_mass_mat(int const dim_index, fk::matrix<P> &&mass,
                                 int const level)
  {
    assert(dim_index >= 0);
    assert(dim_index < num_dims_);

    dimensions_[dim_index].set_mass_matrix(std::move(mass), level);
  }

  P get_dt() const { return dt_; };

  void set_dt(P const dt)
  {
    expect(dt > 0.0);
    dt_ = dt;
  }

  //! Return true if any kind of interpolation has been enabled
  bool has_interp() const
  {
    return !!interp_nox_ or !!interp_x_ or !!interp_initial_ or !!interp_exact_;
  }

  //! Returns the total area/volume of the domain
  void get_domain_bounds(std::array<P, max_num_dimensions> &dmin,
                         std::array<P, max_num_dimensions> &dmax) const
  {
    for (int i = 0; i < num_dims_; i++)
    {
      dmin[i] = dimensions_[i].domain_min;
      dmax[i] = dimensions_[i].domain_max;
    }
  }

  /*!
   * \brief Interpolation operator that does not have explicit dependence on space.
   *
   * Examples of no explicit dependence on x:
   *    F(t, u(t, x)) = t * u(t, x)
   *    F(t, u(t, x)) = u(t, x)^2
   *
   * Set by the derived classes with
   * \code
   *   this->interp_nox_ =
   * [](P t, std::vector<P> const &u, std::vector<P> &F)->void
   * {
   *   for (size_t i = 0; i < u.size(); i++)
   *     F[i] = t * u[i];
   *  // alternative: F[i] = u[i] * u[i];
   * }
   * \endcode
   */
  std::function<void(P t, std::vector<P> const &, std::vector<P> &)> const &
  interp_nox() const { return interp_nox_; }

  /*!
   * \brief Interpolation operator that has explicit dependence on space.
   *
   * Examples of no explicit dependence on x:
   *    F(t, u(t, x)) = t * (x_1 + x_2) * u(t, x)
   *    F(t, u(t, x)) = sin(x_1) * cos(x_2)
   * the second example is a forcing term with no dependence on u(t, x)
   *
   * Set by the derived classes with
   * \code
   *   this->interp_x_ =
   * [](P t, vector2d<P> const &x, std::vector<P> const &u, std::vector<P> &F)
   *   ->void
   * {
   *   for (size_t i = 0; i < u.size(); i++)
   *     F[i] = t * (x[i][0] + x[i][1]) * u[i];
   *  // forcing exmaple: F[i] = std::sin(x[i][0]) * std::cos(x[i][1])
   * }
   * \endcode
   */
  std::function<void(P t, vector2d<P> const &, std::vector<P> const &, std::vector<P> &)> const &
  interp_x() const { return interp_x_; }

  /*!
   * \brief Define non-separable initial conditions.
   *
   * Set by the derived classes with
   * \code
   *   this->interp_initial_ =
   * [](vector2d<P> const &x, std::vector<P> &u)
   *   ->void
   * {
   *   for (size_t i = 0; i < u.size(); i++)
   *     u[i] = x[i][0] + x[i][1];
   * }
   * \endcode
   */
  std::function<void(vector2d<P> const &, std::vector<P> &)> const &
  interp_initial() const { return interp_initial_; }

  /*!
   * \brief Define non-separable exact solution.
   *
   * Set by the derived classes with
   * \code
   *   this->interp_exact_ =
   * [](P t, vector2d<P> const &x, std::vector<P> &u)
   *   ->void
   * {
   *   for (size_t i = 0; i < u.size(); i++)
   *     u[i] = t + x[i][0] + x[i][1];
   * }
   * \endcode
   */
  std::function<void(P t, vector2d<P> const &, std::vector<P> &)> const &
  interp_exact() const { return interp_exact_; }

protected:
  std::function<void(P t, std::vector<P> const &, std::vector<P> &)> interp_nox_;

  std::function<void(P t, vector2d<P> const &, std::vector<P> const &, std::vector<P> &)> interp_x_;

  std::function<void(vector2d<P> const &, std::vector<P> &)> interp_initial_;

  std::function<void(P t, vector2d<P> const &, std::vector<P> &)> interp_exact_;

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
      if (auto const &levels = cli_input.get_max_adapt_levels();
          !levels.empty())
        return *std::max_element(levels.begin(), levels.end());
      else
        return cli_input.get_max_level();
    }
    else
    {
      // if adaptivity is not used, only generate to the highest dim level
      auto const levels = cli_input.get_starting_levels();
      return levels.empty()
                 ? std::max_element(
                       dims.begin(), dims.end(),
                       [](dimension<P> const &a, dimension<P> const &b) {
                         return a.get_level() < b.get_level();
                       })
                       ->get_level()
                 : *std::max_element(levels.begin(), levels.end());
    }
  }

  int num_dims_;
  int num_sources_;
  int num_terms_;
  int max_level_;

  std::vector<source<P>> sources_;
  std::vector<md_func_type<P>> exact_vector_funcs_;

  bool do_poisson_solve_;
  bool do_collision_operator_;
  bool has_analytic_soln_;

  std::vector<dimension<P>> dimensions_;
  term_set<P> terms_;
  P dt_;
};
} // namespace asgard
