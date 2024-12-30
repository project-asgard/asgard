#pragma once

#include "../asgard_dimension.hpp"
#include "../asgard_indexset.hpp"
#include "../asgard_quadrature.hpp"

// the quadrature is needed by some of the pdes to perform internal operations

namespace asgard
{
//
// This file contains all of the interface and object definitions for our
// representation of a PDE
//

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

enum class coefficient_type
{
  grad,
  mass,
  div,
  penalty
};

enum class pterm_dependence
{
  none, // nothing special, uses generic g-func
  electric_field, // depends on the electric field
  electric_field_infnrm, // depends on the max abs( electric_field )
  moment_divided_by_density, // moment divided by moment 0
  lenard_bernstein_diff_theta_1x1v,
  lenard_bernstein_diff_theta_1x2v,
  lenard_bernstein_diff_theta_1x3v,
};

template<coefficient_type>
struct has_flux_t : public std::true_type{};

template<> struct has_flux_t<coefficient_type::mass> : public std::false_type{};

template<coefficient_type t>
constexpr bool has_flux_v = has_flux_t<t>::value;

constexpr bool has_flux(coefficient_type t) {
  return (t != coefficient_type::mass);
}

enum class flux_type
{
  upwind        = -1,
  central       = 0,
  downwind      = 1,
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

//! indicates mass partial term of moment divided by density (moment 0)
struct mass_moment_over_density {
  explicit mass_moment_over_density(int moment_in) : moment(moment_in)
  {
    expect(moment > 0);
  }
  int moment;
};
//! indicates mass partial term of negative moment divided by density (moment 0)
struct mass_moment_over_density_neg {
  explicit mass_moment_over_density_neg(int moment_in) : moment(moment_in)
  {
    expect(moment > 0);
  }
  int moment;
};

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

  partial_term(coefficient_type const coeff_type_in,
               pterm_dependence const depends_in,
               g_func_f_type<P> const g_func_f_in    = nullptr,
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

      : coeff_type_(coeff_type_in), depends_(depends_in), g_func_f_(g_func_f_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(set_flux(flux_in)), left_(left_in),
        right_(right_in), ileft_(set_bilinear_boundary(left_in)),
        iright_(set_bilinear_boundary(right_in)), left_homo_(left_homo_in),
        right_homo_(right_homo_in), left_bc_funcs_(left_bc_funcs_in),
        right_bc_funcs_(right_bc_funcs_in),
        left_bc_time_func_(left_bc_time_func_in),
        right_bc_time_func_(right_bc_time_func_in), dv_func_(dv_func_in)
  {
    expect(depends_ != pterm_dependence::none);
    expect(depends_ != pterm_dependence::moment_divided_by_density);
    expect(coeff_type_ == coefficient_type::mass); // have not done the others yet
    // if this depends on the electric-filed, there should be a g_func_f
    expect(not (depends_ == pterm_dependence::electric_field and !g_func_f_));
    expect(not (depends_ == pterm_dependence::electric_field_infnrm and !g_func_f_));
    // if this depends on Lenard Bernstein collision, cannot have g funcs
    if (depends_ == pterm_dependence::lenard_bernstein_diff_theta_1x1v
        or depends_ == pterm_dependence::lenard_bernstein_diff_theta_1x2v
        or depends_ == pterm_dependence::lenard_bernstein_diff_theta_1x3v) {
      expect(!g_func_ and !g_func_f_);
    }
  }

  //! pterm_dependence that assumes a mass coefficient type
  partial_term(pterm_dependence const depends_in,
               g_func_f_type<P> const g_func_f_in = nullptr)
    : partial_term(coefficient_type::mass, depends_in, g_func_f_in)
  {}

  //! indicates mass term with coefficient mom_in.moment / moment0
  partial_term(mass_moment_over_density mom_in, g_func_type<P> const dv_func_in = nullptr)
      : depends_(pterm_dependence::moment_divided_by_density), mom(mom_in.moment),
        dv_func_(dv_func_in)
  {}
  //! indicates mass term with coefficient - mom_in.moment / moment0
  partial_term(mass_moment_over_density_neg mom_in, g_func_type<P> const dv_func_in = nullptr)
      : depends_(pterm_dependence::moment_divided_by_density), mom(-mom_in.moment),
        dv_func_(dv_func_in)
  {}

  P get_flux_scale() const { return static_cast<P>(flux_); };

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
  pterm_dependence depends() const { return depends_; }

  bool is_identity() const
  {
    if (depends_ != pterm_dependence::none)
      return false;

    return (coeff_type_ == coefficient_type::mass and not g_func_ and not g_func_f_
            and not lhs_mass_func_ and not dv_func_);
  }

  g_func_type<P> const &g_func() const { return g_func_; }
  g_func_f_type<P> const &g_func_f() const { return g_func_f_; }
  g_func_type<P> const &lhs_mass_func() const { return lhs_mass_func_; }

  flux_type flux() const { return flux_; }

  boundary_condition left() const { return left_; }

  boundary_condition right() const { return right_; }

  boundary_condition ileft() const { return ileft_; }
  boundary_condition iright() const { return iright_; }

  homogeneity left_homo() const { return left_homo_; };
  homogeneity right_homo() const { return right_homo_; };

  int mom_index() const { return mom; }

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
  //! can be used to set the dv_func() without a messy constructor
  g_func_type<P> &dv_func()
  {
    return dv_func_;
  }

private:
  coefficient_type coeff_type_ = coefficient_type::mass;

  pterm_dependence depends_ = pterm_dependence::none;

  g_func_type<P> g_func_;
  g_func_f_type<P> g_func_f_;
  g_func_type<P> lhs_mass_func_;

  int mom = 0; // paired with mom-by-density, cannot be zero, sign used too

  flux_type flux_;

  boundary_condition left_  = boundary_condition::neumann;
  boundary_condition right_ = boundary_condition::neumann;

  boundary_condition ileft_  = boundary_condition::neumann;
  boundary_condition iright_ = boundary_condition::neumann;

  homogeneity left_homo_  = homogeneity::homogeneous;
  homogeneity right_homo_ = homogeneity::homogeneous;

  std::vector<vector_func<P>> left_bc_funcs_;
  std::vector<vector_func<P>> right_bc_funcs_;

  scalar_func<P> left_bc_time_func_;
  scalar_func<P> right_bc_time_func_;
  g_func_type<P> dv_func_;
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
  term(bool const time_dependent_in, std::string const name_in,
       partial_term<P> const &partial_terms,
       imex_flag const flag_in = imex_flag::unspecified)
      : time_dependent_(time_dependent_in), name_(name_in), flag_(flag_in),
        partial_terms_({partial_terms, })
  {}
  term(std::string const name_in,
       std::initializer_list<partial_term<P>> const partial_terms,
       imex_flag const flag_in = imex_flag::unspecified)
      : time_dependent_(false), name_(name_in), flag_(flag_in),
        partial_terms_(partial_terms)
  {}
  term(std::string const name_in, partial_term<P> const &partial_terms,
       imex_flag const flag_in = imex_flag::unspecified)
      : time_dependent_(false), name_(name_in), flag_(flag_in),
        partial_terms_({partial_terms, })
  {}

  std::vector<partial_term<P>> const &get_partial_terms() const
  {
    return partial_terms_;
  }

  bool time_dependent() const { return time_dependent_; }
  std::string const &name() const { return name_; }

  imex_flag flag() const { return flag_; }

  bool has_dependence(pterm_dependence dep) const {
    for (auto const &pt : partial_terms_)
      if (pt.depends() == dep)
        return true;
    return false;
  }
  bool is_moment_independant() const {
    for (auto const &pt : partial_terms_)
      if (pt.depends() != pterm_dependence::none)
        return false;
    return true;
  }
  bool is_identity() const {
    for (auto const &pt : partial_terms_)
      if (not pt.is_identity())
        return false;
    return true;
  }

  int max_moment_index() const {
    int mmax = 0;
    for (auto const &pt : partial_terms_)
      mmax = std::max(mmax, std::abs(pt.mom_index()));
    return mmax;
  }

private:
  bool time_dependent_;
  std::string name_;

  imex_flag flag_;

  std::vector<partial_term<P>> partial_terms_;
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

// template<typename P>
// using moment_funcs = std::vector<std::vector<md_func_type<P>>>;

template<typename P>
class PDE
{
public:
  // used for sanity/error checking
  using precision_mode = P;

  PDE() {}
  PDE(prog_opts const &cli_input, int const num_dims_in, int const num_sources_in,
      std::vector<dimension<P>> const dimensions,
      term_set<P> const terms, std::vector<source<P>> const sources_in,
      std::vector<vector_func<P>> const exact_vector_funcs_in,
      dt_func<P> const get_dt,
      bool const has_analytic_soln_in     = false,
      bool const do_collision_operator_in = true)
      : PDE(cli_input, num_dims_in, num_sources_in, dimensions, terms, sources_in,
            std::vector<md_func_type<P>>({exact_vector_funcs_in}),
            get_dt, has_analytic_soln_in, do_collision_operator_in)
  {}
  PDE(prog_opts const &cli_input, int const num_dims_in, int const num_sources_in,
      std::vector<dimension<P>> dimensions,
      term_set<P> terms, std::vector<source<P>> sources_in,
      std::vector<md_func_type<P>> exact_vector_funcs_in,
      dt_func<P> get_dt,
      bool const has_analytic_soln_in     = false,
      bool const do_collision_operator_in = true)
  {
    initialize(cli_input, num_dims_in, num_sources_in,
      std::move(dimensions), std::move(terms), std::move(sources_in),
      std::move(exact_vector_funcs_in),
      std::move(get_dt),
      has_analytic_soln_in,
      do_collision_operator_in);
  }

  void initialize(prog_opts const &cli_input, int const num_dims_in, int const num_sources_in,
      std::vector<dimension<P>> const &dimensions,
      term_set<P> const &terms, std::vector<source<P>> const &sources_in,
      std::vector<vector_func<P>> const &exact_vector_funcs_in,
      dt_func<P> const &get_dt,
      bool const has_analytic_soln_in     = false,
      bool const do_collision_operator_in = true)
  {
    this->initialize(cli_input, num_dims_in, num_sources_in, dimensions,
                     terms, sources_in, std::vector<md_func_type<P>>({exact_vector_funcs_in}),
                     get_dt, has_analytic_soln_in,
                     do_collision_operator_in);
  }
  void initialize(prog_opts const &cli_input, int const num_dims_in, int const num_sources_in,
      std::vector<dimension<P>> const &dimensions,
      term_set<P> const &terms, std::vector<source<P>> const &sources_in,
      std::vector<md_func_type<P>> const &exact_vector_funcs_in,
      dt_func<P> const &get_dt,
      bool const has_analytic_soln_in     = false,
      bool const do_collision_operator_in = true)
  {
    this->initialize(cli_input, num_dims_in, num_sources_in,
                     std::vector<dimension<P>>(dimensions), term_set<P>(terms),
                     std::vector<source<P>>(sources_in),
                     std::vector<md_func_type<P>>(exact_vector_funcs_in),
                     dt_func<P>(get_dt), has_analytic_soln_in,
                     do_collision_operator_in);
  }

  void initialize(prog_opts const &cli_input, int const num_dims_in, int const num_sources_in,
      std::vector<dimension<P>> &&dimensions,
      term_set<P> &&terms, std::vector<source<P>> &&sources_in,
      std::vector<md_func_type<P>> &&exact_vector_funcs_in,
      dt_func<P> &&get_dt,
      bool const has_analytic_soln_in     = false,
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

    options_ = cli_input; // save a copy of the options

    num_dims_    = num_dims_in;
    num_sources_ = num_sources_in;

    sources_            = std::move(sources_in);
    exact_vector_funcs_ = std::move(exact_vector_funcs_in);

    do_collision_operator_ = do_collision_operator_in;
    has_analytic_soln_     = has_analytic_soln_in;
    dimensions_            = std::move(dimensions);
    terms_                 = std::move(terms);

    num_terms_ = static_cast<int>(terms_.size());

    // sanity check and load sane defaults when appropriate
    expect(num_dims_ > 0 and num_dims_ <= max_num_dimensions);
    expect(num_sources_ >= 0);
    expect(num_terms_ > 0 or (num_terms_ == 0 and has_interp()));

    expect(dimensions_.size() == static_cast<unsigned>(num_dims_));
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

    // secondary sanity checking and setting up the defaults
    if (not options_.grid)
      options_.grid = grid_type::sparse;

    // modify for appropriate level/degree
    // if default lev/degree not used
    if (not options_.start_levels.empty())
    {
      size_t num_provided = options_.start_levels.size();
      rassert(not (num_provided >= 2 and num_provided != static_cast<size_t>(num_dims_)),
              "wrong number of starting levels provided, --start_levels, -l, must container either one int or an int per dimension");
      if (num_provided < static_cast<size_t>(num_dims_))
      {
        options_.start_levels.resize(num_dims_, options_.start_levels.front());
      }
      for (auto d : indexof<int>(num_dims_))
      {
        dimensions_[d].set_level(options_.start_levels[d]);
      }
    }
    else
    {
      options_.start_levels.reserve(num_dims_);
      for (auto const &dim : dimensions_)
        options_.start_levels.push_back(dim.get_level());
    }

    if (not options_.max_levels.empty()) // user provided max-level
    {
      size_t num_provided = options_.max_levels.size();
      rassert(not (num_provided >= 2 and num_provided != static_cast<size_t>(num_dims_)),
              "wrong number of max levels provided, must container either one int or an int per dimension");
      if (num_provided < static_cast<size_t>(num_dims_))
      { // resize the vector and fill it up with the first value
        options_.max_levels.resize(num_dims_, options_.max_levels.front());
      }
      for (auto d : indexof<int>(num_dims_))
      {
        rassert(options_.max_levels[d] >= dimensions_[d].get_level(),
                "the max-level cannot be less than the stating level (lower the starting level or increase the max)");
      }
    }
    else
    {
      options_.max_levels = options_.start_levels;
    }

    max_level_ = *std::max_element(options_.max_levels.begin(), options_.max_levels.end());

    if (options_.degree) // user provided degree
    {
      int const degree = options_.degree.value();
      rassert(degree >= 0, "the degree must be non-negative number");
      for (auto &dim : dimensions_)
        dim.set_degree(degree);
    }
    else
      options_.degree = dimensions_.front().get_degree();

    // polynomial degree of freedom in a cell
    int const pdof = dimensions_[0].get_degree() + 1;

    // check all terms
    for (auto &term_list : terms_)
    {
      expect(term_list.size() == static_cast<unsigned>(num_dims_));
      for (auto &term_1D : term_list)
      {
        expect(term_1D.get_partial_terms().size() > 0);

        auto const max_dof = fm::ipow2(static_cast<int64_t>(max_level_)) * pdof;
        expect(max_dof < INT_MAX);

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

    // check all sources
    for (auto const &s : sources_)
    {
      expect(s.source_funcs().size() == static_cast<unsigned>(num_dims_));
    }

    dt_ = (options_.dt) ? options_.dt.value() : get_dt(dimensions_[0]) * 0.01;

    if (not options_.num_time_steps)
      options_.num_time_steps = 10;

    if (options_.step_method)
    {
      use_imex_     = options_.step_method.value() == time_advance::method::imex;
      use_implicit_ = options_.step_method.value() == time_advance::method::imp;
    }
    else
    {
      use_imex_     = false;
      use_implicit_ = false;
      options_.step_method = time_advance::method::exp;
    }

    gmres_outputs.resize(use_imex_ ? 2 : 1);

    expect(not (!!interp_nox_ and !!interp_x_));

    if (options_.adapt_threshold)
    {
      rassert(options_.adapt_threshold.value() > 0,
              "the adapt-threshold should be a positive value");
      if (not options_.anorm)
        options_.anorm = adapt_norm::linf;
    }

    if (use_imex_ or use_implicit_)
      if (not options_.solver)
        options_.solver = solve_opts::bicgstab;

    // missing tolerance will be set within the solver module
    if (not options_.isolver_tolerance)
      options_.isolver_tolerance = solver::notolerance;
    if (not options_.isolver_iterations)
      options_.isolver_iterations = solver::novalue;
    if (not options_.isolver_outer_iterations)
      options_.isolver_outer_iterations = solver::novalue;
  }

  constexpr static int extract_dim0 = 1;
  // copy constructor to create a 1D version of the PDE
  // this is used in the IMEX time advance to help define 1D mapping from
  // wavelet to realspace
  // TODO: there is likely a better way to do this. Another option is to flatten
  // element table to 1D (see hash_table_2D_to_1D.m)
  PDE(const PDE &pde, int)
      : options_(pde.options_), num_dims_(1), num_sources_(pde.sources_.size()),
        max_level_(pde.max_level_),
        sources_(pde.sources_), exact_vector_funcs_(pde.exact_vector_funcs_),
        do_collision_operator_(pde.do_collision_operator()),
        has_analytic_soln_(pde.has_analytic_soln()),
        dimensions_({pde.get_dimensions()[0]}), terms_(pde.get_terms())
  {
    options_.grid          = grid_type::dense;
    options_.start_levels  = {pde.get_dimensions().front().get_level(), };
    options_.max_levels    = {pde.max_level(), };
  }

  const prog_opts &options() const { return options_; }

  // public but const data.
  int num_dims() const { return num_dims_; }
  int num_sources() const { return num_sources_; }
  int num_terms() const { return num_terms_; }
  int max_level() const { return max_level_; }

  bool use_implicit() const { return use_implicit_; }
  bool use_imex() const { return use_imex_; }
  kronmult_mode kron_mod() const { return kmod_; }
  int memory_limit() const { return memory_limit_; }

  bool is_output_step(int i) const
  {
    if (not options_.wavelet_output_freq)
      return false;
    return (i == 0 or i % options_.wavelet_output_freq.value() == 0);
  }

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

  bool skip_old_moments = false; // TODO: remove this once all PDEs have transitioned

  bool do_poisson_solve() const { // TODO: rename to poisson dependence
    for (auto const &terms_md : terms_)
      for (auto const &term1d : terms_md)
        if (term1d.has_dependence(pterm_dependence::electric_field))
          return true;

    // no terms have the poisson dependence
    return false;
  }
  bool do_collision_operator() const { return do_collision_operator_; }
  bool has_analytic_soln() const { return has_analytic_soln_; }

  int required_moments() const
  {
    int num_moments = (this->do_poisson_solve()) ? 1 : 0;
    for (auto const &terms_md : terms_) {
      for (auto const &term1d : terms_md) {
        if (term1d.has_dependence(pterm_dependence::moment_divided_by_density))
        {
          int const mom = term1d.max_moment_index();
          num_moments = std::max(1 + mom / (num_dims_ - 1), num_moments);
        }
        else if (term1d.has_dependence(pterm_dependence::lenard_bernstein_diff_theta_1x1v)
                 or term1d.has_dependence(pterm_dependence::lenard_bernstein_diff_theta_1x2v)
                 or term1d.has_dependence(pterm_dependence::lenard_bernstein_diff_theta_1x3v))
        {
          num_moments = std::max(3, num_moments);
        }
      }
    }
    return num_moments;
  }

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

  void update_dimension(int const dim_index, int const new_level)
  {
    assert(dim_index >= 0);
    assert(dim_index < num_dims_);
    assert(new_level >= 0);

    dimensions_[dim_index].set_level(new_level);
  }

  //! Return the direction of the flux term or -1 for mass term
  int get_flux_direction(int term_id)
  {
    for (int d = 0; d < num_dims_; d++)
      for (auto const &pt : terms_[term_id][d].get_partial_terms())
        if (has_flux(pt.coeff_type()))
          return d;
    return -1;
  }

  P get_dt() const { return dt_; }; //  using default cfl

  void set_dt(P const dt)
  {
    expect(dt > 0.0);
    dt_ = dt;
  }
  //! returns the max of the currently set levels based on the refinement
  int current_max_level() const
  {
    int ml = 0;
    for (auto &dim : dimensions_)
      ml = std::max(ml, dim.get_level());
    return ml;
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

#ifndef KRON_MODE_GLOBAL
  // this is a hack needed for the old local-kronmult, keep it for MPI purposes only
  std::vector<fk::matrix<P>> coeffs_;
  fk::matrix<P> const & get_coefficients(int t, int d) const
  {
    return coeffs_[t * num_dims_ + d];
  }
#endif

  // commonly used building blocks of g_funcs
  static P gfunc_pos1(P const, P const) {
    return P{1};
  }
  static P gfunc_neg1(P const, P const) {
    return P{-1};
  }
  static P gfunc_positive(P const v, P const) {
    return std::max(P{0}, v);
  }
  static P gfunc_negative(P const v, P const) {
    return std::min(P{0}, v);
  }
  static P gfunc_f_field(P const, P const, P const f) {
    return f;
  }
  static P gfunc_f_positive(P const, P const, P const f) {
    return std::max(P{0}, f);
  }
  static P gfunc_f_negative(P const, P const, P const f) {
    return std::min(P{0}, f);
  }

protected:
  std::function<void(P t, std::vector<P> const &, std::vector<P> &)> interp_nox_;

  std::function<void(P t, vector2d<P> const &, std::vector<P> const &, std::vector<P> &)> interp_x_;

  std::function<void(vector2d<P> const &, std::vector<P> &)> interp_initial_;

  std::function<void(P t, vector2d<P> const &, std::vector<P> &)> interp_exact_;

private:
  prog_opts options_;

  int num_dims_    = 0;
  int num_sources_ = 0;
  int num_terms_   = 0;
  int max_level_   = 0;

  std::vector<source<P>> sources_;
  std::vector<md_func_type<P>> exact_vector_funcs_;

  bool do_collision_operator_ = false;
  bool has_analytic_soln_     = false;

  std::vector<dimension<P>> dimensions_;
  term_set<P> terms_;
  P dt_{0};

  // time stepping options
  bool use_implicit_  = false;
  bool use_imex_      = false;
  // those will be removed in near future
  kronmult_mode kmod_ = kronmult_mode::dense;
  int memory_limit_   = 0;
};

//! add the two-part Vlasov operator, periodic boundary
template<typename P>
inline void add_vlassov_1x1v(term_set<P> &terms)
{
  imex_flag constexpr imex = imex_flag::imex_explicit;

  partial_term<P> ptDivU(
      coefficient_type::div, PDE<P>::gfunc_neg1, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  partial_term<P> ptMassP(coefficient_type::mass, PDE<P>::gfunc_positive);

  term<P> div_x_up("div_x_up", ptDivU, imex);

  term<P> massP("mass_positive", ptMassP, imex);

  partial_term<P> ptDivD(
      coefficient_type::div, PDE<P>::gfunc_neg1, nullptr, flux_type::downwind,
      boundary_condition::periodic, boundary_condition::periodic);

  partial_term<P> ptMassN(coefficient_type::mass, PDE<P>::gfunc_negative);

  term<P> div_x_down("div_x_down", ptDivD, imex);

  term<P> massN("mass_negative", ptMassN, imex);

  terms.push_back({div_x_up, massP});
  terms.push_back({div_x_down, massN});
}

//! adds the LB collision operator to the term set
template<typename P>
inline void add_lenard_bernstein_collisions_1x1v(P const nu, term_set<P> &terms)
{
  std::function<P(P const, P const)> const_nu = [nnu = nu](P const, P const = 0)->P{ return nnu; };
  std::function<P(P const, P const)> get_nuv = [nnu = nu](P const v, P const = 0)->P{ return nnu * v; };

  bool constexpr time_depend = true;

  imex_flag constexpr imex = imex_flag::imex_implicit;

  // moment components of the collision operator, split into 3 parts
  // (nu, div_v v) -> (mass_nu, divv)
  // (-u_f, nu * div_v) -> (pt_mass_uf_neg, nu_divv)
  // (mom2/mom0 - u_f^2, nu * div * grad) -> (pt_mass_ef, {pt_div_up, pt_nu_grad_down})

  partial_term<P> pt_divv(
      coefficient_type::div, get_nuv, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  partial_term<P> pt_nu_divv(
      coefficient_type::div, const_nu, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  partial_term<P> pt_div_up(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  partial_term<P> pt_nu_grad_down(
      coefficient_type::grad, const_nu, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  term<P> I("I", partial_term<P>(coefficient_type::mass), imex);

  term<P> divv("LB_divv", {pt_divv}, imex);

  term<P> mass_uf_neg(time_depend, "LB_uf_neg",
                      partial_term<P>(mass_moment_over_density_neg{1}), imex);

  term<P> nu_divv("LB_vdiv", {pt_nu_divv, }, imex);

  term<P> mass_theta(time_depend, "LB_mass_theta",
                     partial_term<P>(pterm_dependence::lenard_bernstein_diff_theta_1x1v), imex);

  term<P> nu_div_grad("LB_nu_div_grad", {pt_div_up, pt_nu_grad_down}, imex);

  terms.push_back({I, divv});
  terms.push_back({mass_uf_neg, nu_divv});
  terms.push_back({mass_theta, nu_div_grad});
}

//! adds the LB collision operator to the term set
template<typename P>
inline void add_lenard_bernstein_collisions_1x2v(P const nu, term_set<P> &terms)
{
  std::function<P(P const, P const)> const_nu = [nnu = nu](P const, P const = 0)->P{ return nnu; };
  std::function<P(P const, P const)> get_nuv = [nnu = nu](P const v, P const = 0)->P{ return nnu * v; };

  bool constexpr time_depend = true;

  imex_flag constexpr imex = imex_flag::imex_implicit;

  term<P> I("I", partial_term<P>(coefficient_type::mass), imex);

  partial_term<P> pt_nu_div_vv(
      coefficient_type::div, get_nuv, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  term<P> nu_div_vv("LB_nu_div_vv", pt_nu_div_vv, imex);


  partial_term<P> pt_nu_div_v(
      coefficient_type::div, const_nu, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  term<P> mass_u1(time_depend, "LB_u1", partial_term<P>(mass_moment_over_density_neg{1}), imex);
  term<P> mass_u2(time_depend, "LB_u2", partial_term<P>(mass_moment_over_density_neg{2}), imex);

  term<P> nu_div_v("LB_nu_div_v", pt_nu_div_v, imex);


  partial_term<P> pt_div_up(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  partial_term<P> pt_nu_grad_down(
      coefficient_type::grad, const_nu, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  term<P> nu_div_grad("LB_nu_div_grad", {pt_div_up, pt_nu_grad_down}, imex);

  term<P> mass_theta(time_depend, "LB_mass_theta",
                     partial_term<P>(pterm_dependence::lenard_bernstein_diff_theta_1x2v), imex);

  terms.push_back({I, nu_div_vv, I});
  terms.push_back({I, I, nu_div_vv});

  terms.push_back({mass_u1, nu_div_v, I});
  terms.push_back({mass_u2, I, nu_div_v});

  terms.push_back({mass_theta, nu_div_grad, I});
  terms.push_back({mass_theta, I, nu_div_grad});
}

//! adds the LB collision operator to the term set
template<typename P>
inline void add_lenard_bernstein_collisions_1x3v(P const nu, term_set<P> &terms)
{
  std::function<P(P const, P const)> const_nu = [nnu = nu](P const, P const = 0)->P{ return nnu; };
  std::function<P(P const, P const)> get_nuv = [nnu = nu](P const v, P const = 0)->P{ return nnu * v; };

  bool constexpr time_depend = true;

  imex_flag constexpr imex = imex_flag::imex_implicit;

  term<P> I("I", partial_term<P>(coefficient_type::mass), imex);

  partial_term<P> pt_nu_div_vv(
      coefficient_type::div, get_nuv, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  term<P> nu_div_vv("LB_nu_div_vv", pt_nu_div_vv, imex);


  partial_term<P> pt_nu_div_v(
      coefficient_type::div, const_nu, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  term<P> mass_u1(time_depend, "LB_u1", partial_term<P>(mass_moment_over_density_neg{1}), imex);
  term<P> mass_u2(time_depend, "LB_u2", partial_term<P>(mass_moment_over_density_neg{2}), imex);
  term<P> mass_u3(time_depend, "LB_u3", partial_term<P>(mass_moment_over_density_neg{3}), imex);

  term<P> nu_div_v("LB_nu_div_v", pt_nu_div_v, imex);


  partial_term<P> pt_div_up(
      coefficient_type::div, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  partial_term<P> pt_nu_grad_down(
      coefficient_type::grad, const_nu, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  term<P> nu_div_grad("LB_nu_div_grad", {pt_div_up, pt_nu_grad_down}, imex);

  term<P> mass_theta(time_depend, "LB_mass_theta",
                     partial_term<P>(pterm_dependence::lenard_bernstein_diff_theta_1x3v), imex);

  terms.push_back({I, nu_div_vv, I, I});
  terms.push_back({I, I, nu_div_vv, I});
  terms.push_back({I, I, I, nu_div_vv});

  terms.push_back({mass_u1, nu_div_v, I, I});
  terms.push_back({mass_u2, I, nu_div_v, I});
  terms.push_back({mass_u3, I, I, nu_div_v});

  terms.push_back({mass_theta, nu_div_grad, I, I});
  terms.push_back({mass_theta, I, nu_div_grad, I});
  terms.push_back({mass_theta, I, I, nu_div_grad});
}

} // namespace asgard
