#pragma once

#include "../asgard_dimension.hpp"
#include "../asgard_indexset.hpp"
#include "../asgard_quadrature.hpp"

// the quadrature is needed by some of the pdes to perform internal operations

namespace asgard
{
#ifndef __ASGARD_DOXYGEN_SKIP

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
  free
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
  electric_field_only, // depends only on the electric filed and not position
  electric_field_infnrm, // depends on the max abs( electric_field )
  moment_divided_by_density, // moment divided by moment 0
  lenard_bernstein_coll_theta_1x1v,
  lenard_bernstein_coll_theta_1x2v,
  lenard_bernstein_coll_theta_1x3v,
};

//! allows grouping terms together
enum class term_groups
{
  //! consider all terms
  all,
  //! imex excplicit terms
  imex_explicit,
  //! imex imex_implicit terms
  imex_implicit,
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
  // lax_friedrich = 0
};

enum class imex_flag
{
  unspecified = 0,
  imex_explicit = 1,
  imex_implicit = 2,
};
int constexpr num_imex_variants = 3;

/*!
 * \brief Indicates wither we need to recompute matrices based different conditions
 *
 * If a term has a fix coefficient, then there will not reason to update the matrices.
 * On ther other end of the spectrum, terms that depend on the PDE solution, e.g.,
 * the moments or the electric field, have to be recomputed on every stage of
 * a time advance algorithm.
 * The penalty terms have to be updated when the mesh discretization level changes.
 */
enum class changes_with
{
  //! no need to update the operator matrices
  none,
  //! assume we must always update on chnge in the time or the solution field
  time
};

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

//! type-tag indicating a mass partial term
struct type_tag_identity_term {};
constexpr type_tag_identity_term pt_identity{};

struct type_tag_mass_term {};
constexpr type_tag_mass_term pt_mass{};

struct type_tag_div_periodic {};
constexpr type_tag_div_periodic pt_div_periodic{};

struct type_tag_div_free {};
constexpr type_tag_div_free pt_div_free{};

struct type_tag_div_dirichlet_zero {};
constexpr type_tag_div_dirichlet_zero pt_div_dirichlet_zero{};

struct type_tag_grad_periodic {};
constexpr type_tag_grad_periodic pt_grad_periodic{};

struct type_tag_grad_free {};
constexpr type_tag_grad_free pt_grad_free{};

struct type_tag_grad_dirichlet_zero {};
constexpr type_tag_grad_dirichlet_zero pt_grad_dirichlet_zero{};

struct type_tag_penalty {};
constexpr type_tag_penalty pt_penalty{};

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

  //! equivalent to creating an identity term
  partial_term() = default;

  //! creates an identity partial term
  partial_term(type_tag_identity_term const &) : coeff_type_(coefficient_type::mass) {}

  partial_term(type_tag_mass_term const &,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               g_func_type<P> const dv_func_in       = nullptr)

      : coeff_type_(coefficient_type::mass), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), dv_func_(dv_func_in)
  {}

  partial_term(type_tag_div_periodic const &,
               flux_type const flux_in,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               g_func_type<P> const dv_func_in       = nullptr)

      : coeff_type_(coefficient_type::div), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(flux_in),
        left_(boundary_condition::periodic), right_(boundary_condition::periodic),
        ileft_(boundary_condition::periodic), iright_(boundary_condition::periodic),
        dv_func_(dv_func_in)
  {}

  partial_term(type_tag_div_free const &,
               flux_type const flux_in,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               g_func_type<P> const dv_func_in       = nullptr)

      : coeff_type_(coefficient_type::div), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(flux_in),
        left_(boundary_condition::free), right_(boundary_condition::free),
        ileft_(boundary_condition::free), iright_(boundary_condition::free),
        dv_func_(dv_func_in)
  {}

  partial_term(type_tag_div_dirichlet_zero const &,
               flux_type const flux_in,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               g_func_type<P> const dv_func_in       = nullptr)

      : coeff_type_(coefficient_type::div), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(flux_in),
        left_(boundary_condition::dirichlet), right_(boundary_condition::dirichlet),
        ileft_(boundary_condition::dirichlet), iright_(boundary_condition::dirichlet),
        dv_func_(dv_func_in)
  {}

  partial_term(type_tag_grad_periodic const &,
               flux_type const flux_in,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               g_func_type<P> const dv_func_in       = nullptr)

      : coeff_type_(coefficient_type::grad), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(grad_flux(flux_in)),
        left_(boundary_condition::periodic), right_(boundary_condition::periodic),
        ileft_(boundary_condition::periodic), iright_(boundary_condition::periodic),
        dv_func_(dv_func_in)
  {}

  partial_term(type_tag_grad_free const &,
               flux_type const flux_in,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               g_func_type<P> const dv_func_in       = nullptr)

      : coeff_type_(coefficient_type::grad), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(grad_flux(flux_in)),
        left_(boundary_condition::free), right_(boundary_condition::free),
        ileft_(set_bilinear_boundary(boundary_condition::free)),
        iright_(set_bilinear_boundary(boundary_condition::free)),
        dv_func_(dv_func_in)
  {}

  partial_term(type_tag_grad_dirichlet_zero const &,
               flux_type const flux_in,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               g_func_type<P> const dv_func_in       = nullptr)

      : coeff_type_(coefficient_type::grad), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(grad_flux(flux_in)),
        left_(boundary_condition::dirichlet), right_(boundary_condition::dirichlet),
        ileft_(boundary_condition::free), iright_(boundary_condition::free),
        dv_func_(dv_func_in)
  {}

  partial_term(coefficient_type const coeff_type_in,
               g_func_type<P> const g_func_in        = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               flux_type const flux_in               = flux_type::central,
               boundary_condition const left_in  = boundary_condition::free,
               boundary_condition const right_in = boundary_condition::free,
               std::vector<vector_func<P>> const left_bc_funcs_in  = {},
               scalar_func<P> const left_bc_time_func_in           = nullptr,
               std::vector<vector_func<P>> const right_bc_funcs_in = {},
               scalar_func<P> const right_bc_time_func_in          = nullptr,
               g_func_type<P> const dv_func_in                     = nullptr)

      : coeff_type_(coeff_type_in), g_func_(g_func_in),
        lhs_mass_func_(lhs_mass_func_in), flux_(set_flux(flux_in)), left_(left_in),
        right_(right_in), ileft_(set_bilinear_boundary(left_in)),
        iright_(set_bilinear_boundary(right_in)), left_bc_funcs_(left_bc_funcs_in),
        right_bc_funcs_(right_bc_funcs_in), left_bc_time_func_(left_bc_time_func_in),
        right_bc_time_func_(right_bc_time_func_in), dv_func_(dv_func_in)
  {}

  partial_term(pterm_dependence const depends_in,
               g_func_f_type<P> const g_func_f_in    = nullptr,
               g_func_type<P> const lhs_mass_func_in = nullptr,
               g_func_type<P> const dv_func_in       = nullptr)

      : coeff_type_(coefficient_type::mass), depends_(depends_in),
        g_func_f_(g_func_f_in), lhs_mass_func_(lhs_mass_func_in),
        dv_func_(dv_func_in)
  {
    expect(depends_ != pterm_dependence::none);
    expect(depends_ != pterm_dependence::moment_divided_by_density);
    // if this depends on the electric-filed, there should be a g_func_f
    expect(not (depends_ == pterm_dependence::electric_field and !g_func_f_));
    expect(not (depends_ == pterm_dependence::electric_field_infnrm and !g_func_f_));
    // if this depends on Lenard Bernstein collision, cannot have g funcs
    if (depends_ == pterm_dependence::lenard_bernstein_coll_theta_1x1v
        or depends_ == pterm_dependence::lenard_bernstein_coll_theta_1x2v
        or depends_ == pterm_dependence::lenard_bernstein_coll_theta_1x3v) {
      expect(!g_func_ and !g_func_f_);
    }
  }

  //! indicates mass term with coefficient mom_in.moment / moment0
  partial_term(mass_moment_over_density mom_in, g_func_type<P> const dv_func_in = nullptr)
      : depends_(pterm_dependence::moment_divided_by_density),
        mom(mom_in.moment), dv_func_(dv_func_in)
  {}
  //! indicates mass term with coefficient - mom_in.moment / moment0
  partial_term(mass_moment_over_density_neg mom_in, g_func_type<P> const dv_func_in = nullptr)
      : depends_(pterm_dependence::moment_divided_by_density), mom(-mom_in.moment),
        dv_func_(dv_func_in)
  {}

  P get_flux_scale() const { return static_cast<P>(flux_); };

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

  bool left_bc_zero() const { return left_bc_funcs_.empty(); }
  bool right_bc_zero() const { return right_bc_funcs_.empty(); };

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
      return (bc == boundary_condition::dirichlet) ? boundary_condition::free
                                                   : boundary_condition::dirichlet;
    }
    return bc;
  }
  flux_type grad_flux(flux_type f) {
    switch(f) {
      case flux_type::upwind: return flux_type::downwind;
      case flux_type::downwind: return flux_type::upwind;
      default:
        return flux_type::central;
    }
  }

  coefficient_type coeff_type_ = coefficient_type::mass;

  pterm_dependence depends_ = pterm_dependence::none;

  g_func_type<P> g_func_;
  g_func_f_type<P> g_func_f_;
  g_func_type<P> lhs_mass_func_;

  int mom = 0; // paired with mom-by-density, cannot be zero, sign used too

  flux_type flux_ = flux_type::central;

  boundary_condition left_  = boundary_condition::free;
  boundary_condition right_ = boundary_condition::free;

  boundary_condition ileft_  = boundary_condition::free;
  boundary_condition iright_ = boundary_condition::free;

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

  int required_moment_indexes() const
  {
    int mom_idx = 0;
    for (auto const &pt : partial_terms_)
    {
      switch (pt.depends())
      {
        case pterm_dependence::electric_field:
        case pterm_dependence::electric_field_infnrm:
          mom_idx = std::max(mom_idx, 1);
          break;
        case pterm_dependence::moment_divided_by_density:
          mom_idx = std::max(mom_idx, std::abs(pt.mom_index()));
          break;
        case pterm_dependence::lenard_bernstein_coll_theta_1x1v:
          mom_idx = std::max(mom_idx, 3);
          break;
        case pterm_dependence::lenard_bernstein_coll_theta_1x2v:
          mom_idx = std::max(mom_idx, 5);
          break;
        case pterm_dependence::lenard_bernstein_coll_theta_1x3v:
          mom_idx = std::max(mom_idx, 7);
          break;
        default: // does not require any moments
          break;
      };
    }
    return mom_idx;
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
    int mom_indexes = 0;
    for (auto &term_list : terms_)
    {
      expect(term_list.size() == static_cast<unsigned>(num_dims_));
      for (auto &term_1D : term_list)
      {
        expect(term_1D.get_partial_terms().size() > 0);

        auto const max_dof = fm::ipow2(static_cast<int64_t>(max_level_)) * pdof;
        expect(max_dof < INT_MAX);

        mom_indexes = std::max(mom_indexes, term_1D.required_moment_indexes());
      }
    }

    if (mom_indexes > 0) {
      num_required_moments_ = 1 + (mom_indexes - 1) / (num_dims_ - 1);
      expect(mom_indexes == 1 + (num_dims_ - 1) * (num_required_moments_ - 1));
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

    if (not options_.step_method)
    {
      // no step method requested, select a default method
      if (num_required_moments_ > 0) {
        // messing with moments, collision and/or poisson solver, default to imex
        options_.step_method = time_method::imex;
      } else {
        // no moments needed for this PDE, use explicit integration
        options_.step_method = time_method::exp;
      }
    }

    use_imex_     = options_.step_method.value() == time_method::imex;
    use_implicit_ = options_.step_method.value() == time_method::imp;

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
        options_.solver = solver_method::bicgstab;

    // missing tolerance will be set within the solver module
    if (not options_.isolver_tolerance)
      options_.isolver_tolerance = solvers::notolerance;
    if (not options_.isolver_iterations)
      options_.isolver_iterations = solvers::novalue;
    if (not options_.isolver_inner_iterations)
      options_.isolver_inner_iterations = solvers::novalue;
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

  bool do_poisson_solve() const { // TODO: rename to poisson dependence
    for (auto const &terms_md : terms_)
      for (auto const &term1d : terms_md)
        if (term1d.has_dependence(pterm_dependence::electric_field)
            or term1d.has_dependence(pterm_dependence::electric_field_infnrm))
          return true;

    // no terms have the poisson dependence
    return false;
  }
  bool do_collision_operator() const { return do_collision_operator_; }
  bool has_analytic_soln() const { return has_analytic_soln_; }

  int required_moments() const { return num_required_moments_; }

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

  int num_required_moments_ = 0;

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

  partial_term<P> ptDivU{pt_div_periodic, flux_type::upwind, PDE<P>::gfunc_neg1};

  partial_term<P> ptMassP{pt_mass, PDE<P>::gfunc_positive};

  term<P> div_x_up("div_x_up", ptDivU, imex);

  term<P> massP("mass_positive", ptMassP, imex);

  partial_term<P> ptDivD{pt_div_periodic, flux_type::downwind, PDE<P>::gfunc_neg1};

  partial_term<P> ptMassN{pt_mass, PDE<P>::gfunc_negative};

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

  partial_term<P> pt_divv{pt_div_dirichlet_zero, flux_type::upwind, get_nuv};

  partial_term<P> pt_nu_divv{pt_div_dirichlet_zero, flux_type::central, const_nu};

  partial_term<P> pt_div_up{pt_div_dirichlet_zero, flux_type::upwind};

  partial_term<P> pt_nu_grad_down{pt_grad_dirichlet_zero, flux_type::downwind, const_nu};

  term<P> I("LB_I", pt_identity, imex);

  term<P> divv("LB_divv", pt_divv, imex);

  term<P> mass_uf_neg(time_depend, "LB_uf_neg", {mass_moment_over_density_neg{1}}, imex);

  term<P> nu_divv("LB_vdiv", pt_nu_divv, imex);

  term<P> mass_theta(time_depend, "LB_mass_theta",
                     {pterm_dependence::lenard_bernstein_coll_theta_1x1v}, imex);

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

  term<P> I("LB_I", pt_identity, imex);

  partial_term<P> pt_nu_div_vv{pt_div_dirichlet_zero, flux_type::upwind, get_nuv};

  term<P> nu_div_vv("LB_nu_div_vv", pt_nu_div_vv, imex);

  partial_term<P> pt_nu_div_v{pt_div_dirichlet_zero, flux_type::central, const_nu};

  term<P> mass_u1(time_depend, "LB_u1", {mass_moment_over_density_neg{1}}, imex);
  term<P> mass_u2(time_depend, "LB_u2", {mass_moment_over_density_neg{2}}, imex);

  term<P> nu_div_v("LB_nu_div_v", pt_nu_div_v, imex);

  partial_term<P> pt_div_up{pt_div_dirichlet_zero, flux_type::upwind};

  partial_term<P> pt_nu_grad_down{pt_grad_dirichlet_zero, flux_type::downwind, const_nu};

  term<P> nu_div_grad("LB_nu_div_grad", {pt_div_up, pt_nu_grad_down}, imex);

  term<P> mass_theta(time_depend, "LB_mass_theta",
                     {pterm_dependence::lenard_bernstein_coll_theta_1x2v}, imex);

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

  term<P> I("LB_I", pt_identity, imex);

  partial_term<P> pt_nu_div_vv{pt_div_dirichlet_zero, flux_type::upwind, get_nuv};

  term<P> nu_div_vv("LB_nu_div_vv", pt_nu_div_vv, imex);

  partial_term<P> pt_nu_div_v{pt_div_dirichlet_zero, flux_type::central, const_nu};

  term<P> mass_u1(time_depend, "LB_u1", {mass_moment_over_density_neg{1}}, imex);
  term<P> mass_u2(time_depend, "LB_u2", {mass_moment_over_density_neg{2}}, imex);
  term<P> mass_u3(time_depend, "LB_u3", {mass_moment_over_density_neg{3}}, imex);

  term<P> nu_div_v("LB_nu_div_v", pt_nu_div_v, imex);

  partial_term<P> pt_div_up{pt_div_dirichlet_zero, flux_type::upwind};

  partial_term<P> pt_nu_grad_down{pt_grad_dirichlet_zero, flux_type::downwind, const_nu};

  term<P> nu_div_grad("LB_nu_div_grad", {pt_div_up, pt_nu_grad_down}, imex);

  term<P> mass_theta(time_depend, "LB_mass_theta",
                     {pterm_dependence::lenard_bernstein_coll_theta_1x3v}, imex);

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

/*!
 * \ingroup asgard_pde_definition
 * \brief Signature for a non-separable function
 */
template<typename P>
using md_func = std::function<void(P t, vector2d<P> const &, std::vector<P> &)>;
/*!
 * \ingroup asgard_pde_definition
 * \brief Signature for a non-separable function that accepts an additional field parameter
 */
template<typename P>
using md_func_f = std::function<void(P t, vector2d<P> const &,
                                     std::vector<P> const &, std::vector<P> &)>;

#endif // doxygen skip

/*!
 * \defgroup asgard_pde_definition ASGarD PDE Definition
 *
 * Tools for defining a PDE description and discretization scheme.
 */

/*!
 * \ingroup asgard_pde_definition
 * \brief Defines the boundary conditions for separable operator
 *
 * The separable operators are always defined on a 1d interval. Periodic conditions
 * "connect" the flux on the left-most and right-most cells, so that information
 * moving out through the boundary is added to the other side. Dirichlet boundary
 * imposes zero flux through the matrix entries and non-homogeneous conditions
 * have to appear as a source. Free boundary conditions allow the flux to be
 * determined by the dynamics of the PDE and the current field.
 */
enum class boundary_type
{
  //! periodic boundary conditions
  periodic,
  //! fixed flux at the boundary
  dirichlet,
  //! free boundary condition
  free,
  //! free boundary on the left, Dirichlet on the right
  left_free,
  //! free boundary on the right, Dirichlet on the left
  right_free,
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Defines the type of one-dimensional operation
 */
enum class operation_type
{
  //! identity term
  identity,
  //! mass term
  mass,
  //! grad term, derivative on the basis function
  grad,
  //! div term, derivative on the test function
  div,
  //! penalty term, regularizer used for stability purposes
  penalty,
  //! chain term, product of two or more one dimensional terms
  chain
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Helper allowing to specify left boundary conditions
 */
template<typename P>
struct left_boundary_cond {
  //! constant left boundary
  explicit left_boundary_cond(P cnt) : c(cnt) {}
  //! time-dependent left boundary boundary
  explicit left_boundary_cond(scalar_func<P> t) : time_(std::move(t)) {}
  //! the constant
  P c = 0;
  //! the time function
  scalar_func<P> time_;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Helper allowing to specify right boundary conditions
 */
template<typename P>
struct right_boundary_cond {
  //! constant left boundary
  explicit right_boundary_cond(P cnt) : c(cnt) {}
  //! time-dependent left boundary boundary
  explicit right_boundary_cond(scalar_func<P> t) : time_(std::move(t)) {}
  //! the constant
  P c = 0;
  //! the time function
  scalar_func<P> time_;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Defines the separable boundary conditions for div and grad
 */
template<typename P = default_precision>
struct dirichelt_boundary1d {
  //! default case, identical to homogeneous boundary
  dirichelt_boundary1d() = default;
  //! constant in time
  dirichelt_boundary1d(P left, P right)
      : const_left(left), const_right(right)
  {}
  //! left constant, right variable
  dirichelt_boundary1d(P left, scalar_func<P> right)
      : const_left(left), right_t(std::move(right))
  {}
  //! left variable, right constant
  dirichelt_boundary1d(scalar_func<P> left, P right)
      : const_right(right), left_t(std::move(left))
  {}
  //! left variable, right constant
  dirichelt_boundary1d(scalar_func<P> left, scalar_func<P> right)
      : left_t(std::move(left)), right_t(std::move(right))
  {}
  //! move constant boundary term with different precision
  template<typename otherP>
  dirichelt_boundary1d(dirichelt_boundary1d<otherP> other)
      : const_left(static_cast<P>(other.const_left)),
        const_right(static_cast<P>(other.const_right))
  {
    rassert(not other.left_t and not other.right_t,
            "type mismatch using dirichelt_boundary1d, "
            "see the type-safety documentation of term_1d");
  }
  //! left boundary
  dirichelt_boundary1d(left_boundary_cond<P> left_bc)
    : const_left(left_bc.c), left_t(std::move(left_bc.time_))
  {}
  //! left boundary, different precision
  template<typename otherP>
  dirichelt_boundary1d(left_boundary_cond<otherP> left_bc)
    : const_left(left_bc.c)
  {
    rassert(not left_bc.time_,
            "type mismatch using left_boundary_cond and dirichelt_boundary1d "
            "see the type-safety documentation of term_1d");
  }
  //! right boundary
  dirichelt_boundary1d(right_boundary_cond<P> right_bc)
    : const_right(right_bc.c), right_t(std::move(right_bc.time_))
  {}
  //! left boundary, different precision
  template<typename otherP>
  dirichelt_boundary1d(right_boundary_cond<otherP> right_bc)
    : const_right(right_bc.c)
  {
    rassert(not right_bc.time_,
            "type mismatch using right_boundary_cond and dirichelt_boundary1d "
            "see the type-safety documentation of term_1d");
  }

  //! constant left boundary condition
  P const_left = 0;
  //! constant right boundary condition
  P const_right = 0;
  //! time-dependent left-boundary condition
  scalar_func<P> left_t;
  //! time-dependent right-boundary condition
  scalar_func<P> right_t;
  //! (non-const) separable boundary conditions on the left
  std::vector<separable_func<P>> sep_left;
  //! (non-const) separable boundary conditions on the right
  std::vector<separable_func<P>> sep_right;

  //! add left separable function
  void add_left(separable_func<P> f) { sep_left.emplace_back(std::move(f)); }
  //! add right separable function
  void add_right(separable_func<P> f) { sep_right.emplace_back(std::move(f)); }
  //! get the number of left separable functions
  int num_left() const { return static_cast<int>(sep_left.size()); }
  //! get the number of left separable functions
  int num_right() const { return static_cast<int>(sep_right.size()); }
  //! get the number of all separable functions
  int num_sep() const { return num_left() + num_right(); }

  //! returns true if there is left boundary condition
  bool has_left() const { return (const_left != 0 or left_t); }
  //! returns true if there is right boundary condition
  bool has_right() const { return (const_right != 0 or right_t); }
  //! return true if either left or right boundary has been set
  bool has_any() const { return has_left() or has_right(); }
  //! converts to true if either left or right boundary has been set
  operator bool() const { return has_any(); }
  //! throws if the boundary type is incompatible, i.e., inhomogeneous Dirichlet set for free boundary
  void throw_if_invalid(boundary_type bnd) {
    switch (bnd) {
      case boundary_type::periodic:
      case boundary_type::free:
        rassert(num_sep() == 0, "cannot specify dirichelt_boundary1d with boundary_type periodic or free");
        rassert(not has_any(), "cannot specify dirichelt_boundary1d with boundary_type periodic or free");
        break;
      case boundary_type::left_free:
        rassert(num_left() == 0, "cannot specify left separable dirichelt_boundary1d with boundary_type::left_free");
        rassert(not has_left(), "cannot specify left dirichelt_boundary1d with boundary_type::left_free");
        break;
      case boundary_type::right_free:
        rassert(num_right() == 0, "cannot specify right separable dirichelt_boundary1d with boundary_type::right_free");
        rassert(not has_right(), "cannot specify right dirichelt_boundary1d with boundary_type::right_free");
        break;
      default:
        break;
    };
  }
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for an identity mass term
 */
struct term_identity {};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for a mass term
 */
template<typename P = default_precision>
struct term_mass {
  //! make a mass term with constant coefficient
  term_mass(no_deduce<P> cc) : const_coeff(cc) {}
  //! make a mass term with given right hand side coefficient
  term_mass(sfixed_func1d<P> rhs) : right(std::move(rhs)) {}

  //! constant coefficient, if left/right-hand-side functions are null
  P const_coeff = 0;
  //! right-hand-side function
  sfixed_func1d<P> right;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for a grad term, includes flux and boundary conditions
 */
template<typename P = default_precision>
struct term_grad {
  //! make a grad term with constant coefficient
  term_grad(no_deduce<P> cc, flux_type flx, boundary_type bnd,
            dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
    : const_coeff(cc), flux(flx), boundary(bnd), dirichlet(std::move(dir))
  {
    dirichlet.throw_if_invalid(boundary);
  }
  //! make a grad term with constant coefficient 1
  term_grad(flux_type flx, boundary_type bnd, dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
    : flux(flx), boundary(bnd), dirichlet(std::move(dir))
  {
    dirichlet.throw_if_invalid(boundary);
  }
  //! make a grad term with given right hand side coefficient
  term_grad(sfixed_func1d<P> frhs, flux_type flx, boundary_type bnd,
            dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
    : const_coeff(0), right(std::move(frhs)), flux(flx), boundary(bnd), dirichlet(std::move(dir))
  {
    dirichlet.throw_if_invalid(boundary);
  }

  //! constant coefficient, if left/right-hand-side functions are null
  P const_coeff = 1;
  //! right-hand-side function
  sfixed_func1d<P> right;

  //! flux type
  flux_type flux;
  //! boundary type
  boundary_type boundary;
  //! non-zero Dirichlet boundary
  dirichelt_boundary1d<P> dirichlet;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for a div term, includes flux and boundary conditions
 */
template<typename P = default_precision>
struct term_div {
  //! make a grad term with constant coefficient
  term_div(no_deduce<P> cc, flux_type flx, boundary_type bnd,
           dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
    : const_coeff(cc), flux(flx), boundary(bnd), dirichlet(std::move(dir))
  {
    dirichlet.throw_if_invalid(boundary);
  }
  //! make a grad term with constant coefficient 1
  term_div(flux_type flx, boundary_type bnd, dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
    : const_coeff(1), flux(flx), boundary(bnd), dirichlet(std::move(dir))
  {
    dirichlet.throw_if_invalid(boundary);
  }
  //! make a grad term with given right hand side coefficient
  term_div(sfixed_func1d<P> frhs, flux_type flx, boundary_type bnd,
           dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
    : right(std::move(frhs)), flux(flx), boundary(bnd), dirichlet(std::move(dir))
  {
    dirichlet.throw_if_invalid(boundary);
  }

  //! constant coefficient, if left/right-hand-side functions are null
  P const_coeff = 0;
  //! right-hand-side function
  sfixed_func1d<P> right;

  //! flux type
  flux_type flux;
  //! boundary type
  boundary_type boundary;
  //! non-zero Dirichlet boundary
  dirichelt_boundary1d<P> dirichlet;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for a div term, includes flux and boundary conditions
 */
template<typename P = default_precision>
struct term_penalty {
  //! make a grad term with constant coefficient
  term_penalty(no_deduce<P> cc, flux_type flx, boundary_type bnd,
               dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
    : const_coeff(cc), flux(flx), boundary(bnd), dirichlet(std::move(dir))
  {
    dirichlet.throw_if_invalid(boundary);
  }
  //! make a grad term with constant coefficient 1
  term_penalty(flux_type flx, boundary_type bnd, dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
    : const_coeff(1), flux(flx), boundary(bnd), dirichlet(std::move(dir))
  {
    dirichlet.throw_if_invalid(boundary);
  }

  //! constant coefficient, if left/right-hand-side functions are null
  P const_coeff = 0;

  //! flux type
  flux_type flux;
  //! boundary type
  boundary_type boundary;
  //! non-zero Dirichlet boundary
  dirichelt_boundary1d<P> dirichlet;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for chain of one-dimensional terms
 *
 * Example usage:
 * \code
 *   // declare a chain term
 *   term_1d t1d(term_chain{});
 *
 *   // add the 1d terms later
 *   t1d += term_div{-2, flux_type::upwind, boundary_type::free};
 *   t1d += term_grad{2, flux_type::upwind, boundary_type::dirichlet};
 * \endcode
 */
struct term_chain {};

/*!
 * \ingroup asgard_pde_definition
 * \brief Mass term that depends on the electric field
 */
template<typename P = default_precision>
struct mass_electric {
  //! mass based only on the electric field, same as rhs being the identity function y = x
  mass_electric() {}
  //! right side depends only on the field
  mass_electric(sfixed_func1d<P> rhs) : right(std::move(rhs)) {}
  //! right side depends on the field and position
  mass_electric(sfixed_func1d_f<P> rhs_f) : right_f(std::move(rhs_f)) {}

  //! right-hand-side function, field only no spatial dependence
  sfixed_func1d<P> right;
  //! right-hand-side function, depends on position and field
  sfixed_func1d_f<P> right_f;
};

// forward declaration so it can be set as a friend
template<typename P>
struct term_manager;

/*!
 * \ingroup asgard_pde_definition
 * \brief One dimensional term, building block of separable operators
 *
 * \par Main usage
 * This class has two main modes of operation, first is as a single term representing
 * mass, div, grad, or penalty operation. The simple operations are best created
 * using the helper structs term_identity, term_mass, term_div and term_grad.
 *
 * \par
 * The second mode is to represent a chain of simple terms.
 * The operators in the chain will be multiplied together using small-matrix
 * logic in a local cell-by-cell algorithm.
 * Chains of partial terms are computationally more efficient than chains
 * of multidimensional terms, but have more restrictions on the types
 * of terms that can be chained:
 * - div or grad partial term with central flux can only chain with a mass term
 * - div or grad partial term with upwind/downwind flux can only chain with
 *   grad or div with opposing downwind/upwind flux
 * - a penalty term is equivalent to div/grad with central flux
 *
 * \par
 * Chain-of-chains is not allowed as it is unnecessary.
 *
 * \par Type-safety
 * ASGarD classes are templated to use either double or float precision, which can
 * create inconveniences with the C++ type system.
 * Consider the following code that creates a 1d mass term with coefficient 3.
 * \code
 *   // type explicitly set to float
 *   term_mass<float> fmass{3};
 *   // type explicitly set to double
 *   term_mass<double> dmass{3};
 *   // if double is available, type is double, else use float
 *   term_mass<asgard::default_precision> long_name_mass{3};
 *   // same as above but with less typing
 *   term_mass amass{3};
 * \endcode
 * In all cases the coefficient is set from a constant with type int and converted
 * to either float or double. Now take one more step and add the term to a 1D pde.
 * \code
 *   // assuming both double and float are available
 *   // the domain is 1d and the options are not important here
 *   asgard::PDEv2<float> fpde(options, domain1d);
 *
 *   // add mass term, type matches but uses lots of typing
 *   fpde += term_mass<float>{3};
 *
 *   // different types, there will be a conversion but that's fine
 *   // since ASGarD defaults to the higher precision, there may be an additional
 *   // (cheap) conversion of a single constant but no loss of precision
 *   fpde += term_mass{3};
 * \endcode
 * While this works fine for constants, it cannot be done if the coefficient is replaced
 * by a function, since there is no automatic conversion between std::vector<double>
 * and std::vector<float> and automatically doing such conversion is meaninglessly
 * expensive.
 * \code
 *   sfixed_func1d<float> rhs =
 *      [](std::vector<float> const &x, std::vector<float> &y)-> void {...};
 *
 *   // here mass will have type term_mass<float>
 *   auto mass = term_mass{rhs};
 *
 *   // using PDE of matching type
 *   PDEv2<float> fpde(options, domain1d);
 *
 *   fpde += term_1d{mass}; // OK, no need to explicitly specify 'float'
 *
 *   // creating PDE with default double precision
 *   PDEv2 pde(options, domain1d);
 *
 *   // rhs will not be converted to using std::vector<double>
 *   // a wrapper can be written but this has to be done explicitly
 *   // pde += term_1d<float> mass; // will fail to compile
 *   pde += term_1d{mass}; // will compile but yield runtime_error
 * \endcode
 * When using constant coefficients, ASGarD will handle most type conversion automatically
 * and when using variable coefficients the type does not need to be explicitly carried
 * for each template, which is convenient.
 * However, as a trade-off, when using variable coefficients and incorrect types sometimes
 * the error will occur at runtime, as opposed to compile time, since using constant vs.
 * variable coefficient is not known until runtime.
 */
template<typename P = default_precision>
class term_1d
{
public:
  //! make an identity term
  term_1d() = default;
  //! make an identity term
  term_1d(term_identity) {}
  //! make a general term
  term_1d(operation_type opt, flux_type flx, boundary_type bnd,
          sfixed_func1d<P> frhs, P crhs, dirichelt_boundary1d<P> dir = dirichelt_boundary1d<P>{})
      : optype_(opt), flux_(flx), boundary_(bnd),
        rhs_(std::move(frhs)), rhs_const_(crhs), dirichlet_(std::move(dir))
  {
    expect(optype_ != operation_type::identity);

    if (optype_ != operation_type::div and
        optype_ != operation_type::grad and
        optype_ != operation_type::penalty)
      rassert(not dirichlet_.has_any(),
              "cannot set boundary conditions for term_1d with operation_type "
              "that is not div, grad or penalty");

    if (optype_ == operation_type::grad) {
      if (flux_ == flux_type::upwind)
        flux_ = flux_type::downwind;
      else if (flux_ == flux_type::downwind)
        flux_ = flux_type::upwind;
    }
  }
  //! make a term that depends on coupled fields, e.g., moments or electric field
  term_1d(pterm_dependence dep, sfixed_func1d_f<P> ffunc)
    : optype_(operation_type::mass), depends_(dep), field_f_(std::move(ffunc))
  {}

  //! make a mass term
  term_1d(term_mass<P> mt)
    : term_1d(operation_type::mass, flux_type::central, boundary_type::free,
              std::move(mt.right), mt.const_coeff)
  {}
  //! make a mass term, hack around creating term_1d<float> from term_mass<double>
  template<typename otherP>
  term_1d(term_mass<otherP> mt)
    : term_1d(operation_type::mass, flux_type::central, boundary_type::free,
              nullptr, static_cast<P>(mt.const_coeff))
  {
    rassert(not mt.right, "type mismatch using term_mass to create term_1d, "
                          "see the type-safety documentation of term_1d");
  }
  //! make a grad term
  term_1d(term_grad<P> grd)
    : term_1d(operation_type::grad, grd.flux, grd.boundary,
              std::move(grd.right), grd.const_coeff, std::move(grd.dirichlet))
  {}
  //! make a div term
  term_1d(term_div<P> divt)
    : term_1d(operation_type::div, divt.flux, divt.boundary,
              std::move(divt.right), divt.const_coeff, std::move(divt.dirichlet))
  {}
  //! make a penalty term
  term_1d(term_penalty<P> pent)
    : term_1d(operation_type::penalty, pent.flux, pent.boundary,
              nullptr, pent.const_coeff, std::move(pent.dirichlet))
  {}
  //! make a penalty term
  template<typename otherP>
  term_1d(term_penalty<otherP> pent)
    : term_1d(operation_type::penalty, pent.flux, pent.boundary,
              nullptr, static_cast<P>(pent.const_coeff), std::move(pent.dirichlet))
  {}
  //! make a chain term
  term_1d(std::vector<term_1d<P>> tvec)
    : optype_(operation_type::chain), chain_(std::move(tvec))
  {
    // remove the identity terms in the chain
    int numid = 0;
    for (auto const &c : chain_) {
      if (c.is_identity())
        numid += 1;
      if (c.is_chain())
        throw std::runtime_error("cannot create a chain-of-chains of term1d");
    }

    int const num_chain = this->num_chain();
    if (num_chain == numid) {
      // all identities, nothing to chain
      optype_ = operation_type::identity;
      chain_.resize(0);
    } else if (num_chain - numid == 1) {
      // chain has only one non-identity term
      for (auto &c : chain_) {
        if (not c.is_identity()) {
          term_1d<P> temp = std::move(c);
          *this = std::move(temp);
          break;
        }
      }
    } else if (numid > 0) {
      std::vector<term_1d<P>> vec = std::move(chain_);
      chain_ = std::vector<term_1d<P>>();
      chain_.reserve(vec.size() - numid);
      for (auto &c : vec) {
        if (not c.is_identity())
          chain_.emplace_back(std::move(c));
      }
    }

    if (not check_chain())
      throw std::runtime_error("incompatible flux combination used in a term_1d chain, "
                               "must split into a term_md chain");
  }
  //! make a chain term, add terms later with add_term() or +=
  term_1d(term_chain) : optype_(operation_type::chain) {}
  //! make a term that depends on the electric field
  term_1d(mass_electric<P> elmass)
    : optype_(operation_type::mass), change_(changes_with::time),
      rhs_(std::move(elmass.right)), field_f_(std::move(elmass.right_f))
  {
    depends_ = (field_f_) ? pterm_dependence::electric_field
                          : pterm_dependence::electric_field_only;
  }

  //! indicates whether this is an identity term
  bool is_identity() const { return (optype_ == operation_type::identity); }
  //! indicates whether this is a mass term
  bool is_mass() const { return (optype_ == operation_type::mass); }
  //! indicates whether this is a grad term
  bool is_grad() const { return (optype_ == operation_type::grad); }
  //! indicates whether this is a div term
  bool is_div() const { return (optype_ == operation_type::div); }
  //! indicates whether this is a penalty term
  bool is_penalty() const { return (optype_ == operation_type::penalty); }
  //! indicates whether this is a chain term
  bool is_chain() const { return (optype_ == operation_type::chain); }
  //! returns the operation type
  operation_type optype() const { return optype_; }
  //! returns the boundary type
  boundary_type boundary() const { return boundary_; }
  //! returns the flux type
  flux_type flux() const { return flux_; }

  //! returns the right-hand-side function
  sfixed_func1d<P> const &rhs() const { return rhs_; }
  //! calls the right-hand-side function
  void rhs(std::vector<P> const &x, std::vector<P> &fx) const {
    return rhs_(x, fx);
  }

  //! returns the required moment, if any
  int moment() const { return mom; }

  //! returns the rhs function that calls the field
  sfixed_func1d_f<P> const &field() const { return field_f_; }
  //! calls the rhs function that depends on the field
  void field(std::vector<P> const &x, std::vector<P> const &f, std::vector<P> &fx) const {
    return field_f_(x, f, fx);
  }

  //! returns the constant right-hand-side
  P rhs_const() const { return rhs_const_; }

  //! can read or set the the change option
  changes_with &change() { return change_; }
  //! can read the change option
  changes_with change() const { return change_; }

  //! returns the extra dependence
  pterm_dependence depends() const { return depends_; }

  //! (chain-mode only) number of chained terms
  int num_chain() const { return static_cast<int>(chain_.size()); }
  //! (chain-mode only) get the vector of the chain
  std::vector<term_1d<P>> const &chain() const { return chain_; }
  //! (chain-mode only) get the i-th term in the chain
  term_1d<P> const &operator[](int i) const { return chain_[i]; }
  //! (chain-mode only) add one more term to the chain
  void add_term(term_1d<P> tm) {
    chain_.emplace_back(std::move(tm));
    if (not check_chain())
      throw std::runtime_error("incompatible flux combination used in a term_1d chain, "
                               "must split into a term_md chain");
  }
  //! (chain-mode only) add one more term to the chain
  term_1d<P> & operator += (term_1d<P> tm) {
    this->add_term(std::move(tm));
    return *this;
  }
  //! returns true if the term has a flux
  bool has_flux() const {
    if (optype_ == operation_type::chain) {
      for (auto const &cc : chain_)
        if (cc.optype_ != operation_type::mass and cc.optype_ != operation_type::identity)
          return true;
      return false;
    } else {
      return (optype_ != operation_type::mass and optype_ != operation_type::identity);
    }
  }
  //! add penalty to a div or grad term, more efficient than adding additional term
  void set_penalty(P penalty_coefficient) {
    rassert(optype_ == operation_type::div or optype_ == operation_type::grad
            or optype_ == operation_type::chain,
            "penalty can be added only to div grad or chain terms, if added to a chain "
            "flux and boundary condition will be taken from the back of the chain");
    rassert(penalty_coefficient > 0, "penalty coefficient has to be positive");
    penalty_ = penalty_coefficient;
    if (is_chain()) {
      flux_      = chain_.back().flux();
      boundary_  = chain_.back().boundary();
      dirichlet_ = chain_.back().dirichlet(); // TODO: this should not copy or move
      // the problem above is that adding the boundary condition to a term
      // yields a source term, must combine the scale factors for to terms ...
    }
  }
  //! add penalty to a chain term, more efficient than adding additional term
  void set_penalty(P penalty_coefficient, flux_type flx, boundary_type bnd,
                   dirichelt_boundary1d<P> dir = {}) {
    rassert(optype_ == operation_type::chain,
            "penalty with specified flux can be added only to a chain term, adding "
            "penalty to div or grad terms matches the provided flux");
    rassert(penalty_coefficient > 0, "penalty coefficient has to be positive");
    penalty_   = penalty_coefficient;
    flux_      = flx;
    boundary_  = bnd;
    dirichlet_ = std::move(dir);
  }
  //! get the current penalty coefficient
  P penalty() const { return penalty_; }

  //! returns the boundary conditions
  dirichelt_boundary1d<P> const &dirichlet() const { return dirichlet_; }

  //! returns if has Dirichlet or any chain terms have Dirichlet bc
  bool has_dirichlet() const {
    if (dirichlet_.has_any())
      return true;
    if (is_chain()) {
      for (auto const &c : chain_) {
        if (c.dirichlet().has_any())
          return true;
      }
    }
    return false;
  }
  //! return number of separable Dirichlet sources
  int num_sep_dirichlet() const {
    int num_sep = dirichlet_.num_sep();
    for (auto const &c : chain_) // if not a chain, then chain_ is empty
      num_sep += c.dirichlet().num_sep();
    return num_sep;
  }

  // allow direct access to the private data
  friend struct term_manager<P>;

private:
  //! (chain-mode only) access the i-th term in the chain, allows mods
  term_1d<P> &chain(int i) { return chain_[i]; }

  bool check_chain() {
    return true;
    int fluxdir = 2; // no flux direction found, two available
    for (int i : iindexof(chain_)) {
      term_1d<P> const &pt = chain_[i];
      // get the int-value of the flux, flip for grad
      int const fdir = static_cast<int>(pt.flux())
                      * ((pt.optype() == operation_type::grad) ? -1 : 1);
      switch (pt.optype())
      {
        case operation_type::penalty:
          if (fluxdir == 2) // have two flux dirs available
            fluxdir = 0; // take both dirs
          else
            return false; // conflict, no flux-dirs left
          break;
        case operation_type::div:
        case operation_type::grad:
          // if the fdir direction is already taken, bad setup
          if (fluxdir == 0 or fluxdir == fdir)
            return false;
          else if (fdir == 0) { // requested central flux, takes two dirs
            if (fluxdir != 2) // one flux dir already used, bad
              return false;
            else
              fluxdir = 0; // take all flux dirs
          } else { // requested up/down flux, take one dir
            if (fluxdir != 2) // one flux already used
              fluxdir = 0; // ok, but no flux available anymore
            else
              fluxdir = fdir; // two available, take one flux direction
          }
        default: // mass term has no flux, nothing to do
          break;
      }
    }
    return true;
  }

  operation_type optype_ = operation_type::identity;
  pterm_dependence depends_ = pterm_dependence::none;

  flux_type flux_ = flux_type::central;
  boundary_type boundary_ = boundary_type::free;

  changes_with change_ = changes_with::none;

  sfixed_func1d<P> rhs_;
  P rhs_const_ = 1;
  P penalty_   = 0;

  int mom = 0;
  sfixed_func1d_f<P> field_f_;

  dirichelt_boundary1d<P> dirichlet_;

  std::vector<term_1d<P>> chain_;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Separable mass term, i.e., num-dims mass 1d terms
 *
 * A regular term_1d that is mass or a term_md with with mass terms can depend on time,
 * e.g., via moments. The mass-md term contains only mass terms that are time-independent
 */
template<typename P = default_precision>
class mass_md
{
public:
  //! constructs an empty term, nothing selected
  mass_md() = default;
  //! constructs an identity term
  mass_md(int dims) : num_dims_(dims) {}
  //! construct a term from the given list
  mass_md(std::initializer_list<term_1d<P>> list)
      : num_dims_(static_cast<int>(list.size()))
  {
    expect(num_dims_ <= max_num_dimensions);
    for (int d : iindexof(num_dims_)) {
      rassert((list.begin() + d)->is_mass() or (list.begin() + d)->is_identity(),
              "mass_md terms must be mass or identity");
      rassert((list.begin() + d)->depends() == pterm_dependence::none,
              "the mass_md terms cannot depend on moments or the electric field")
      terms_[d] = std::move(*((list.begin() + d)));
    }
  }
  //! construct a term from the given list
  mass_md(std::vector<term_1d<P>> list)
      : num_dims_(static_cast<int>(list.size()))
  {
    expect(num_dims_ <= max_num_dimensions);
    for (int d : iindexof(num_dims_)) {
      rassert(list[d].is_mass() or list[d].is_identity(),
              "mass_md terms must be mass or identity");
      terms_[d] = std::move(list[d]);
    }
  }
  //! returns the number of dimensions
  int num_dims() const { return num_dims_; }

  //! indicates whether the dimension and terms have been initialized
  operator bool () const { return (num_dims_ > 0); }

  //! returns true if all terms are identity
  bool is_identity() const {
    for (int d : iindexof(num_dims_))
      if (not terms_[d].is_identity())
        return false;
    return true;
  }
  //! access the d-th term
  term_1d<P> const &operator[] (int d) const { return terms_[d]; }
  //! access the d-th term
  term_1d<P> const &dim(int d) const { return terms_[d]; }

private:
  int num_dims_ = 0;

  std::array<term_1d<P>, max_num_dimensions> terms_;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Multidimensional term of the partial differential equation
 *
 * The term can be one of three modes:
 * - a separable term consisting of a number of one-dimensional chains, one per dimension
 * - an interpolation term, possibly non-linear and non-separable
 * - a chain of separable or interpolation terms
 *
 * Cannot create a separable term with all pterm_chain as identity.
 * A chain can be build only of separable and interpolation terms, recursive chains
 * are not allowed.
 */
template<typename P = default_precision>
class term_md
{
public:
  //! indicates the mode of the multi-dimensional term
  enum class mode { separable, interpolatory, chain };

  //! empty term, to be reinitialized later
  term_md() = default;

  //! 1d separable case
  template<typename otherP>
  term_md(term_1d<otherP> trm)
    : term_md({std::move(trm), })
  {}
  //! multi-dimensional separable case, using initializer list
  term_md(std::initializer_list<term_1d<P>> clist)
    : mode_(mode::separable), num_dims_(static_cast<int>(clist.size()))
  {
    int num_identity = 0;
    expect(num_dims_ <= max_num_dimensions);
    for (int i : iindexof(num_dims_)) {
      sep[i] = std::move(*(clist.begin() + i));
      if (sep[i].is_identity())
        num_identity++;
    }

    if (num_identity == num_dims_)
      throw std::runtime_error("cannot create term_md with all terms being identities");
  }
  //! multi-dimensional separable case, using std::vector
  term_md(std::vector<term_1d<P>> clist)
    : mode_(mode::separable), num_dims_(static_cast<int>(clist.size()))
  {
    int num_identity = 0;
    expect(num_dims_ <= max_num_dimensions);
    for (int i : iindexof(num_dims_)) {
      sep[i] = std::move(*(clist.begin() + i));
      if (sep[i].is_identity())
        num_identity++;
    }

    if (num_identity == num_dims_)
      throw std::runtime_error("cannot create term_md with all terms being identities");
  }
  //! list of multi-dimensional terms, from initializer list
  term_md(std::initializer_list<term_md<P>> clist)
    : mode_(mode::chain), chain_(std::move(clist))
  {
    // first pass, look for term with set dimensions and disallow recursive chaining
    for (auto const &ch : chain_)
    {
      switch (ch.term_mode())
      {
        case mode::chain:
          throw std::runtime_error("recursive chains (chain with chains) of term_md are not supported");
          break;
        case mode::separable:
          num_dims_ = ch.num_dims();
          break;
        default: // work on interpolation later
          break;
      }
    }

    for (auto const &ch : chain_)
    {
      if (ch.term_mode() == mode::separable and ch.num_dims() != num_dims_)
        throw std::runtime_error("inconsistent dimension of terms in the chain");
    }
  }
  //! list of multi-dimensional terms, from std::vector
  term_md(std::vector<term_md<P>> clist)
    : mode_(mode::chain), chain_(std::move(clist))
  {
    // first pass, look for term with set dimensions and disallow recursive chaining
    for (auto const &ch : chain_)
    {
      switch (ch.term_mode())
      {
        case mode::chain:
          throw std::runtime_error("recursive chains (chain with chains) of term_md are not supported");
          break;
        case mode::separable:
          num_dims_ = ch.num_dims();
          break;
        default: // work on interpolation later
          break;
      }
    }

    for (auto const &ch : chain_)
    {
      if (ch.term_mode() == mode::separable and ch.num_dims() != num_dims_)
        throw std::runtime_error("inconsistent dimension of terms in the chain");
    }
  }

  //! (separable mode only) get the 1d term with index i
  term_1d<P> &dim(int i) {
    expect(mode_ == mode::separable);
    return sep[i];
  }
  //! (separable mode only) get the 1d term with index i, const overload
  term_1d<P> const &dim(int i) const {
    expect(mode_ == mode::separable);
    return sep[i];
  }

  //! get the chain term with index i
  term_md<P> &chain(int i) {
    expect(mode_ == mode::chain);
    return chain_[i];
  }
  //! get the chain term with index i, const-overload
  term_md<P> const &chain(int i) const {
    expect(mode_ == mode::chain);
    return chain_[i];
  }

  //! returns true if the term has been set, i.e., dims is non-zero
  operator bool () const { return (num_dims_ > 0); }

  //! indicate which mode is being used
  mode term_mode() const { return mode_; }
  //! returns true if the terms is chain term
  bool is_chain() const { return (mode_ == mode::chain); }
  //! returns true if the terms is separable
  bool is_separable() const { return (mode_ == mode::separable); }
  //! return true if the term uses interpolation
  bool is_interpolatory() const { return (mode_ == mode::interpolatory); }

  //! sets the mass term
  void set_mass(mass_md<P> tmass) {
    rassert(is_separable(), "mass can only be set for a separable term");
    rassert(tmass.num_dims() == num_dims_, "the mass for term_md must have matching dimensions");
    mass_ = std::move(tmass);
  }
  //! returns the stored mass term
  mass_md<P> const &mass() const { return mass_; }

  //! separable case only, the number of dimensions
  int num_dims() const { return num_dims_; }
  //! (internal use) interpolation or chain mode only, set the number of dimensions
  void set_num_dimensions(int dims) {
    if (num_dims_ == dims)
      return;

    switch (mode_) {
      case mode::separable:
        throw std::runtime_error("wrong number of dimensions of separable term");
      case mode::interpolatory:
        num_dims_ = dims;
        break;
      default: // case mode::chain:
        num_dims_ = dims;
        for (auto &ch : chain_)
        {
          if (ch.term_mode() == mode::separable and ch.num_dims() != num_dims_)
            throw std::runtime_error("wrong number of dimensions of separable term in a chain");
          ch.set_num_dimensions(num_dims_);
        }
        break;
    }
  }
  //! chain case only, the number of chained terms
  int num_chain() const { return static_cast<int>(chain_.size()); }
  //! returns the dimension with flux, only one such is allowed, returns -1 if no flux is used
  int flux_dim() const {
    if (is_chain()) {
      int dir  = -1;
      size_t c = 0;
      while (dir == -1 and c < chain_.size())
        dir = chain_[c++].flux_dim();
      return dir;
    } else {
      for (int d : iindexof(num_dims)) {
        if (sep[d].has_flux())
          return d;
      }
      return -1;
    }
  }

  //! mode for the imex time-stepping
  imex_flag imex = imex_flag::unspecified;

  // allow direct access to the private data
  friend struct term_manager<P>;

private:
  // mode for the term
  mode mode_ = mode::interpolatory;
  // separable case
  int num_dims_ = 0;
  std::array<term_1d<P>, max_num_dimensions> sep;
  mass_md<P> mass_;
  // non-separable/interpolation case
  md_func_f<P> interp_;
  // chain of other terms
  std::vector<term_md<P>> chain_;
};

/*!
 * \ingroup asgard_discretization
 * \brief Holds initial time, final time, time-step, etc.
 *
 * When constructed, it takes 2 of 3 parameters, stop-time, time-step and number
 * of time steps. Then sets the correct third parameter.
 *
 * When remaining steps hits 0, current_time is equal to final_time,
 * give or take some machine precision.
 */
template<typename P = default_precision>
class time_data
{
public:
  //! type-tag for specifying  dt
  struct input_dt {
    //! explicit constructor, temporarily stores dt
    explicit input_dt(P v) : value(v) {}
    //! stored value
    P value;
  };
  //! type-tag for specifying stop-time
  struct input_stop_time {
    //! explicit constructor, temporarily stores the stop-time
    explicit input_stop_time(P v) : value(v) {}
    //! stored value
    P value;
  };
  //! unset time-data, all entries are negative, must be set later
  time_data() = default;
  //! steady state case, sets only the end time and num-steps to 1
  time_data(P endt)
      : smethod_(time_method::steady), stop_time_(endt), time_(0), step_(0), num_remain_(1)
  {}
  //! specify time-step and final time
  time_data(time_method smethod, input_dt dt, input_stop_time stop_time)
      : smethod_(smethod), dt_(dt.value), stop_time_(stop_time.value),
        time_(0), step_(0)
  {
    // assume we are doing at least 1 step and round down end_time / dt
    num_remain_ = static_cast<int64_t>(stop_time_ / dt_);
    if (num_remain_ == 0)
      num_remain_ = 1;
    // readjust dt to minimize rounding error
    if (dt_ * num_remain_ < stop_time_)
      num_remain_ += 1;
    dt_ = stop_time_ / static_cast<P>(num_remain_);
  }
  //! specify number of steps and final time
  time_data(time_method smethod, int64_t num_steps, input_stop_time stop_time)
    : smethod_(smethod), stop_time_(stop_time.value), time_(0), step_(0),
      num_remain_(num_steps)
  {
    dt_ = stop_time_ / static_cast<P>(num_remain_);
  }
  //! specify time-step and number of steps
  time_data(time_method smethod, input_dt dt, int64_t num_steps)
    : smethod_(smethod), dt_(dt.value), time_(0), step_(0),
      num_remain_(num_steps)
  {
    stop_time_ = num_remain_ * dt_;
  }

  //! return the time-advance method
  time_method step_method() const { return smethod_; }

  //! returns the time-step
  P dt() const { return dt_; }
  //! returns the stop-time
  P stop_time() const { return stop_time_; }
  //! returns the current time
  P time() const { return time_; }
  //! returns the current time, non-const ref that can reset the time
  P &time() { return time_; }
  //! returns the current step number
  int64_t step() const { return step_; }
  //! returns the number of remaining time-steps
  int64_t num_remain() const { return num_remain_; }
  //! set final time and zero out the num_remain
  void set_final_time() {
    time_       = stop_time_;
    num_remain_ = 0;
  }

  //! advances the time and updates the current and remaining steps
  void take_step() {
    ++step_;
    --num_remain_;
    time_ += dt_;
  }

  //! prints the stepping data to a stream (human readable format)
  void print_time(std::ostream &os) const {
    os << "  time (t)        " << time_
       << "\n  stop-time (T)   " << stop_time_
       << "\n  num-steps (n)   " << tools::split_style(num_remain_)
       << "\n  time-step (dt)  " << dt_ << '\n';
  }

  //! allows writer to save/load the time data
  friend class h5manager<P>;

private:
  time_method smethod_ = time_method::exp;
  // the following entries cannot be negative, negative means "not-set"

  //! current time-step
  P dt_ = -1;
  //! currently set final time
  P stop_time_ = -1;
  //! current time for the simulation
  P time_ = -1;
  //! current number of steps taken
  int64_t step_ = -1;
  //! remaining steps
  int64_t num_remain_ = -1;
};

/*!
 * \ingroup asgard_discretization
 * \brief Allows writing time-data to a stream
 */
template<typename P>
inline std::ostream &operator<<(std::ostream &os, time_data<P> const &dtime)
{
  dtime.print_time(os);
  return os;
}

/*!
 * \ingroup asgard_pde_definition
 * \brief Contains shorthand notation for common operators
 *
 * Many PDEs are build from common building blocks, divergence, Laplacian, etc.
 * ASGarD has a list of commonly used operators as a shorthand when defining
 * custom PDEs.
 */
namespace operators {

/*!
 * \ingroup asgard_pde_definition
 * \brief The divergence operator, sum of derivatives in each dimension
 *
 * The divergence operator in general form for d dimensions:
 * \f[ \nabla \cdot f = \frac{\partial}{\partial x_1} f + \frac{\partial}{\partial x_2} f + \cdots + \frac{\partial}{\partial x_d} f \f]
 * Each term can be assigned a separate coefficient.
 */
struct divergence {
  //! boundary condition to use for all divergence terms
  boundary_type btype;
  //! coefficients of the divergence terms
  std::vector<double> coeffs;
};


}

/*!
 * \ingroup asgard_pde_definition
 * \brief Container for terms, sources, boundary conditions, etc.
 *
 * The PDE descriptor only indirectly specifies a partial differential equation,
 * the primary objective is to specify the discretization scheme.
 *
 * The main components are:
 * - asgard::pde_domain defining the dimensions and ranges for each dimension
 * - asgard::prog_opts defining user options for sparse grid, time-stepping scheme
 *   and many others
 * - initial conditions
 * - terms indicating differential and integral operators
 * - source terms that appear on the right-hand-side of the equation
 *
 * The first two are defined in the constructor of the object and the others
 * can be specified later. See the included examples.
 */
template<typename P = default_precision>
class PDEv2
{
public:
  //! used for sanity/error checking
  using precision_mode = P;

  //! creates an empty pde
  PDEv2() = default;
  //! initialize the pde over the domain
  PDEv2(prog_opts opts, pde_domain<P> domain)
    : options_(std::move(opts)), domain_(std::move(domain)),
      mass_(domain_.num_dims())
  {
    int const numd = domain_.num_dims();
    if (domain_.num_dims() == 0)
      throw std::runtime_error("the pde cannot be initialized with an empty domain");

    if (options_.restarting()) {
      // more error checking is done during the file reading process
      if (not std::filesystem::exists(options_.restart_file))
        throw std::runtime_error("Cannot find file: '" + options_.restart_file + "'");
    } else {
      // cold start, apply defaults and sanitize
      if (options_.start_levels.empty()) {
        if (options_.default_start_levels.empty())
          throw std::runtime_error("must specify start levels for the grid");
        else
          options_.start_levels = options_.default_start_levels;
      }

      if (options_.start_levels.size() == 1) {
        int const l = options_.start_levels.front(); // isotropic level
        if (numd > 1)
          options_.start_levels.resize(numd, l); // fill vector with l
      } else {
        if (numd != static_cast<int>(options_.start_levels.size()))
          throw std::runtime_error("the starting levels must include either a single entry"
                                   "indicating uniform/isotropic grid or one entry per dimension");
      }

      if (options_.max_levels.empty()) {
        options_.max_levels = options_.start_levels; // if unspecified, use max for start
      } else {
        if (options_.max_levels.size() == 1) {
          int const l = options_.max_levels.front(); // uniform max
          if (numd > 1)
            options_.max_levels.resize(numd, l); // fill vector with l
        } else {
          if (options_.max_levels.size() != options_.start_levels.size())
            throw std::runtime_error("the max levels must include either a single entry"
                                     "indicating uniform max or one entry per dimension");
        }
        // use the initial as max, if the max is less than the initial level
        for (int d : iindexof(numd))
          options_.max_levels[d] = std::max(options_.max_levels[d], options_.start_levels[d]);
      }

      max_level_ = *std::max_element(options_.max_levels.begin(), options_.max_levels.end());

      if (not options_.degree) {
        if (options_.default_degree)
          options_.degree = options_.default_degree.value();
        else
          throw std::runtime_error("must provide a polynomial degree with -d or default_degree()");
      }

      // setting step method
      if (not options_.step_method and options_.default_step_method)
        options_.step_method = options_.default_step_method.value();

      // setting solver for the time-stepper
      if (not options_.solver and options_.default_solver)
        options_.solver = options_.default_solver.value();
      // setting up preconditioner for a possibly iterative solver
      if (not options_.precon and options_.default_precon)
        options_.precon = options_.default_precon.value();
      // setting up the solver options
      if (not options_.isolver_tolerance and options_.default_isolver_tolerance)
        options_.isolver_tolerance = options_.default_isolver_tolerance.value();
      if (not options_.isolver_iterations and options_.default_isolver_iterations)
        options_.isolver_iterations = options_.default_isolver_iterations.value();
      if (not options_.isolver_inner_iterations and options_.default_isolver_inner_iterations)
        options_.isolver_inner_iterations = options_.default_isolver_inner_iterations.value();
    }

    // don't support l-inf norm yet
    if (options_.adapt_threshold) {
      if (options_.anorm and options_.anorm.value() == adapt_norm::linf)
        std::cerr << "warning: l-inf norm not implemented for pde-version 2, switching to l2\n";
      options_.anorm = adapt_norm::l2;
    }
  }

  //! shortcut for the number of dimensions
  int num_dims() const { return domain_.num_dims(); }
  //! shortcut for the number of terms
  int num_terms() const { return static_cast<int>(terms_.size()); }
  //! indicates whether the pde was initialized with a domain
  operator bool () const { return (domain_.num_dims() > 0); }
  //! return the max level that can be used by the grid
  int max_level() const { return max_level_; }
  //! returns the degree for the polynomial basis
  int degree() const { return options_.degree.value(); }

  //! returns the options, modded to normalize
  prog_opts const &options() const { return options_; }
  //! returns the domain loaded in the constructor
  pde_domain<P> const &domain() const { return domain_; }

  //! set non-separable initial condition, can have only one
  void set_initial(md_func<P> ic_md) {
    initial_md_ = std::move(ic_md);
  }
  //! add separable initial condition, can have multiple
  void add_initial(separable_func<P> ic_md) {
    initial_sep_.emplace_back(std::move(ic_md));
  }
  //! returns the separable initial conditions
  std::vector<separable_func<P>> const &ic_sep() const { return initial_sep_; }
  //! returns the non-separable initial condition
  md_func<P> const &ic_md() const { return initial_md_; }

  //! set the mass (density) of the pde
  void set_mass(mass_md<P> tmass) {
    rassert(tmass.num_dims() == num_dims(), "mass number of dimensions must match the domain");
    mass_ = std::move(tmass);
  }
  //! returns the mass term
  mass_md<P> const &mass() const { return mass_; }

  //! adding a term to the pde
  PDEv2<P> & operator += (term_md<P> tmd) {
    rassert(not tmd.mass(), "only terms in a chain can have a mass_md");
    if (tmd.is_chain())
      rassert(not tmd.chain(0).mass(), "the 0-th term of a chain cannot have a mass_md")
    tmd.set_num_dimensions(domain_.num_dims());
    terms_.emplace_back(std::move(tmd));
    return *this;
  }
  //! adding a term to the pde
  void add_term(term_md<P> tmd) {
    tmd.set_num_dimensions(domain_.num_dims());
    terms_.emplace_back(std::move(tmd));
  }
  //! returns the loaded terms
  std::vector<term_md<P>> const &terms() const { return terms_; }
  //! returns the i-th term
  term_md<P> const &term(int i) const { return terms_[i]; }

  //! set non-separable right-hand-source, can have only one
  void set_source(md_func<P> smd) {
    sources_md_ = std::move(smd);
  }
  //! add separable right-hand-source, can have multiple
  void add_source(separable_func<P> smd) {
    sources_sep_.emplace_back(std::move(smd));
  }
  //! add separable right-hand-source, can have multiple
  PDEv2<P> & operator += (separable_func<P> tmd) {
    this->add_source(std::move(tmd));
    return *this;
  }
  //! returns the separable sources
  std::vector<separable_func<P>> const &source_sep() const { return sources_sep_; }
  //! returns the i-th separable sources
  separable_func<P> const &source_sep(int i) const { return sources_sep_[i]; }
  //! returns the non-separable source
  md_func<P> const &source_md() const { return sources_md_; }

  //! returns the smallest cell size in given dimension and level, , uses max-level by default
  P cell_size(int dim, int level = -1) const {
    if (level < 0)
      level = max_level_;
    return domain_.cell_size(dim, level);
  }
  //! returns the smallest cell size across all dimensions, uses max-level by default
  P min_cell_size(int level = -1) const {
    if (level < 0)
      level = max_level_;
    return domain_.min_cell_size(level);
  }


  //! allows writer to save/load the pde and options
  friend class h5manager<P>;
  //! allows the term_manager to access the terms
  friend struct term_manager<P>;

private:
  prog_opts options_;
  pde_domain<P> domain_;
  int max_level_ = 1;

  md_func<P> initial_md_;
  std::vector<separable_func<P>> initial_sep_;

  mass_md<P> mass_;
  std::vector<term_md<P>> terms_;

  md_func<P> sources_md_;
  std::vector<separable_func<P>> sources_sep_;
};

} // namespace asgard
