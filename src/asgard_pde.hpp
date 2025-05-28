#pragma once

#include "asgard_dimension.hpp"
#include "asgard_indexset.hpp"
#include "asgard_quadrature.hpp"

// the quadrature is needed by some of the pdes to perform internal operations

/*!
 * \defgroup asgard_pde_definition ASGarD PDE Definition
 *
 * Tools for defining a PDE description and discretization scheme.
 */

namespace asgard
{
/*!
 * \ingroup asgard_pde_definition
 * \brief Indicates special dependence in the terms
 *
 * Used to do coupling with moments.
 */
enum class term_dependence
{
  //! no coupling, just a regular term
  none,
  //! dependence on the electric field with a coefficient
  electric_field,
  //! dependence on the electric field only, coefficient is 1
  electric_field_only,
  //! moment divided by moment 0
  moment_divided_by_density,
  //! Lenard-Bernstein theta term, 1x1v term
  lenard_bernstein_coll_theta_1x1v,
  //! Lenard-Bernstein theta term, 1x2v term
  lenard_bernstein_coll_theta_1x2v,
  //! Lenard-Bernstein theta term, 1x3v term
  lenard_bernstein_coll_theta_1x3v,
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Adjust the edge flux for the operator matrices
 *
 * The direction of the flux at the cell edges is determined by the sign
 * of the operator coefficient, for separable terms that is the product
 * of the one-dimensional coefficients.
 *
 * The default upwind flux will yield highest convergence rate, but it will take
 * into account only the coefficient from this dimension, i.e., assuming
 * the other dimensions have positive coefficients.
 * If a term in another dimension has a negative sign, then the upwind has
 * to manually swapped with a downwind flux.
 * If the coefficient is unknown or indeterminate, e.g., depends on the moments,
 * or the coefficient in another dimension can change sign,
 * then the central flux should be used.
 *
 * Here are some examples, where the coefficients can always be variable,
 * so long as the sign remain positive or negative or oscillates pos-neg.
 * In the example, dim 1 is always the dimension with the flux.
 *
 * <table>
 *  <tr><th> dim 1 </th><th> dim 2 </th><th> dim 3 </th><th> flux type </th></tr>
 *  <tr><th> pos-neg </th><th> positive </th><th> N/A </th><th> upwind </th></tr>
 *  <tr><th> pos-neg </th><th> negative </th><th> N/A </th><th> downwind </th></tr>
 *  <tr><th> pos-neg </th><th> negative </th><th> negative </th><th> upwind </th></tr>
 *  <tr><th> negative </th><th> positive </th><th> N/A </th><th> upwind </th></tr>
 *  <tr><th> negative </th><th> positive </th><th> negative </th><th> downwind </th></tr>
 *  <tr><th> positive </th><th> pos-neg </th><th> N/A </th><th> central </th></tr>
 * </table>
 *
 */
enum class flux_type
{
  //! default flux
  upwind   = -1,
  //! other dimensions have variable positive/negative coefficients
  central  = 0,
  //! other dimensions yield negative coefficient
  downwind = 1,
};

/*!
 * \ingroup asgard_pde_definition
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
using md_func_f = std::function<void(P t, vector2d<P> const &x,
                                     std::vector<P> const &f, std::vector<P> &vals)>;

/*!
 * \ingroup asgard_pde_definition
 * \brief Defines the boundary conditions for separable operator
 *
 * The separable operators are always defined on a 1d interval. Periodic conditions
 * "connect" the flux on the left-most and right-most cells, so that information
 * moving out through the boundary is added to the other side.
 * The flux can be "fixed" or "pinned" at either the left, right or bothsides,
 * defaulting to zero (homogeneous) boundary condition, but a non-zero
 * (inhomogeneous) flux can also be specified, e.g., see asgard::left_boundary_flux
 * and asgard::right_boundary_flux.
 * Finally, the flux can be unspecified, which allows the for the internal
 * dynamics of the PDE to define the actual value, e.g., an outflow condition.
 */
enum class boundary_type
{
  //! periodic boundary conditions
  periodic,
  //! fixed flux on the left end of the boundary
  left,
  //! fixed flux on the right end of the boundary
  right,
  //! fixed flux at both ends of the boundary
  bothsides,
  //! do not fix the flux on either end of the domain
  none
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Defines the type of one-dimensional operation
 */
enum class operation_type
{
  //! identity term
  identity,
  //! volume term, i.e., no derivative or boundary flux
  volume,
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
 * \brief Intermediate container for an identity term
 */
struct term_identity {};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for a volume term, no boundary or flux types
 */
template<typename P = default_precision>
struct term_volume {
  //! make a volume term with constant coefficient
  term_volume(no_deduce<P> cc) : const_coeff(cc) {}
  //! make a volume term with given right hand side coefficient
  term_volume(sfixed_func1d<P> rhs) : right(std::move(rhs)) {}

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
  //! make a grad term with constant coefficient 1, upwind flux and given boundary_type
  term_grad(boundary_type bnd = boundary_type::none)
    : boundary(bnd)
  {}
  //! make a grad term with constant coefficient 1, and given flux and boundary_type
  term_grad(flux_type flx, boundary_type bnd = boundary_type::none)
    : flux(flx), boundary(bnd)
  {}
  //! make a grad term with given constant coefficient, upwind flux, and given boundary_type
  term_grad(no_deduce<P> cc, boundary_type bnd = boundary_type::none)
    : const_coeff(cc), boundary(bnd)
  {}
  //! make a grad term with given constant coefficient, flux_type and boundary_type
  term_grad(no_deduce<P> cc, flux_type flx, boundary_type bnd = boundary_type::none)
    : const_coeff(cc), flux(flx), boundary(bnd)
  {}
  //! make a grad term with given coefficient, upwind flux, and given boundary_type
  term_grad(sfixed_func1d<P> cc, boundary_type bnd = boundary_type::none)
    : const_coeff(0), var_coeff(std::move(cc)), boundary(bnd)
  {}
  //! make a grad term with given coefficient, flux_type and boundary_type
  term_grad(sfixed_func1d<P> cc, flux_type flx, boundary_type bnd = boundary_type::none)
    : const_coeff(0), var_coeff(std::move(cc)), flux(flx), boundary(bnd)
  {}

  //! constant coefficient, used if var_coeff is not set
  P const_coeff = 1;
  //! non-constant coefficient function
  sfixed_func1d<P> var_coeff;

  //! flux type
  flux_type flux = flux_type::upwind;
  //! boundary type
  boundary_type boundary = boundary_type::none;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for a div term, includes flux and boundary conditions
 */
template<typename P = default_precision>
struct term_div {
  //! make a div term with constant coefficient 1, upwind flux and given boundary_type
  term_div(boundary_type bnd = boundary_type::none)
    : boundary(bnd)
  {}
  //! make a div term with constant coefficient 1, and given flux and boundary_type
  term_div(flux_type flx, boundary_type bnd = boundary_type::none)
    : flux(flx), boundary(bnd)
  {}
  //! make a div term with given constant coefficient, upwind flux, and given boundary_type
  term_div(no_deduce<P> cc, boundary_type bnd = boundary_type::none)
    : const_coeff(cc), boundary(bnd)
  {}
  //! make a div term with given constant coefficient, flux_type and boundary_type
  term_div(no_deduce<P> cc, flux_type flx, boundary_type bnd = boundary_type::none)
    : const_coeff(cc), flux(flx), boundary(bnd)
  {}
  //! make a div term with given coefficient, upwind flux, and given boundary_type
  term_div(sfixed_func1d<P> cc, boundary_type bnd = boundary_type::none)
    : var_coeff(std::move(cc)), boundary(bnd)
  {}
  //! make a div term with given coefficient, flux_type and boundary_type
  term_div(sfixed_func1d<P> cc, flux_type flx, boundary_type bnd = boundary_type::none)
    : var_coeff(std::move(cc)), flux(flx), boundary(bnd)
  {}

  //! constant coefficient, used if var_coeff is not set
  P const_coeff = 1;
  //! non-constant coefficient function
  sfixed_func1d<P> var_coeff;

  //! flux type
  flux_type flux = flux_type::upwind;
  //! boundary type
  boundary_type boundary;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Intermediate container for a div term, includes flux and boundary conditions
 */
template<typename P = default_precision>
struct term_penalty {
  //! make a penalty term with upwind flux and given boundary type
  term_penalty(no_deduce<P> cc, boundary_type bnd = boundary_type::none)
    : const_coeff(cc), boundary(bnd)
  {}
  //! make a penalty term with given flux_type and boundary_type
  term_penalty(no_deduce<P> cc, flux_type flx, boundary_type bnd = boundary_type::none)
    : const_coeff(cc), flux(flx), boundary(bnd)
  {}

  //! coefficient
  P const_coeff = 1;

  //! flux type
  flux_type flux = flux_type::upwind;
  //! boundary type
  boundary_type boundary;
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
 *   t1d += term_div{-2, flux_type::upwind, boundary_type::none};
 *   t1d += term_grad{2, flux_type::upwind, boundary_type::bothsides};
 * \endcode
 */
struct term_chain {};

/*!
 * \ingroup asgard_pde_definition
 * \brief Volume term that depends on the electric field
 */
template<typename P = default_precision>
struct volume_electric {
  //! mass based only on the electric field, same as rhs being the identity function y = x
  volume_electric() {}
  //! right side depends only on the field
  volume_electric(sfixed_func1d<P> rhs) : right(std::move(rhs)) {}
  //! right side depends on the field and position
  volume_electric(sfixed_func1d_f<P> rhs_f) : right_f(std::move(rhs_f)) {}

  //! right-hand-side function, field only no spatial dependence
  sfixed_func1d<P> right;
  //! right-hand-side function, depends on position and field
  sfixed_func1d_f<P> right_f;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Volume term that depends on a given moment divided by the density (moment 0)
 */
struct term_moment_over_density {
  //! constructor, sets the moment
  explicit term_moment_over_density(int mom) : moment(mom) {
    rassert(moment > 0, "The moment over density must be at least 1");
  }
  //! the moment to be used, must use something other than 0
  int moment = 0;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Volume term that depends on the negative of a moment divided by the density (moment 0)
 */
struct term_moment_over_density_neg {
  //! set the negative moment over density
  explicit term_moment_over_density_neg(int mom) : moment(mom) {
    rassert(moment > 0, "The moment over density must be at least 1");
  }
  //! the moment to be used, must use something other than 0
  int moment = 0;
};

// forward declaration so it can be set as a friend
template<typename P>
struct term_manager;
// forward declaration so it can be set as a friend
template<typename P>
class discretization_manager;

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
  //! make a general term, prefer using the helper structs term_(volume,grad,div,penalty)
  term_1d(operation_type opt, flux_type flx, boundary_type bnd, sfixed_func1d<P> frhs, P crhs)
      : optype_(opt), flux_(flx), boundary_(bnd),
        rhs_(std::move(frhs)), rhs_const_(crhs)
  {
    expect(optype_ != operation_type::identity);

    if (optype_ == operation_type::grad) {
      if (flux_ == flux_type::upwind)
        flux_ = flux_type::downwind;
      else if (flux_ == flux_type::downwind)
        flux_ = flux_type::upwind;
    }
  }
  //! make a term that depends on coupled fields, e.g., moments or electric field
  term_1d(term_dependence dep, sfixed_func1d_f<P> ffunc = nullptr)
    : optype_(operation_type::volume), depends_(dep), field_f_(std::move(ffunc))
  {}

  //! make a volume term
  term_1d(term_volume<P> mt)
    : term_1d(operation_type::volume, flux_type::central, boundary_type::none,
              std::move(mt.right), mt.const_coeff)
  {}
  //! make a volume term, hack around creating term_1d<float> from term_mass<double>
  template<typename otherP>
  term_1d(term_volume<otherP> mt)
    : term_1d(operation_type::volume, flux_type::central, boundary_type::none,
              nullptr, static_cast<P>(mt.const_coeff))
  {
    rassert(not mt.right, "type mismatch using term_mass to create term_1d, "
                          "see the type-safety documentation of term_1d");
  }
  //! make a grad term
  term_1d(term_grad<P> grd)
    : term_1d(operation_type::grad, grd.flux, grd.boundary,
              std::move(grd.var_coeff), grd.const_coeff)
  {}
  //! make a div term
  term_1d(term_div<P> divt)
    : term_1d(operation_type::div, divt.flux, divt.boundary,
              std::move(divt.var_coeff), divt.const_coeff)
  {}
  //! make a penalty term
  term_1d(term_penalty<P> pent)
    : term_1d(operation_type::penalty, pent.flux, pent.boundary, nullptr, pent.const_coeff)
  {}
  //! make a penalty term
  template<typename otherP>
  term_1d(term_penalty<otherP> pent)
    : term_1d(operation_type::penalty, pent.flux, pent.boundary,
              nullptr, static_cast<P>(pent.const_coeff))
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

    check_chain();
  }
  //! make a chain term, add terms later with add_term() or +=
  term_1d(term_chain) : optype_(operation_type::chain) {}
  //! make a term that depends on the electric field
  term_1d(volume_electric<P> elmass)
    : optype_(operation_type::volume), change_(changes_with::time),
      rhs_(std::move(elmass.right)), field_f_(std::move(elmass.right_f))
  {
    depends_ = (field_f_) ? term_dependence::electric_field
                          : term_dependence::electric_field_only;
  }
  //! make moment over density dependence term
  term_1d(term_moment_over_density moment)
    : optype_(operation_type::volume),
      depends_(term_dependence::moment_divided_by_density),
      change_(changes_with::time), mom(moment.moment)
  {}
  //! make moment over density dependence term, with negative sign
  term_1d(term_moment_over_density_neg moment)
    : optype_(operation_type::volume),
      depends_(term_dependence::moment_divided_by_density),
      change_(changes_with::time), mom(-moment.moment)
  {}

  //! indicates whether this is an identity term
  bool is_identity() const { return (optype_ == operation_type::identity); }
  //! indicates whether this is a mass term
  bool is_volume() const { return (optype_ == operation_type::volume); }
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
  term_dependence depends() const { return depends_; }

  //! (chain-mode only) number of chained terms
  int num_chain() const { return static_cast<int>(chain_.size()); }
  //! (chain-mode only) get the vector of the chain
  std::vector<term_1d<P>> const &chain() const { return chain_; }
  //! (chain-mode only) get the i-th term in the chain
  term_1d<P> const &operator[](int i) const { return chain_[i]; }
  //! (chain-mode only) add one more term to the chain
  void add_term(term_1d<P> tm) {
    chain_.emplace_back(std::move(tm));
    check_chain();
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
        if (not cc.is_identity() and not cc.is_volume())
          return true;
      return false;
    } else {
      return (not this->is_identity() and not this->is_volume());
    }
  }
  //! add penalty to a div or grad term, more efficient than adding additional term
  void set_penalty(P penalty_coefficient) {
    rassert(optype_ == operation_type::div or optype_ == operation_type::grad
            or optype_ == operation_type::chain,
            "penalty can be added only to div grad or chain terms, if added to a chain, "
            "the flux and boundary condition will be taken from the back of the chain");
    rassert(penalty_coefficient > 0, "penalty coefficient has to be positive");
    penalty_ = penalty_coefficient;
  }
  //! get the current penalty coefficient
  P penalty() const { return penalty_; }

  // allow direct access to the private data
  friend struct term_manager<P>;

private:
  //! (chain-mode only) access the i-th term in the chain, allows mods
  term_1d<P> &chain(int i) { return chain_[i]; }
  //! check if the chain has wrong set of fluxes
  void check_chain() {
    int side = 0;
    int central = 0;
    for (int i : iindexof(chain_)) {
      if (chain_[i].is_volume())
        continue;
      if (chain_[i].flux() == flux_type::central)
        central ++;
      else
        side ++;
    }
    rassert(central <= 1, "cannot chain two central fluxes together");
    rassert(not (central == 1 and side > 0),
            "cannot chain a central flux with a side flux");
    rassert(side <= 2, "cannot chain more than two non-central fluxes");
  }

  operation_type optype_ = operation_type::identity;
  term_dependence depends_ = term_dependence::none;

  flux_type flux_ = flux_type::central;
  boundary_type boundary_ = boundary_type::none;

  changes_with change_ = changes_with::none;

  sfixed_func1d<P> rhs_;
  P rhs_const_ = 1;
  P penalty_   = 0;

  int mom = 0;
  sfixed_func1d_f<P> field_f_;

  std::vector<term_1d<P>> chain_;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Separable mass term, i.e., num-dims volume 1d terms
 *
 * A regular term_1d that is volume or a term_md with with mass terms can depend on time,
 * e.g., via moments, or can be negative in some areas of the domain.
 * The mass-md term contains only volume terms that are time-independent and have
 * an always positive coefficient.
 */
template<typename P = default_precision>
class mass_md {
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
      rassert((list.begin() + d)->is_volume() or (list.begin() + d)->is_identity(),
              "mass_md terms must be volume or identity");
      rassert((list.begin() + d)->depends() == term_dependence::none,
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
      rassert(list[d].is_volume() or list[d].is_identity(),
              "mass_md terms must be volume or identity");
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
 * \brief Intermediate container for a multidimensional interpolation term
 */
template<typename P>
struct term_interp {
  //! create the intermediate term and set the interpolation function
  term_interp(md_func_f<P> itep) : interp(std::move(itep)) {}
  //! holds the interpolation function
  md_func_f<P> interp;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Helper struct to make boundary_flux and set the left flag
 *
 */
template<typename P = default_precision>
struct left_boundary_flux {
  //! make a null term, will have to be reinitialized
  left_boundary_flux() = default;
  //! set boundary condition with the given separable function
  explicit left_boundary_flux(separable_func<P> f)
    : func(std::move(f))
  {
    chain_level.fill(-1);
  }
  //! create a new term and set the chain levels
  explicit left_boundary_flux(separable_func<P> f, std::vector<int> const &clevel)
    : func(std::move(f))
  {
    rassert(clevel.size() == static_cast<size_t>(func.num_dims()),
            "the number of specified chain levels must match dimension of "
            "the separable_func in construction of left_boundary_flux");
    chain_level.fill(-1);
    for (int d : iindexof(clevel))
      chain_level[d] = clevel[d];
  }
  //! the separable function
  separable_func<P> func;
  //! the chain levels
  std::array<int, max_num_dimensions> chain_level;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Helper struct to make boundary_flux and set the right flag
 *
 */
template<typename P = default_precision>
struct right_boundary_flux {
  //! make a null term, will have to be reinitialized
  right_boundary_flux() = default;
  //! set boundary condition with the given separable function
  explicit right_boundary_flux(separable_func<P> f)
    : func(std::move(f))
  {
    chain_level.fill(-1);
  }
  //! create a new term and set the chain levels
  explicit right_boundary_flux(separable_func<P> f, std::vector<int> const &clevel)
    : func(std::move(f))
  {
    rassert(clevel.size() == static_cast<size_t>(func.num_dims()),
            "the number of specified chain levels must match dimension of "
            "the separable_func in construction of right_boundary_flux");
    chain_level.fill(-1);
    for (int d : iindexof(clevel))
      chain_level[d] = clevel[d];
  }
  //! the separable function
  separable_func<P> func;
  //! the chain levels
  std::array<int, max_num_dimensions> chain_level = {-1};
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Helper struct to make boundary_flux and set both left and right flags
 *
 * This is useful when the boundary condition at the left and right points
 * have exactly the same value.
 * If the term is using boundary_type::bothsides but the left and right
 * values are different, then left_boundary_flux and right_boundary_flux
 * have to be specified separately.
 */
template<typename P = default_precision>
struct sym_boundary_flux {
  //! make a null term, will have to be reinitialized
  sym_boundary_flux() = default;
  //! set boundary condition with the given separable function
  explicit sym_boundary_flux(separable_func<P> f)
    : func(std::move(f))
  {
    chain_level.fill(-1);
  }
  //! create a new term and set the chain levels
  explicit sym_boundary_flux(separable_func<P> f, std::vector<int> const &clevel)
    : func(std::move(f))
  {
    rassert(clevel.size() == static_cast<size_t>(func.num_dims()),
            "the number of specified chain levels must match dimension of "
            "the separable_func in construction of sym_boundary_flux");
    chain_level.fill(-1);
    for (int d : iindexof(clevel))
      chain_level[d] = clevel[d];
  }
  //! the separable function
  separable_func<P> func;
  //! the chain levels
  std::array<int, max_num_dimensions> chain_level = {-1};
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Specifies the flux at the boundary, e.g., Dirichlet boundary condition
 *
 * Construct an instance using one of the helpers.
 */
template<typename P = default_precision>
class boundary_flux {
public:
  //! makes default, zero boundary flux
  boundary_flux() { ch_level_.fill(-1); }
  //! make a left boundary flux
  boundary_flux(left_boundary_flux<P> lbf)
    : side_(left_side), func_(std::move(lbf.func)), ch_level_(lbf.chain_level)
  {}
  //! make a right boundary flux
  boundary_flux(right_boundary_flux<P> rbf)
    : side_(right_side), func_(std::move(rbf.func)), ch_level_(rbf.chain_level)
  {}
  //! make a symmetric boundary flux
  boundary_flux(sym_boundary_flux<P> sbf)
    : side_(both_sides), func_(std::move(sbf.func)), ch_level_(sbf.chain_level)
  {}

  //! true if this is left flux
  bool is_left() const { return (side_ != right_side); }
  //! true if this is right flux
  bool is_right() const { return (side_ != left_side); }

  //! check if object has been initialized
  operator bool () const { return (side_ != unset); }
  //! returns const-ref to the stored function
  separable_func<P> const &func() const { return func_; }
  //! return the chain level for the given dimension, allows modification
  int &chain_level(int dim) { return ch_level_[dim]; }
  //! return the chain level for the given dimension
  int const &chain_level(int dim) const { return ch_level_[dim]; }

  // allow access by the term_manager
  friend struct term_manager<P>;

private:
  enum bf_mode { left_side, right_side, both_sides, unset };

  bf_mode side_ = unset;
  separable_func<P> func_;
  std::array<int, max_num_dimensions> ch_level_ = {-1};
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
  //! set an interpolation term
  term_md(term_interp<P> tint)
    : mode_(mode::interpolatory), interp_(std::move(tint.interp))
  {}

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
      for (int d : iindexof(num_dims_)) {
        if (sep[d].has_flux())
          return d;
      }
      return -1;
    }
  }
  //! add new inhomogeneous boundary function to the term
  term_md<P> operator += (boundary_flux<P> bf) {
    rassert(is_separable(), "cannot add separable boundary conditions to non-separable term_md");
    rassert(bf.func().num_dims() == num_dims_,
            "wrong dimension set for boundary flux given to term_md");
    int fd = flux_dim();
    rassert(fd != -1,
            "cannot set boundary conditions for term_md with no derivatives");
    rassert(bf.func().is_const(fd),
            "the flux function has to be constant in the dimension of term_md::flux_dim()")
    bc_flux_.emplace_back(std::move(bf));
    return *this;
  }
  //! returns the interpolation matrix
  md_func_f<P> const &interp() const { return interp_; }
  //! applies the interpolation function, f = f(t, x, nu)
  void interp(P t, vector2d<P> const &x, std::vector<P> const &f, std::vector<P> &vals) const {
    expect(!!interp_);
    interp_(t, x, f, vals);
  }

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
  // boundary conditions
  std::vector<boundary_flux<P>> bc_flux_;
};


#ifndef __ASGARD_DOXYGEN_SKIP
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

/*!
 * \ingroup asgard_pde_definition
 * \brief Adds the Lenard-Bernstein collision operator to the PDE
 *
 * Currently sets homogeneous (zero) boundary conditions at the edge of the velocity domain.
 */
struct lenard_bernstein_collisions {
  //! sets the Lenard-Bernstein collision operator with the given collision frequency
  lenard_bernstein_collisions(double coll_frequency) : nu(coll_frequency) {}
  //! collision frequency
  double nu = 0;
};

} // namespace::operators
#endif

/*!
 * \ingroup asgard_pde_definition
 * \brief Container for group id associated with imex explicit step
 */
struct imex_explicit_group {
  //! sets the explicit group
  explicit imex_explicit_group(int g = -1) : gid(g) {}
  //! the group id
  int gid = -1;
};
/*!
 * \ingroup asgard_pde_definition
 * \brief Container for group id associated with imex implicit step
 */
 struct imex_implicit_group {
  //! sets the implicit group
  explicit imex_implicit_group(int g = -1) : gid(g) {}
  //! the group id
  int gid = -1;
};
/*!
 * \ingroup asgard_pde_definition
 * \brief Strong type for a group-id, implicit, explicit, custom, etc.
 */
struct group_id {
  //! make a generic id from an explicit group
  explicit group_id(imex_explicit_group ie) : gid(ie.gid) {}
  //! make a generic id from an implicit group
  explicit group_id(imex_implicit_group ii) : gid(ii.gid) {}
  //! sets the implicit group
  explicit group_id(int g = -1) : gid(g) {}
  //! the group id
  int gid = -1;
};

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
class pde_scheme
{
public:
  //! used for sanity/error checking
  using precision_mode = P;

  //! creates an empty pde
  pde_scheme() = default;
  //! initialize the pde over the domain
  pde_scheme(prog_opts opts, pde_domain<P> domain)
    : options_(std::move(opts)), domain_(std::move(domain)),
      mass_(domain_.num_dims()), sources_md_(1)
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
    has_interp_funcs = true;
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
  pde_scheme<P> &operator += (term_md<P> tmd) {
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

  //! set non-separable right-hand-source, can have only one per term-group
  void set_source(md_func<P> smd) {
    has_interp_funcs = true;
    sources_md_[std::max(current_term_group, 0)] = std::move(smd);
  }
  //! add separable right-hand-source, can have multiple
  void add_source(separable_func<P> smd) {
    sources_sep_.emplace_back(std::move(smd));
  }
  //! add separable right-hand-source, can have multiple
  pde_scheme<P> & operator += (separable_func<P> tmd) {
    this->add_source(std::move(tmd));
    return *this;
  }
  //! add collision operator
  pde_scheme<P> & operator += (operators::lenard_bernstein_collisions lbc);
  //! returns the separable sources
  std::vector<separable_func<P>> const &source_sep() const { return sources_sep_; }
  //! returns the i-th separable sources
  separable_func<P> const &source_sep(int i) const { return sources_sep_[i]; }
  //! returns the non-separable source
  md_func<P> const &source_md(int i) const { return sources_md_[i]; }

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
  //! begin a new term group, returns the index-id of the new group
  int new_term_group() {
    if (current_term_group == -1) { // initialize group engine
      rassert(terms_.empty() and sources_sep_.empty(),
              "if using term-groups, new_term_group() must be called before any terms/sources are added");
      current_term_group = 0;
    } else { // new group
      finalize_term_groups();
      current_term_group ++;
      sources_md_.push_back(nullptr); // add empty interpolatory source
    }
    return current_term_group;
  }

  //! forces the use of IMEX time-stepping and sets the implicit and explicit modes
  void set(imex_implicit_group im, imex_explicit_group ex) {
    expect(is_imex(options_.step_method.value()));
    im_ = im;
    ex_ = ex;
  }
  //! returns the implicit group
  imex_implicit_group imex_im() const { return im_; }
  //! returns the explicit group
  imex_explicit_group imex_ex() const { return ex_; }

  //! allows writer to save/load the pde and options
  friend class h5manager<P>;
  //! allows the term_manager to access the terms
  friend struct term_manager<P>;
  //! allows the discretization_manager to access the options
  friend class discretization_manager<P>;

private:
  void finalize_term_groups() {
    if (current_term_group == -1) // no groups being used
      return;
    if (current_term_group == 0) {
      term_groups.emplace_back(0, static_cast<int>(terms_.size()));
      source_groups.emplace_back(0, static_cast<int>(sources_sep_.size()));
    } else {
      term_groups.emplace_back(term_groups.back().end(), static_cast<int>(terms_.size()));
      source_groups.emplace_back(source_groups.back().end(),
                                 static_cast<int>(sources_sep_.size()));
    }
  }

  prog_opts options_;
  pde_domain<P> domain_;
  int max_level_ = 1;

  bool has_interp_funcs = false;

  md_func<P> initial_md_;
  std::vector<separable_func<P>> initial_sep_;

  mass_md<P> mass_;
  std::vector<term_md<P>> terms_;

  std::vector<md_func<P>> sources_md_;
  std::vector<separable_func<P>> sources_sep_;

  int current_term_group = -1;
  std::vector<irange> term_groups;
  std::vector<irange> source_groups;

  imex_implicit_group im_;
  imex_explicit_group ex_;
};

/*!
 * \brief Alias for backwards computationally
 *
 * Will be removed in an upcoming release.
 */
template<typename P>
using PDEv2 = pde_scheme<P>;

} // namespace asgard
