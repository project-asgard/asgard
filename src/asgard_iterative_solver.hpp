#include "asgard_term_build.hpp"

namespace asgard::solvers
{

/*!
 * \internal
 * \brief Signature for the left-hand linear operation for a solver, raw-array variant
 *
 * Computes `y = alpha * A * x + beta * y`
 *
 * \endinternal
 */
template<typename P>
using operation_apply_lhs =
  std::function<void(P alpha, P const x[], P beta, P y[])>;

/*!
 * \internal
 * \brief Signature for the preconditioner
 *
 * Computes `y = inverse-P * y` and do not need the constants
 * from the asgard::operation_apply_lhs
 *
 * \endinternal
 */
template<typename P>
using operation_apply_precon = std::function<void(P y[])>;

/*!
 * \internal
 * \brief Conjugate Gradient method
 *
 * The class mostly holds workspace vectors.
 * \endinternal
 */
template<typename P>
class cg
{
public:
  //! default constructor, nothing to do
  cg() = default;

  //! construct and set the tolerance and maximum number of iterations
  cg(P tolerance, int maxi) : tolerance_(tolerance), max_iter_(maxi) {
    #ifdef ASGARD_USE_GPU
    grho.resize(1);
    grho_new.resize(1);
    gp_dot_gq.resize(1);
    #endif
  }

  //! solve for the given linear operator, right-hand-side and initial iterate
  int solve(operation_apply_precon<P> precon, operation_apply_lhs<P> apply_lhs,
            std::vector<P> const &rhs, std::vector<P> &x) const;

#ifdef ASGARD_USE_GPU
  //! solve for the given linear operator, right-hand-side and initial iterate, uses gpus
  int solve(operation_apply_precon<P> precon, operation_apply_lhs<P> apply_lhs,
            gpu::vector<P> const &rhs, gpu::vector<P> &x) const;
#endif

  //! preconditioning requires three extra workspace vectors
  mutable std::vector<P> prec_rhs;
  //! preconditioning requires three extra workspace vectors
  mutable std::vector<P> prec_y;
  //! preconditioning requires three extra workspace vectors
  mutable std::vector<P> prec_yb;
  //! returns the set tolerance
  P tolerance() const { return tolerance_; }
  //! returns the set max-number of iterations
  int max_iter() const { return max_iter_; }

  #ifdef ASGARD_USE_GPU
  //! preconditioning requires three extra workspace vectors
  mutable gpu::vector<P> prec_rhs_gpu;
  //! preconditioning requires three extra workspace vectors
  mutable gpu::vector<P> prec_y_gpu;
  //! preconditioning requires three extra workspace vectors
  mutable gpu::vector<P> prec_yb_gpu;
  #endif

  //! computes approximate memory usage by the object
  size_t used_bytes() const;

private:
  P tolerance_ = 0;
  int max_iter_ = 0;

  //! CPU workspace
  mutable std::vector<P> r, p, q, z;
#ifdef ASGARD_USE_GPU
  //! GPU workspace
  mutable gpu::vector<P> gr, gp, gq, gz;
  //! VRAM Scalars
  mutable gpu::vector<P> grho, grho_new, gp_dot_gq;
#endif
};

/*!
 * \internal
 * \brief BiCGSTAB method combines Conjugate-Gradient and GMRES
 *
 * The class mostly holds workspace vectors.
 * \endinternal
 */
template<typename P>
class bicgstab
{
public:
  //! default constructor, nothing to do
  bicgstab() = default;

  //! construct and set the tolerance and maximum number of iterations
  bicgstab(P tol, int maxi) : tolerance_(tol), max_iter_(maxi) {};

  //! solve for the given linear operator, right-hand-side and initial iterate
  int solve(operation_apply_lhs<P> apply_lhs, std::vector<P> const &rhs,
            std::vector<P> &x) const;

  #ifdef ASGARD_USE_GPU
  //! solve for the given linear operator, right-hand-side and initial iterate, uses the gpus
  int solve(operation_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
            gpu::vector<P> &x) const;
  #endif

  //! preconditioning requires three extra workspace vectors
  mutable std::vector<P> prec_rhs;
  //! preconditioning requires three extra workspace vectors
  mutable std::vector<P> prec_y;
  //! preconditioning requires three extra workspace vectors
  mutable std::vector<P> prec_yb;
  //! returns the set tolerance
  P tolerance() const { return tolerance_; }
  //! returns the set max-number of iterations
  int max_iter() const { return max_iter_; }

  #ifdef ASGARD_USE_GPU
  //! preconditioning requires three extra workspace vectors
  mutable gpu::vector<P> prec_rhs_gpu;
  //! preconditioning requires three extra workspace vectors
  mutable gpu::vector<P> prec_y_gpu;
  //! preconditioning requires three extra workspace vectors
  mutable gpu::vector<P> prec_yb_gpu;
  #endif

  //! computes approximate memory usage by the object
  size_t used_bytes() const;

private:
  P tolerance_  = 0;
  int max_iter_ = 0;

  mutable std::vector<P> rref, r, p, v, t;

  #ifdef ASGARD_USE_GPU
  mutable gpu::vector<P> grref, gr, gp, gv, gt;
  #endif
};

/*!
 * \internal
 * \brief General Minimum Residual solver - GMRES
 *
 * Implements the restarted version with given max-number of inner and outer
 * iterations. The class mostly holds workspace vectors.
 * \endinternal
 */
template<typename P>
class gmres
{
public:
  //! default constructor, nothing to do
  gmres() = default;

  //! construct and set the tolerance and maximum number of iterations
  gmres(P tol, int maxi, int maxo)
    : tolerance_(tol), max_inner_(maxi), max_outer_(maxo)
  {
    krylov_data.resize(3 * (max_inner_ + 1) + ((max_inner_ + 1) * max_inner_) / 2);

    #ifdef ASGARD_USE_GPU
    gpu_coeffs.resize(max_inner_ + 1);
    #endif

    P *data = krylov_data.data();
    krylov_proj = std::exchange(data, data + ((max_inner_ + 1) * max_inner_ / 2));
    sines       = std::exchange(data, data + max_inner_ + 1);
    cosines     = std::exchange(data, data + max_inner_ + 1);
    krylov_sol  = std::exchange(data, data + max_inner_ + 1);
    assert(data == krylov_data.data() + krylov_data.size());
  }

  //! solve for the given linear operators, right-hand-side and initial iterate
  int solve(operation_apply_precon<P> apply_precon,
            operation_apply_lhs<P> apply_lhs, std::vector<P> const &rhs,
            std::vector<P> &x) const;

  #ifdef ASGARD_USE_GPU
  //! solve for the given linear operators, right-hand-side and initial iterate
  int solve(operation_apply_precon<P> apply_precon,
            operation_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
            gpu::vector<P> &x) const;
  #endif

  //! returns the set tolerance
  P tolerance() const { return tolerance_; }
  //! returns the set max-number of iterations
  int max_inner() const { return max_inner_; }
  //! returns the max-number of restarts
  int max_outer() const { return max_outer_; }
  //! computes approximate memory usage by the object
  size_t used_bytes() const;

private:
  P tolerance_   = 0;
  int max_inner_ = 0;
  int max_outer_ = 0;

  mutable std::vector<P> basis;
  #ifdef ASGARD_USE_GPU
  mutable gpu::vector<P> gpu_basis;
  mutable gpu::vector<P> gpu_coeffs;
  #endif

  mutable std::vector<P> krylov_data;
  mutable P *krylov_proj = nullptr;
  mutable P *sines       = nullptr;
  mutable P *cosines     = nullptr;
  mutable P *krylov_sol  = nullptr;
};
} // namespace::asgard::solvers

namespace asgard
{

/*!
 * \internal
 * \brief Stores precondioner data
 *
 * Holds the variants for the precondioner.
 * \endinternal
 */
template<typename P>
struct preconditioner_data {
  //! default constructor
  preconditioner_data() = default;
  //! set the method for the precondioner
  preconditioner_data(precon_method precon) {
    if (precon == precon_method::jacobi) {
      data.template emplace<std::vector<P>>();
      #ifdef ASGARD_USE_GPU
      gpu_data.template emplace<gpu::vector<P>>();
      #endif
    }
  }

  //! returns the precondioner method
  precon_method method() const { return static_cast<precon_method>(data.index()); }
  //! extracts the precon_method
  operator precon_method() const { return static_cast<precon_method>(data.index()); }
  //! print the human-readable name of the precondioner
  void print_method(std::ostream &os = std::cout) const {
    switch(method()) {
      case precon_method::none:
        os << "  no preconditioner";
        break;
      case precon_method::jacobi:
        os << "   jacobi diagonal preconditioner";
        break;
      default:
        os << "unknown"; // should never happen
        break;
    }
  }
  /*!
   * \internal
   * \brief Write the preconditioner name to a stream
   *
   * \endinternal
   */
  friend std::ostream &operator<<(std::ostream &os, preconditioner_data<P> const &precon) {
    precon.print_method(os);
    return os;
  }

  //! indicates whether the precondioner is set to none
  operator bool () const { return not std::holds_alternative<std::monostate>(data); }

  //! indicates whether the precondioner is valid for this sparse grid
  bool valid_for(sparse_grid const &grid) const {
    return (std::holds_alternative<std::monostate>(data) or grid_gen == grid.generation());
  }

  //! returns the data for the Jacobi precondioner
  std::vector<P> &jacobi() { return std::get<std::vector<P>>(data); }
  #ifdef ASGARD_USE_GPU
  gpu::vector<P> &gpu_jacobi() { return std::get<gpu::vector<P>>(gpu_data); }
  #endif

  //! computes approximate memory usage by the object
  size_t used_bytes() const {
    if (std::holds_alternative<std::vector<P>>(data))
      return std::get<std::vector<P>>(data).size() * sizeof(P);
    return 0;
  }

  //! sparse grid generation
  int grid_gen = -1;
  //! holds the cpu data for the precondioner
  std::variant<std::monostate, std::vector<P>> data = std::monostate{};
  #ifdef ASGARD_USE_GPU
  // holds the gpu data for the precondioner
  std::variant<std::monostate, gpu::vector<P>> gpu_data = std::monostate{};
  #endif
};
} // namespace::asgard
