#pragma once
#include "asgard_kronmult.hpp"
#include "asgard_pde.hpp"
#include "asgard_pde_functions.hpp"
#include "asgard_wavelet_basis.hpp"
#include "asgard_block_matrix.hpp"
#include "asgard_term_manager.hpp"

namespace asgard::solvers
{
/*!
 * \internal
 * \brief Direct solver, explicitly forms the dense matrix, very expensive
 *
 * The dense solver is intended for testing and prototyping purposes,
 * since it explicitly forms the dense matrix, the cost of the actual linear
 * algebra operations is orders of magnitude more than alternatives.
 *
 * The alpha parameter indicates the type of operator needed.
 * If alpha is non-zero, this will build the matrix I + alpha * terms,
 * e.g., use alpha = dt for backwards Euler method with the native being canceled
 * since the terms appear on the left side of the equation.
 * If alpha is zero, the matrix being build will correspond to just terms.
 * \endinternal
 */
template<typename P>
class direct
{
public:
  //! make a default, empty solver
  direct() = default;

  //! updates the matrix for the given group
  void update(group_id group, size_t stage, sparse_grid const &grid,
              connection_patterns const &conn,
              term_manager<P> const &terms, P alpha);

  void update(group_id group, sparse_grid const &grid,
              connection_patterns const &conn,
              term_manager<P> const &terms, P alpha)
  {
    update(group, 0, grid, conn, terms, alpha);
  }

  //! returns the currently loaded grid generation
  int grid_gen(group_id group, size_t stage) const {
    size_t const idx = mat_index(group, stage);
    if (idx < mats.size()) {
      return mats[idx].grid_gen;
    } else {
      return -1; // no generation loaded, no matrix at all
    }
  }
  //! returns the currently loaded grid generation, group_id::all() and stage 0
  int grid_gen() const { return grid_gen(group_id::all(), 0); }
  //! solves Ax = b
  void operator() (group_id group, size_t stage, std::vector<P> &b) const {
    size_t const idx = mat_index(group, stage);
    assert(idx < mats.size());
    mats[idx].dense_mat.solve(b);
  }
  //! solves Ax = b
  void operator() (group_id group, std::vector<P> &b) const { (*this)(group, 0, b); }

  #ifdef ASGARD_USE_GPU
  //! solves Ax = b, b is in GPU memory
  void operator() (group_id group, size_t stage, P b[]) const {
    size_t const idx = mat_index(group, stage);
    assert(idx < mats.size());
    mats[idx].dense_mat.solve(b);
  }
  //! solves Ax = b, b is in GPU memory
  void operator() (group_id group, P b[]) const { (*this)(group, 0, b); }
  #endif

  //! set the number of stages
  void set_num_stages(size_t num) { num_stages = num; }

  //! computes approximate memory usage by the object
  size_t used_bytes() const;

private:
  //! returns the index for the given group
  size_t mat_index(group_id group, int stage) const {
    return static_cast<size_t>((group() + 1) * num_stages + stage);
  }
  //! instance of a matrix for a given term generation
  struct matrix_instance {
    //! holds the sparse grid generation
    int grid_gen = -1;
    //! holds the factorized dense matrix
    dense_matrix<P> dense_mat;
  };
  //! holds the number of stages, e.g., IMEX needs two matrices
  size_t num_stages = 1;
  //! holds the matrices for the different matrix terms
  std::vector<matrix_instance> mats;
};

/*!
 * \internal
 * \brief Signature for the left-hand linear operation for a solver, raw-array variant
 *
 * Computes `y = alpha * A * x + beta * y`
 *
 * \endinternal
 */
template<typename P>
using operatoin_apply_lhs =
  std::function<void(P alpha, P const x[], P beta, P y[])>;

/*!
 * \internal
 * \brief Signature for the preconditioner
 *
 * Computes `y = inverse-P * y` and do not need the constants
 * from the asgard::operatoin_apply_lhs
 *
 * \endinternal
 */
template<typename P>
using operatoin_apply_precon = std::function<void(P y[])>;

template<typename P>
class cg
{
public:
  cg(P tolerance, int max_iter = 1000)
      : tolerance_(tolerance), max_iter_(max_iter) {}

  // CPU Signature
  int solve(operatoin_apply_lhs<P> apply_lhs, std::vector<P> const &rhs, std::vector<P> &x) const;

#ifdef ASGARD_USE_GPU
  // GPU Signature
  int solve(operatoin_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs, gpu::vector<P> &x) const;
#endif

  P tolerance() const { return tolerance_; }
  int max_iter() const { return max_iter_; }
  size_t used_bytes() const;

private:
  P tolerance_ = 0.0;
  int max_iter_ = 0;

  // CPU workspace
  mutable std::vector<P> r, p, q;

#ifdef ASGARD_USE_GPU
  // GPU workspace
  mutable gpu::vector<P> gr, gp, gq;
  
  // VRAM-trapped Scalars
  mutable gpu::vector<P> d_rho, d_rho_new, d_p_dot_q, d_alpha, d_beta;
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
  int solve(operatoin_apply_lhs<P> apply_lhs, std::vector<P> const &rhs,
            std::vector<P> &x) const;

  #ifdef ASGARD_USE_GPU
  //! solve for the given linear operator, right-hand-side and initial iterate, uses the gpus
  int solve(operatoin_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
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
  int solve(operatoin_apply_precon<P> apply_precon,
            operatoin_apply_lhs<P> apply_lhs, std::vector<P> const &rhs,
            std::vector<P> &x) const;

  #ifdef ASGARD_USE_GPU
  //! solve for the given linear operators, right-hand-side and initial iterate
  int solve(operatoin_apply_precon<P> apply_precon,
            operatoin_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
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

/*!
 * \internal
 * \brief Specialized solver for problem with form nu * identity * f = rhs
 *
 * Certain PDEs, e.g., BGK, are using IMEX setup where the implicit solve
 * uses a term of the form, nu * I * f = rhs.
 * The solution in that case is trivial and does not need the extra overhead
 * of the more sophisticated solvers.
 * \endinternal
 */
template<typename P>
class scaled_identity
{
public:
  //! default constructor, nothing to do
  scaled_identity() = default;
  //! update the solver, i.e., recompute the scale
  void update(group_id group, size_t stage, sparse_grid const &grid,
              term_manager<P> const &terms, P alpha);
  //! update the solver, i.e., recompute the scale, stage 0
  void update(group_id group, sparse_grid const &grid, term_manager<P> const &terms, P alpha) {
    update(group, 0, grid, terms, alpha);
  }

  //! returns the currently loaded grid generation
  int grid_gen(group_id group, size_t stage) const {
    size_t const idx = s_index(group, stage);
    if (idx < scale_.size()) {
      return scale_[idx].grid_gen;
    } else {
      return -1; // no generation loaded, no matrix at all
    }
  }
  //! return the currently loaded grid generation for group_id::all() and stage 0
  int grid_gen() const { return grid_gen(group_id::all(), 0); }

  //! set the scale of the left-hand-side, solution is scaled by 1/alpha
  void set_alpha(group_id group, size_t stage, P alpha) {
    size_t const idx = s_index(group, stage);
    if (scale_.size() <= idx) scale_.resize(idx + 1);
    scale_[idx].value = P{1} / alpha;
  }
  //! set the scale of the left-hand-side, solution is scaled by 1/alpha
  void set_alpha(P alpha) {
    set_alpha(group_id::all(), 0, alpha);
  }
  //! returns the current scale
  P scale(group_id group, size_t stage) const { return scale_[s_index(group, stage)].value; }
  //! returns the current scale
  P scale() const { return scale_[s_index(group_id::all(), 0)].value; }

  //! solve for the given linear operators, right-hand-side and initial iterate
  void operator() (std::vector<P> &x) const {
    return (*this)(group_id::all(), 0, x);
  }
  //! solve for the given linear operators, right-hand-side and initial iterate
  void operator() (group_id group, size_t stage, std::vector<P> &x) const;

  #ifdef ASGARD_USE_GPU
  //! solve for the given linear operators, right-hand-side and initial iterate
  void operator() (group_id group, size_t stage, gpu::vector<P> &x) const;
  //! solve for the given linear operators, right-hand-side and initial iterate
  void operator() (group_id group, size_t stage, P x[]) const;
  #endif

  //! set the number of stages
  void set_num_stages(size_t num) { num_stages = num; }
  //! computes approximate memory usage by the object
  size_t used_bytes() const { return 0; }

private:
  struct scales {
    P value = 0;
    int grid_gen = -1;
  };
  //! returns the index for the given group
  size_t s_index(group_id group, size_t stage) const {
    return static_cast<size_t>((group() + 1) * num_stages + stage);
  }
  #ifdef ASGARD_USE_GPU
  //! expected state size on the GPU
  int64_t num_entries = 0;
  #endif
  //! number of stored stages
  size_t num_stages = 1;
  //! scale factor
  std::vector<scales> scale_;
};

} // namespace asgard::solvers

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

/*!
 * \internal
 * \brief Allows a time-stepper to take a hold of some solver
 *
 * Variant that represents any of the available asgard solvers.
 * Each time-stepper is expected to handle the intricacies of setting
 * the correct solver parameters, this is just the container.
 * \endinternal
 */
template<typename P>
struct solver_manager
{
  //! default constructor, probably not the best idea
  solver_manager() = default;
  //! create a new solver
  solver_manager(prog_opts const &options)
  {
    rassert(options.solver, "steady state and implicit time-stepping require a solver, e.g., "
            "'-sv direct' or '-sv gmres -ist 1.E-6 -isi 300 -isn 50 "
            "see --help for list of available solvers");

    solver_method opt = options.solver.value();

    switch (opt) {
      case solver_method::direct:
        var = solvers::direct<P>(); // will be initialized later
        break;
      case solver_method::bicgstab:
        rassert(options.isolver_tolerance,
                "missing tolerance for the iterative solver bicgstab");
        rassert(options.isolver_iterations,
                "missing number of iterations for the iterative solver bicgstab");
        var = solvers::bicgstab<P>(options.isolver_tolerance.value(),
                                   options.isolver_iterations.value());
        break;
      case solver_method::gmres:
        rassert(options.isolver_tolerance,
                "missing tolerance for the iterative solver gmres");
        rassert(options.isolver_iterations,
                "missing number of iterations for the iterative solver gmres");
        rassert(options.isolver_inner_iterations,
                "missing number of outer iterations for the iterative solver gmres");
        var = solvers::gmres<P>(options.isolver_tolerance.value(),
                                options.isolver_inner_iterations.value(),
                                options.isolver_iterations.value());
        break;
      case solver_method::scaled_identity:
        var = solvers::scaled_identity<P>{};
        break;
      default: // unreachable
        break;
    }
  }

  //! for multi-stage solves, e.g., IMEX-2, set the number of stages
  void set_num_stages(size_t num) {
    if (method() == solver_method::direct)
      std::get<solvers::direct<P>>(var).set_num_stages(num);
  }

  //! returns true if the solver is one-shot in-place, i.e., direct or scaled_identity
  bool uses_inplace_solve() const {
    solver_method const m = method();
    return (m == solver_method::direct or m == solver_method::scaled_identity);
  }

  //! direct solver only, just call the matrix inversion method
  void solve_inplace(std::vector<P> &x) {
    if (method() == solver_method::direct)
      std::get<solvers::direct<P>>(var)(group_id::all(), 0, x);
    else
      std::get<solvers::scaled_identity<P>>(var)(group_id::all(), 0, x);
  }
  //! direct solver only, just call the matrix inversion method
  void solve_inplace(group_id group, size_t stage, std::vector<P> &x) {
    if (method() == solver_method::direct)
      std::get<solvers::direct<P>>(var)(group, stage, x);
    else
      std::get<solvers::scaled_identity<P>>(var)(group, stage, x);
  }

  #ifdef ASGARD_USE_GPU
  //! direct solver only, just call the matrix inversion method
  void solve_inplace(P x[]) {
    if (method() == solver_method::direct)
      std::get<solvers::direct<P>>(var)(group_id::all(), 0, x);
    else
      std::get<solvers::scaled_identity<P>>(var)(group_id::all(), 0, x);
  }
  //! direct solver only, just call the matrix inversion method
  void solve_inplace(group_id group, size_t stage, P x[]) {
    if (method() == solver_method::direct)
      std::get<solvers::direct<P>>(var)(group, stage, x);
    else
      std::get<solvers::scaled_identity<P>>(var)(group, stage, x);
  }
  #endif

  //! iterative solver, calls the appropriate iterative solver
  void iterate_solve(solvers::operatoin_apply_lhs<P> apply_lhs,
                     std::vector<P> const &rhs, std::vector<P> &x) const
  {
    iterate_solve(nullptr, apply_lhs, rhs, x);
  }

  //! iterative solver, calls the appropriate iterative solver
  void iterate_solve(solvers::operatoin_apply_precon<P> prec,
                     solvers::operatoin_apply_lhs<P> apply_lhs,
                     std::vector<P> const &rhs, std::vector<P> &x) const
  {
    if (method() == solver_method::bicgstab) {
      if (prec) {
        solvers::bicgstab<P> const &bicg = std::get<solvers::bicgstab<P>>(var);

        bicg.prec_y.resize(rhs.size());

        bicg.prec_rhs = rhs;
        prec(bicg.prec_rhs.data());

        num_apply += bicg.solve([&](P alpha, P const xx[], P beta, P y[])
            -> void {
              if (beta == 0) {
                apply_lhs(alpha, xx, 0, y);
                prec(y);
              } else {
                apply_lhs(alpha, xx, 0, bicg.prec_y.data());
                prec(bicg.prec_y.data());
                xpby(bicg.prec_y, beta, y);
              }
            }, bicg.prec_rhs, x);
      } else {
        num_apply += std::get<solvers::bicgstab<P>>(var).solve(apply_lhs, rhs, x);
      }
    } else { // if (opt == solve_opts::gmres)
      if (prec) {
        solvers::gmres<P> const &gmres = std::get<solvers::gmres<P>>(var);

        num_apply += gmres.solve(prec, apply_lhs, rhs, x);
      } else {
        num_apply += std::get<solvers::gmres<P>>(var).solve(
          [](P *)->void{ /* no preconditioner */ }, apply_lhs, rhs, x);
      }
    }
  }

  #ifdef ASGARD_USE_GPU
  //! iterative solver, calls the appropriate iterative solver, gpu variant
  void iterate_solve(solvers::operatoin_apply_precon<P> prec,
                     solvers::operatoin_apply_lhs<P> apply_lhs,
                     gpu::vector<P> const &rhs, gpu::vector<P> &x) const;
  //! iterative solver, calls the appropriate iterative solver
  void iterate_solve(solvers::operatoin_apply_lhs<P> apply_lhs,
                     gpu::vector<P> const &rhs, gpu::vector<P> &x) const
  {
    iterate_solve(nullptr, apply_lhs, rhs, x);
  }
  #endif

  //! updates the internals for the current grid generation
  void update_grid(sparse_grid const &grid,
                   connection_patterns const &conn,
                   term_manager<P> const &terms, P alpha,
                   preconditioner_data<P> &precon)
  {
    update_grid(group_id::all(), 0, grid, conn, terms, alpha, precon);
  }
  //! updates the internals for the current grid generation
  void update_grid(group_id groupid, size_t stage, sparse_grid const &grid,
                   connection_patterns const &conn,
                   term_manager<P> const &terms, P alpha,
                   preconditioner_data<P> &precon);

  //! write the solver options in human-readable format
  void print_opts(std::ostream &os) const;
  /*!
   * \internal
   * \brief Write the options to a stream
   *
   * \endinternal
   */
  friend std::ostream &operator<<(std::ostream &os, solver_manager<P> const &solver) {
    solver.print_opts(os);
    return os;
  }
  //! get the currently set method
  solver_method method() const { return static_cast<solver_method>(var.index()); }

  //! remember the total mat-vec products
  mutable int64_t num_apply = 0;
  //! holds the actual solver instance
  std::variant<solvers::direct<P>,
               solvers::bicgstab<P>,
               solvers::gmres<P>,
               solvers::scaled_identity<P>> var;

  //! computes approximate memory usage by the object
  size_t used_bytes() const {
    return std::visit([](auto const &v) -> size_t {
        return v.used_bytes();
      }, var);
  }

  //! helper method, y = x + beta * y, compiles with OpenMP and SIMD
  static void xpby(std::vector<P> const &x, P beta, P y[]);
};

}
