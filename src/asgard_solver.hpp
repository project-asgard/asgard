#pragma once
#include "asgard_adapt.hpp"
#include "asgard_kron_operators.hpp"
#include "asgard_term_manager.hpp"

namespace asgard::solvers
{
enum class poisson_bc
{
  dirichlet,
  periodic
};

inline bool is_direct(solver_method s)
{
  return (s == solver_method::direct);
}

// simple, node-local test version of gmres
template<typename P>
gmres_info<P>
simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
             fk::matrix<P> const &M, int const restart, int const max_iter,
             P const tolerance);
// simple, node-local test version of bicgstab
template<typename P>
gmres_info<P>
simple_bicgstab(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
                fk::matrix<P> const &M, int const max_iter,
                P const tolerance);

// solves ( I - dt * mat ) * x = b
template<typename P, resource resrc>
gmres_info<P>
simple_gmres_euler(const P dt, imex_flag imex,
                   kron_operators<P> const &ops,
                   fk::vector<P, mem_type::owner, resrc> &x,
                   fk::vector<P, mem_type::owner, resrc> const &b,
                   int const restart, int const max_iter, P const tolerance);

// solves ( I - dt * mat ) * x = b
template<typename P, resource resrc>
gmres_info<P>
bicgstab_euler(const P dt, imex_flag imex,
               kron_operators<P> const &ops,
               fk::vector<P, mem_type::owner, resrc> &x,
               fk::vector<P, mem_type::owner, resrc> const &b,
               int const max_iter, P const tolerance);

template<typename P>
int default_gmres_restarts(int num_cols);

/*!
 * \brief Stores the data for a poisson solver
 *
 * Holds the domain size, the factor of the operator matrices, etc.
 */
template<typename P>
class poisson
{
public:
  //! default, uninitialized constructor
  poisson() = default;
  //! initialize Poisson solver over the domain with given min/max, level and degree of input basis
  poisson(int pdegree, P domain_min, P domain_max, int level)
    : degree(pdegree), xmin(domain_min), xmax(domain_max), current_level(level)
  {
    if (current_level == 0) return; // nothing to solve

    remake_factors();
  }
  //! change the level, called on refinement
  void update_level(int new_level) {
    if (current_level == new_level)
      return;
    current_level = new_level;
    remake_factors();
  }
  /*!
  * \brief Given the Legendre expansion of the density, find the electric field
  *
  * The density is given as a cell-by-cell Legendre expansion with the given degree.
  * The result is a piece-wise constant approximation to the electric field
  * over each cell.
  *
  * dleft/dright are the values for the Dirichlet boundary conditions,
  * if using periodic boundary, dleft/dright are not used (assumed zero).
  */
  void solve(std::vector<P> const &density, P dleft, P dright, poisson_bc const bc,
            std::vector<P> &efield);

  //! poisson solve using periodic boundary conditions
  void solve_periodic(std::vector<P> const &density, std::vector<P> &efield) {
    solve(density, 0, 0, poisson_bc::periodic, efield);
  }
  //! indicates whether the solver has been initialized
  operator bool() const { return (current_level >= 0); }

private:
  //! set the solver for the current level
  void remake_factors()
  {
    if (current_level == 0)
      return; // nothing to do
    int const nnodes = fm::ipow2(current_level) - 1;
    P const dx = (xmax - xmin) / (nnodes + 1);

    diag = std::vector<P>(nnodes, P{2} / dx);
    subdiag = std::vector<P>(nnodes - 1, -P{1} / dx);

    rhs.resize(nnodes);

    fm::pttrf(diag, subdiag);
  }

  int degree = -1;
  P xmin = 0, xmax = 0;
  int current_level = -1;
  std::vector<P> diag, subdiag, rhs;
};

/*!
 * \internal
 * \brief Direct solver, explicitly forms the dense matrix, very expensive
 *
 * The dense solver is intended for testing and prototyping purposes,
 * since it explicitly forms the dense matrix, the cost of the actual linear
 * algebra operations is orders of magnitude more than alternatives.
 *
 * \endinternal
 */
template<typename P>
class direct
{
public:
  //! make a default, empty solver
  direct() = default;
  //! build a dense solver for the system I + alpha * terms
  direct(sparse_grid const &grid, connection_patterns const &conn,
         term_manager<P> const &terms, P alpha);

  //! inverts the stored matrix
  void operator() (std::vector<P> &x) const
  {
    expect(dense_mat.is_factorized());
    dense_mat.solve(x);
  }
  //! checks whether the solver has been set
  operator bool () const { return dense_mat; }

private:
  //! holds the factor of the dense matrix
  dense_matrix<P> dense_mat;
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

private:
  P tolerance_  = 0;
  int max_iter_ = 0;

  mutable std::vector<P> rref, r, p, v, t;
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

    P *data = krylov_data.data();
    krylov_proj = std::exchange(data, data + ((max_inner_ + 1) * max_inner_ / 2));
    sines       = std::exchange(data, data + max_inner_ + 1);
    cosines     = std::exchange(data, data + max_inner_ + 1);
    krylov_sol  = std::exchange(data, data + max_inner_ + 1);
    expect(data == krylov_data.data() + krylov_data.size());
  }

  //! solve for the given linear operators, right-hand-side and initial iterate
  int solve(operatoin_apply_precon<P> apply_precon,
            operatoin_apply_lhs<P> apply_lhs, std::vector<P> const &rhs,
            std::vector<P> &x) const;

  //! returns the set tolerance
  P tolerance() const { return tolerance_; }
  //! returns the set max-number of iterations
  int max_inner() const { return max_inner_; }
  //! returns the max-number of restarts
  int max_outer() const { return max_outer_; }

private:
  P tolerance_   = 0;
  int max_inner_ = 0;
  int max_outer_ = 0;

  mutable std::vector<P> basis;

  mutable std::vector<P> krylov_data;
  mutable P *krylov_proj = nullptr;
  mutable P *sines       = nullptr;
  mutable P *cosines     = nullptr;
  mutable P *krylov_sol  = nullptr;
};

} // namespace asgard::solvers

namespace asgard
{

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

    opt = options.solver.value();

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
        precon = options.precon.value_or(precon_method::none);
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
        precon = options.precon.value_or(precon_method::none);
        break;
      default: // unreachable
        break;
    }
  }

  //! direct solver only, just call the matrix inversion method
  void direct_solve(std::vector<P> &x) {
    expect(opt == solver_method::direct);
    std::get<solvers::direct<P>>(var)(x);
  }

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
    expect(opt != solver_method::direct);
    if (opt == solver_method::bicgstab) {
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

  //! updates the internals for the current grid generation
  void update_grid(sparse_grid const &grid,
                   connection_patterns const &conn,
                   term_manager<P> const &terms, P alpha);

  //! write the solver options in human-readable format
  void print_opts(std::ostream &os) const;

  //! selected solver
  solver_method opt = solver_method::direct;
  //! selected solver
  precon_method precon = precon_method::none;
  //! remember the total mat-vec products
  mutable int64_t num_apply = 0;
  //! remembers the generation of the grid that was used to last set the manager
  int grid_gen = -1;
  //! holds the actual solver instance
  std::variant<solvers::direct<P>, solvers::gmres<P>, solvers::bicgstab<P>> var;
  //! holds data for the jacobi preconditioner
  std::vector<P> jacobi;

  //! helper method, y = x + beta * y, compiles with OpenMP and SIMD
  static void xpby(std::vector<P> const &x, P beta, P y[]);
};

/*!
 * \internal
 * \brief Write the options to a stream
 *
 * \endinternal
 */
template<typename P>
inline std::ostream &operator<<(std::ostream &os, solver_manager<P> const &solver)
{
  solver.print_opts(os);
  return os;
}

}
