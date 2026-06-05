#pragma once

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

} // namespace asgard::solvers

namespace asgard::poisson
{

/*!
 * \internal
 * \brief Boundary conditions for the Poisson solver
 *
 * \endinternal
 */
enum class poisson_bc
{
  //! Dirichlet, set left-right values
  dirichlet,
  //! Periodic, implemented with zero-Dirichlet and removed average
  periodic
};

//! An empty Poisson class for problems which do not require a Poisson solve
class poisson_none {};

/*!
 * \internal
 * \brief Signature for the build term function
 *
 * Builds the matrices needed for Kronecker operations from a term entry
 *
 * \endinternal
 */
template<typename P>
using build_term_func =
  std::function<void(term_entry<P> &tentry, int const dim, int const level)>;

  /*!
 * \internal
 * \brief Signature for the iterative solve function
 *
 * Uses an iterative solver to solve the Poisson equation
 *
 * \endinternal
 */
template<typename P>
using iter_solve_func =
  std::function<int(solvers::operation_apply_lhs<P> apply_lhs,
                     std::vector<P> const &rhs, std::vector<P> &x)>;

/*!
 * \brief Stores the data for a multi-dimensional poisson solver
 *
 * Solves the Laplace equation ∇²u = ρ using a chained div-grad operator,
 * matching the approach used in diffusion.cpp.
 * The solver applies: ∇² = div(grad(f)) in each dimension
 */
template<typename P>
class poisson_md
{
public:
  //! default, uninitialized constructor
  poisson_md() = default;
  //! initialize Poisson solver over the multi-dimensional domain
  poisson_md(int const num_pos, int const max_level, std::array<P, max_num_dimensions> const &xleft,
             std::array<P, max_num_dimensions> const &xright, connection_patterns const &conn,
             hierarchy_manipulator<P> const &hier, moments_list const &mlist,
             build_term_func<P> build, iter_solve_func<P> iter_solve, moment_id const m0);
  /*!
  * \brief Given the wavelet representation of the density, find the electric field also in wavelet space
  */
  void solve(std::vector<P> const &density, momentset<P> &moms,
             sparse_grid const &position_grid, connection_patterns const &conn, 
             kronmult::workspace<P> &work, poisson_bc const bc);
  //! poisson solve using periodic boundary conditions
  void solve_periodic(std::vector<P> const &density, momentset<P> &moms,
                      sparse_grid const &position_grid, connection_patterns const &conn,
                      kronmult::workspace<P> &work)
  {
    solve(density, moms, position_grid, conn, work, poisson_bc::periodic);
  }

  //! indicates whether the solver has been initialized
  operator bool() const { return (num_dims >= 0); }

  //! returns the id for the zero moment
  moment_id const &moment0() const { return mom0; }

private:
  #ifdef ASGARD_USE_GPU
  void apply_terms_gpu(int const pdof, sparse_grid const &grid, connection_patterns const &conn,
  P alpha, P const x[], P beta, P y[]);
  #endif

  // Solves for just the electric potential, used as a substep inside the solver
  void solve_potential_(std::vector<P> const &density, sparse_grid const &grid,
                        connection_patterns const &conn, kronmult::workspace<P> &work, poisson_bc const bc);

  int num_dims = -1;
  int pdof = -1; 
  moment_id mom0 = moment_id{-1};
  std::array<moment_id, max_pos_dims> moms_electric;
  std::vector<term_entry<P>> laplacian_terms;
  block_sparse_matrix<P> derivative_mat;
  std::array<P, max_pos_dims> dim_scalings;
  std::vector<P> rhs;
  std::vector<P> potential;
  iter_solve_func<P> iter_solve;
  #ifdef ASGARD_USE_GPU
  gpu::vector<P> d_rhs;
  gpu::vector<P> d_efield;
  mutable std::array<gpu::vector<P>, max_num_gpus> gpu_x, gpu_y; // for out-of-core evals
  #endif
};

/*!
 * \brief Stores the data for a 1D Poisson solver
 *
 * Holds the domain size, the factor of the operator matrices, etc.
 */
template<typename P>
class poisson_1d
{
public:
  //! default, uninitialized constructor
  poisson_1d() = default;
  //! initialize Poisson solver over the domain with given min/max, level and degree of input basis
  poisson_1d(int pdegree, P domain_min, P domain_max, int level, moment_id m0, moment_id melectric)
    : pdof(pdegree + 1), xmin(domain_min), xmax(domain_max), current_level(level), mom0(m0), mom_electric(melectric)
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

  //! resize the vector the current efield size
  void resize_vector(std::vector<P> &eflield) {
    eflield.resize(fm::ipow2(current_level));
  }
  //! returns the id for the zero moment
  moment_id const &moment0() const { return mom0; }
  //! returns the id for the electric field moment
  moment_id const &moment_electric() const { return mom_electric; };

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

    compute->pttrf(diag, subdiag);
  }

  int pdof = -1;
  P xmin = 0, xmax = 0;
  int current_level = -1;
  moment_id mom0 = moment_id::unset();
  moment_id mom_electric = moment_id::unset();
  std::vector<P> diag, subdiag, rhs;
};

} // asgard::poisson