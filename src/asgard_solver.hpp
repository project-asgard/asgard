#pragma once
#include "asgard_batch.hpp"
#include "asgard_kron_operators.hpp"

namespace asgard::solver
{
enum class poisson_bc
{
  dirichlet,
  periodic
};

inline bool is_direct(solve_opts s)
{
  return (s == solve_opts::direct);
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
bicgstab(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
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
class poisson_data
{
public:
  //! initialize Poisson solver over the domain with given min/max, level and degree of input basis
  poisson_data(int pdegree, P domain_min, P domain_max, int level)
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

  int degree;
  P xmin, xmax;
  int current_level;
  std::vector<P> diag, subdiag, rhs;
};

template<typename P>
void setup_poisson(const int N_nodes, P const x_min, P const x_max,
                   fk::vector<P> &diag, fk::vector<P> &off_diag);

template<typename P>
void poisson_solver(fk::vector<P> const &source, fk::vector<P> const &A_D,
                    fk::vector<P> const &A_E, fk::vector<P> &phi,
                    fk::vector<P> &E, int const degree, int const N_elements,
                    P const x_min, P const x_max, P const phi_min,
                    P const phi_max, poisson_bc const bc);

} // namespace asgard::solver
