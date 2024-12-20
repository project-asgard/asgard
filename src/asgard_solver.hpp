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
struct poisson_data
{
// Updating the Poisson logic
// 1. Have a new type of term, mass that depends on E-pos and E-neg
// 2. Split the term construction, update that term later (chains?)
//    - construct from operators that assume constant over domain
//    - relate the cell number to the index of the value
// 3. Do poisson solve using Fi, update only the poisson terms

poisson_data(P domain_min, P domain_max, int level)
  : xmin(domain_min), xmax(domain_max), current_level(level)
{
  if (current_level == 0) return; // nothing to solve

  remake_factors();
}
void update_level(int new_level) {
  current_level = new_level;
  remake_factors();
}
void remake_factors()
{
  int const nnodes = fm::ipow2(current_level) - 1;
  P const dx = (xmax - xmin) / (nnodes + 1);

  diag = std::vector<P>(nnodes, P{2} / dx);
  subdiag = std::vector<P>(nnodes - 1, -P{1} / dx);

  fm::pttrf(diag, subdiag);
}

P xmin, xmax;
int current_level;
std::vector<P> diag, subdiag;
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
