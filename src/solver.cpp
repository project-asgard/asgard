#include "solver.hpp"
#include "distribution.hpp"
#include "fast_math.hpp"
#include "kronmult.hpp"
#include "quadrature.hpp"
#include "tools.hpp"
#include <algorithm>
#include <stdexcept>

namespace asgard::solver
{
// simple, node-local test version
template<typename P>
gmres_info<P>
simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
             fk::matrix<P> const &M, int const restart, int const max_iter,
             P const tolerance)
{
  auto dense_matrix_wrapper = [&A](fk::vector<P> const &x_in, fk::vector<P> &y,
                                   P const alpha = 1.0, P const beta = 0.0) {
    bool const trans_A = false;
    fm::gemv(A, x_in, y, trans_A, alpha, beta);
  };
  return simple_gmres(dense_matrix_wrapper, x, b, M, restart, max_iter,
                      tolerance);
}

template<typename P>
gmres_info<P>
simple_gmres(PDE<P> const &pde, elements::table const &elem_table,
             options const &program_options, element_subgrid const &my_subgrid,
             fk::vector<P> &x, fk::vector<P> const &b, fk::matrix<P> const &M,
             int const restart, int const max_iter, P const tolerance,
             imex_flag const imex, P const dt_factor)
{
  auto euler_operator = [&pde, &elem_table, &program_options, &my_subgrid, imex,
                         dt_factor](fk::vector<P> const &x_in, fk::vector<P> &y,
                                    P const alpha = 1.0, P const beta = 0.0) {
    auto tmp = kronmult::execute(pde, elem_table, program_options, my_subgrid,
                                 x_in, imex);
    tmp      = x_in - tmp * pde.get_dt() * dt_factor;
    y        = tmp * alpha + y * beta;
  };
  return simple_gmres(euler_operator, x, b, M, restart, max_iter, tolerance);
}

/*! Generates a default number inner iterations when no use input is given
 * \param num_cols Number of columns in the A matrix.
 * \returns default number of iterations before restart
 */
template<typename P>
static int default_gmres_restarts(int num_cols)
{
  // at least 10 iterations before restart but not more than num_cols
  int minimum = std::min(10, num_cols);
  // No more than 200 iterations before restart but not more than num_cols
  int maximum = std::min(200, num_cols);
  // Don't go over 512 MB.
  return std::clamp(static_cast<int>(512. / get_MB<P>(num_cols)), minimum,
                    maximum);
}

// simple, node-local test version
template<typename P, typename matrix_replacement>
gmres_info<P>
simple_gmres(matrix_replacement mat, fk::vector<P> &x, fk::vector<P> const &b,
             fk::matrix<P> const &M, int restart, int max_iter, P tolerance)
{
  if (tolerance == parser::NO_USER_VALUE_FP)
    tolerance = std::is_same_v<float, P> ? 1e-6 : 1e-12;
  expect(tolerance >= std::numeric_limits<P>::epsilon());
  int const n = b.size();
  expect(n == x.size());

  bool const do_precond = M.size() > 0;
  std::vector<int> precond_pivots(n);
  if (do_precond)
  {
    expect(M.ncols() == n);
    expect(M.nrows() == n);
  }
  fk::matrix<P> precond(M);
  bool precond_factored = false;

  if (restart == parser::NO_USER_VALUE)
    restart = default_gmres_restarts<P>(n);
  expect(restart > 0); // checked in program_options
  if (restart > n)
  {
    std::ostringstream err_msg;
    err_msg << "Number of inner iterations " << restart << " must be less than "
            << n << "!";
    throw std::invalid_argument(err_msg.str());
  }
  if (max_iter == parser::NO_USER_VALUE)
    max_iter = n;
  if (max_iter < restart)
  {
    std::ostringstream err_msg;
    err_msg << "Number of outer iterations " << max_iter
            << " must be greater than " << restart << "!";
    throw std::invalid_argument(err_msg.str());
  }
  if (max_iter > n)
  {
    std::ostringstream err_msg;
    err_msg << "Number of outer iterations " << max_iter
            << " must be less than " << n << "!";
    throw std::invalid_argument(err_msg.str());
  }
  P const norm_b = [&b]() {
    P const norm = fm::nrm2(b);
    return (norm == 0.0) ? static_cast<P>(1.0) : norm;
  }();

  fk::vector<P> residual(b);
  auto const compute_residual = [&]() {
    P const alpha = -1.0;
    P const beta  = 1.0;
    residual      = b;
    mat(x, residual, alpha, beta);
    if (do_precond)
    {
      precond_factored ? fm::getrs(precond, residual, precond_pivots)
                       : fm::gesv(precond, residual, precond_pivots);
      precond_factored = true;
    }
    return fm::nrm2(residual);
  };

  auto const done = [](P const error, int const outer_iters,
                       int const inner_iters) -> gmres_info<P> {
    std::cout << "GMRES complete with error: " << error << '\n';
    std::cout << outer_iters << " outer iterations, " << inner_iters
              << " inner iterations\n";
    return gmres_info<P>{error, outer_iters, inner_iters};
  };

  P error = compute_residual() / norm_b;
  if (error < tolerance)
  {
    return done(error, 0, 0);
  }

  fk::matrix<P> basis(n, restart + 1);
  fk::matrix<P> krylov_proj(restart + 1, restart);
  fk::vector<P> sines(restart + 1);
  fk::vector<P> cosines(restart + 1);

  int it = 0;
  int i  = 0;
  for (; it < max_iter; ++it)
  {
    P const norm_r = compute_residual();

    basis.update_col(0, residual * (1 / norm_r));

    fk::vector<P> krylov_sol(n + 1);
    krylov_sol(0) = norm_r;
    for (i = 0; i < restart; ++i)
    {
      auto tmp = fk::vector<P>(
          fk::vector<P, mem_type::view>(basis, i, 0, basis.nrows() - 1));
      fk::vector<P> new_basis(tmp.size());
      mat(tmp, new_basis, P{1.0}, P{0.0});

      if (do_precond)
      {
        fm::getrs(precond, new_basis, precond_pivots);
      }

      for (int k = 0; k <= i; ++k)
      {
        fk::vector<P, mem_type::const_view> const basis_vect(basis, k, 0,
                                                             basis.nrows() - 1);
        krylov_proj(k, i) = new_basis * basis_vect;
        new_basis         = new_basis - (basis_vect * krylov_proj(k, i));
      }
      krylov_proj(i + 1, i) = fm::nrm2(new_basis);

      basis.update_col(i + 1, new_basis * (1 / krylov_proj(i + 1, i)));
      for (int k = 0; k <= i - 1; ++k)
      {
        P const temp =
            cosines(k) * krylov_proj(k, i) + sines(k) * krylov_proj(k + 1, i);
        krylov_proj(k + 1, i) =
            -sines(k) * krylov_proj(k, i) + cosines(k) * krylov_proj(k + 1, i);
        krylov_proj(k, i) = temp;
      }

      // compute given's rotation
      lib_dispatch::rotg(krylov_proj.data(i, i), krylov_proj.data(i + 1, i),
                         cosines.data(i), sines.data(i));

      krylov_proj(i + 1, i) = 0.0;
      P const temp          = cosines(i) * krylov_sol(i);
      krylov_sol(i + 1)     = -sines(i) * krylov_sol(i);
      krylov_sol(i)         = temp;
      error                 = std::abs(krylov_sol(i + 1)) / norm_b;

      if (error <= tolerance)
      {
        auto proj = fk::matrix<P, mem_type::view>(krylov_proj, 0, i, 0, i);
        std::vector<int> pivots(i + 1);

        auto s_view = fk::vector<P, mem_type::view>(krylov_sol, 0, i);
        fm::gesv(proj, s_view, pivots);
        x = x +
            (fk::matrix<P, mem_type::view>(basis, 0, basis.nrows() - 1, 0, i) *
             s_view);
        break; // depart the inner iteration loop
      }
    } // end of inner iteration loop

    if (error <= tolerance)
    {
      return done(error, it, i); // all done!
    }
    auto proj   = fk::matrix<P, mem_type::view>(krylov_proj, 0, restart - 1, 0,
                                              restart - 1);
    auto s_view = fk::vector<P, mem_type::view>(krylov_sol, 0, restart - 1);
    std::vector<int> pivots(restart);
    fm::gesv(proj, s_view, pivots);
    x = x + (fk::matrix<P, mem_type::view>(basis, 0, basis.nrows() - 1, 0,
                                           restart - 1) *
             s_view);
    P const norm_r_outer                               = compute_residual();
    krylov_sol(std::min(krylov_sol.size() - 1, i + 1)) = norm_r_outer;
    error                                              = norm_r_outer / norm_b;

    if (error <= tolerance)
    {
      return done(error, it, i);
    }
  } // end outer iteration

  return done(error, it, i);
}

template<typename P>
void setup_poisson(const int N_elements, P const x_min, P const x_max,
                   fk::vector<P> &diag, fk::vector<P> &off_diag)
{
  // sets up and factorizes the matrix to use in the poisson solver
  const P dx = (x_max - x_min) / static_cast<P>(N_elements);

  const int N_nodes = N_elements - 1;

  diag.resize(N_nodes);
  off_diag.resize(N_nodes - 1);

  for (int i = 0; i < N_nodes; ++i)
  {
    diag[i] = 2.0 / dx;
  }

  for (int i = 0; i < N_nodes - 1; ++i)
  {
    off_diag[i] = -1.0 / dx;
  }

  fm::pttrf(diag, off_diag);
}

template<typename P>
void poisson_solver(fk::vector<P> const &source, fk::vector<P> const &A_D,
                    fk::vector<P> const &A_E, fk::vector<P> &phi,
                    fk::vector<P> &E, int const degree, int const N_elements,
                    P const x_min, P const x_max, P const phi_min,
                    P const phi_max, poisson_bc const bc)
{
  tools::timer.start("poisson_solver");
  // Solving: - phi_xx = source Using Linear Finite Elements
  // Boundary Conditions: phi(x_min)=phi_min and phi(x_max)=phi_max
  // Returns phi and E = - Phi_x in Gauss-Legendre Nodes

  P const dx = (x_max - x_min) / static_cast<P>(N_elements);

  auto const lgwt = legendre_weights<P>(degree + 1, -1.0, 1.0, true);

  int N_nodes = N_elements - 1;

  // Average the Source Vector (if Periodic) //
  double ave_source = 0.0;
  if (bc == poisson_bc::periodic)
  {
    for (int i = 0; i < N_elements; i++)
    {
      for (int q = 0; q < degree + 1; q++)
      {
        ave_source += 0.5 * dx * lgwt[1][q] * source[i * (degree + 1) + q];
      }
    }
    ave_source /= (x_max - x_min);
  }

  // Set the Source Vector //
  fk::vector<P> b(N_nodes);
  for (int i = 0; i < N_nodes; i++)
  {
    b[i] = 0.0;
    for (int q = 0; q < degree + 1; q++)
    {
      b[i] += 0.25 * dx * lgwt[1][q] *
              (source[(i) * (degree + 1) + q] * (1.0 + lgwt[0][q]) +
               source[(i + 1) * (degree + 1) + q] * (1.0 - lgwt[0][q]) -
               2.0 * ave_source);
    }
  }

  // Linear Solve //
  fm::pttrs(A_D, A_E, b);

  // Set Potential and Electric Field in DG Nodes //
  P const dg = (phi_max - phi_min) / (x_max - x_min);

  // First Element //
  for (int k = 0; k < degree + 1; k++)
  {
    P const x_k = x_min + 0.5 * dx * (1.0 + lgwt[0][k]);
    P const g_k = phi_min + dg * (x_k - x_min);

    phi[k] = 0.5 * b[0] * (1.0 + lgwt[0][k]) + g_k;

    E[k] = -b[0] / dx - dg;
  }

  // Interior Elements //
  for (int i = 1; i < N_elements - 1; i++)
  {
    for (int q = 0; q < degree + 1; q++)
    {
      int const k = i * (degree + 1) + q;
      P const x_k = (x_min + i * dx) + 0.5 * dx * (1.0 + lgwt[0][q]);
      P const g_k = phi_min + dg * (x_k - x_min);

      phi[k] =
          0.5 * (b[i - 1] * (1.0 - lgwt[0][q]) + b[i] * (1.0 + lgwt[0][q])) +
          g_k;

      E[k] = -(b[i] - b[i - 1]) / dx - dg;
    }
  }

  // Last Element //
  int const i = N_elements - 1;
  for (int q = 0; q < degree + 1; q++)
  {
    int const k = i * (degree + 1) + q;
    P const x_k = (x_min + i * dx) + 0.5 * dx * (1.0 + lgwt[0][q]);
    P const g_k = phi_min + dg * (x_k - x_min);

    phi[k] = 0.5 * b[i - 1] * (1.0 - lgwt[0][q]) + g_k;

    E[k] = b[i - 1] / dx - dg;
  }
  tools::timer.stop("poisson_solver");
}

template gmres_info<float>
simple_gmres(fk::matrix<float> const &A, fk::vector<float> &x,
             fk::vector<float> const &b, fk::matrix<float> const &M,
             int const restart, int const max_iter, float const tolerance);

template gmres_info<double>
simple_gmres(fk::matrix<double> const &A, fk::vector<double> &x,
             fk::vector<double> const &b, fk::matrix<double> const &M,
             int const restart, int const max_iter, double const tolerance);

template gmres_info<float>
simple_gmres(PDE<float> const &pde, elements::table const &elem_table,
             options const &program_options, element_subgrid const &my_subgrid,
             fk::vector<float> &x, fk::vector<float> const &b,
             fk::matrix<float> const &M, int const restart, int const max_iter,
             float const tolerance, imex_flag const imex,
             const float dt_factor);

template gmres_info<double>
simple_gmres(PDE<double> const &pde, elements::table const &elem_table,
             options const &program_options, element_subgrid const &my_subgrid,
             fk::vector<double> &x, fk::vector<double> const &b,
             fk::matrix<double> const &M, int const restart, int const max_iter,
             double const tolerance, imex_flag const imex,
             const double dt_factor);

template void setup_poisson(const int N_elements, float const x_min,
                            float const x_max, fk::vector<float> &diag,
                            fk::vector<float> &off_diag);
template void setup_poisson(const int N_elements, double const x_min,
                            double const x_max, fk::vector<double> &diag,
                            fk::vector<double> &off_diag);

template void
poisson_solver(fk::vector<float> const &source, fk::vector<float> const &A_D,
               fk::vector<float> const &A_E, fk::vector<float> &phi,
               fk::vector<float> &E, int const degree, int const N_elements,
               float const x_min, float const x_max, float const phi_min,
               float const phi_max, poisson_bc const bc);
template void
poisson_solver(fk::vector<double> const &source, fk::vector<double> const &A_D,
               fk::vector<double> const &A_E, fk::vector<double> &phi,
               fk::vector<double> &E, int const degree, int const N_elements,
               double const x_min, double const x_max, double const phi_min,
               double const phi_max, poisson_bc const bc);

} // namespace asgard::solver
