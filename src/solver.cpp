#include "solver.hpp"
#include "distribution.hpp"
#include "fast_math.hpp"
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

template<typename P, resource resrc>
gmres_info<P>
simple_gmres_euler(const P dt, kronmult_matrix<P> const &mat,
                   fk::vector<P, mem_type::owner, resrc> &x,
                   fk::vector<P, mem_type::owner, resrc> const &b,
                   int const restart, int const max_iter, P const tolerance)
{
  return simple_gmres(
      [&](fk::vector<P, mem_type::owner, resrc> const &x_in,
          fk::vector<P, mem_type::owner, resrc> &y, P const alpha,
          P const beta) -> void {
        mat.template apply<resrc>(-dt * alpha, x_in.data(), beta, y.data());
        int one = 1, n = y.size();
        lib_dispatch::axpy<resrc>(n, alpha, x_in.data(), one, y.data(), one);
      },
      x, b, fk::matrix<P>(), restart, max_iter, tolerance);
}
/*! Generates a default number inner iterations when no use input is given
 * \param num_cols Number of columns in the A matrix.
 * \returns default number of iterations before restart
 */
template<typename P>
int default_gmres_restarts(int num_cols)
{
  // at least 10 iterations before restart but not more than num_cols
  int minimum = std::min(10, num_cols);
  // No more than 200 iterations before restart but not more than num_cols
  int maximum = std::min(200, num_cols);
  // Don't go over 512 MB.
  return std::clamp(static_cast<int>(512. / get_MB<P>(num_cols)), minimum,
                    maximum);
}

static int pos_from_indices(int i, int j) { return i + j * (j + 1) / 2; }

// simple, node-local test version
template<typename P, typename matrix_replacement, resource resrc>
gmres_info<P>
simple_gmres(matrix_replacement mat, fk::vector<P, mem_type::owner, resrc> &x,
             fk::vector<P, mem_type::owner, resrc> const &b,
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

  fk::vector<P, mem_type::owner, resrc> residual(b);
  auto const compute_residual = [&]() {
    P const alpha = -1.0;
    P const beta  = 1.0;
    residual      = b;
    mat(x, residual, alpha, beta);
    if (do_precond)
    {
      if constexpr (resrc == resource::device)
      {
#ifdef ASGARD_USE_CUDA
        static_assert(resrc == resource::device);
        auto res = residual.clone_onto_host();
        precond_factored ? fm::getrs(precond, res, precond_pivots)
                         : fm::gesv(precond, res, precond_pivots);
        fk::copy_vector(residual, res);
        precond_factored = true;
#endif
      }
      else if constexpr (resrc == resource::host)
      {
        precond_factored ? fm::getrs(precond, residual, precond_pivots)
                         : fm::gesv(precond, residual, precond_pivots);
      }
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

  fk::matrix<P, mem_type::owner, resrc> basis(n, restart + 1);
  fk::vector<P> krylov_proj(restart * (restart + 1) / 2);
  fk::vector<P> sines(restart + 1);
  fk::vector<P> cosines(restart + 1);

  int it = 0;
  int i  = 0;
  for (; it < max_iter; ++it)
  {
    P const norm_r = compute_residual();

    auto scaled = residual;
    scaled.scale(1. / norm_r);
    basis.update_col(0, scaled);

    fk::vector<P> krylov_sol(n + 1);
    krylov_sol(0) = norm_r;
    for (i = 0; i < restart; ++i)
    {
      fk::vector<P, mem_type::owner, resrc> tmp(
          fk::vector<P, mem_type::view, resrc>(basis, i, 0, basis.nrows() - 1));
      fk::vector<P, mem_type::owner, resrc> new_basis(tmp.size());
      mat(tmp, new_basis, P{1.0}, P{0.0});

      if (do_precond)
      {
        if constexpr (resrc == resource::device)
        {
#ifdef ASGARD_USE_CUDA
          static_assert(resrc == resource::device);
          auto new_basis_h = new_basis.clone_onto_host();
          fm::getrs(precond, new_basis_h, precond_pivots);
          fk::copy_vector(new_basis, new_basis_h);
#endif
        }
        else if constexpr (resrc == resource::host)
        {
          fm::getrs(precond, new_basis, precond_pivots);
        }
      }

      fk::matrix<P, mem_type::const_view, resrc> basis_v(basis, 0, n - 1, 0, i);
      fk::vector<P, mem_type::view> coeffs(krylov_proj, pos_from_indices(0, i),
                                           pos_from_indices(i, i));
      if constexpr (resrc == resource::device)
      {
#ifdef ASGARD_USE_CUDA
        static_assert(resrc == resource::device);
        auto coeffs_d = coeffs.clone_onto_device();
        fm::gemv(basis_v, new_basis, coeffs_d, true, P{1.0}, P{0.0});
        fm::gemv(basis_v, coeffs_d, new_basis, false, P{-1.0}, P{1.0});
        fk::copy_vector(coeffs, coeffs_d);
#endif
      }
      else if constexpr (resrc == resource::host)
      {
        fm::gemv(basis_v, new_basis, coeffs, true, P{1.0}, P{0.0});
        fm::gemv(basis_v, coeffs, new_basis, false, P{-1.0}, P{1.0});
      }
      auto nrm = fm::nrm2(new_basis);

      basis.update_col(i + 1, new_basis.scale(1 / nrm));
      for (int k = 0; k < i; ++k)
      {
        lib_dispatch::rot(1, coeffs.data(k), 1, coeffs.data(k + 1), 1,
                          cosines[k], sines[k]);
      }

      // compute given's rotation
      lib_dispatch::rotg(coeffs.data(i), &nrm, cosines.data(i), sines.data(i));

      P const temp      = cosines(i) * krylov_sol(i);
      krylov_sol(i + 1) = -sines(i) * krylov_sol(i);
      krylov_sol(i)     = temp;
      error             = std::abs(krylov_sol(i + 1)) / norm_b;

      if (error <= tolerance)
      {
        auto proj   = fk::vector<P, mem_type::view>(krylov_proj, 0,
                                                  pos_from_indices(i, i));
        auto s_view = fk::vector<P, mem_type::view>(krylov_sol, 0, i);
        fm::tpsv(proj, s_view);
        fk::matrix<P, mem_type::view, resrc> m(basis, 0, basis.nrows() - 1, 0,
                                               i);
        if constexpr (resrc == resource::device)
          fm::gemv(m, s_view.clone_onto_device(), x, false, P{1.0}, P{1.0});
        else if constexpr (resrc == resource::host)
          fm::gemv(m, s_view, x, false, P{1.0}, P{1.0});
        break; // depart the inner iteration loop
      }
    } // end of inner iteration loop

    if (error <= tolerance)
    {
      return done(error, it, i); // all done!
    }

    auto proj = fk::vector<P, mem_type::view>(
        krylov_proj, 0, pos_from_indices(restart - 1, restart - 1));
    auto s_view = fk::vector<P, mem_type::view>(krylov_sol, 0, restart - 1);
    fm::tpsv(proj, s_view);

    fk::matrix<P, mem_type::view, resrc> m(basis, 0, basis.nrows() - 1, 0,
                                           restart - 1);
    if constexpr (resrc == resource::device)
      fm::gemv(m, s_view.clone_onto_device(), x, false, P{1.0}, P{1.0});
    else if constexpr (resrc == resource::host)
      fm::gemv(m, s_view, x, false, P{1.0}, P{1.0});
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

#ifdef ASGARD_ENABLE_DOUBLE

template gmres_info<double>
simple_gmres(fk::matrix<double> const &A, fk::vector<double> &x,
             fk::vector<double> const &b, fk::matrix<double> const &M,
             int const restart, int const max_iter, double const tolerance);

template gmres_info<double>
simple_gmres_euler(const double dt, kronmult_matrix<double> const &mat,
                   fk::vector<double> &x, fk::vector<double> const &b,
                   int const restart, int const max_iter,
                   double const tolerance);

template int default_gmres_restarts<double>(int num_cols);

#ifdef ASGARD_USE_CUDA
template gmres_info<double> simple_gmres_euler(
    const double dt, kronmult_matrix<double> const &mat,
    fk::vector<double, mem_type::owner, resource::device> &x,
    fk::vector<double, mem_type::owner, resource::device> const &b,
    int const restart, int const max_iter, double const tolerance);
#endif
template void setup_poisson(const int N_elements, double const x_min,
                            double const x_max, fk::vector<double> &diag,
                            fk::vector<double> &off_diag);

template void
poisson_solver(fk::vector<double> const &source, fk::vector<double> const &A_D,
               fk::vector<double> const &A_E, fk::vector<double> &phi,
               fk::vector<double> &E, int const degree, int const N_elements,
               double const x_min, double const x_max, double const phi_min,
               double const phi_max, poisson_bc const bc);

#endif

#ifdef ASGARD_ENABLE_FLOAT

template gmres_info<float>
simple_gmres(fk::matrix<float> const &A, fk::vector<float> &x,
             fk::vector<float> const &b, fk::matrix<float> const &M,
             int const restart, int const max_iter, float const tolerance);

template gmres_info<float>
simple_gmres_euler(const float dt, kronmult_matrix<float> const &mat,
                   fk::vector<float> &x, fk::vector<float> const &b,
                   int const restart, int const max_iter,
                   float const tolerance);

template int default_gmres_restarts<float>(int num_cols);

#ifdef ASGARD_USE_CUDA
template gmres_info<float> simple_gmres_euler(
    const float dt, kronmult_matrix<float> const &mat,
    fk::vector<float, mem_type::owner, resource::device> &x,
    fk::vector<float, mem_type::owner, resource::device> const &b,
    int const restart, int const max_iter, float const tolerance);
#endif
template void setup_poisson(const int N_elements, float const x_min,
                            float const x_max, fk::vector<float> &diag,
                            fk::vector<float> &off_diag);

template void
poisson_solver(fk::vector<float> const &source, fk::vector<float> const &A_D,
               fk::vector<float> const &A_E, fk::vector<float> &phi,
               fk::vector<float> &E, int const degree, int const N_elements,
               float const x_min, float const x_max, float const phi_min,
               float const phi_max, poisson_bc const bc);

#endif

} // namespace asgard::solver
