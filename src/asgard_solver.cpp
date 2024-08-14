#include "asgard_solver.hpp"

namespace asgard::solver
{
template<typename P>
class dense_preconditioner
{
public:
  dense_preconditioner(fk::matrix<P> const &M)
      : precond(M), precond_pivots(M.ncols())
  {
    expect(static_cast<size_t>(M.nrows()) == precond_pivots.size());
    fm::getrf(precond, precond_pivots);
  }
  template<mem_type bmem>
  void operator()(fk::vector<P, bmem, resource::host> &b_h) const
  {
    fm::getrs(precond, b_h, precond_pivots);
  }
#ifdef ASGARD_USE_CUDA
  template<mem_type bmem>
  void operator()(fk::vector<P, bmem, resource::device> &b_d) const
  {
    auto b_h = b_d.clone_onto_host();
    fm::getrs(precond, b_h, precond_pivots);
    fk::copy_vector(b_d, b_h);
  }
#endif
private:
  fk::matrix<P> precond;
  std::vector<int> precond_pivots;
};

template<typename P>
class no_op_preconditioner
{
public:
  template<mem_type bmem>
  void operator()(fk::vector<P, bmem, resource::host> &) const
  {}
#ifdef ASGARD_USE_CUDA
  template<mem_type bmem>
  void operator()(fk::vector<P, bmem, resource::device> &) const
  {}
#endif
};

// simple, node-local test version
template<typename P>
gmres_info<P>
simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
             fk::matrix<P> const &M, int const restart, int const max_iter,
             P const tolerance)
{
  auto dense_matrix_wrapper =
      [&A](P const alpha, fk::vector<P, mem_type::view> const x_in,
           P const beta, fk::vector<P, mem_type::view> y) {
        fm::gemv(A, x_in, y, false, alpha, beta);
      };
  if (M.size() > 0)
    return simple_gmres(dense_matrix_wrapper, fk::vector<P, mem_type::view>(x),
                        b, dense_preconditioner(M), restart, max_iter,
                        tolerance);
  else
    return simple_gmres(dense_matrix_wrapper, fk::vector<P, mem_type::view>(x),
                        b, no_op_preconditioner<P>(), restart, max_iter,
                        tolerance);
}

// simple, node-local test version
template<typename P>
gmres_info<P>
bicgstab(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
         fk::matrix<P> const &M, int const max_iter,
         P const tolerance)
{
  auto dense_matrix_wrapper =
      [&A](P const alpha, fk::vector<P, mem_type::view> const x_in,
           P const beta, fk::vector<P, mem_type::view> y) {
        fm::gemv(A, x_in, y, false, alpha, beta);
      };
  if (M.size() > 0)
    return bicgstab(dense_matrix_wrapper, fk::vector<P, mem_type::view>(x),
                    b, dense_preconditioner(M), max_iter,
                    tolerance);
  else
    return bicgstab(dense_matrix_wrapper, fk::vector<P, mem_type::view>(x),
                    b, no_op_preconditioner<P>(), max_iter,
                    tolerance);
}

// preconditiner is only available in global mode
template<typename P>
void apply_diagonal_precond(std::vector<P> const &pc, P dt,
                            fk::vector<P, mem_type::view, resource::host> &x)
{
#pragma omp parallel for
  for (size_t i = 0; i < pc.size(); i++)
    x[i] /= (1.0 - dt * pc[i]);
}
#ifdef ASGARD_USE_CUDA
template<typename P>
void apply_diagonal_precond(gpu::vector<P> const &pc, P dt,
                            fk::vector<P, mem_type::view, resource::device> &x)
{
  kronmult::gpu_precon_jacobi(pc.size(), dt, pc.data(), x.data());
}
#endif

template<typename P, resource resrc>
gmres_info<P>
simple_gmres_euler(const P dt, imex_flag imex,
                   kron_operators<P> const &ops,
                   fk::vector<P, mem_type::owner, resrc> &x,
                   fk::vector<P, mem_type::owner, resrc> const &b,
                   int const restart, int const max_iter, P const tolerance)
{
  auto const &pc = ops.template get_diagonal_preconditioner<resrc>();

  return simple_gmres(
      [&](P const alpha, fk::vector<P, mem_type::view, resrc> const x_in,
          P const beta, fk::vector<P, mem_type::view, resrc> y) -> void {
        tools::time_event performance("kronmult - implicit", ops.flops(imex));
        ops.template apply<resrc>(imex, -dt * alpha, x_in.data(), beta, y.data());
        lib_dispatch::axpy<resrc>(y.size(), alpha, x_in.data(), 1, y.data(), 1);
      },
      fk::vector<P, mem_type::view, resrc>(x), b,
      [&](fk::vector<P, mem_type::view, resrc> &x_in) -> void {
        tools::time_event performance("kronmult - preconditioner", pc.size());
        apply_diagonal_precond(pc, dt, x_in);
      },
      restart, max_iter, tolerance, ops.verbosity);
}
template<typename P, resource resrc>
gmres_info<P>
bicgstab_euler(const P dt, imex_flag imex,
               kron_operators<P> const &ops,
               fk::vector<P, mem_type::owner, resrc> &x,
               fk::vector<P, mem_type::owner, resrc> const &b,
               int const max_iter, P const tolerance)
{
  auto const &pc = ops.template get_diagonal_preconditioner<resrc>();

  return bicgstab(
    [&](P const alpha, fk::vector<P, mem_type::view, resrc> const x_in,
          P const beta, fk::vector<P, mem_type::view, resrc> y) -> void {
        tools::time_event performance("kronmult - implicit", ops.flops(imex));
        ops.template apply<resrc>(imex, -dt * alpha, x_in.data(), beta, y.data());
        lib_dispatch::axpy<resrc>(y.size(), alpha, x_in.data(), 1, y.data(), 1);
      },
      fk::vector<P, mem_type::view, resrc>(x), b,
      [&](fk::vector<P, mem_type::view, resrc> &x_in) -> void {
        tools::time_event performance("kronmult - preconditioner", pc.size());
        apply_diagonal_precond(pc, dt, x_in);
      }, max_iter, tolerance);
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
template<typename P, resource resrc, typename matrix_abstraction,
         typename preconditioner_abstraction>
gmres_info<P>
simple_gmres(matrix_abstraction mat, fk::vector<P, mem_type::view, resrc> x,
             fk::vector<P, mem_type::owner, resrc> const &b,
             preconditioner_abstraction precondition, int restart,
             int max_outer_iterations, P tolerance,
             verbosity_level verbosity = verbosity_level::high)
{
  if (tolerance <= notolerance + std::numeric_limits<P>::epsilon())
    tolerance = std::is_same_v<float, P> ? 1e-6 : 1e-12;
  expect(tolerance >= std::numeric_limits<P>::epsilon());

  int const n = b.size();
  expect(n == x.size());

  if (restart == novalue)
    restart = default_gmres_restarts<P>(n);
  expect(restart > 0); // checked in program_options

  if (max_outer_iterations == novalue)
    max_outer_iterations = n;
  expect(max_outer_iterations > 0); // checked in program_options

  // controls how often the inner residual print occurs
  int const print_freq = restart / 3;

  fk::matrix<P, mem_type::owner, resrc> basis(n, restart + 1);
  fk::vector<P> krylov_proj(restart * (restart + 1) / 2);
  fk::vector<P> sines(restart + 1);
  fk::vector<P> cosines(restart + 1);
  fk::vector<P> krylov_sol(restart + 1);

  int total_iterations = 0;
  int outer_iterations = 0;
  int inner_iterations = 0;

  P inner_res = 0.;
  P outer_res = tolerance + 1.;
  while ((outer_res > tolerance) && (outer_iterations < max_outer_iterations))
  {
    fk::vector<P, mem_type::view, resrc> scaled(basis, 0, 0, basis.nrows() - 1);
    scaled = b;
    mat(P{-1.}, x, P{1.}, scaled);
    precondition(scaled);
    ++total_iterations;

    inner_res = fm::nrm2(scaled);
    scaled.scale(P{1.} / inner_res);
    krylov_sol[0] = inner_res;

    inner_iterations = 0;
    while ((inner_res > tolerance) && (inner_iterations < restart))
    {
      fk::vector<P, mem_type::view, resrc> const tmp(basis, inner_iterations, 0,
                                                     basis.nrows() - 1);
      fk::vector<P, mem_type::view, resrc> new_basis(
          basis, inner_iterations + 1, 0, basis.nrows() - 1);
      mat(P{1.}, tmp, P{0.}, new_basis);
      precondition(new_basis);
      ++total_iterations;
      fk::matrix<P, mem_type::const_view, resrc> basis_v(basis, 0, n - 1, 0,
                                                         inner_iterations);
      fk::vector<P, mem_type::view> coeffs(
          krylov_proj, pos_from_indices(0, inner_iterations),
          pos_from_indices(inner_iterations, inner_iterations));
      if constexpr (resrc == resource::device)
      {
#ifdef ASGARD_USE_CUDA
        static_assert(resrc == resource::device);
        fk::vector<P, mem_type::owner, resrc> coeffs_d(coeffs.size());
        fm::gemv(basis_v, new_basis, coeffs_d, true, P{1.}, P{0.});
        fm::gemv(basis_v, coeffs_d, new_basis, false, P{-1.}, P{1.});
        fk::copy_vector(coeffs, coeffs_d);
#endif
      }
      else if constexpr (resrc == resource::host)
      {
        fm::gemv(basis_v, new_basis, coeffs, true, P{1.}, P{0.});
        fm::gemv(basis_v, coeffs, new_basis, false, P{-1.}, P{1.});
      }
      P const nrm = fm::nrm2(new_basis);
      new_basis.scale(P{1.} / nrm);
      for (int k = 0; k < inner_iterations; ++k)
      {
        lib_dispatch::rot(1, coeffs.data(k), 1, coeffs.data(k + 1), 1,
                          cosines[k], sines[k]);
      }

      // compute given's rotation
      P beta = nrm;
      lib_dispatch::rotg(coeffs.data(inner_iterations), &beta,
                         cosines.data(inner_iterations),
                         sines.data(inner_iterations));

      inner_res =
          std::abs(sines[inner_iterations] * krylov_sol[inner_iterations]);

      if ((inner_res > tolerance) && (inner_iterations < restart))
      {
        krylov_sol[inner_iterations + 1] = 0.;
        lib_dispatch::rot(1, krylov_sol.data(inner_iterations), 1,
                          krylov_sol.data(inner_iterations + 1), 1,
                          cosines[inner_iterations], sines[inner_iterations]);
      }

      if (inner_iterations % print_freq == 0 and verbosity == verbosity_level::high)
      {
        std::cout << "   -- GMRES inner iteration " << inner_iterations << " / "
                  << restart << " w/ residual " << inner_res << std::endl;
      }
      ++inner_iterations;
    } // end of inner iteration loop

    if (inner_iterations > 0)
    {
      auto proj = fk::vector<P, mem_type::view>(
          krylov_proj, 0,
          pos_from_indices(inner_iterations - 1, inner_iterations - 1));
      auto s_view =
          fk::vector<P, mem_type::view>(krylov_sol, 0, inner_iterations - 1);
      fm::tpsv(proj, s_view);
      fk::matrix<P, mem_type::view, resrc> m(basis, 0, basis.nrows() - 1, 0,
                                             inner_iterations - 1);
      if constexpr (resrc == resource::device)
        fm::gemv(m, s_view.clone_onto_device(), x, false, P{1.}, P{1.});
      else if constexpr (resrc == resource::host)
        fm::gemv(m, s_view, x, false, P{1.}, P{1.});
    }
    ++outer_iterations;
    outer_res = inner_res;
  } // end outer iteration
  if (verbosity == verbosity_level::high)
  {
    std::cout << "GMRES complete with error: " << outer_res << '\n';
    std::cout << total_iterations << " iterations\n";
  }
  return gmres_info<P>{outer_res, total_iterations};
}

//*****************************************************************
// Iterative template routine -- BiCGSTAB
//
// BiCGSTAB solves the unsymmetric linear system Ax = b
// using the Preconditioned BiConjugate Gradient Stabilized method
//
// BiCGSTAB follows the algorithm described on p. 27 of the
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//
//*****************************************************************
template<typename P, resource resrc, typename matrix_abstraction,
         typename preconditioner_abstraction>
gmres_info<P>
bicgstab(matrix_abstraction mat, fk::vector<P, mem_type::view, resrc> x,
         fk::vector<P, mem_type::owner, resrc> const &b,
         preconditioner_abstraction precondition,
         int max_iter, P tol)
{
  if (tol <= notolerance + std::numeric_limits<P>::epsilon())
    tol = std::is_same_v<float, P> ? 1e-6 : 1e-12;
  expect(tol >= std::numeric_limits<P>::epsilon());

  int const n = b.size();
  expect(n == x.size());

  if (max_iter == novalue)
    max_iter = n;
  expect(max_iter > 0); // checked in program_options

  fk::vector<P, mem_type::owner, resrc> p(n), phat(n), s(n), shat(n), t(n), v(n);

  P normb                                 = fm::nrm2(b);
  fk::vector<P, mem_type::owner, resrc> r = b;
  mat(P{-1.}, x, P{1.}, fk::vector<P, mem_type::view, resrc>(r));

  fk::vector<P, mem_type::owner, resrc> rtilde = r;

  if (normb == 0.)
    normb = 1.;

  P resid = fm::nrm2(r) / normb;
  if (resid <= tol)
  {
    return gmres_info<P>{resid, 0};
  }

  P rho_2 = 0;
  P alpha = 0;
  P omega = 0;
  for (int i = 1; i <= max_iter; i++)
  {
    P rho_1 = rtilde * r;
    if (rho_1 == 0)
    {
      throw std::runtime_error("BiCGSTAB method failed. rho_1 == 0");
    }
    if (i == 1)
    {
      p = r;
    }
    else
    {
      P const beta = (rho_1 / rho_2) * (alpha / omega);
      phat         = p;
      fm::axpy(v, phat, P{-1} * omega);
      p = r;
      fm::axpy(phat, p, beta);
    }
    phat = p;
    fk::vector<P, mem_type::view, resrc> phat_v(phat);
    precondition(phat_v);
    mat(P{1.}, phat_v, P{0.}, fk::vector<P, mem_type::view, resrc>(v));
    alpha = rho_1 / (rtilde * v);
    s     = r;
    fm::axpy(v, s, P{-1} * alpha);
    resid = fm::nrm2(s) / normb;
    if (resid < tol)
    {
      fm::axpy(phat, x, alpha);
      return gmres_info<P>{resid, i};
    }
    shat = s;
    fk::vector<P, mem_type::view, resrc> shat_v(shat);
    precondition(shat_v);
    mat(P{1.}, shat_v, P{0.}, fk::vector<P, mem_type::view, resrc>(t));
    omega = (t * s) / (t * t);
    fm::axpy(phat, x, alpha);
    fm::axpy(shat, x, omega);
    r = s;
    fm::axpy(t, r, P{-1} * omega);

    rho_2 = rho_1;
    resid = fm::nrm2(r) / normb;
    if (resid < tol)
    {
      return gmres_info<P>{resid, i};
    }
    if (omega == 0)
    {
      throw std::runtime_error("BiCGSTAB method failed. omega == 0");
    }
  }
  std::cerr << "Warning: No convergence within max_iter = " << max_iter << " iterations\n";
  return gmres_info<P>{resid, max_iter};
}

template<typename P>
void setup_poisson(const int N_elements, P const x_min, P const x_max,
                   fk::vector<P> &diag, fk::vector<P> &off_diag)
{
  // no need to solve on one element
  if (N_elements == 1)
  {
    return;
  }
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
  tools::time_event psolve_("poisson_solver");
  // Solving: - phi_xx = source Using Linear Finite Elements
  // Boundary Conditions: phi(x_min)=phi_min and phi(x_max)=phi_max
  // Returns phi and E = - Phi_x in Gauss-Legendre Nodes

  P const dx = (x_max - x_min) / static_cast<P>(N_elements);

  int const pdof = degree + 1;

  auto const lgwt =
      legendre_weights<P>(pdof, -1.0, 1.0, quadrature_mode::use_degree);

  // If only one element, skip poisson solve and solve via BCs
  if (N_elements == 1)
  {
    for (int k = 0; k < pdof + 1; k++)
    {
      P const x_k = x_min + 0.5 * dx * (1.0 + lgwt[0][k]);
      phi[k]      = ((phi_max - phi_min) / (x_max - x_min)) * (x_k - x_min) + x_min;
      E[k]        = -(phi_max - phi_min) / (x_max - x_min);
    }
    tools::timer.stop("poisson_solver");
    return;
  }

  int N_nodes = N_elements - 1;

  // Average the Source Vector (if Periodic) //
  double ave_source = 0.0;
  if (bc == poisson_bc::periodic)
  {
    for (int i = 0; i < N_elements; i++)
    {
      for (int q = 0; q < pdof + 1; q++)
      {
        ave_source += 0.5 * dx * lgwt[1][q] * source[i * (pdof + 1) + q];
      }
    }
    ave_source /= (x_max - x_min);
  }

  // Set the Source Vector //
  fk::vector<P> b(N_nodes);
  for (int i = 0; i < N_nodes; i++)
  {
    b[i] = 0.0;
    for (int q = 0; q < pdof + 1; q++)
    {
      b[i] += 0.25 * dx * lgwt[1][q] *
              (source[(i) * (pdof + 1) + q] * (1.0 + lgwt[0][q]) +
               source[(i + 1) * (pdof + 1) + q] * (1.0 - lgwt[0][q]) -
               2.0 * ave_source);
    }
  }

  // Linear Solve //
  fm::pttrs(A_D, A_E, b);

  // Set Potential and Electric Field in DG Nodes //
  P const dg = (phi_max - phi_min) / (x_max - x_min);

  // First Element //
  for (int k = 0; k < pdof + 1; k++)
  {
    P const x_k = x_min + 0.5 * dx * (1.0 + lgwt[0][k]);
    P const g_k = phi_min + dg * (x_k - x_min);

    phi[k] = 0.5 * b[0] * (1.0 + lgwt[0][k]) + g_k;

    E[k] = -b[0] / dx - dg;
  }

  // Interior Elements //
  for (int i = 1; i < N_elements - 1; i++)
  {
    for (int q = 0; q < pdof + 1; q++)
    {
      int const k = i * (pdof + 1) + q;
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
  for (int q = 0; q < pdof + 1; q++)
  {
    int const k = i * (pdof + 1) + q;
    P const x_k = (x_min + i * dx) + 0.5 * dx * (1.0 + lgwt[0][q]);
    P const g_k = phi_min + dg * (x_k - x_min);

    phi[k] = 0.5 * b[i - 1] * (1.0 - lgwt[0][q]) + g_k;

    E[k] = b[i - 1] / dx - dg;
  }
}

#ifdef ASGARD_ENABLE_DOUBLE

template gmres_info<double>
simple_gmres(fk::matrix<double> const &A, fk::vector<double> &x,
             fk::vector<double> const &b, fk::matrix<double> const &M,
             int const restart, int const max_iter, double const tolerance);
template gmres_info<double>
bicgstab(fk::matrix<double> const &A, fk::vector<double> &x,
         fk::vector<double> const &b, fk::matrix<double> const &M,
         int const max_iter, double const tolerance);

template gmres_info<double>
simple_gmres_euler(const double dt, imex_flag imex,
                   kron_operators<double> const &ops,
                   fk::vector<double, mem_type::owner, resource::host> &x,
                   fk::vector<double, mem_type::owner, resource::host> const &b,
                   int const restart, int const max_iter, double const tolerance);

template gmres_info<double>
bicgstab_euler(const double dt, imex_flag imex,
               kron_operators<double> const &ops,
               fk::vector<double, mem_type::owner, resource::host> &x,
               fk::vector<double, mem_type::owner, resource::host> const &b,
               int const max_iter, double const tolerance);

#ifdef ASGARD_USE_CUDA
template gmres_info<double>
simple_gmres_euler(const double dt, imex_flag imex,
                   kron_operators<double> const &ops,
                   fk::vector<double, mem_type::owner, resource::device> &x,
                   fk::vector<double, mem_type::owner, resource::device> const &b,
                   int const restart, int const max_iter, double const tolerance);
template gmres_info<double>
bicgstab_euler(const double dt, imex_flag imex,
               kron_operators<double> const &ops,
               fk::vector<double, mem_type::owner, resource::device> &x,
               fk::vector<double, mem_type::owner, resource::device> const &b,
               int const max_iter, double const tolerance);
#endif

template int default_gmres_restarts<double>(int num_cols);

template void setup_poisson(const int N_elements, double const x_min,
                            double const x_max, fk::vector<double> &diag,
                            fk::vector<double> &off_diag);

template void
poisson_solver(fk::vector<double> const &source, fk::vector<double> const &A_D,
               fk::vector<double> const &A_E, fk::vector<double> &phi,
               fk::vector<double> &E, int const degree, int const N_elements,
               double const x_min, double const x_max, double const phi_min,
               double const phi_max, poisson_bc const bc);
#endif // ASGARD_ENABLE_DOUBLE

#ifdef ASGARD_ENABLE_FLOAT
template gmres_info<float>
simple_gmres(fk::matrix<float> const &A, fk::vector<float> &x,
             fk::vector<float> const &b, fk::matrix<float> const &M,
             int const restart, int const max_iter, float const tolerance);

template gmres_info<float>
bicgstab(fk::matrix<float> const &A, fk::vector<float> &x,
         fk::vector<float> const &b, fk::matrix<float> const &M,
         int const max_iter, float const tolerance);

template gmres_info<float>
simple_gmres_euler(const float dt, imex_flag imex,
                   kron_operators<float> const &ops,
                   fk::vector<float, mem_type::owner, resource::host> &x,
                   fk::vector<float, mem_type::owner, resource::host> const &b,
                   int const restart, int const max_iter, float const tolerance);

template gmres_info<float>
bicgstab_euler(const float dt, imex_flag imex,
               kron_operators<float> const &ops,
               fk::vector<float, mem_type::owner, resource::host> &x,
               fk::vector<float, mem_type::owner, resource::host> const &b,
               int const max_iter, float const tolerance);

#ifdef ASGARD_USE_CUDA
template gmres_info<float>
simple_gmres_euler(const float dt, imex_flag imex,
                   kron_operators<float> const &ops,
                   fk::vector<float, mem_type::owner, resource::device> &x,
                   fk::vector<float, mem_type::owner, resource::device> const &b,
                   int const restart, int const max_iter, float const tolerance);

template gmres_info<float>
bicgstab_euler(const float dt, imex_flag imex,
               kron_operators<float> const &ops,
               fk::vector<float, mem_type::owner, resource::device> &x,
               fk::vector<float, mem_type::owner, resource::device> const &b,
               int const max_iter, float const tolerance);
#endif

template int default_gmres_restarts<float>(int num_cols);

template void setup_poisson(const int N_elements, float const x_min,
                            float const x_max, fk::vector<float> &diag,
                            fk::vector<float> &off_diag);

template void
poisson_solver(fk::vector<float> const &source, fk::vector<float> const &A_D,
               fk::vector<float> const &A_E, fk::vector<float> &phi,
               fk::vector<float> &E, int const degree, int const N_elements,
               float const x_min, float const x_max, float const phi_min,
               float const phi_max, poisson_bc const bc);
#endif // ASGARD_ENABLE_FLOAT

} // namespace asgard::solver
