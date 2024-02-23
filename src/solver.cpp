#include "solver.hpp"
#include "distribution.hpp"
#include "fast_math.hpp"
#include "quadrature.hpp"
#include "tools.hpp"
#include <algorithm>
#include <stdexcept>

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

#ifdef KRON_MODE_GLOBAL
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
simple_gmres_euler(const P dt, matrix_entry mentry,
                   global_kron_matrix<P> const &mat,
                   fk::vector<P, mem_type::owner, resrc> &x,
                   fk::vector<P, mem_type::owner, resrc> const &b,
                   int const restart, int const max_iter, P const tolerance)
{
  auto const &pc = mat.template get_diagonal_preconditioner<resrc>();

  return simple_gmres(
      [&](P const alpha, fk::vector<P, mem_type::view, resrc> const x_in,
          P const beta, fk::vector<P, mem_type::view, resrc> y) -> void {
        tools::time_event performance("kronmult - implicit", mat.flops(mentry));
        mat.template apply<resrc>(mentry, -dt * alpha, x_in.data(), beta, y.data());
        lib_dispatch::axpy<resrc>(y.size(), alpha, x_in.data(), 1, y.data(), 1);
      },
      fk::vector<P, mem_type::view, resrc>(x), b,
      [&](fk::vector<P, mem_type::view, resrc> &x_in) -> void {
        tools::time_event performance("kronmult - preconditioner", pc.size());
        apply_diagonal_precond(pc, dt, x_in);
      },
      restart, max_iter, tolerance);
}
#else
template<typename P, resource resrc>
gmres_info<P>
simple_gmres_euler(const P dt, kronmult_matrix<P> const &mat,
                   fk::vector<P, mem_type::owner, resrc> &x,
                   fk::vector<P, mem_type::owner, resrc> const &b,
                   int const restart, int const max_iter, P const tolerance)
{
  return simple_gmres(
      [&](P const alpha, fk::vector<P, mem_type::view, resrc> const x_in,
          P const beta, fk::vector<P, mem_type::view, resrc> y) -> void {
        tools::time_event performance("kronmult - implicit", mat.flops());
        mat.template apply<resrc>(-dt * alpha, x_in.data(), beta, y.data());
        lib_dispatch::axpy<resrc>(y.size(), alpha, x_in.data(), 1, y.data(), 1);
      },
      fk::vector<P, mem_type::view, resrc>(x), b, no_op_preconditioner<P>(),
      restart, max_iter, tolerance);
}

template<typename P, resource resrc>
gmres_info<P>
simple_gmres_euler(adapt::distributed_grid<P> const &adaptive_grid, int const elem_size,
                   P dt, kronmult_matrix<P> const &mat,
                   fk::vector<P, mem_type::owner, resrc> &x,
                   fk::vector<P, mem_type::owner, resrc> const &b,
                   int const restart, int const max_iter, P const tolerance)
{
#ifdef ASGARD_USE_MPI
  if (get_num_ranks() > 1)
    return simple_gmres(
        adaptive_grid, elem_size,
        [&](P const alpha, fk::vector<P, mem_type::view, resrc> const x_in,
            P const beta, fk::vector<P, mem_type::view, resrc> y) -> void {
          tools::time_event performance("kronmult - implicit", mat.flops());
          auto plan = adaptive_grid.get_distrib_plan();
          // switch to elem_size?
          fk::vector<P, mem_type::owner, resrc> y_local(y.size()), y_tmp(y.size());
          mat.template apply<resrc>(-dt * alpha, x_in.data(), P{0}, y_local.data());
          reduce_results(y_local, y_tmp, plan, get_rank());
          exchange_results(y_tmp, y_local, elem_size, plan, get_rank());
          lib_dispatch::axpy<resrc>(y.size(), beta, y.data(), 1, y_local.data(), 1);
          lib_dispatch::axpy<resrc>(y.size(), alpha, x_in.data(), 1, y_local.data(), 1);
          y = y_local;
        },
        fk::vector<P, mem_type::view, resrc>(x), b, no_op_preconditioner<P>(),
        restart, max_iter, tolerance);
#else
  ignore(adaptive_grid);
  ignore(elem_size);
#endif
  return simple_gmres_euler(dt, mat, x, b, restart, max_iter, tolerance);
}
#endif

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
             int max_outer_iterations, P tolerance)
{
  if (tolerance == parser::NO_USER_VALUE_FP)
    tolerance = std::is_same_v<float, P> ? 1e-6 : 1e-12;
  expect(tolerance >= std::numeric_limits<P>::epsilon());

  int const n = b.size();
  expect(n == x.size());

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

  if (max_outer_iterations == parser::NO_USER_VALUE)
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
    fk::vector<P, mem_type::view, resrc> scaled(basis, 0, 0, n - 1);
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
                                                     n - 1);
      fk::vector<P, mem_type::view, resrc> new_basis(
          basis, inner_iterations + 1, 0, n - 1);
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

      if (inner_iterations % print_freq == 0)
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
      fk::matrix<P, mem_type::view, resrc> m(basis, 0, n - 1, 0,
                                             inner_iterations - 1);
      if constexpr (resrc == resource::device)
        fm::gemv(m, s_view.clone_onto_device(), x, false, P{1.}, P{1.});
      else if constexpr (resrc == resource::host)
        fm::gemv(m, s_view, x, false, P{1.}, P{1.});
    }
    ++outer_iterations;
    outer_res = inner_res;
  } // end outer iteration
  std::cout << "GMRES complete with error: " << outer_res << '\n';
  std::cout << total_iterations << " iterations\n";
  return gmres_info<P>{outer_res, total_iterations};
}

#ifdef ASGARD_USE_MPI
// simple, node-local test version
template<typename P, resource resrc, typename matrix_abstraction,
         typename preconditioner_abstraction>
gmres_info<P>
simple_gmres(adapt::distributed_grid<P> const &adaptive_grid, int const elem_size,
             matrix_abstraction mat, fk::vector<P, mem_type::view, resrc> x,
             fk::vector<P, mem_type::owner, resrc> const &b,
             preconditioner_abstraction precondition, int restart,
             int max_outer_iterations, P tolerance)
{
  int num_rows = 0;
  int num_cols = 0;
  for (int i = 0; i < get_num_ranks(); ++i)
  {
    auto const &grid = adaptive_grid.get_subgrid(i);
    num_rows         = std::max(num_rows, grid.row_stop + 1);
    num_cols         = std::max(num_cols, grid.col_stop + 1);
  }
  expect(num_rows == num_cols);
  num_rows *= elem_size;

  if (tolerance == parser::NO_USER_VALUE_FP)
    tolerance = std::is_same_v<float, P> ? 1e-6 : 1e-12;
  expect(tolerance >= std::numeric_limits<P>::epsilon());

  int const n = b.size();
  expect(n == x.size());
  {
    auto const &grid = adaptive_grid.get_subgrid(get_rank());
    expect(n == elem_size*(grid.col_stop - grid.col_start + 1));
  }
  if (restart == parser::NO_USER_VALUE)
    restart = default_gmres_restarts<P>(num_rows);
  expect(restart > 0); // checked in program_options
  if (restart > num_rows)
  {
    std::ostringstream err_msg;
    err_msg << "Number of inner iterations " << restart << " must be less than "
            << num_rows << "!";
    throw std::invalid_argument(err_msg.str());
  }

  if (max_outer_iterations == parser::NO_USER_VALUE)
    max_outer_iterations = num_rows;
  expect(max_outer_iterations > 0); // checked in program_options

  // controls how often the inner residual print occurs
  int const print_freq = restart / 3;
  fk::matrix<P, mem_type::owner, resrc> basis(n, restart + 1);
  fk::vector<P> krylov_proj(restart * (restart + 1) / 2);
  fk::vector<P> sines(restart + 1);
  fk::vector<P> cosines(restart + 1);
  fk::vector<P> krylov_sol(restart + 1);
  auto plan = adaptive_grid.get_distrib_plan();

  int total_iterations = 0;
  int outer_iterations = 0;
  int inner_iterations = 0;

  P inner_res = 0.;
  P outer_res = tolerance + 1.;
  while ((outer_res > tolerance) && (outer_iterations < max_outer_iterations))
  {
    fk::vector<P, mem_type::view, resrc> scaled(basis, 0, 0, n - 1);
    scaled = b;
    mat(P{-1.}, x, P{1.}, scaled);
    precondition(scaled);
    ++total_iterations;
    P const local_inner_res = std::accumulate(scaled.begin(), scaled.end(), P{0}, [](P sum, P elem) { return sum + elem * elem; });
    reduce_results(local_inner_res, inner_res, plan, get_rank());
    inner_res = std::sqrt(inner_res);
    //std::cout << get_rank() << " " << inner_res << std::endl;
    scaled.scale(P{1.} / inner_res);
    krylov_sol[0] = inner_res;

    inner_iterations = 0;
    while ((inner_res > tolerance) && (inner_iterations < restart))
    {
      fk::vector<P, mem_type::view, resrc> const tmp(basis, inner_iterations, 0, n - 1);
      fk::vector<P, mem_type::view, resrc> new_basis(basis, inner_iterations + 1, 0, n - 1);
      mat(P{1.}, tmp, P{0.}, new_basis);
      precondition(new_basis);
      ++total_iterations;
      fk::matrix<P, mem_type::const_view, resrc> basis_v(basis, 0, n - 1, 0, inner_iterations);
      fk::vector<P, mem_type::view> coeffs(krylov_proj, pos_from_indices(0, inner_iterations), pos_from_indices(inner_iterations, inner_iterations));
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
        fk::vector<P, mem_type::owner, resrc> coeffs_tmp(coeffs.size());
        fm::gemv(basis_v, new_basis, coeffs_tmp, true, P{1.}, P{0.});
        reduce_results(coeffs_tmp, coeffs, plan, get_rank());
        fm::gemv(basis_v, coeffs, new_basis, false, P{-1.}, P{1.});
      }
      P const local_nrm = std::accumulate(new_basis.begin(), new_basis.end(), P{0}, [](P sum, P elem) { return sum + elem * elem; });
      P nrm;
      reduce_results(local_nrm, nrm, plan, get_rank());
      nrm = std::sqrt(nrm);
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

      if (get_rank() == 0 && inner_iterations % print_freq == 0)
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
      fk::matrix<P, mem_type::view, resrc> m(basis, 0, n - 1, 0,
                                             inner_iterations - 1);
      if constexpr (resrc == resource::device)
        fm::gemv(m, s_view.clone_onto_device(), x, false, P{1.}, P{1.});
      else if constexpr (resrc == resource::host)
        fm::gemv(m, s_view, x, false, P{1.}, P{1.});
    }
    ++outer_iterations;
    outer_res = inner_res;
  } // end outer iteration
  std::cout << "GMRES complete with error: " << outer_res << '\n';
  std::cout << total_iterations << " iterations\n";
  return gmres_info<P>{outer_res, total_iterations};
}
#endif

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

  auto const lgwt =
      legendre_weights<P>(degree + 1, -1.0, 1.0, quadrature_mode::use_degree);

  // If only one element, skip poisson solve and solve via BCs
  if (N_elements == 1)
  {
    for (int k = 0; k < degree + 1; k++)
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
}

#ifdef ASGARD_ENABLE_DOUBLE

template gmres_info<double>
simple_gmres(fk::matrix<double> const &A, fk::vector<double> &x,
             fk::vector<double> const &b, fk::matrix<double> const &M,
             int const restart, int const max_iter, double const tolerance);

#ifdef KRON_MODE_GLOBAL
template gmres_info<double>
simple_gmres_euler(const double dt, matrix_entry mentry,
                   global_kron_matrix<double> const &mat,
                   fk::vector<double, mem_type::owner, resource::host> &x,
                   fk::vector<double, mem_type::owner, resource::host> const &b,
                   int const restart, int const max_iter, double const tolerance);
#ifdef ASGARD_USE_CUDA
template gmres_info<double>
simple_gmres_euler(const double dt, matrix_entry mentry,
                   global_kron_matrix<double> const &mat,
                   fk::vector<double, mem_type::owner, resource::device> &x,
                   fk::vector<double, mem_type::owner, resource::device> const &b,
                   int const restart, int const max_iter, double const tolerance);
#endif
#else
template gmres_info<double>
simple_gmres_euler(adapt::distributed_grid<double> const &adaptive_grid, int const elem_size,
                   double const dt, kronmult_matrix<double> const &mat,
                   fk::vector<double> &x, fk::vector<double> const &b,
                   int const restart, int const max_iter,
                   double const tolerance);
#ifdef ASGARD_USE_CUDA
template gmres_info<double> simple_gmres_euler(
    adapt::distributed_grid<double> const &adaptive_grid, int const elem_size,
    double const dt, kronmult_matrix<double> const &mat,
    fk::vector<double, mem_type::owner, resource::device> &x,
    fk::vector<double, mem_type::owner, resource::device> const &b,
    int const restart, int const max_iter, double const tolerance);
#endif
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

#endif

#ifdef ASGARD_ENABLE_FLOAT

template gmres_info<float>
simple_gmres(fk::matrix<float> const &A, fk::vector<float> &x,
             fk::vector<float> const &b, fk::matrix<float> const &M,
             int const restart, int const max_iter, float const tolerance);

#ifdef KRON_MODE_GLOBAL
template gmres_info<float>
simple_gmres_euler(const float dt, matrix_entry mentry,
                   global_kron_matrix<float> const &mat,
                   fk::vector<float, mem_type::owner, resource::host> &x,
                   fk::vector<float, mem_type::owner, resource::host> const &b,
                   int const restart, int const max_iter, float const tolerance);
#ifdef ASGARD_USE_CUDA
template gmres_info<float>
simple_gmres_euler(const float dt, matrix_entry mentry,
                   global_kron_matrix<float> const &mat,
                   fk::vector<float, mem_type::owner, resource::device> &x,
                   fk::vector<float, mem_type::owner, resource::device> const &b,
                   int const restart, int const max_iter, float const tolerance);
#endif
#else
template gmres_info<float>
simple_gmres_euler(adapt::distributed_grid<float> const &adaptive_grid, int const elem_size,
                   float const dt, kronmult_matrix<float> const &mat,
                   fk::vector<float> &x, fk::vector<float> const &b,
                   int const restart, int const max_iter,
                   float const tolerance);
#ifdef ASGARD_USE_CUDA
template gmres_info<float> simple_gmres_euler(
    adapt::distributed_grid<float> const &adaptive_grid, int const elem_size,
    float const dt, kronmult_matrix<float> const &mat,
    fk::vector<float, mem_type::owner, resource::device> &x,
    fk::vector<float, mem_type::owner, resource::device> const &b,
    int const restart, int const max_iter, float const tolerance);
#endif
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

#endif

} // namespace asgard::solver
