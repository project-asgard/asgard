#include "asgard_solver.hpp"

#include "asgard_small_mats.hpp"

namespace asgard::solvers
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
simple_bicgstab(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
         fk::matrix<P> const &M, int const max_iter,
         P const tolerance)
{
  auto dense_matrix_wrapper =
      [&A](P const alpha, fk::vector<P, mem_type::view> const x_in,
           P const beta, fk::vector<P, mem_type::view> y) {
        fm::gemv(A, x_in, y, false, alpha, beta);
      };
  if (M.size() > 0)
    return simple_bicgstab(dense_matrix_wrapper, fk::vector<P, mem_type::view>(x),
                           b, dense_preconditioner(M), max_iter,
                           tolerance);
  else
    return simple_bicgstab(dense_matrix_wrapper, fk::vector<P, mem_type::view>(x),
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

  return simple_bicgstab(
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
simple_bicgstab(matrix_abstraction mat, fk::vector<P, mem_type::view, resrc> x,
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
void poisson<P>::solve(std::vector<P> const &density, P dleft, P dright,
                       poisson_bc const bc, std::vector<P> &efield)
{
  tools::time_event psolve_("poisson_solver");

  if (current_level == 0)
  {
    efield.resize(1);
    efield[0] = -(dright - dleft) / (xmax - xmin);
    return;
  }

  int const nelem = fm::ipow2(current_level);

  P const dx = (xmax - xmin) / static_cast<P>(nelem);

  int const pdof = degree + 1;

  int const nnodes = nelem - 1;

  // integrals of hat-basis functions x, 1-x vs Legendre basis 1, sqrt(3) * 2x-1
  // over canonical element (0, 1)
  // the input coefficients are l-2 normalized over sub-cells, hence the sqrt-scaling
  P const c0 = std::sqrt(dx) * 0.5;
  P const c1 = std::sqrt(dx) * std::sqrt(3.0) * P{1} / P{6}; // the integral with the left basis is negative

  span2d<P const> rho(pdof, nelem, density.data());

  // building the right-hand-side vector
  if (bc == poisson_bc::periodic)
  {
    dleft = dright = P{0};

    P average = 0;
    for (int i : iindexof(nelem))
      average += rho[i][0]; // reading the constant
    // the extra 2 below is because the correction is applied to 2 elements
    average *= P{2} * dx / (xmax - xmin);

    if (pdof == 1) { // consider only constant functions
      for (int i : iindexof(nnodes))
        rhs[i] = c0 * (rho[i][0] + rho[i + 1][0] - average);
    } else {
      for (int i : iindexof(nnodes))
        rhs[i] = c0 * (rho[i][0] + rho[i + 1][0] - average)
                + c1 * rho[i][1] - c1 * rho[i + 1][1];
    }
  }
  else
  {
    if (pdof == 1) { // consider only constant functions
      for (int i : iindexof(nnodes))
        rhs[i] = c0 * (rho[i][0] + rho[i + 1][0]);
    } else {
      for (int i : iindexof(nnodes))
        rhs[i] = c0 * (rho[i][0] + rho[i + 1][0])
                + c1 * rho[i][1] - c1 * rho[i + 1][1];
    }
    rhs.front() += dleft / dx;
    rhs.back()  += dright / dx;
  }

  // // Linear Solve //
  fm::pttrs(diag, subdiag, rhs);

  // Set Potential and Electric Field in DG Nodes //
  efield.resize(nelem);

  efield[0] = - (rhs[0] - dleft) / dx;
  for (int i = 1; i < nelem - 1; i++)
    efield[i] = - (rhs[i] - rhs[i - 1]) / dx;
  efield.back() = - (dright - rhs.back()) / dx;
}

template<typename P>
direct<P>::direct(sparse_grid const &grid, connection_patterns const &conn,
                  term_manager<P> const &terms, P alpha)
{
  tools::time_event timing_("forming dense matrix");
  int const num_dims    = grid.num_dims();
  int const num_indexes = grid.num_indexes();
  int const pdof        = terms.legendre.pdof;

  int const n = fm::ipow(pdof, num_dims);

  block_matrix<P> bmat(n * n, num_indexes, num_indexes);
  block_matrix<P> wmat(n * n, num_indexes, num_indexes);

  std::array<block_matrix<P>, max_num_dimensions> ids; // identity coefficients
  for (int d : iindexof(num_dims)) {
    int const size = fm::ipow2(grid.current_level(d));
    ids[d] = block_matrix<P>(pdof * pdof, size, size);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < pdof; j++)
        ids[d](i, i)[j * pdof + j] = 1;
    }
  }

  // work coefficients, full-block matrices that will be used for this term_md
  std::array<block_matrix<P> const *, max_num_dimensions> wcoeffs;

  std::array<block_matrix<P>, max_num_dimensions> temp_mats;

  auto kron_mats = [&](block_matrix<P> &mat)
      -> void
    {
      if (num_dims == 1) {
#pragma omp parallel for
        for (int c = 0; c < num_indexes; c++) {
          for (int r = 0; r < num_indexes; r++) {
            int const ic = grid[c][0];
            int const ir = grid[r][0];

            std::copy_n((*wcoeffs[0])(ir, ic), pdof * pdof, mat(r, c));
          }
        }
      } else {
#pragma omp parallel for
        for (int c = 0; c < num_indexes; c++) {
          for (int r = 0; r < num_indexes; r++) {
            int const *ic = grid[c];
            int const *ir = grid[r];

            int cyc    = 1;
            int stride = fm::ipow(pdof, num_dims - 1);
            int repeat = stride;
            for (int d : iindexof(num_dims)) {
              smmat::kron_block(pdof, cyc, stride, repeat,
                                (*wcoeffs[d])(ir[d], ic[d]), mat(r, c));
              stride /= pdof;
              cyc    *= pdof;
            }
          }
        }
      }
    };

  auto set_wcoeff = [&](term_entry<P> const &te)
      -> void
    {
      for (int d : iindexof(num_dims)) {
        if (te.coeffs[d].nblock() > 0) {
          temp_mats[d] = te.coeffs[d].to_full(conn);
          wcoeffs[d]   = &temp_mats[d];
        } else {
          wcoeffs[d] = &ids[d];
        }
      }
    };

  auto it = terms.terms.begin();
  while (it < terms.terms.end())
  {
    if (it->num_chain == 1) {
      set_wcoeff(*it);
      wmat.fill(1);
      kron_mats(wmat);

      int64_t const size = n * n * num_indexes * num_indexes;
      P *mat_data        = bmat.data();
      P const *wmat_data = wmat.data();

      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < size; i++)
        mat_data[i] += wmat_data[i];

      ++it;
    } else {
      if (it->num_chain == 2) {
        // need two temp matrices
        block_matrix<P> t1(n * n, num_indexes, num_indexes);
        set_wcoeff(*it);
        t1.fill(1);
        kron_mats(t1);

        block_matrix<P> t2(n * n, num_indexes, num_indexes);
        set_wcoeff(*(it + 1));
        t2.fill(1);
        kron_mats(t2);

        gemm1(n, t1, t2, bmat);
      } else {
        throw std::runtime_error(
            "term_md chains with num_chain >= 3 are not yet implemented "
            "for the direct solver");
      }

      it += it->num_chain;
    }
  }

  dense_mat = bmat.to_dense_matrix(n);

  if (alpha != 0)
  {
    int64_t const size = n * num_indexes;

    #pragma omp parallel for
    for (int64_t c = 0; c < size - 1; c++) {
      P *dd = dense_mat.data() + c * (size + 1);
      dd[0] = P{1} + alpha * dd[0];
      dd += 1;
      ASGARD_OMP_SIMD
      for (int64_t i = 0; i < size; i++) {
        dd[i] *= alpha;
      }
    }

    dense_mat(size - 1, size - 1) = P{1} + alpha * dense_mat(size - 1, size - 1);
  }

  dense_mat.factorize();
}

template<typename P>
int bicgstab<P>::solve(
    operatoin_apply_lhs<P> apply_lhs, std::vector<P> const &rhs, std::vector<P> &x) const
{
  tools::time_event timing_("bicgstab::solve");
  int64_t const n = static_cast<int64_t>(rhs.size());
  if (v.size() != rhs.size()) // the other temps are initialized with a copy
    v.resize(n);
  if (t.size() != rhs.size()) // the other temps are initialized with a copy
    t.resize(n);

  auto dot = [&](std::vector<P> const &a, std::vector<P> const &b)
    -> P {
      P sum = 0;
ASGARD_OMP_PARFOR_SIMD_EXTRA(reduction(+:sum))
      for (int64_t i = 0; i < n; i++)
        sum += a[i] * b[i];
      return sum;
    };
  auto dot1 = [&](std::vector<P> const &a)
    -> P {
      P sum = 0;
ASGARD_OMP_PARFOR_SIMD_EXTRA(reduction(+:sum))
      for (int64_t i = 0; i < n; i++)
        sum += a[i] * a[i];
      return sum;
    };
  auto nrm = [&](std::vector<P> const &a)
    -> P {
      return std::sqrt(dot1(a));
    };
  auto axpy = [&](P alpha, std::vector<P> const &a, std::vector<P> &b)
    -> void {
ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < n; i++)
        b[i] += alpha * a[i];
    };

  r = rhs;

  int num_appy = 1;
  apply_lhs(-1, x.data(), 1, r.data()); // r0 = b - A * x0

  P rho = dot1(r);

  rref = r; // initialize rref (hat-r-0) and p
  p    = r;

  for (int i = 0; i < max_iter_; i++) {
    ++num_appy;
    apply_lhs(1, p.data(), 0, v.data()); // v = A * p

    P const alpha = rho / dot(rref, v);

    axpy(alpha, p, x);
    axpy(-alpha, v, r);

    if (nrm(r) < tolerance_) {
      return num_appy;
    }

    ++num_appy;
    apply_lhs(1, r.data(), 0, t.data()); // t = A * p

    P const omega = dot(r, t) / dot1(t);

    axpy(omega, r, x);
    axpy(-omega, t, r);

    if (nrm(r) < tolerance_) {
      return num_appy;
    }

    P const rho1 = dot(rref, r);
    P const beta = (rho1 / rho) * (alpha / omega);

ASGARD_OMP_PARFOR_SIMD
    for (int64_t k = 0; k < n; k++)
      p[k] = r[k] + beta * (p[k] - omega * v[k]);

    rho = rho1;
  }
  std::cerr << "Warning: ASGarD BiCGSTAB solver failed to converge within "
            << max_iter_ << " iterations.\n";
  return num_appy;
}

template<typename P>
int gmres<P>::solve(
    operatoin_apply_precon<P> apply_precon,
    operatoin_apply_lhs<P> apply_lhs, std::vector<P> const &rhs,
    std::vector<P> &x) const
{
  tools::time_event timing_("gmres::solve");
  int const n = static_cast<int>(rhs.size());
  expect(n == static_cast<int>(x.size()));

  basis.resize(static_cast<int64_t>(n) * (max_inner_ + 1));

  int num_appy = 0;

  // int total_iterations = 0;
  int outer_iterations = 0;
  int inner_iterations = 0;

  P inner_res = 0.;
  P outer_res = tolerance_ + 1.;
  while ((outer_res > tolerance_) && (outer_iterations < max_outer_))
  {
    std::copy(rhs.begin(), rhs.end(), basis.begin());
    apply_lhs(-1, x.data(), 1, basis.data());
    apply_precon(basis.data());
    ++num_appy;

    inner_res = lib_dispatch::nrm2(n, basis.data(), 1);
    lib_dispatch::scal(n, P{1} / inner_res, basis.data(), 1);
    krylov_sol[0] = inner_res;

    inner_iterations = 0;
    while (inner_res > tolerance_ and inner_iterations < max_inner_)
    {
      P *r = basis.data() + static_cast<int64_t>(n) * (inner_iterations + 1);
      apply_lhs(1, basis.data() + static_cast<int64_t>(n) * inner_iterations, 0, r);
      apply_precon(r);
      ++num_appy;

      // krylov projection coefficients for this iteration
      P *coeff = krylov_proj + (inner_iterations * (inner_iterations + 1)) / 2;

      lib_dispatch::gemv('T', n, inner_iterations + 1, P{1}, basis.data(), n,
                         r, 1, P{0}, coeff, 1);
      lib_dispatch::gemv('N', n, inner_iterations + 1, P{-1}, basis.data(), n,
                         coeff, 1, P{1}, r, 1);

      P const nrm = lib_dispatch::nrm2(n, r, 1);
      lib_dispatch::scal(n, P{1} / nrm, r, 1);
      for (int k = 0; k < inner_iterations; k++)
        lib_dispatch::rot(1, coeff + k, 1, coeff + k + 1, 1,
                          cosines[k], sines[k]);

      // compute given's rotation
      P beta = nrm;
      lib_dispatch::rotg(coeff + inner_iterations, &beta,
                         cosines + inner_iterations,
                         sines + inner_iterations);

      inner_res =
          std::abs(sines[inner_iterations] * krylov_sol[inner_iterations]);

      if (inner_res > tolerance_ and inner_iterations < max_inner_)
      {
        krylov_sol[inner_iterations + 1] = 0.;
        lib_dispatch::rot(1, krylov_sol + inner_iterations, 1,
                          krylov_sol + inner_iterations + 1, 1,
                          cosines[inner_iterations], sines[inner_iterations]);
      }

      ++inner_iterations;
    } // end of inner iteration loop

    if (inner_iterations > 0)
    {
      lib_dispatch::tpsv('U', 'N', 'N', inner_iterations, krylov_proj, krylov_sol, 1);
      lib_dispatch::gemv('N', n, inner_iterations, P{1}, basis.data(), n,
                         krylov_sol, 1, P{1}, x.data(), 1);
    }
    ++outer_iterations;
    outer_res = inner_res;
  } // end outer iteration

  return num_appy;
}

#ifdef ASGARD_ENABLE_DOUBLE
template class direct<double>;
template class bicgstab<double>;

template gmres_info<double>
simple_gmres(fk::matrix<double> const &A, fk::vector<double> &x,
             fk::vector<double> const &b, fk::matrix<double> const &M,
             int const restart, int const max_iter, double const tolerance);
template gmres_info<double>
simple_bicgstab(fk::matrix<double> const &A, fk::vector<double> &x,
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

template void poisson<double>::solve(
    std::vector<double> const &, double, double, poisson_bc const, std::vector<double> &);

#endif // ASGARD_ENABLE_DOUBLE

#ifdef ASGARD_ENABLE_FLOAT
template class direct<float>;
template class bicgstab<float>;

template gmres_info<float>
simple_gmres(fk::matrix<float> const &A, fk::vector<float> &x,
             fk::vector<float> const &b, fk::matrix<float> const &M,
             int const restart, int const max_iter, float const tolerance);

template gmres_info<float>
simple_bicgstab(fk::matrix<float> const &A, fk::vector<float> &x,
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

template void poisson<float>::solve(
    std::vector<float> const &, float, float, poisson_bc const, std::vector<float> &);

#endif // ASGARD_ENABLE_FLOAT

} // namespace asgard::solvers

namespace asgard
{

template<typename P>
void solver_manager<P>::update_grid(
    sparse_grid const &grid, connection_patterns const &conn,
    term_manager<P> const &terms, P alpha)
{
  tools::time_event timing_("updating solver");
  if (opt == solver_method::direct)
    var = solvers::direct<P>(grid, conn, terms, alpha);

  if (precon == precon_method::jacobi) {
    terms.make_jacobi(grid, conn, jacobi);
    if (alpha == 0) { // steady state solver
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < jacobi.size(); i++)
        jacobi[i] = P{1} / jacobi[i];
    } else {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < jacobi.size(); i++)
        jacobi[i] = P{1} / (P{1} + alpha * jacobi[i]);
    }
  }

  grid_gen = grid.generation();
}

template<typename P>
void solver_manager<P>::xpby(std::vector<P> const &x, P beta, P y[]) {
ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = x[i] + beta * y[i];
}

template<typename P>
void solver_manager<P>::print_opts(std::ostream &os) const
{
  os << "solver:\n";
  bool has_precon = false;
  switch (var.index()) {
    case 0:
      os << "  direct\n";
      break;
    case 1:
      os << "  gmres\n";
      os << "  tolerance: " << std::get<solvers::gmres<P>>(var).tolerance() << '\n';
      os << "  max inner: " << std::get<solvers::gmres<P>>(var).max_inner() << '\n';
      os << "  max outer: " << std::get<solvers::gmres<P>>(var).max_outer() << '\n';
      has_precon = true;
      break;
    case 2:
      os << "  bicgstab\n";
      os << "  tolerance:      " << std::get<solvers::bicgstab<P>>(var).tolerance() << '\n';
      os << "  max iterations: " << std::get<solvers::bicgstab<P>>(var).max_iter() << '\n';
      has_precon = true;
      break;
    default:
      break;
  }
  if (has_precon) {
    switch (precon) {
      case precon_method::none:
        os << "  no preconditioner\n";
        break;
      case precon_method::jacobi:
        os << "  jacobi diagonal preconditioner\n";
        break;
      case precon_method::adi:
        os << "  adi preconditioner\n";
        break;
      default: // unreachable
        break;
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct solver_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct solver_manager<float>;
#endif

}
