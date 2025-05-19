#include "asgard_solver.hpp"

#include "asgard_blas.hpp"
#include "asgard_small_mats.hpp"

namespace asgard::solvers
{

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
  compute->pttrs(diag, subdiag, rhs);

  // Set Potential and Electric Field in DG Nodes //
  efield.resize(nelem);

  efield[0] = - (rhs[0] - dleft) / dx;
  for (int i = 1; i < nelem - 1; i++)
    efield[i] = - (rhs[i] - rhs[i - 1]) / dx;
  efield.back() = - (dright - rhs.back()) / dx;
}

template<typename P>
direct<P>::direct(
    int groupid, sparse_grid const &grid, connection_patterns const &conn,
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


  int const iend = (groupid == -1) ? static_cast<int>(terms.terms.size())
                                   : terms.term_groups[groupid].end();

  int tid = (groupid == -1) ? 0 : terms.term_groups[groupid].begin();
  while (tid < iend)
  {
    auto it = terms.terms.begin() + tid;

    #ifdef ASGARD_USE_MPI
    if (not terms.resources.owns(it->rec)) {
      tid += it->num_chain;
      continue;
    }
    #endif

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

      ++tid;
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

      tid += it->num_chain;
    }
  }

  dense_mat = bmat.to_dense_matrix(n);

  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    if (terms.resources.is_leader()) {
      dense_matrix<P> mat = dense_mat;
      terms.resources.reduce_add(static_cast<int>(dense_mat.nrows() * dense_mat.ncols()),
                                 mat.data(), dense_mat.data());
    } else {
      terms.resources.reduce_add(static_cast<int>(dense_mat.nrows() * dense_mat.ncols()),
                                 dense_mat.data());
      return;
    }
  }
  #endif

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
  P outer_res = tolerance_ + 1.0;
  while (outer_res > tolerance_ and outer_iterations < max_outer_)
  {
    std::copy(rhs.begin(), rhs.end(), basis.begin());
    apply_lhs(-1, x.data(), 1, basis.data());
    apply_precon(basis.data());
    ++num_appy;

    inner_res = fm::nrm2(n, basis.data());

    fm::scal(n, P{1} / inner_res, basis.data());
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

      fm::gemv('T', n, inner_iterations + 1, P{1}, basis.data(), r, P{0}, coeff);
      fm::gemv('N', n, inner_iterations + 1, P{-1}, basis.data(), coeff, P{1}, r);

      P const nrm = fm::nrm2(n, r);
      fm::scal(n, P{1} / nrm, r);
      for (int k = 0; k < inner_iterations; k++)
        fm::rot(1, coeff + k, coeff + k + 1, cosines[k], sines[k]);

      // compute given's rotation
      P beta = nrm;
      fm::rotg(coeff + inner_iterations, &beta, cosines + inner_iterations, sines + inner_iterations);

      inner_res = std::abs(sines[inner_iterations] * krylov_sol[inner_iterations]);

      if (inner_res > tolerance_ and inner_iterations < max_inner_)
      {
        krylov_sol[inner_iterations + 1] = 0.;
        fm::rot(1, krylov_sol + inner_iterations, krylov_sol + inner_iterations + 1,
                cosines[inner_iterations], sines[inner_iterations]);
      }

      ++inner_iterations;
    } // end of inner iteration loop

    if (inner_iterations > 0)
    {
      fm::tpsv('U', 'N', 'N', inner_iterations, krylov_proj, krylov_sol);
      fm::gemv('N', n, inner_iterations, P{1}, basis.data(), krylov_sol, P{1}, x.data());
    }
    ++outer_iterations;
    outer_res = inner_res;
  } // end outer iteration

  return num_appy;
}

#ifdef ASGARD_ENABLE_DOUBLE
template class direct<double>;
template class bicgstab<double>;
template class gmres<double>;

template void poisson<double>::solve(
    std::vector<double> const &, double, double, poisson_bc const, std::vector<double> &);

#endif // ASGARD_ENABLE_DOUBLE

#ifdef ASGARD_ENABLE_FLOAT
template class direct<float>;
template class bicgstab<float>;
template class gmres<float>;

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
    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1) {
      if (terms.resources.is_leader()) {
        terms.make_jacobi(grid, conn, terms.mpiwork);
        terms.resources.reduce_add(terms.mpiwork, jacobi);
      } else {
        terms.make_jacobi(grid, conn, jacobi);
        terms.resources.reduce_add(jacobi);
        grid_gen = grid.generation();
        return;
      }
    } else {
      terms.make_jacobi(grid, conn, jacobi);
    }
    #else
    terms.make_jacobi(grid, conn, jacobi);
    #endif

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
void solver_manager<P>::update_grid(
    int groupid, sparse_grid const &grid,
    connection_patterns const &conn, term_manager<P> const &terms, P alpha)
{
  tools::time_event timing_("updating solver");
  if (opt == solver_method::direct)
    var = solvers::direct<P>(grid, conn, terms, alpha);

  if (precon == precon_method::jacobi) {
    terms.make_jacobi(groupid, grid, conn, jacobi);
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
