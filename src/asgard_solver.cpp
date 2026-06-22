#include "asgard_solver.hpp"

#include "asgard_blas.hpp"
#include "asgard_small_mats.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_algorithms.hpp"
#endif

namespace asgard::solvers
{

template<typename P>
int cg<P>::solve(
    operatoin_apply_lhs<P> apply_lhs, std::vector<P> const &rhs, std::vector<P> &x) const
{
  tools::time_event timing_("cg::solve");
  int64_t const n = static_cast<int64_t>(rhs.size());

  if (r.size() != rhs.size()) r.resize(n);
  if (p.size() != rhs.size()) p.resize(n);
  if (q.size() != rhs.size()) q.resize(n);

  auto dot = [&](std::vector<P> const &a, std::vector<P> const &b) -> P {
      P sum = 0;
      ASGARD_OMP_PARFOR_SIMD_EXTRA(reduction(+:sum))
      for (int64_t i = 0; i < n; i++) sum += a[i] * b[i];
      return sum;
  };

  r = rhs;
  int num_appy = 1;
  apply_lhs(-1.0, x.data(), 1.0, r.data()); // r = b - A * x

  p = r;
  P rho = dot(r, r);

  for (int i = 0; i < max_iter_; i++) {
    ++num_appy;
    apply_lhs(1.0, p.data(), 0.0, q.data()); // q = A * p

    P const p_dot_q = dot(p, q);
    P const alpha = rho / p_dot_q;

    ASGARD_OMP_PARFOR_SIMD
    for (int64_t k = 0; k < n; k++) {
        x[k] += alpha * p[k];
        r[k] -= alpha * q[k];
    }

    P const rho_new = dot(r, r);

    // Exact check based on the new residual
    if (std::sqrt(rho_new) < tolerance_) {
      return num_appy;
    }

    P const beta = rho_new / rho;

    ASGARD_OMP_PARFOR_SIMD
    for (int64_t k = 0; k < n; k++) {
        p[k] = r[k] + beta * p[k];
    }

    rho = rho_new;
  }
  std::cerr << "Warning: ASGarD CPU CG solver failed to converge within " << max_iter_ << " iterations.\n";
  return num_appy;
}

#ifdef ASGARD_USE_GPU
template<typename P>
int cg<P>::solve(operatoin_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
                 gpu::vector<P> &x) const
{
  tools::time_event timing_("cg::solve-gpu");
  int64_t const n = rhs.size();

  if (gq.size() != n) {
    gr.resize(n);
    gp.resize(n);
    gq.resize(n);
    d_rho.resize(1);
    d_rho_new.resize(1);
    d_p_dot_q.resize(1);
    d_alpha.resize(1);
    d_beta.resize(1);
  }

  gr = rhs;

  int num_appy = 1;
  apply_lhs(-1.0, x.data(), 1.0, gr.data()); 

  gp = gr;

  compute->dot1_device(n, gr.data(), d_rho.data());

  for (int i = 0; i < max_iter_; i++) {
    ++num_appy;

    apply_lhs(1.0, gp.data(), 0.0, gq.data()); 

    compute->dot_device(n, gp.data(), gq.data(), d_p_dot_q.data());

    gpu::cg_calc_alpha(d_rho.data(), d_p_dot_q.data(), d_alpha.data());

    gpu::cg_update_x_r(n, d_alpha.data(), gp.data(), gq.data(), x.data(), gr.data());

    compute->dot1_device(n, gr.data(), d_rho_new.data());

    if (i % 10 == 0) {
        P cpu_rho_new;
        d_rho_new.copy_to_host(1, &cpu_rho_new); 
        if (std::sqrt(cpu_rho_new) < tolerance_) {
            return num_appy;
        }
    }

    gpu::cg_calc_beta(d_rho_new.data(), d_rho.data(), d_beta.data());

    gpu::cg_update_p(n, d_beta.data(), gr.data(), gp.data());

    gpu::cg_update_rho(d_rho_new.data(), d_rho.data());
  }

  std::cerr << "Warning: ASGarD GPU CG solver failed to converge within " << max_iter_ << " iterations.\n";
  return num_appy;
}
#endif

template<typename P>
size_t cg<P>::used_bytes() const {
  size_t total = r.size() + p.size() + q.size();
  return total * sizeof(P);
}

template<typename P>
void direct<P>::update(
    group_id group, size_t stage, sparse_grid const &grid, connection_patterns const &conn,
    term_manager<P> const &terms, P alpha)
{
  tools::time_event timing_("forming dense matrix");
  int const num_dims    = grid.num_dims();
  int const num_indexes = grid.num_indexes();
  int const pdof        = terms.basis.pdof;

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

  indexrange trange = terms.terms_group_range(group);

  int icurrent = trange.ibegin();
  while (icurrent < trange.iend())
  {
    auto it = terms.terms.begin() + icurrent;

    #ifdef ASGARD_USE_MPI
    if (not terms.resources.owns(it->rec)) {
      icurrent += it->num_chain;
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

      ++icurrent;
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

      icurrent += it->num_chain;
    }
  }

  size_t const idx = mat_index(group, stage); // index for the matrix entry
  if (mats.size() <= idx)
    mats.resize(idx + 1);

  // must happen before the potential MPI return down below
  mats[idx].grid_gen = grid.generation();

  dense_matrix<P> &dense_mat = mats[idx].dense_mat;
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
size_t direct<P>::used_bytes() const {
  size_t t = 0;
  for (auto const &m : mats) t += m.dense_mat.used_bytes();
  return t;
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
size_t bicgstab<P>::used_bytes() const {
  size_t total = rref.size() + r.size() + p.size() + v.size() + t.size();
  return total * sizeof(P);
}

#ifdef ASGARD_USE_GPU
template<typename P>
int bicgstab<P>::solve(operatoin_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
                       gpu::vector<P> &x) const
{
  tools::time_event timing_("bicgstab::solve-gpu");
  int64_t const n = rhs.size();
  if (gv.size() != rhs.size()) // the other temps are initialized with a copy
    gv.resize(n);
  if (gt.size() != rhs.size()) // the other temps are initialized with a copy
    gt.resize(n);

  gr = rhs;

  int num_appy = 1;
  apply_lhs(-1, x.data(), 1, gr.data()); // r0 = b - A * x0

  P rho = compute->dot1(n, gr.data());

  grref = gr; // initialize rref (hat-r-0) and p
  gp    = gr;

  for (int i = 0; i < max_iter_; i++) {
    ++num_appy;
    apply_lhs(1, gp.data(), 0, gv.data()); // v = A * p

    P const alpha = rho / compute->dot(n, grref.data(), gv.data());

    compute->axpy(n, alpha, gp.data(), x.data());
    compute->axpy(n, -alpha, gv.data(), gr.data());

    if (compute->nrm2(n, gr.data()) < tolerance_) {
      return num_appy;
    }

    ++num_appy;
    apply_lhs(1, gr.data(), 0, gt.data()); // t = A * p

    P const omega = compute->dot(n, gr.data(), gt.data()) / compute->dot1(n, gt.data());

    compute->axpy(n, omega, gr.data(), x.data());
    compute->axpy(n, -omega, gt.data(), gr.data());

    if (compute->nrm2(n, gr.data()) < tolerance_) {
      return num_appy;
    }

    P const rho1 = compute->dot(n, grref.data(), gr.data());
    P const beta = (rho1 / rho) * (alpha / omega);

    gpu::compute_last_bicgstab(beta, omega, gr, gv, gp);

    rho = rho1;
  }
  std::cerr << "Warning: ASGarD BiCGSTAB solver failed to converge within "
            << max_iter_ << " iterations.\n";
  return num_appy;
}
#endif


template<typename P>
int gmres<P>::solve(
    operatoin_apply_precon<P> apply_precon,
    operatoin_apply_lhs<P> apply_lhs, std::vector<P> const &rhs,
    std::vector<P> &x) const
{
  #ifdef ASGARD_HAS_FAST_INSTERNAL_BLAS2
  bool constexpr use_asgard_blas = true;
  #else
  bool constexpr use_asgard_blas = false;
  #endif

  tools::time_event timing_("gmres::solve");
  int const n = static_cast<int>(rhs.size());
  assert(x.size() == rhs.size());

  basis.resize(static_cast<int64_t>(n) * (max_inner_ + 1));

  int num_appy = 0;

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

    if constexpr (use_asgard_blas)
      fm::scal_omp(n, P{1} / inner_res, basis.data());
    else
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

      if constexpr (use_asgard_blas)
      {
        fm::gemv_omp('T', n, inner_iterations + 1, P{1}, basis.data(), r, P{0}, coeff);
        fm::gemv_omp('N', n, inner_iterations + 1, P{-1}, basis.data(), coeff, P{1}, r);
      }
      else
      {
        fm::gemv('T', n, inner_iterations + 1, P{1}, basis.data(), r, P{0}, coeff);
        fm::gemv('N', n, inner_iterations + 1, P{-1}, basis.data(), coeff, P{1}, r);
      }

      P const nrm = fm::nrm2(n, r);

      if constexpr (use_asgard_blas)
        fm::scal_omp(n, P{1} / nrm, r);
      else
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
      if constexpr (use_asgard_blas)
        fm::gemv_omp('N', n, inner_iterations, P{1}, basis.data(), krylov_sol, P{1}, x.data());
      else
        fm::gemv('N', n, inner_iterations, P{1}, basis.data(), krylov_sol, P{1}, x.data());
    }
    ++outer_iterations;
    outer_res = inner_res;
  } // end outer iteration

  return num_appy;
}

template<typename P>
size_t gmres<P>::used_bytes() const {
  return (basis.size() + krylov_data.size()) * sizeof(P);
}

#ifdef ASGARD_USE_GPU
template<typename P>
int gmres<P>::solve(
    operatoin_apply_precon<P> apply_precon,
    operatoin_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
    gpu::vector<P> &x) const
{
  tools::time_event timing_("gmres::solve");
  int const n = static_cast<int>(rhs.size());
  assert(n == static_cast<int>(x.size()));

  gpu_basis.resize(static_cast<int64_t>(n) * (max_inner_ + 1));

  int num_appy = 0;

  int outer_iterations = 0;
  int inner_iterations = 0;

  P inner_res = 0.;
  P outer_res = tolerance_ + 1.0;
  while (outer_res > tolerance_ and outer_iterations < max_outer_)
  {
    gpu::memcopy_dev2dev(rhs.size(), rhs.data(), gpu_basis.data());
    apply_lhs(-1, x.data(), 1, gpu_basis.data());
    apply_precon(gpu_basis.data());
    ++num_appy;

    inner_res = compute->nrm2(n, gpu_basis.data());

    compute->scal(n, P{1} / inner_res, gpu_basis.data());
    krylov_sol[0] = inner_res;

    inner_iterations = 0;
    while (inner_res > tolerance_ and inner_iterations < max_inner_)
    {
      P *r = gpu_basis.data() + static_cast<int64_t>(n) * (inner_iterations + 1);
      apply_lhs(1, gpu_basis.data() + static_cast<int64_t>(n) * inner_iterations, 0, r);
      apply_precon(r);
      ++num_appy;

      compute->gemtv(n, inner_iterations + 1, P{1}, gpu_basis.data(), r, P{0}, gpu_coeffs.data());
      compute->gemv(n, inner_iterations + 1, P{-1}, gpu_basis.data(), gpu_coeffs.data(), P{1}, r);

      P const nrm = compute->nrm2(n, r);
      compute->scal(n, P{1} / nrm, r);

      // krylov projection coefficients for this iteration
      P *coeff = krylov_proj + (inner_iterations * (inner_iterations + 1)) / 2;

      gpu_coeffs.copy_to_host(inner_iterations + 1, coeff);

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
      gpu_coeffs.copy_from_host(inner_iterations, krylov_sol);
      compute->gemv(n, inner_iterations, P{1}, gpu_basis.data(), gpu_coeffs.data(), P{1}, x.data());
    }
    ++outer_iterations;
    outer_res = inner_res;
  } // end outer iteration

  return num_appy;
}
#endif

template<typename P>
void scaled_identity<P>::update(group_id group, size_t stage, sparse_grid const &grid,
                                term_manager<P> const &terms, P alpha)
{
  #ifdef ASGARD_USE_GPU
  num_entries = grid.num_indexes() * fm::ipow(terms.basis.pdof, grid.num_dims());
  #endif

  int const num_dims = terms.grid.num_dims();

  indexrange trange = terms.terms_group_range(group);

  if (grid_gen(group, stage) == -1) {
    // first time getting a call here, verify that the operators are scaled identity
    for (int i : trange) {
      term_md<P> const &term = terms.terms[i].tmd;
      rassert(term.is_separable(), "non-separable term detected in the scaled-identity solver");
      for (int d : iindexof(num_dims)) {
        term_1d<P> const &t1d = term.dim(d);
        rassert(t1d.is_identity() or t1d.is_volume(),
                "scaled-identity solver can be used only with volume and identity instances of term1d");
        if (t1d.is_volume())
          rassert(not t1d.rhs(), "detected non-constant coefficient for the scaled-identity solver");
      }
    }
  }

  P scal = 1;
  for (int i : trange) {
    term_md<P> const &term = terms.terms[i].tmd;
    assert(term.is_separable() and term.flux_dim() == -1);
    for (int d : iindexof(num_dims)) {
      term_1d<P> const &t1d = term.dim(d);
      if (t1d.is_volume())
        scal *= t1d.rhs_const();
    }
  }

  if (alpha != 0)
    set_alpha(group, stage, P{1} + alpha * scal); // Euler I + alpha * nu * I
  else
    set_alpha(group, stage, scal); // steady state, nu * I

  size_t const idx = s_index(group, stage);
  scale_[idx].grid_gen = grid.generation();
}

template<typename P>
void scaled_identity<P>::operator()(group_id group, size_t stage, std::vector<P> &x) const
{
  P const s = scale(group, stage);
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    x[i] = s * x[i];
}

#ifdef ASGARD_USE_GPU
template<typename P>
void scaled_identity<P>::operator()(group_id group, size_t stage, gpu::vector<P> &x) const
{
  assert(num_entries == x.size());
  gpu::set_scal(x.size(), scale(group, stage), x.data());
}
template<typename P>
void scaled_identity<P>::operator()(group_id group, size_t stage, P x[]) const
{
  gpu::set_scal(num_entries, scale(group, stage), x);
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template class direct<double>;
template class cg<double>;
template class bicgstab<double>;
template class gmres<double>;
template class scaled_identity<double>;
#endif // ASGARD_ENABLE_DOUBLE

#ifdef ASGARD_ENABLE_FLOAT
template class direct<float>;
template class cg<float>;
template class bicgstab<float>;
template class gmres<float>;
template class scaled_identity<float>;
#endif // ASGARD_ENABLE_FLOAT

} // namespace asgard::solvers

namespace asgard
{

template<typename P>
void solver_manager<P>::update_grid(
    group_id group, size_t stage, sparse_grid const &grid,
    connection_patterns const &conn, term_manager<P> const &terms, P alpha,
    preconditioner_data<P> &precon)
{
  tools::time_event timing_("updating solver");
  // assuming that the moments will cause the terms to change all the time
  // therefore, we need to update the matrices and preconditioners
  bool const needs_update = terms.has_sep_moments(group);

  // first, update the solver itself
  switch (method()) {
    case solver_method::direct: {
      solvers::direct<P> &solver = std::get<solvers::direct<P>>(var);
      if (needs_update or solver.grid_gen(group, stage) != grid.generation())
        solver.update(group, stage, grid, conn, terms, alpha);
    }
    break;
    case solver_method::scaled_identity: {
      solvers::scaled_identity<P> &solver = std::get<solvers::scaled_identity<P>>(var);
      if (needs_update or solver.grid_gen(group, stage) != grid.generation())
        solver.update(group, stage, grid, terms, alpha);
    }
    break;
    default: // iterative solvers don't need updating
      break;
  };

  // second, update the preconditioner
  // return if nothing more to do
  if (not needs_update and precon.valid_for(grid))
    return;

  precon.grid_gen = grid.generation(); // update the grid gen

  precon_method const method = precon; // get the precon method

  if (method == precon_method::jacobi) {
    std::vector<P> &jacobi = precon.jacobi();

    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1) {
      if (terms.resources.is_leader()) {
        terms.make_jacobi(group, terms.mpiwork);
        terms.resources.reduce_add(terms.mpiwork, jacobi);
      } else {
        terms.make_jacobi(group, jacobi);
        terms.resources.reduce_add(jacobi);
        return;
      }
    } else {
      terms.make_jacobi(group, jacobi);
    }
    #else
    terms.make_jacobi(group, jacobi);
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

    #ifdef ASGARD_USE_GPU
    if (terms.resources.is_leader()) {
      compute->set_device(gpu::device{0});
      precon.gpu_jacobi() = jacobi;
    }
    #endif
  }
}

template<typename P>
void solver_manager<P>::xpby(std::vector<P> const &x, P beta, P y[]) {
ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = x[i] + beta * y[i];
}

#ifdef ASGARD_USE_GPU
template<typename P>
void solver_manager<P>::iterate_solve(
    solvers::operatoin_apply_precon<P> prec, solvers::operatoin_apply_lhs<P> apply_lhs,
    gpu::vector<P> const &rhs, gpu::vector<P> &x) const
{
  if (method() == solver_method::bicgstab) {
    if (prec) {
      solvers::bicgstab<P> const &bicg = std::get<solvers::bicgstab<P>>(var);

      bicg.prec_y_gpu.resize(rhs.size());

      bicg.prec_rhs_gpu = rhs;
      prec(bicg.prec_rhs_gpu.data());

      num_apply += bicg.solve([&](P alpha, P const xx[], P beta, P y[])
          -> void {
            if (beta == 0) {
              apply_lhs(alpha, xx, 0, y);
              prec(y);
            } else {
              apply_lhs(alpha, xx, 0, bicg.prec_y_gpu.data());
              prec(bicg.prec_y_gpu.data());
              gpu::xpby(bicg.prec_y_gpu, beta, y);
            }
          }, bicg.prec_rhs_gpu, x);
    } else {
      num_apply += std::get<solvers::bicgstab<P>>(var).solve(apply_lhs, rhs, x);
    }
  } else { // if (opt == solve_opts::gmres)
    if (prec) {
      num_apply += std::get<solvers::gmres<P>>(var).solve(prec, apply_lhs, rhs, x);
    } else {
      num_apply += std::get<solvers::gmres<P>>(var).solve(
        [](P *)->void{ /* no preconditioner */ }, apply_lhs, rhs, x);
    }
  }
}
#endif

template<typename P>
void solver_manager<P>::print_opts(std::ostream &os) const
{
  os << "solver:\n";
  switch (method()) {
    case solver_method::direct:
      os << "  direct\n";
      break;
    case solver_method::bicgstab:
      os << "  bicgstab\n";
      os << "  tolerance:      " << std::get<solvers::bicgstab<P>>(var).tolerance() << '\n';
      os << "  max iterations: " << std::get<solvers::bicgstab<P>>(var).max_iter() << '\n';
      break;
    case solver_method::gmres:
      os << "  gmres\n";
      os << "  tolerance: " << std::get<solvers::gmres<P>>(var).tolerance() << '\n';
      os << "  max inner: " << std::get<solvers::gmres<P>>(var).max_inner() << '\n';
      os << "  max outer: " << std::get<solvers::gmres<P>>(var).max_outer() << '\n';
      break;
    case solver_method::scaled_identity:
      os << "  scaled-identity\n";
      break;
    default:
      break;
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct solver_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct solver_manager<float>;
#endif

}
