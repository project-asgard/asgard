#include "asgard_iterative_solver.hpp"

#include "asgard_blas.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_algorithms.hpp"
#endif

namespace asgard::solvers
{

template<typename P>
int cg<P>::solve(operation_apply_precon<P> precon, operation_apply_lhs<P> apply_lhs,
                 std::vector<P> const &rhs, std::vector<P> &x) const
{
  tools::time_event timing_("cg::solve");
  int64_t const n = static_cast<int64_t>(rhs.size());

  auto dot = [&](std::vector<P> const &a, std::vector<P> const &b) -> P {
      P sum = 0;
      ASGARD_OMP_PARFOR_SIMD_EXTRA(reduction(+:sum))
      for (int64_t i = 0; i < n; i++) sum += a[i] * b[i];
      return sum;
  };

  q.resize(n);
  r = rhs;
  int num_apply = 1;
  apply_lhs(-1.0, x.data(), 1.0, r.data()); // r = b - A * x

  P rho = dot(r, r);

  if (precon != nullptr) {
    z = r;
    precon(z.data());
    p = z;
    rho = dot(r, z);
  } else {
    p = r;
  }

  for (int i = 0; i < max_iter_; i++) {
    ++num_apply;
    apply_lhs(1.0, p.data(), 0.0, q.data()); // q = A * p

    P const p_dot_q = dot(p, q);
    P const alpha = rho / p_dot_q;

    ASGARD_OMP_PARFOR_SIMD
    for (int64_t k = 0; k < n; k++) {
        x[k] += alpha * p[k];
        r[k] -= alpha * q[k];
    }

    P rho_new = dot(r, r);

    // Exact check based on the new residual
    if (rho_new < tolerance_) {
      return num_apply;
    }

    if (precon != nullptr) {
      z = r;
      precon(z.data());
      rho_new = dot(r, z);
    }

    P const beta = rho_new / rho;

    if (precon != nullptr) {
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t k = 0; k < n; k++) {
          p[k] = z[k] + beta * p[k];
      }
    } else {
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t k = 0; k < n; k++) {
          p[k] = r[k] + beta * p[k];
      }
    }

    rho = rho_new;
  }
  std::cerr << "Warning: ASGarD CPU CG solver failed to converge within " << max_iter_ << " iterations.\n";
  return num_apply;
}

#ifdef ASGARD_USE_GPU
template<typename P>
int cg<P>::solve(operation_apply_precon<P> precon, operation_apply_lhs<P> apply_lhs,
                 gpu::vector<P> const &rhs, gpu::vector<P> &x) const
{
  tools::time_event timing_("cg::solve-gpu");
  int64_t const n = rhs.size();

  if (gq.size() != n)
    gq.resize(n);

  gr = rhs;
  int num_apply = 1;
  apply_lhs(-1.0, x.data(), 1.0, gr.data());
  compute->dot1_device(n, gr.data(), grho.data());

  if (precon != nullptr) {
    gz = gr;
    precon(gz.data());
    gp = gz;
    compute->dot_device(n, gr.data(), gz.data(), grho.data());
  } else {
    gp = gr;
  }

  for (int i = 0; i < max_iter_; i++) {
    ++num_apply;

    apply_lhs(1.0, gp.data(), 0.0, gq.data());

    compute->dot_device(n, gp.data(), gq.data(), gp_dot_gq.data());
    gpu::cg_update_x_r(n, grho.data(), gp_dot_gq.data(), gp.data(), gq.data(), x.data(), gr.data());
    compute->dot1_device(n, gr.data(), grho_new.data());

    if (i % 10 == 0) {
      P rho_new;
      grho_new.copy_to_host(1, &rho_new);
      if (rho_new < tolerance_) {
        return num_apply;
      }
    }

    if (precon != nullptr) {
      gz = gr;
      precon(gz.data());
      compute->dot_device(n, gr.data(), gz.data(), grho_new.data());
    }

    if (precon != nullptr) {
      gpu::cg_update_p(n, grho_new.data(), grho.data(), gz.data(), gp.data());
    } else {
      gpu::cg_update_p(n, grho_new.data(), grho.data(), gr.data(), gp.data());
    }
    std::swap(grho_new, grho);
  }

  std::cerr << "Warning: ASGarD GPU CG solver failed to converge within " << max_iter_ << " iterations.\n";
  return num_apply;
}
#endif

template<typename P>
size_t cg<P>::used_bytes() const {
  size_t total = r.size() + p.size() + q.size();
  return total * sizeof(P);
}

template<typename P>
int bicgstab<P>::solve(
    operation_apply_lhs<P> apply_lhs, std::vector<P> const &rhs, std::vector<P> &x) const
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

  int num_apply = 1;
  apply_lhs(-1, x.data(), 1, r.data()); // r0 = b - A * x0

  P rho = dot1(r);

  rref = r; // initialize rref (hat-r-0) and p
  p    = r;

  for (int i = 0; i < max_iter_; i++) {
    ++num_apply;
    apply_lhs(1, p.data(), 0, v.data()); // v = A * p

    P const alpha = rho / dot(rref, v);

    axpy(alpha, p, x);
    axpy(-alpha, v, r);

    if (nrm(r) < tolerance_) {
      return num_apply;
    }

    ++num_apply;
    apply_lhs(1, r.data(), 0, t.data()); // t = A * p

    P const omega = dot(r, t) / dot1(t);

    axpy(omega, r, x);
    axpy(-omega, t, r);

    if (nrm(r) < tolerance_) {
      return num_apply;
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
  return num_apply;
}

template<typename P>
size_t bicgstab<P>::used_bytes() const {
  size_t total = rref.size() + r.size() + p.size() + v.size() + t.size();
  return total * sizeof(P);
}

#ifdef ASGARD_USE_GPU
template<typename P>
int bicgstab<P>::solve(operation_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
                       gpu::vector<P> &x) const
{
  tools::time_event timing_("bicgstab::solve-gpu");
  int64_t const n = rhs.size();
  if (gv.size() != rhs.size()) // the other temps are initialized with a copy
    gv.resize(n);
  if (gt.size() != rhs.size()) // the other temps are initialized with a copy
    gt.resize(n);

  gr = rhs;

  int num_apply = 1;
  apply_lhs(-1, x.data(), 1, gr.data()); // r0 = b - A * x0

  P rho = compute->dot1(n, gr.data());

  grref = gr; // initialize rref (hat-r-0) and p
  gp    = gr;

  for (int i = 0; i < max_iter_; i++) {
    ++num_apply;
    apply_lhs(1, gp.data(), 0, gv.data()); // v = A * p

    P const alpha = rho / compute->dot(n, grref.data(), gv.data());

    compute->axpy(n, alpha, gp.data(), x.data());
    compute->axpy(n, -alpha, gv.data(), gr.data());

    if (compute->nrm2(n, gr.data()) < tolerance_) {
      return num_apply;
    }

    ++num_apply;
    apply_lhs(1, gr.data(), 0, gt.data()); // t = A * p

    P const omega = compute->dot(n, gr.data(), gt.data()) / compute->dot1(n, gt.data());

    compute->axpy(n, omega, gr.data(), x.data());
    compute->axpy(n, -omega, gt.data(), gr.data());

    if (compute->nrm2(n, gr.data()) < tolerance_) {
      return num_apply;
    }

    P const rho1 = compute->dot(n, grref.data(), gr.data());
    P const beta = (rho1 / rho) * (alpha / omega);

    gpu::compute_last_bicgstab(beta, omega, gr, gv, gp);

    rho = rho1;
  }
  std::cerr << "Warning: ASGarD BiCGSTAB solver failed to converge within "
            << max_iter_ << " iterations.\n";
  return num_apply;
}
#endif

template<typename P>
int gmres<P>::solve(
    operation_apply_precon<P> apply_precon,
    operation_apply_lhs<P> apply_lhs, std::vector<P> const &rhs,
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

  int num_apply = 0;

  int outer_iterations = 0;
  int inner_iterations = 0;

  P inner_res = 0.;
  P outer_res = tolerance_ + 1.0;
  while (outer_res > tolerance_ and outer_iterations < max_outer_)
  {
    std::copy(rhs.begin(), rhs.end(), basis.begin());
    apply_lhs(-1, x.data(), 1, basis.data());
    apply_precon(basis.data());
    ++num_apply;

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
      ++num_apply;

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

  return num_apply;
}

template<typename P>
size_t gmres<P>::used_bytes() const {
  return (basis.size() + krylov_data.size()) * sizeof(P);
}

#ifdef ASGARD_USE_GPU
template<typename P>
int gmres<P>::solve(
    operation_apply_precon<P> apply_precon,
    operation_apply_lhs<P> apply_lhs, gpu::vector<P> const &rhs,
    gpu::vector<P> &x) const
{
  tools::time_event timing_("gmres::solve");
  int const n = static_cast<int>(rhs.size());
  assert(n == static_cast<int>(x.size()));

  gpu_basis.resize(static_cast<int64_t>(n) * (max_inner_ + 1));

  int num_apply = 0;

  int outer_iterations = 0;
  int inner_iterations = 0;

  P inner_res = 0.;
  P outer_res = tolerance_ + 1.0;
  while (outer_res > tolerance_ and outer_iterations < max_outer_)
  {
    gpu::memcopy_dev2dev(rhs.size(), rhs.data(), gpu_basis.data());
    apply_lhs(-1, x.data(), 1, gpu_basis.data());
    apply_precon(gpu_basis.data());
    ++num_apply;

    inner_res = compute->nrm2(n, gpu_basis.data());

    compute->scal(n, P{1} / inner_res, gpu_basis.data());
    krylov_sol[0] = inner_res;

    inner_iterations = 0;
    while (inner_res > tolerance_ and inner_iterations < max_inner_)
    {
      P *r = gpu_basis.data() + static_cast<int64_t>(n) * (inner_iterations + 1);
      apply_lhs(1, gpu_basis.data() + static_cast<int64_t>(n) * inner_iterations, 0, r);
      apply_precon(r);
      ++num_apply;

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

  return num_apply;
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template class cg<double>;
template class bicgstab<double>;
template class gmres<double>;
#endif // ASGARD_ENABLE_DOUBLE

#ifdef ASGARD_ENABLE_FLOAT
template class cg<float>;
template class bicgstab<float>;
template class gmres<float>;
#endif // ASGARD_ENABLE_FLOAT

} // namespace::asgard::solvers
