#include "asgard_bicgstab.hpp"
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
bicgstab_euler(const P dt, matrix_entry mentry,
               global_kron_matrix<P> const &mat,
               fk::vector<P, mem_type::owner, resrc> &x,
               fk::vector<P, mem_type::owner, resrc> const &b,
               int const max_iter, P const tolerance)
{
  auto const &pc = mat.template get_diagonal_preconditioner<resrc>();

  return bicgstab(
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
      max_iter, tolerance);
}
#else
template<typename P, resource resrc>
gmres_info<P>
bicgstab_euler(const P dt, kronmult_matrix<P> const &mat,
               fk::vector<P, mem_type::owner, resrc> &x,
               fk::vector<P, mem_type::owner, resrc> const &b,
               int const max_iter, P const tolerance)
{
  return bicgstab(
      [&](P const alpha, fk::vector<P, mem_type::view, resrc> const x_in,
          P const beta, fk::vector<P, mem_type::view, resrc> y) -> void {
        tools::time_event performance("kronmult - implicit", mat.flops());
        mat.template apply<resrc>(-dt * alpha, x_in.data(), beta, y.data());
        lib_dispatch::axpy<resrc>(y.size(), alpha, x_in.data(), 1, y.data(), 1);
      },
      fk::vector<P, mem_type::view, resrc>(x), b, no_op_preconditioner<P>(),
      max_iter, tolerance);
}
#endif

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
  if (tol == parser::NO_USER_VALUE_FP)
    tol = std::is_same_v<float, P> ? 1e-6 : 1e-12;
  expect(tol >= std::numeric_limits<P>::epsilon());

  int const n = b.size();
  expect(n == x.size());

  if (max_iter == parser::NO_USER_VALUE)
    max_iter = n;
  expect(max_iter > 0); // checked in program_options

  fk::vector<P> rho_1(1), rho_2(1), alpha(1), beta(1), omega(1);
  fk::vector<P, mem_type::owner, resrc> p, phat, s, shat, t, v;

  P normb = fm::nrm2(b);
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

  for (int i = 1; i <= max_iter; i++)
  {
    rho_1(0) = lib_dispatch::dot<resrc, P>(rtilde.size(), rtilde.data(), 1, r.data(), 1);
    if (rho_1(0) == 0)
    {
      return gmres_info<P>{resid, i};
    }
    if (i == 1)
    {
      p.resize(r.size()) = r;
    }
    else
    {
      beta(0) = (rho_1(0) / rho_2(0)) * (alpha(0) / omega(0));
      fm::axpy(v, p, P{-1} * omega(0));
      p = r + beta(0) * p;
    }
    phat.resize(p.size()) = p;
    fk::vector<P, mem_type::view, resrc> phat_v(phat);
    precondition(phat_v);
    v.resize(phat_v.size());
    mat(P{1.}, phat_v, P{0.}, fk::vector<P, mem_type::view, resrc>(v));
    alpha(0) = rho_1(0) / lib_dispatch::dot<resrc, P>(rtilde.size(), rtilde.data(), 1, v.data(), 1);
    s.resize(r.size()) = r;
    fm::axpy(v, s, P{-1} * alpha(0));
    resid    = fm::nrm2(s) / normb;
    if (resid < tol)
    {
      fm::axpy(phat, x, alpha(0));
      return gmres_info<P>{resid, i};
    }
    shat.resize(s.size()) = s;
    fk::vector<P, mem_type::view, resrc> shat_v(shat);
    precondition(shat_v);
    t.resize(shat.size());
    mat(P{1.}, shat_v, P{0.}, fk::vector<P, mem_type::view, resrc>(t));
    omega(0) = lib_dispatch::dot<resrc, P>(t.size(), t.data(), 1, s.data(), 1) / lib_dispatch::dot<resrc, P>(t.size(), t.data(), 1, t.data(), 1);
    fm::axpy(phat, x, alpha(0));
    fm::axpy(shat, x, omega(0));
    r = s;
    fm::axpy(t, r, P{-1} * omega(0));

    rho_2(0) = rho_1(0);
    resid    = fm::nrm2(r) / normb;
    if (resid < tol)
    {
      return gmres_info<P>{resid, i};
    }
    if (omega(0) == 0)
    {
      return gmres_info<P>{fm::nrm2(r) / normb, i};
    }
  }
  return gmres_info<P>{resid, max_iter};
}

#ifdef ASGARD_ENABLE_DOUBLE

template gmres_info<double>
bicgstab(fk::matrix<double> const &A, fk::vector<double> &x,
         fk::vector<double> const &b, fk::matrix<double> const &M,
         int const max_iter, double const tolerance);

#ifdef KRON_MODE_GLOBAL
template gmres_info<double>
bicgstab_euler(const double dt, matrix_entry mentry,
               global_kron_matrix<double> const &mat,
               fk::vector<double, mem_type::owner, resource::host> &x,
               fk::vector<double, mem_type::owner, resource::host> const &b,
               int const max_iter, double const tolerance);
#ifdef ASGARD_USE_CUDA
template gmres_info<double>
bicgstab_euler(const double dt, matrix_entry mentry,
               global_kron_matrix<double> const &mat,
               fk::vector<double, mem_type::owner, resource::device> &x,
               fk::vector<double, mem_type::owner, resource::device> const &b,
               int const max_iter, double const tolerance);
#endif
#else
template gmres_info<double>
bicgstab_euler(const double dt, kronmult_matrix<double> const &mat,
               fk::vector<double> &x, fk::vector<double> const &b,
               int const max_iter,
               double const tolerance);
#ifdef ASGARD_USE_CUDA
template gmres_info<double> bicgstab_euler(
    const double dt, kronmult_matrix<double> const &mat,
    fk::vector<double, mem_type::owner, resource::device> &x,
    fk::vector<double, mem_type::owner, resource::device> const &b,
    int const max_iter, double const tolerance);
#endif
#endif
#endif

#ifdef ASGARD_ENABLE_FLOAT

template gmres_info<float>
bicgstab(fk::matrix<float> const &A, fk::vector<float> &x,
         fk::vector<float> const &b, fk::matrix<float> const &M,
         int const max_iter, float const tolerance);

#ifdef KRON_MODE_GLOBAL
template gmres_info<float>
bicgstab_euler(const float dt, matrix_entry mentry,
               global_kron_matrix<float> const &mat,
               fk::vector<float, mem_type::owner, resource::host> &x,
               fk::vector<float, mem_type::owner, resource::host> const &b,
               int const max_iter, float const tolerance);
#ifdef ASGARD_USE_CUDA
template gmres_info<float>
bicgstab_euler(const float dt, matrix_entry mentry,
               global_kron_matrix<float> const &mat,
               fk::vector<float, mem_type::owner, resource::device> &x,
               fk::vector<float, mem_type::owner, resource::device> const &b,
               int const max_iter, float const tolerance);
#endif
#else
template gmres_info<float>
bicgstab_euler(const float dt, kronmult_matrix<float> const &mat,
               fk::vector<float> &x, fk::vector<float> const &b,
               int const max_iter,
               float const tolerance);
#ifdef ASGARD_USE_CUDA
template gmres_info<float> bicgstab_euler(
    const float dt, kronmult_matrix<float> const &mat,
    fk::vector<float, mem_type::owner, resource::device> &x,
    fk::vector<float, mem_type::owner, resource::device> const &b,
    int const max_iter, float const tolerance);
#endif
#endif
#endif

} // namespace asgard::solver
