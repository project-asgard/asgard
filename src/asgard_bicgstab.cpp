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
bicgstab_info<P>
bicgstab(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
         fk::matrix<P> const &M, int const restart, int const max_iter,
         P const tolerance)
{
  auto dense_matrix_wrapper =
      [&A](P const alpha, fk::vector<P, mem_type::view> const x_in,
           P const beta, fk::vector<P, mem_type::view> y) {
        fm::gemv(A, x_in, y, false, alpha, beta);
      };
  if (M.size() > 0)
    return bicgstab(dense_matrix_wrapper, fk::vector<P, mem_type::view>(x),
                    b, dense_preconditioner(M), restart, max_iter,
                    tolerance);
  else
    return bicgstab(dense_matrix_wrapper, fk::vector<P, mem_type::view>(x),
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
bicgstab_info<P>
bicgstab_euler(const P dt, matrix_entry mentry,
               global_kron_matrix<P> const &mat,
               fk::vector<P, mem_type::owner, resrc> &x,
               fk::vector<P, mem_type::owner, resrc> const &b,
               int const restart, int const max_iter, P const tolerance)
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
      restart, max_iter, tolerance);
}
#else
template<typename P, resource resrc>
bicgstab_info<P>
bicgstab_euler(const P dt, kronmult_matrix<P> const &mat,
               fk::vector<P, mem_type::owner, resrc> &x,
               fk::vector<P, mem_type::owner, resrc> const &b,
               int const restart, int const max_iter, P const tolerance)
{
  return bicgstab(
      [&](P const alpha, fk::vector<P, mem_type::view, resrc> const x_in,
          P const beta, fk::vector<P, mem_type::view, resrc> y) -> void {
        tools::time_event performance("kronmult - implicit", mat.flops());
        mat.template apply<resrc>(-dt * alpha, x_in.data(), beta, y.data());
        lib_dispatch::axpy<resrc>(y.size(), alpha, x_in.data(), 1, y.data(), 1);
      },
      fk::vector<P, mem_type::view, resrc>(x), b, no_op_preconditioner<P>(),
      restart, max_iter, tolerance);
}
#endif
} // namespace asgard::solver
