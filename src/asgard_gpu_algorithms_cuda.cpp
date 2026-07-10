#include "asgard_gpu_algorithms.hpp"

namespace asgard::gpu
{

inline int round_up(int64_t num, int max_threads) {
  return (num + max_threads - 1) / max_threads;
}

template<typename P, int num_threads>
__global__ void kernel_jacobi_apply(int64_t num, P const jacobi[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] *= jacobi[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void jacobi_apply(gpu::vector<P> const &jacobi, P y[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(jacobi.size(), max_threads);

  kernel_jacobi_apply<P, max_threads><<<num_blocks, max_threads>>>
      (jacobi.size(), jacobi.data(), y);
}

template<typename P, int num_threads>
__global__ void kernel_bicgstab_last(int64_t num, P beta, P omega, P const r[], P const v[], P p[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    p[i] = r[i] + beta * (p[i] - omega * v[i]);
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void compute_last_bicgstab(P beta, P omega, gpu::vector<P> const &r,
                           gpu::vector<P> const &v, gpu::vector<P> &p)
{
  assert(r.size() == v.size() and r.size() == p.size());
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(r.size(), max_threads);

  kernel_bicgstab_last<P, max_threads><<<num_blocks, max_threads>>>
      (r.size(), beta, omega, r.data(), v.data(), p.data());
}

template<typename P, int num_threads>
__global__ void kernel_xpby(int64_t num, P beta, P const x[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = x[i] + beta * y[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void xpby(gpu::vector<P> const &x, P beta, P y[]) {
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(x.size(), max_threads);

  kernel_xpby<P, max_threads><<<num_blocks, max_threads>>>(x.size(), beta, x.data(), y);
}

template<typename P, int num_threads>
__global__ void kernel_axpby(int64_t num, P alpha, P const x[], P beta, P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = alpha * x[i] + beta * y[i];
    i += num_threads * gridDim.x;
  }
}
template<typename P, int num_threads>
__global__ void kernel_axpby0(int64_t num, P alpha, P const x[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = alpha * x[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void axpby(int64_t num, P alpha, P const x[], P beta, P y[]) {
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(num, max_threads);

  if (beta == 0)
    kernel_axpby0<P, max_threads><<<num_blocks, max_threads>>>(num, alpha, x, y);
  else
    kernel_axpby<P, max_threads><<<num_blocks, max_threads>>>(num, alpha, x, beta, y);
}

template<typename P, int num_threads>
__global__ void kernel_axpbygz(int64_t num, P alpha, P const x[], P beta, P const y[], P gamma, P z[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    z[i] = alpha * x[i] + beta * y[i] + gamma * z[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void axpbygz(int64_t num, no_deduce<P> alpha, P const x[], no_deduce<P> beta, P const y[],
             no_deduce<P> gamma, P z[]) {
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(num, max_threads);

  kernel_axpbygz<P, max_threads><<<num_blocks, max_threads>>>(num, alpha, x, beta, y, gamma, z);
}

template<typename P, int num_threads>
__global__ void kernel_setscal(int64_t num, P alpha, P x[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    x[i] = alpha * x[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void set_scal(int64_t num, P alpha, P x[]) {
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(num, max_threads);
  kernel_setscal<P, max_threads><<<num_blocks, max_threads>>>(num, alpha, x);
}

template<typename P, int num_threads>
__global__ void kernel_sum2(int64_t num, P const x[], P a1, P const x1[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = x[i] + a1 * x1[i];
    i += num_threads * gridDim.x;
  }
}
template<typename P, int num_threads>
__global__ void kernel_sum3(int64_t num, P const x[], P a1, P const x1[],
                            P a2, P const x2[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = x[i] + a1 * x1[i] + a2 * x2[i];
    i += num_threads * gridDim.x;
  }
}
template<typename P, int num_threads>
__global__ void kernel_sum4(int64_t num, P const x[], P a1, P const x1[],
                            P a2, P const x2[], P a3, P const x3[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = x[i] + a1 * x1[i] + a2 * x2[i] + a3 * x3[i];
    i += num_threads * gridDim.x;
  }
}
template<typename P, int num_threads>
__global__ void kernel_sum5(int64_t num, P const x[], P a1, P const x1[],
                            P a2, P const x2[], P a3, P const x3[],
                            P a4, P const x4[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = x[i] + a1 * x1[i] + a2 * x2[i] + a3 * x3[i] + a4 * x4[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void sum2(gpu::vector<P> const &x, no_deduce<P> a1, gpu::vector<P> const &x1, gpu::vector<P> &y) {
  constexpr int max_threads = 1024;
  int64_t const num = x.size();
  int const num_blocks = round_up(num, max_threads);
  kernel_sum2<P, max_threads><<<num_blocks, max_threads>>>(num, x.data(), a1, x1.data(), y.data());
}
template<typename P>
void sum3(gpu::vector<P> const &x, no_deduce<P> a1, gpu::vector<P> const &x1, no_deduce<P> a2, gpu::vector<P> const &x2,
          gpu::vector<P> &y) {
  constexpr int max_threads = 1024;
  int64_t const num = x.size();
  int const num_blocks = round_up(num, max_threads);
  kernel_sum3<P, max_threads><<<num_blocks, max_threads>>>(
      num, x.data(), a1, x1.data(), a2, x2.data(), y.data());
}
template<typename P>
void sum4(gpu::vector<P> const &x, no_deduce<P> a1, gpu::vector<P> const &x1, no_deduce<P> a2, gpu::vector<P> const &x2,
          no_deduce<P> a3, gpu::vector<P> const &x3, gpu::vector<P> &y) {
  constexpr int max_threads = 1024;
  int64_t const num = x.size();
  int const num_blocks = round_up(num, max_threads);
  kernel_sum4<P, max_threads><<<num_blocks, max_threads>>>(
      num, x.data(), a1, x1.data(), a2, x2.data(), a3, x3.data(), y.data());
}
template<typename P>
void sum5(gpu::vector<P> const &x, no_deduce<P> a1, gpu::vector<P> const &x1, no_deduce<P> a2, gpu::vector<P> const &x2,
          no_deduce<P> a3, gpu::vector<P> const &x3, no_deduce<P> a4, gpu::vector<P> const &x4, gpu::vector<P> &y) {
  constexpr int max_threads = 1024;
  int64_t const num = x.size();
  int const num_blocks = round_up(num, max_threads);
  kernel_sum5<P, max_threads><<<num_blocks, max_threads>>>(
      num, x.data(), a1, x1.data(), a2, x2.data(), a3, x3.data(), a4, x4.data(), y.data());
}

template<typename P, int num_threads>
__global__ void kernel_num_non_finite(int64_t num, P const x[], int *sum)
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    if (not isfinite(x[i]))
      atomicAdd(sum, int{1});
    i += num_threads * gridDim.x;
  }
}

template<typename P>
int num_non_finite(int64_t num, P const x[]) {
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(num, max_threads);
  static ::asgard::gpu::vector<int> sum = []() -> ::asgard::gpu::vector<int> {
        ::asgard::gpu::vector<int> result(1);
        compute->fill_zeros(result);
        return result;
      }();
  kernel_num_non_finite<P, max_threads><<<num_blocks, max_threads>>>(num, x, sum.data());
  int cpu_res = 0;
  gpu::memcopy_dev2host(1, sum.data(), &cpu_res);
  return cpu_res;
}

template<typename P>
__global__ void cg_calc_alpha_kernel(P const *rho, P const *p_dot_q, P *alpha) {
  *alpha = *rho / *p_dot_q;
}

template<typename P>
__global__ void cg_calc_beta_kernel(P const *rho_new, P const *rho, P *beta) {
  *beta = *rho_new / *rho;
}

template<typename P>
__global__ void cg_update_x_r_kernel(int64_t n, P const *alpha, P const p[], P const q[], P x[], P r[]) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    P a = *alpha;
    x[i] += a * p[i];
    r[i] -= a * q[i];
  }
}

template<typename P>
__global__ void cg_update_p_kernel(int64_t n, P const *beta, P const r[], P p[]) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = r[i] + (*beta) * p[i];
  }
}

template<typename P>
__global__ void cg_update_rho_kernel(P const *rho_new, P *rho) {
  *rho = *rho_new;
}

template<typename P>
__global__ void zero_kernel(P *x) {
  *x = 0;
}

template<typename P>
void cg_calc_alpha(P const *rho, P const *p_dot_q, P *alpha) {
  cg_calc_alpha_kernel<<<1, 1>>>(rho, p_dot_q, alpha);
}
template<typename P>
void cg_calc_beta(P const *rho_new, P const *rho, P *beta) {
  cg_calc_beta_kernel<<<1, 1>>>(rho_new, rho, beta);
}
template<typename P>
void cg_update_x_r(int64_t n, P const *alpha, P const p[], P const q[], P x[], P r[]) {
  int const threads = 1024;
  int const blocks = (n + threads - 1) / threads;
  cg_update_x_r_kernel<<<blocks, threads>>>(n, alpha, p, q, x, r);
}
template<typename P>
void cg_update_p(int64_t n, P const *beta, P const r[], P p[]) {
  int const threads = 1024;
  int const blocks = (n + threads - 1) / threads;
  cg_update_p_kernel<<<blocks, threads>>>(n, beta, r, p);
}
template<typename P>
void cg_update_rho(P const *rho_new, P *rho) {
  cg_update_rho_kernel<<<1, 1>>>(rho_new, rho);
}
template<typename P>
void set_zero(P *x) {
  zero_kernel<<<1, 1>>>(x);
}

#ifdef ASGARD_ENABLE_DOUBLE
template void jacobi_apply<double>(gpu::vector<double> const &, double[]);

template void compute_last_bicgstab<double>(double, double, gpu::vector<double> const &,
                                            gpu::vector<double> const &, gpu::vector<double> &);

template void xpby(gpu::vector<double> const &x, double beta, double y[]);
template void axpby(int64_t, double, double const[], double, double[]);
template void axpbygz(int64_t, double, double const[], double, double const[], double, double[]);
template void set_scal(int64_t, double, double[]);

template void sum2(gpu::vector<double> const &, double, gpu::vector<double> const &, gpu::vector<double> &);
template void sum3(gpu::vector<double> const &, double, gpu::vector<double> const &,
                   double, gpu::vector<double> const &, gpu::vector<double> &);
template void sum4(gpu::vector<double> const &, double, gpu::vector<double> const &,
                   double, gpu::vector<double> const &, double, gpu::vector<double> const &, gpu::vector<double> &);
template void sum5(gpu::vector<double> const &, double, gpu::vector<double> const &, double, gpu::vector<double> const &,
                   double, gpu::vector<double> const &, double, gpu::vector<double> const &, gpu::vector<double> &);

template int num_non_finite(int64_t num, double const x[]);

template void cg_calc_alpha<double>(double const*, double const*, double*);
template void cg_calc_beta<double>(double const*, double const*, double*);
template void cg_update_x_r<double>(int64_t, double const*, double const*, double const*, double*, double*);
template void cg_update_p<double>(int64_t, double const*, double const*, double*);
template void cg_update_rho<double>(double const*, double*);
template void set_zero<double>(double*);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void jacobi_apply<float>(gpu::vector<float> const &, float[]);

template void compute_last_bicgstab<float>(float, float, gpu::vector<float> const &,
                                           gpu::vector<float> const &, gpu::vector<float> &);

template void xpby(gpu::vector<float> const &x, float beta, float y[]);
template void axpby(int64_t, float, float const[], float, float[]);
template void axpbygz(int64_t, float, float const[], float, float const[], float, float[]);
template void set_scal(int64_t, float, float[]);

template void sum2(gpu::vector<float> const &, float, gpu::vector<float> const &, gpu::vector<float> &);
template void sum3(gpu::vector<float> const &, float, gpu::vector<float> const &,
                   float, gpu::vector<float> const &, gpu::vector<float> &);
template void sum4(gpu::vector<float> const &, float, gpu::vector<float> const &,
                   float, gpu::vector<float> const &, float, gpu::vector<float> const &, gpu::vector<float> &);
template void sum5(gpu::vector<float> const &, float, gpu::vector<float> const &, float, gpu::vector<float> const &,
                   float, gpu::vector<float> const &, float, gpu::vector<float> const &, gpu::vector<float> &);

template int num_non_finite(int64_t num, float const x[]);

template void cg_calc_alpha<float>(float const*, float const*, float*);
template void cg_calc_beta<float>(float const*, float const*, float*);
template void cg_update_x_r<float>(int64_t, float const*, float const*, float const*, float*, float*);
template void cg_update_p<float>(int64_t, float const*, float const*, float*);
template void cg_update_rho<float>(float const*, float*);
template void set_zero<float>(float*);
#endif
}
