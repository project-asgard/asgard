#include "asgard_gpu_algorithms.hpp"

namespace asgard::gpu
{

inline int round_up(int64_t num, int max_threads) {
  int r = num / max_threads;
  if (r * max_threads < num) ++r;
  return r;
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
  expect(r.size() == v.size() and r.size() == p.size());
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


#ifdef ASGARD_ENABLE_DOUBLE

template void jacobi_apply<double>(gpu::vector<double> const &, double[]);

template void compute_last_bicgstab<double>(double, double, gpu::vector<double> const &,
                                            gpu::vector<double> const &, gpu::vector<double> &);

template void xpby(gpu::vector<double> const &x, double beta, double y[]);
template void axpby(int64_t, double, double const[], double, double[]);

#endif

#ifdef ASGARD_ENABLE_FLOAT

template void jacobi_apply<float>(gpu::vector<float> const &, float[]);

template void compute_last_bicgstab<float>(float, float, gpu::vector<float> const &,
                                           gpu::vector<float> const &, gpu::vector<float> &);

template void xpby(gpu::vector<float> const &x, float beta, float y[]);
template void axpby(int64_t, float, float const[], float, float[]);

#endif
}
