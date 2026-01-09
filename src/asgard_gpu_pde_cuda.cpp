#include "asgard_gpu_algorithms.hpp"

namespace asgard::gpu
{

template<typename P, int num_threads>
__global__ void kernel_moment_ratio(int64_t num, P nu, P const momX[], P const mom0[],
                                    P const f[], P vals[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    vals[i] = (nu * momX[i] * f[i]) / mom0[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void moment_ratio(P nu, vector<P> const &momX, vector<P> const &mom0, P const f[], P vals[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = (momX.size() + max_threads - 1) / max_threads;

  kernel_moment_ratio<P, max_threads><<<num_blocks, max_threads>>>
      (momX.size(), nu, momX.data(), mom0.data(), f, vals);
}

template<typename P, int num_threads>
__global__ void kernel_lbc_vel1(int64_t num, P nu, P const mom0[], P const mom1[],
                                P const mom2[], P const f[], P vals[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    vals[i] = nu * (mom2[i] / mom0[i] - (mom1[i] * mom1[i]) / (mom0[i] * mom0[i])) * f[i];;
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void lbc_vel1(P nu, vector<P> const &mom0, vector<P> const &mom1,
              vector<P> const &mom2, P const f[], P vals[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = (mom0.size() + max_threads - 1) / max_threads;

  kernel_lbc_vel1<P, max_threads><<<num_blocks, max_threads>>>
      (mom0.size(), nu, mom0.data(), mom1.data(), mom2.data(), f, vals);
}

template<typename P, int num_threads>
__global__ void kernel_lbc_vel2(
    int64_t num, P nu, P const mom0[], P const mom10[], P const mom01[], P const mom20[],
    P const mom02[], P const f[], P vals[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    vals[i] = 0.5 *  nu * f[i] *
              ((mom20[i] + mom02[i]) / mom0[i] -
               (mom10[i] * mom10[i] + mom01[i] * mom01[i]) / (mom0[i] * mom0[i]));
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void lbc_vel2(P nu, vector<P> const &mom0, vector<P> const &mom10, vector<P> const &mom01,
              vector<P> const &mom20, vector<P> const &mom02, P const f[], P vals[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = (mom0.size() + max_threads - 1) / max_threads;

  kernel_lbc_vel2<P, max_threads><<<num_blocks, max_threads>>>
      (mom0.size(), nu, mom0.data(), mom10.data(), mom01.data(),
       mom20.data(), mom02.data(), f, vals);
}

template<typename P, int num_threads>
__global__ void kernel_lbc_vel3(
    int64_t num, P nu, P const mom0[], P const mom100[], P const mom010[], P const mom001[],
    P const mom200[], P const mom020[], P const mom002[], P const f[], P vals[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    vals[i] = (P{1} / P{3}) *  nu * f[i] *
               ((mom200[i] + mom020[i] + mom002[i]) / mom0[i] -
                (mom100[i] * mom100[i] + mom010[i] * mom010[i] + mom001[i] * mom001[i]) / (mom0[i] * mom0[i]));
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void lbc_vel3(P nu, vector<P> const &mom0, vector<P> const &mom100, vector<P> const &mom010,
              vector<P> const &mom001, vector<P> const &mom200, vector<P> const &mom020,
              vector<P> const &mom002, P const f[], P vals[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = (mom0.size() + max_threads - 1) / max_threads;

  kernel_lbc_vel3<P, max_threads><<<num_blocks, max_threads>>>
      (mom0.size(), nu, mom0.data(), mom100.data(), mom010.data(), mom001.data(),
       mom200.data(), mom020.data(), mom002.data(), f, vals);
}

#ifdef ASGARD_ENABLE_DOUBLE
template void moment_ratio(double, gpu::vector<double> const &, gpu::vector<double> const &,
                           double const[], double[]);

template void lbc_vel1(double, vector<double> const &, vector<double> const &,
                       vector<double> const &, double const[], double[]);

template void lbc_vel2(double, vector<double> const &, vector<double> const &,
                       vector<double> const &, vector<double> const &, vector<double> const &,
                       double const[], double[]);

template void lbc_vel3(double, vector<double> const &, vector<double> const &,
                       vector<double> const &, vector<double> const &, vector<double> const &,
                       vector<double> const &, vector<double> const &, double const[], double[]);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void moment_ratio(float, gpu::vector<float> const &, gpu::vector<float> const &,
                           float const[], float[]);

template void lbc_vel1(float nu, vector<float> const &mom0, vector<float> const &mom1,
                       vector<float> const &mom2, float const f[], float vals[]);

template void lbc_vel2(float, vector<float> const &, vector<float> const &,
                       vector<float> const &, vector<float> const &, vector<float> const &,
                       float const[], float[]);

template void lbc_vel3(float, vector<float> const &, vector<float> const &,
                       vector<float> const &, vector<float> const &, vector<float> const &,
                       vector<float> const &, vector<float> const &, float const[], float[]);
#endif
}
