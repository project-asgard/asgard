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

template<typename P, int num_threads, int num_pos>
__global__ void kernel_bgk_vel1(
    int64_t num, P nu, P const nodes[],
    P const mom0[], P const mom1[], P const mom2[], P vals[])
{
   P constexpr PI_ = 3.141592653589793;

  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num)
  {
    P const v = nodes[i * (num_pos + 1) + num_pos];

    P const n = mom0[i];
    P const u = mom1[i] / n;
    P const d = v - u;
    P const t = mom2[i] / n - u * u;

    vals[i] = exp(- P{0.5} * d * d / t) * nu * n / sqrt(2 * PI_ * t);

    i += num_threads * gridDim.x;
  }
}

template<typename P>
void bgk_vel1(P nu, int num_pos, P const nodes[], vector<P> const &mom0,
              vector<P> const &mom1, vector<P> const &mom2, P vals[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = (mom0.size() + max_threads - 1) / max_threads;

  switch (num_pos) {
  case 1:
    kernel_bgk_vel1<P, max_threads, 1><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom1.data(), mom2.data(), vals);
    break;
  case 2:
    kernel_bgk_vel1<P, max_threads, 2><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom1.data(), mom2.data(), vals);
    break;
  case 3:
    kernel_bgk_vel1<P, max_threads, 3><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom1.data(), mom2.data(), vals);
    break;
  default: // unreachable
    break;
  }
}

template<typename P, int num_threads, int num_pos>
__global__ void kernel_bgk_vel2(
    int64_t num, P nu, P const nodes[], P const mom0[],
    P const mom10[], P const mom01[], P const mom20[], P const mom02[], P vals[])
{
   P constexpr PI_ = 3.141592653589793;

  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num)
  {
    P const n = mom0[i];
    P const u0 = mom10[i] / n;
    P const u1 = mom01[i] / n;
    P const t = 0.5 * ((mom20[i] + mom02[i]) / n - u0 * u0 - u1 * u1);

    P const vu0 = nodes[i * (num_pos + 2) + num_pos] - u0;
    P const vu1 = nodes[i * (num_pos + 2) + num_pos + 1] - u1;
    P const d = vu0 * vu0 + vu1 * vu1;

    vals[i] = exp(- P{0.5} * d / t) * nu * n / (2 * PI_ * t);

    i += num_threads * gridDim.x;
  }
}

template<typename P>
void bgk_vel2(P nu, int num_pos, P const nodes[], vector<P> const &mom0,
              vector<P> const &mom10, vector<P> const &mom01,
              vector<P> const &mom20, vector<P> const &mom02, P vals[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = (mom0.size() + max_threads - 1) / max_threads;

  switch (num_pos) {
  case 1:
    kernel_bgk_vel2<P, max_threads, 1><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom10.data(), mom01.data(),
         mom20.data(), mom02.data(), vals);
    break;
  case 2:
    kernel_bgk_vel2<P, max_threads, 2><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom10.data(), mom01.data(),
         mom20.data(), mom02.data(), vals);
    break;
  case 3:
    kernel_bgk_vel2<P, max_threads, 3><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom10.data(), mom01.data(),
         mom20.data(), mom02.data(), vals);
    break;
  default: // unreachable
    break;
  }
}

template<typename P, int num_threads, int num_pos>
__global__ void kernel_bgk_vel3(
    int64_t num, P nu, P const nodes[], P const mom0[],
    P const mom100[], P const mom010[], P const mom001[],
    P const mom200[], P const mom020[], P const mom002[], P vals[])
{
   P constexpr PI_ = 3.141592653589793;

  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num)
  {
    P const n = mom0[i];
    P const u0 = mom100[i] / n;
    P const u1 = mom010[i] / n;
    P const u2 = mom001[i] / n;
    P const t = ((mom200[i] + mom020[i] + mom002[i]) / n - u0 * u0 - u1 * u1 - u2 * u2) / P{3};

    P const vu0 = nodes[i * (num_pos + 3) + num_pos] - u0;
    P const vu1 = nodes[i * (num_pos + 3) + num_pos + 1] - u1;
    P const vu2 = nodes[i * (num_pos + 3) + num_pos + 2] - u2;
    P const d = vu0 * vu0 + vu1 * vu1 + vu2 * vu2;
    P const pit = 2 * PI_ * t;

    vals[i] = exp(- P{0.5} * d / t) * nu * n / (pit * std::sqrt(pit));

    i += num_threads * gridDim.x;
  }
}

template<typename P>
void bgk_vel3(P nu, int num_pos, P const nodes[], vector<P> const &mom0,
              vector<P> const &mom100, vector<P> const &mom010, vector<P> const &mom001,
              vector<P> const &mom200, vector<P> const &mom020, vector<P> const &mom002,
              P vals[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = (mom0.size() + max_threads - 1) / max_threads;

  switch (num_pos) {
  case 1:
    kernel_bgk_vel3<P, max_threads, 1><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom100.data(), mom010.data(),
         mom001.data(), mom200.data(), mom020.data(), mom002.data(), vals);
    break;
  case 2:
    kernel_bgk_vel3<P, max_threads, 2><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom100.data(), mom010.data(),
         mom001.data(), mom200.data(), mom020.data(), mom002.data(), vals);
    break;
  case 3:
    kernel_bgk_vel3<P, max_threads, 3><<<num_blocks, max_threads>>>
        (mom0.size(), nu, nodes, mom0.data(), mom100.data(), mom010.data(),
         mom001.data(), mom200.data(), mom020.data(), mom002.data(), vals);
    break;
  default: // unreachable
    break;
  };
}

template<typename A, typename B>
struct gpu_is_same {
  static constexpr bool value = false;
};
template<typename A>
struct gpu_is_same<A, A> {
  static constexpr bool value = true;
};


template<typename P, int pdof>
__device__ inline void subkernel_wavelet_maxwell(
    P const gbasis[], int idx, P xleft, P xright, P u, P th, P temp[], P out[])
{
  if (idx == 0) {
    // Integrate from -infty to infty to preserve collision invariants
    P const mid = 0.5*(xleft + xright);

    // Integral of 1/√(2πt)(1,v,v^2)exp(-(v-u)^2/2t) for v=\pm\infty
    P const I0 = 1.0;
    P const I1 = u;
    P const I2 = th + u*u;
    P const I3 = 3.0*u*th + u*u*u;

    // Integral of 1/√(2πt)(phi_0,phi_1,phi_2)exp(-(v-u)^2/2t) where
    //   phi_i are the orthonormal Legendre polynomials on each element
    P const jv2 = 2.0 / (xleft - xright);
    P const jv  = sqrt(jv2); // inverse root jacobian

    out[0] = sqrt(1.0/2.0)*jv*I0;
    if constexpr (pdof > 1)
      out[1] = sqrt(3.0/2.0)*jv*jv2*( -I0*mid + I1 );
    if constexpr (pdof > 2)
      out[2] = sqrt(5.0/8.0)*jv*( (3.0*jv2*jv2*mid*mid-1.0)*I0 - 6.0*jv2*jv2*mid*I1 + 3.0*jv2*jv2*I2);
    if constexpr (pdof > 3)
      out[3] = sqrt(7.0/8.0)*jv*(  (3.0*jv2*mid - 5.0*jv2*jv2*jv2*mid*mid*mid)*I0
                               + (15.0*jv2*jv2*jv2*mid*mid - 3.0*jv2)*I1
                               -  15.0*jv2*jv2*jv2*mid*I2
                               +  5.0*jv2*jv2*jv2*I3
                               );

  } else {

    P constexpr s2    = 1.414213562373095; // sqrt(2)
    P constexpr is2   = 7.071067811865475e-01; // 1 / sqrt(2)
    P constexpr is2pi = 3.989422804014327e-01; // 1.0/std::sqrt(2.0*PI)

    int lvl = idx;
    int num_idx_above = 1;
    while (lvl >>= 1)
      num_idx_above <<= 1;

    P const dv = (xright - xleft) / num_idx_above; // (b-a)/2^(lev-1)
    // Endpoints of wavelet element
    P const loca = xleft + (idx - num_idx_above) * dv;

    // For v_lev > 0 wavlets are piecewise polynomials
    // Integrate legendre polynomials the map to wavelets
    for (int l=0; l<2; l++)
    {
      P left  = loca + l*dv/2;
      P right = loca + (l+1)*dv/2;
      P mid = P{0.5} * (left+right);

      // v -> (v-u)/sqrt(t)
      P z_l = (left-u) / sqrt(th);
      P z_r = (right-u) / sqrt(th);

      // Integrals of 1/√(2π)(1,z,z^2)exp(-z^2/2) from z=a..b
      P T0_l, T0_r, T1_l, T1_r;

      if constexpr (std::is_same_v<P, double>) {

        T0_l = 0.5 * erf(is2 * z_l); // P T0_l = 0.5*std::erf(z_l/std::sqrt(2.0));
        T0_r = 0.5 * erf(is2 * z_r);  // P T0_r = 0.5*std::erf(z_r/std::sqrt(2.0));

        T1_l = is2pi * exp(-0.5*z_l*z_l); // P T1_l = 1.0/std::sqrt(2.0*PI)*std::exp(-0.5*z_l*z_l);
        T1_r = is2pi * exp(-0.5*z_r*z_r); // P T1_r = 1.0/std::sqrt(2.0*PI)*std::exp(-0.5*z_r*z_r);

      } else {

        T0_l = P{0.5} * erff(is2 * z_l);
        T0_r = P{0.5} * erff(is2 * z_r);

        T1_l = is2pi * expf(-P{0.5} * z_l * z_l);
        T1_r = is2pi * expf(-P{0.5} * z_r * z_r);
      }


      P C0 =   T0_r - T0_l;
      P C1 = -(T1_r - T1_l);
      P C2 = -(z_r * T1_r - z_l * T1_l) + C0;
      P C3 =  (z_l * z_l + P{2}) * T1_l - (z_r * z_r + P{2}) * T1_r;

      // Integral of 1/√(2πt)(1,v,v^2)exp(-(v-u)^2/2t) dv using z = (v-u)/√t
      P I0 =       C0;
      P I1 =     u*C0 +          sqrt(th)*C1;
      P I2 =   u*u*C0 +   P{2}*u*sqrt(th)*C1 +        th*C2;
      P I3 = u*u*u*C0 + P{3}*u*u*sqrt(th)*C1 + P{3}*u*th*C2 + sqrt(th)*th*C3;

      // Integral of 1/√(2πt)(phi_0,phi_1,phi_2)exp(-(v-u)^2/2t) where
      //   phi_i are the orthonormal Legendre polynomials on each element
      P jv = sqrt(4.0/dv); // inverse root jacobian
      P jv2= jv*jv;

      out[0] = sqrt(1.0/2.0)*jv*I0;
      if (pdof > 1)
        out[1] = sqrt(3.0/2.0)*jv*jv*jv*( -I0*mid + I1 );
      if (pdof > 2)
        out[2] = sqrt(5.0/8.0)*jv*( (3.0*jv*jv*jv*jv*mid*mid-1.0)*I0 - 6.0*jv*jv*jv*jv*mid*I1 + 3.0*jv*jv*jv*jv*I2);
      if (pdof > 3)
        out[3] = sqrt(7.0/8.0)*jv*( (3.0*jv2*mid - 5.0*jv2*jv2*jv2*mid*mid*mid)*I0
                                   + (15.0*jv2*jv2*jv2*mid*mid - 3.0*jv2)*I1
                                   -  15.0*jv2*jv2*jv2*mid*I2
                                   +   5.0*jv2*jv2*jv2*I3
                                  );

    }

  }

}


template<typename P, int num_threads>
__global__ void kernel_bgk_hybrid1D(
    int64_t num, P nu, P const nodes[],
    P const mom0[], P const mom1[], P const mom2[], P vals[])
{
  P constexpr PI_ = 3.141592653589793;

  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num)
  {
    // P const v = nodes[i * (num_pos + 1) + num_pos];
    //
    // P const n = mom0[i];
    // P const u = mom1[i] / n;
    // P const d = v - u;
    // P const t = mom2[i] / n - u * u;
    //
    // vals[i] = exp(- P{0.5} * d * d / t) * nu * n / sqrt(2 * PI_ * t);

    i += num_threads * gridDim.x;
  }
}

template<typename P>
void bgk_hybrid1D(int pdof, vector<P> const &gbasis, P nu, P const nodes[], vector<P> const &mom0,
                  vector<P> const &mom1, vector<P> const &mom2, P vals[])
{
  switch (pdof) {
  case 1:
    break;
  case 2:
    break;
  case 3:
    break;
  case 4:
    break;
  default: // unreachable
    break;
  };
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

template void bgk_vel1(double, int, double const[], vector<double> const &,
                       vector<double> const &, vector<double> const &, double vals[]);

template void bgk_vel2(double, int, double const[], vector<double> const &,
                       vector<double> const &, vector<double> const &, vector<double> const &,
                       vector<double> const &, double vals[]);

template void bgk_vel3(double, int, double const[], vector<double> const &,
                       vector<double> const &, vector<double> const &, vector<double> const &,
                       vector<double> const &, vector<double> const &, vector<double> const &,
                       double vals[]);

template void bgk_hybrid1D(int, vector<double> const &, double, double const[], vector<double> const &,
                           vector<double> const &, vector<double> const &, double[]);
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

template void bgk_vel1(float, int, float const[], vector<float> const &,
                       vector<float> const &, vector<float> const &, float vals[]);

template void bgk_vel2(float, int, float const[], vector<float> const &,
                       vector<float> const &, vector<float> const &, vector<float> const &,
                       vector<float> const &, float vals[]);

template void bgk_vel3(float, int, float const[], vector<float> const &,
                       vector<float> const &, vector<float> const &, vector<float> const &,
                       vector<float> const &, vector<float> const &, vector<float> const &,
                       float vals[]);

template void bgk_hybrid1D(int, vector<float> const &, float, float const[], vector<float> const &,
                           vector<float> const &, vector<float> const &, float[]);
#endif
}
