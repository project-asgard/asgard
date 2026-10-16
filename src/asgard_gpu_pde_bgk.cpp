#include "asgard_gpu_algorithms.hpp"

namespace asgard::gpu
{

template<typename P>
__device__ inline P subkernel_wavelet_maxwell_pdof1(
    P const gbasis[], int idx, P xleft, P xright, P u, P th)
{
  // pdof == 1, then each team has only one thread
  // this is a special case, no need to work so much with vectors and can do things more cleanly
  if (idx == 0) {

    return sqrt(P{1} / (xleft - xright));

  } else {

    P constexpr is2   = 7.071067811865475e-01; // 1 / sqrt(2)

    int lvl = idx;
    int num_idx_above = 1;
    while (lvl >>= 1)
      num_idx_above <<= 1;

    P const dv = (xright - xleft) / num_idx_above; // (b-a)/2^(lev-1)
    // Endpoints of wavelet element

    P left  = xleft + (idx - num_idx_above) * dv;;
    P right = left + dv/2;

    P z_l = (left-u) / sqrt(th);
    P z_r = (right-u) / sqrt(th);

    // Integrals of 1/√(2π)(1,z,z^2)exp(-z^2/2) from z=a..b
    P T0_l, T0_r;

    if constexpr (std::is_same_v<P, double>) {
      T0_l = 0.5 * erf(is2 * z_l); // P T0_l = 0.5*std::erf(z_l/std::sqrt(2.0));
      T0_r = 0.5 * erf(is2 * z_r);  // P T0_r = 0.5*std::erf(z_r/std::sqrt(2.0));
    } else {
      T0_l = 0.5f * erff(is2 * z_l);
      T0_r = 0.5f * erff(is2 * z_r);
    }

    P vleft = sqrt(P{2} / dv) * (T0_r - T0_l);

    left += dv;
    right += dv;

    z_l = (left-u) / sqrt(th);
    z_r = (right-u) / sqrt(th);

    if constexpr (std::is_same_v<P, double>) {
      T0_l = 0.5 * erf(is2 * z_l); // P T0_l = 0.5*std::erf(z_l/std::sqrt(2.0));
      T0_r = 0.5 * erf(is2 * z_r);  // P T0_r = 0.5*std::erf(z_r/std::sqrt(2.0));
    } else {
      T0_l = 0.5f * erff(is2 * z_l);
      T0_r = 0.5f * erff(is2 * z_r);
    }

    P vright = sqrt(P{2} / dv) * (T0_r - T0_l);

    return gbasis[0] * vleft + gbasis[1] * vright;
  }
}

template<typename P, int pdof>
__device__ inline void subkernel_wavelet_maxwell(int idx, P xleft, P xright, P u, P th, P out[])
{
  static_assert(pdof > 1, "the pdof == 1 case is handled separately");
  if (idx == 0) {
    // Integrate from -infty to infty to preserve collision invariants
    P const mid = 0.5 * (xright + xleft);

    // Integral of 1/√(2πt)(1,v,v^2)exp(-(v-u)^2/2t) for v=\pm\infty
    P const I0 = 1;
    P const I1 = u;
    P const I2 = th + u*u;
    P const I3 = 3.0*u*th + u*u*u;

    // Integral of 1/√(2πt)(phi_0,phi_1,phi_2)exp(-(v-u)^2/2t) where
    //   phi_i are the orthonormal Legendre polynomials on each element
    P const jv2 = 2.0 / (xright - xleft);
    P const jv  = sqrt(jv2); // inverse root jacobian

    out[0] = sqrt(1.0/2.0)*jv*I0;
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

      out[0 + pdof * l] = sqrt(1.0/2.0)*jv*I0;
      out[1 + pdof * l] = sqrt(3.0/2.0)*jv*jv*jv*( -I0*mid + I1 );

      if constexpr (pdof > 2)
        out[2 + pdof * l] = sqrt(5.0/8.0)*jv*( (3.0*jv*jv*jv*jv*mid*mid-1.0)*I0
                                               - 6.0*jv*jv*jv*jv*mid*I1 + 3.0*jv*jv*jv*jv*I2);
      if constexpr (pdof > 3)
        out[3 + pdof * l] = sqrt(7.0/8.0)*jv*( (3.0*jv2*mid - 5.0*jv2*jv2*jv2*mid*mid*mid)*I0
                                               + (15.0*jv2*jv2*jv2*mid*mid - 3.0*jv2)*I1
                                               -  15.0*jv2*jv2*jv2*mid*I2
                                               +   5.0*jv2*jv2*jv2*I3
                                              );

    }
  }
}

// the total number of threads must be at least 2 * pdof * pdof + 2, even if num is less
// num_teams is the number of teams per block
// num is the total vector size divided by pdof
template<typename P, int num_teams, int pdof, int num_threads>
__global__ void kernel_bgk_hybrid1D(
    int64_t num, P nu, P const mbasis[], int const cells[],
    P const mom0[], P const mom1[], P const mom2[], P vals[])
{
  static_assert(num_threads >= 2 * pdof * pdof + 2,
                "num_threads is used for sanity check in this kernel, "
                "must have enough threads to read mbasis into shared-cache as a one-shot operation");
  static_assert(num_teams == num_threads / pdof, "sanity check on the threads and teams");

  // team ID: threadIdx.y
  // member number within team: threadIdx.x
  // num teams in block: blockDim.y
  // num blocks: gridDim.x
  __shared__ P gbasis[2 * pdof * pdof];
  __shared__ P domain[2]; // left/right end-points of the domain
  __shared__ P out[num_teams][2 * pdof];

  // read the mbasis into __shared__ memory, all teams read the same data
  int64_t i = threadIdx.x + threadIdx.y * blockDim.x; // total tread index in block
  if (i < 2 * pdof * pdof) {
    gbasis[i] = mbasis[i];
  } else if (i < 2 * pdof * pdof + 2) {
    domain[i - 2 * pdof * pdof] = mbasis[i];
  }
  __syncthreads(); // ensure all i/o in __shared__ memory has been complete

  i = threadIdx.y + blockIdx.x * blockDim.y;

  while (i < num)
  {
    int const j = i * pdof; // j is the index in the global vector
    int const icell = i / pdof; // icell is the cell index

    P const u1 = mom1[j] / mom0[j];
    P const t  = mom2[j] / mom0[j] - u1 * u1;

    int const idx = cells[2 * icell + 1];

    if constexpr (pdof == 1)
    {
      vals[j] = nu * mom0[j] * subkernel_wavelet_maxwell_pdof1<P>
                                    (gbasis, idx, domain[0], domain[1], u1, t);
    }
    else
    {
      // this is more complex here ...
      if (threadIdx.x == 0) { // only one thread can populate the temp workspace
        subkernel_wavelet_maxwell<P, pdof>(idx, domain[0], domain[1], u1, t, out[threadIdx.y]);
      }
      __syncthreads(); // ensure the temp/io threads have been set

      P v = 0;
      if (idx == 0) {
        v = out[threadIdx.y][threadIdx.x]; // already in out
      } else { // out has 2 * pdof entries and must multiply by gbasis
        for (int k = 0; k < 2 * pdof; k++)
          v += gbasis[k * pdof + threadIdx.x] * out[threadIdx.y][k];
      }

      vals[j + threadIdx.x] = nu * mom0[j] * v;
    }

    i += gridDim.x * blockDim.y;
  }
}

template<typename P, int pdof>
void launch_kernel_bgk_hybrid1D(
    int64_t num, P nu, P const mbasis[], int const cells[],
    P const mom0[], P const mom1[], P const mom2[], P vals[])
{
  constexpr int num_threads = 1024;
  constexpr int num_teams = num_threads / pdof;

  // constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;
  // max_teams is the number of teams that we need to use, that cannot exceed 2^31
  // so capt it at 2^30
  int64_t const max_teams = (num + num_teams - 1) / num_teams;
  int launch_blocks = (1 << 30);
  if (max_teams < (1l << 30))
    launch_blocks = static_cast<int>(max_teams);

  dim3 const launch_grid(pdof, num_teams);

  kernel_bgk_hybrid1D<P, num_teams, pdof, num_threads>
    <<<launch_blocks, launch_grid>>>
    (num, nu, mbasis, cells, mom0, mom1, mom2, vals);
}

template<typename P>
void bgk_hybrid1D(int pdof, vector<P> const &gbasis, P nu, int const cells[], vector<P> const &mom0,
                  vector<P> const &mom1, vector<P> const &mom2, P vals[])
{
  int64_t const num = mom0.size() / pdof;

  switch (pdof) {
  case 1:
    launch_kernel_bgk_hybrid1D<P, 1>(num, nu, gbasis.data(), cells, mom0.data(), mom1.data(), mom2.data(), vals);
    break;
  case 2:
    launch_kernel_bgk_hybrid1D<P, 2>(num, nu, gbasis.data(), cells, mom0.data(), mom1.data(), mom2.data(), vals);
    break;
  case 3:
    launch_kernel_bgk_hybrid1D<P, 3>(num, nu, gbasis.data(), cells, mom0.data(), mom1.data(), mom2.data(), vals);
    break;
  case 4:
    launch_kernel_bgk_hybrid1D<P, 4>(num, nu, gbasis.data(), cells, mom0.data(), mom1.data(), mom2.data(), vals);
    break;
  default: // unreachable
    break;
  };
}

// num_teams is the number of teams per block
// num is the total vector size divided by pdof
template<typename P, int num_teams, int pdof, int num_threads>
__global__ void kernel_bgk_hybrid2D(
    int64_t num, P nu, P const mbasis[], int const cells[],
    P const mom0[], P const mom10[], P const mom01[], P const mom20[], P const mom02[], P vals[])
{
  static_assert(num_threads >= 2 * pdof * pdof + 4,
                "num_threads is used for sanity check in this kernel, "
                "must have enough threads to read mbasis into shared-cache as a one-shot operation");
  static_assert(num_teams == num_threads / (pdof * pdof), "sanity check on the threads and teams");

  // team ID: threadIdx.y
  // member number within team: threadIdx.x
  // num teams in block: blockDim.y
  // num blocks: gridDim.x
  __shared__ P gbasis[2 * pdof * pdof];
  __shared__ P domain[4]; // left/right end-points of the domain
  __shared__ P out[num_teams][2][2 * pdof];

  // read the mbasis into __shared__ memory, all teams read the same data
  int64_t i = threadIdx.x + threadIdx.y * blockDim.x; // total tread index in block
  if (i < 2 * pdof * pdof) {
    gbasis[i] = mbasis[i];
  } else if (i < 2 * pdof * pdof + 4) {
    domain[i - 2 * pdof * pdof] = mbasis[i];
  }
  __syncthreads(); // ensure all i/o in __shared__ memory has been complete

  i = threadIdx.y + blockIdx.x * blockDim.y;

  int const iv1 = threadIdx.x / pdof;
  int const iv2 = threadIdx.x % pdof;

  while (i < num)
  {
    int const j = i * pdof * pdof; // j is the index in the global vector
    int const icell = i / (pdof * pdof); // icell is the cell index

    P const u1 = mom10[j] / mom0[j];
    P const u2 = mom01[j] / mom0[j];
    P const t  = P{0.5} * ((mom20[j] + mom02[j]) / mom0[j] - u1 * u1 - u2 * u2);

    int const cell1 = cells[4 * icell + 2];
    int const cell2 = cells[4 * icell + 3];

    if constexpr (pdof == 1)
    {
      P v1 = subkernel_wavelet_maxwell_pdof1<P>(gbasis, cell1, domain[0], domain[1], u1, t);
      P v2 = subkernel_wavelet_maxwell_pdof1<P>(gbasis, cell2, domain[2], domain[3], u2, t);

      vals[j] = nu * mom0[j] * v1 * v2;
    }
    else
    {
      // this is more complex here ...
      if (threadIdx.x == 0) { // only one thread can populate the temp workspace
        subkernel_wavelet_maxwell<P, pdof>(cell1, domain[0], domain[1], u1, t, out[threadIdx.y][0]);
        subkernel_wavelet_maxwell<P, pdof>(cell2, domain[2], domain[3], u2, t, out[threadIdx.y][1]);
      }
      __syncthreads(); // ensure the temp/io threads have been set

      P v1 = 0;
      if (cell1 == 0) {
        v1 = out[threadIdx.y][0][iv1]; // already in out
      } else { // out has 2 * pdof entries and must multiply by gbasis
        for (int k = 0; k < 2 * pdof; k++)
          v1 += gbasis[k * pdof + iv1] * out[threadIdx.y][0][k];
      }

      P v2 = 0;
      if (cell2 == 0) {
        v2 = out[threadIdx.y][1][iv2]; // already in out
      } else { // out has 2 * pdof entries and must multiply by gbasis
        for (int k = 0; k < 2 * pdof; k++)
          v2 += gbasis[k * pdof + iv2] * out[threadIdx.y][1][k];
      }

      vals[j + threadIdx.x] = nu * mom0[j] * v1 * v2;
    }

    i += gridDim.x * blockDim.y;
  }
}

template<typename P, int pdof>
void launch_kernel_bgk_hybrid2D(
    int64_t num, P nu, P const mbasis[], int const cells[],
    P const mom0[], P const mom10[], P const mom01[], P const mom20[], P const mom02[], P vals[])
{
  constexpr int num_threads = 1024;
  constexpr int num_teams = num_threads / (pdof * pdof);

  // constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;
  // max_teams is the number of teams that we need to use, that cannot exceed 2^31
  // so capt it at 2^30
  int64_t const max_teams = (num + num_teams - 1) / num_teams;
  int launch_blocks = (1 << 30);
  if (max_teams < (1l << 30))
    launch_blocks = static_cast<int>(max_teams);

  dim3 const launch_grid(pdof * pdof, num_teams);

  kernel_bgk_hybrid2D<P, num_teams, pdof, num_threads>
    <<<launch_blocks, launch_grid>>>
    (num, nu, mbasis, cells, mom0, mom10, mom01, mom20, mom02, vals);
}

template<typename P>
void bgk_hybrid2D(int pdof, vector<P> const &gbasis, P nu, int const cells[], vector<P> const &mom0,
                  vector<P> const &mom10, vector<P> const &mom01,
                  vector<P> const &mom20, vector<P> const &mom02, P vals[])
{
  int64_t const num = mom0.size() / (pdof * pdof);

  switch (pdof) {
  case 1:
    launch_kernel_bgk_hybrid2D<P, 1>(num, nu, gbasis.data(), cells, mom0.data(),
                                     mom10.data(), mom01.data(), mom20.data(), mom02.data(), vals);
    break;
  case 2:
    launch_kernel_bgk_hybrid2D<P, 2>(num, nu, gbasis.data(), cells, mom0.data(),
                                     mom10.data(), mom01.data(), mom20.data(), mom02.data(), vals);
    break;
  case 3:
    launch_kernel_bgk_hybrid2D<P, 3>(num, nu, gbasis.data(), cells, mom0.data(),
                                     mom10.data(), mom01.data(), mom20.data(), mom02.data(), vals);
    break;
  case 4:
    launch_kernel_bgk_hybrid2D<P, 4>(num, nu, gbasis.data(), cells, mom0.data(),
                                     mom10.data(), mom01.data(), mom20.data(), mom02.data(), vals);
    break;
  default: // unreachable
    break;
  };
}

// num_teams is the number of teams per block
// num is the total vector size divided by pdof
template<typename P, int num_teams, int pdof, int num_threads>
__global__ void kernel_bgk_hybrid3D(
    int64_t num, P nu, P const mbasis[], int const cells[], P const mom0[],
    P const mom100[], P const mom010[], P const mom001[],
    P const mom200[], P const mom020[], P const mom002[], P vals[])
{
  static_assert(num_threads >= 2 * pdof * pdof + 6,
                "num_threads is used for sanity check in this kernel, "
                "must have enough threads to read mbasis into shared-cache as a one-shot operation");
  static_assert(num_teams == num_threads / (pdof * pdof * pdof), "sanity check on the threads and teams");

  // team ID: threadIdx.y
  // member number within team: threadIdx.x
  // num teams in block: blockDim.y
  // num blocks: gridDim.x
  __shared__ P gbasis[2 * pdof * pdof];
  __shared__ P domain[6]; // left/right end-points of the domain
  __shared__ P out[num_teams][3][2 * pdof];

  // read the mbasis into __shared__ memory, all teams read the same data
  int64_t i = threadIdx.x + threadIdx.y * blockDim.x; // total tread index in block
  if (i < 2 * pdof * pdof) {
    gbasis[i] = mbasis[i];
  } else if (i < 2 * pdof * pdof + 6) {
    domain[i - 2 * pdof * pdof] = mbasis[i];
  }
  __syncthreads(); // ensure all i/o in __shared__ memory has been complete

  i = threadIdx.y + blockIdx.x * blockDim.y;

  int const iv1 = threadIdx.x / (pdof * pdof);
  int const iv2 = (threadIdx.x / pdof) % pdof;
  int const iv3 = threadIdx.x % pdof;

  while (i < num)
  {
    int const j = i * pdof * pdof * pdof; // j is the index in the global vector
    int const icell = i / (pdof * pdof * pdof); // icell is the cell index

    P const u1 = mom100[j] / mom0[j];
    P const u2 = mom010[j] / mom0[j];
    P const u3 = mom001[j] / mom0[j];
    P const t  =  (1.0/3.0) * ((mom200[j] + mom020[j] + mom002[j]) / mom0[j] - u1 * u1 - u2 * u2 - u3 * u3);

    int const cell1 = cells[6 * icell + 3];
    int const cell2 = cells[6 * icell + 4];
    int const cell3 = cells[6 * icell + 5];

    if constexpr (pdof == 1)
    {
      P v1 = subkernel_wavelet_maxwell_pdof1<P>(gbasis, cell1, domain[0], domain[1], u1, t);
      P v2 = subkernel_wavelet_maxwell_pdof1<P>(gbasis, cell2, domain[2], domain[3], u2, t);
      P v3 = subkernel_wavelet_maxwell_pdof1<P>(gbasis, cell3, domain[4], domain[5], u3, t);

      vals[j] = nu * mom0[j] * v1 * v2 * v3;
    }
    else
    {
      // this is more complex here ...
      if (threadIdx.x == 0) { // only one thread can populate the temp workspace
        subkernel_wavelet_maxwell<P, pdof>(cell1, domain[0], domain[1], u1, t, out[threadIdx.y][0]);
        subkernel_wavelet_maxwell<P, pdof>(cell2, domain[2], domain[3], u2, t, out[threadIdx.y][1]);
        subkernel_wavelet_maxwell<P, pdof>(cell3, domain[4], domain[5], u3, t, out[threadIdx.y][2]);
      }
      __syncthreads(); // ensure the temp/io threads have been set

      P v1 = 0;
      if (cell1 == 0) {
        v1 = out[threadIdx.y][0][iv1]; // already in out
      } else { // out has 2 * pdof entries and must multiply by gbasis
        for (int k = 0; k < 2 * pdof; k++)
          v1 += gbasis[k * pdof + iv1] * out[threadIdx.y][0][k];
      }

      P v2 = 0;
      if (cell2 == 0) {
        v2 = out[threadIdx.y][1][iv2]; // already in out
      } else { // out has 2 * pdof entries and must multiply by gbasis
        for (int k = 0; k < 2 * pdof; k++)
          v2 += gbasis[k * pdof + iv2] * out[threadIdx.y][1][k];
      }

      P v3 = 0;
      if (cell3 == 0) {
        v3 = out[threadIdx.y][2][iv3]; // already in out
      } else { // out has 2 * pdof entries and must multiply by gbasis
        for (int k = 0; k < 2 * pdof; k++)
          v3 += gbasis[k * pdof + iv3] * out[threadIdx.y][2][k];
      }

      vals[j + threadIdx.x] = nu * mom0[j] * v1 * v2 * v3;
    }

    i += gridDim.x * blockDim.y;
  }
}

template<typename P, int pdof>
void launch_kernel_bgk_hybrid3D(
    int64_t num, P nu, P const mbasis[], int const cells[], P const mom0[],
    P const mom100[], P const mom010[], P const mom001[],
    P const mom200[], P const mom020[], P const mom002[], P vals[])
{
  constexpr int num_threads = 1024;
  constexpr int team_size = pdof * pdof * pdof;
  constexpr int num_teams = num_threads / team_size;

  // constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;
  // max_teams is the number of teams that we need to use, that cannot exceed 2^31
  // so capt it at 2^30
  int64_t const max_teams = (num + num_teams - 1) / num_teams;
  int launch_blocks = (1 << 30);
  if (max_teams < (1l << 30))
    launch_blocks = static_cast<int>(max_teams);

  dim3 const launch_grid(team_size, num_teams);

  kernel_bgk_hybrid3D<P, num_teams, pdof, num_threads>
    <<<launch_blocks, launch_grid>>>
    (num, nu, mbasis, cells, mom0, mom100, mom010, mom001, mom200, mom020, mom002, vals);

  compute->device_synchronize();
}

template<typename P>
void bgk_hybrid3D(int pdof, vector<P> const &gbasis, P nu, int const cells[], vector<P> const &mom0,
                  vector<P> const &mom100, vector<P> const &mom010, vector<P> const &mom001,
                  vector<P> const &mom200, vector<P> const &mom020, vector<P> const &mom002,
                  P vals[])
{
  int64_t const num = mom0.size() / (pdof * pdof * pdof);

  switch (pdof) {
  case 1:
    launch_kernel_bgk_hybrid3D<P, 1>(num, nu, gbasis.data(), cells, mom0.data(),
                                     mom100.data(), mom010.data(), mom001.data(),
                                     mom200.data(), mom020.data(), mom002.data(), vals);
    break;
  case 2:
    launch_kernel_bgk_hybrid3D<P, 2>(num, nu, gbasis.data(), cells, mom0.data(),
                                     mom100.data(), mom010.data(), mom001.data(),
                                     mom200.data(), mom020.data(), mom002.data(), vals);
    break;
  case 3:
    launch_kernel_bgk_hybrid3D<P, 3>(num, nu, gbasis.data(), cells, mom0.data(),
                                     mom100.data(), mom010.data(), mom001.data(),
                                     mom200.data(), mom020.data(), mom002.data(), vals);
    break;
  case 4:
    launch_kernel_bgk_hybrid3D<P, 4>(num, nu, gbasis.data(), cells, mom0.data(),
                                     mom100.data(), mom010.data(), mom001.data(),
                                     mom200.data(), mom020.data(), mom002.data(), vals);
    break;
  default: // unreachable
    break;
  };
}

#ifdef ASGARD_ENABLE_DOUBLE

template void bgk_hybrid1D(int, vector<double> const &, double, int const[], vector<double> const &,
                           vector<double> const &, vector<double> const &, double[]);

template void bgk_hybrid2D(int, vector<double> const &, double, int const[], vector<double> const &,
                           vector<double> const &, vector<double> const &,
                           vector<double> const &, vector<double> const &, double[]);

template void bgk_hybrid3D(int, vector<double> const &, double, int const[], vector<double> const &,
                           vector<double> const &, vector<double> const &, vector<double> const &,
                           vector<double> const &, vector<double> const &, vector<double> const &,
                           double[]);
#endif

#ifdef ASGARD_ENABLE_FLOAT

template void bgk_hybrid1D(int, vector<float> const &, float, int const[], vector<float> const &,
                           vector<float> const &, vector<float> const &, float[]);

template void bgk_hybrid2D(int, vector<float> const &, float, int const[], vector<float> const &,
                           vector<float> const &, vector<float> const &,
                           vector<float> const &, vector<float> const &, float[]);

template void bgk_hybrid3D(int, vector<float> const &, float, int const[], vector<float> const &,
                           vector<float> const &, vector<float> const &, vector<float> const &,
                           vector<float> const &, vector<float> const &, vector<float> const &,
                           float[]);
#endif
}
