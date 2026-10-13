#include "asgard_pde_functions.hpp"
#include "asgard_small_mats.hpp"
#include "asgard_wavelet_basis.hpp"
#include "asgard_discretization.hpp"
#include "asgard_interp.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_pde.hpp"
#endif

namespace asgard
{

int prog_opts::max_level(dimension_id const &dim) const {
  int lstart = 0;
  if (start_levels.empty() and not default_start_levels.empty()) {
    lstart = (static_cast<size_t>(dim()) < default_start_levels.size())
              ? default_start_levels[dim()]
              : default_start_levels.front();
  }
  if (not start_levels.empty()) {
    lstart = (static_cast<size_t>(dim()) < start_levels.size())
              ? start_levels[dim()]
              : start_levels.front();

  }
  int lmax = 0;
  if (not max_levels.empty()) {
    lmax = (static_cast<size_t>(dim()) < max_levels.size())
            ? max_levels[dim()]
            : max_levels.front();

  }
  return std::max(lmax, lstart);
}

template<typename P>
template<typename opmode>
void pde_scheme<P>::process(operators::lenard_bernstein_collisions lbc)
{
  static_assert(std::is_same_v<opmode, source<P>> or std::is_same_v<opmode, term_md<P>>);

  rassert(domain_.num_pos() > 0, "cannot set lenard_bernstein_collisions operator for a pde_domain with no position dimensions");
  rassert(domain_.num_vel() > 0, "cannot set lenard_bernstein_collisions operator for a pde_domain with no velocity dimensions");
  rassert(domain_.num_pos() <= 3, "cannot set lenard_bernstein_collisions operator for a pde_domain with more than 3 position dimensions");
  rassert(domain_.num_vel() <= 3, "cannot set lenard_bernstein_collisions operator for a pde_domain with more than 3 velocity dimensions");
  rassert(lbc.nu > 0, "the collision frequency has to be positive");

  P const nu = static_cast<P>(lbc.nu);

  auto vnu = [=](std::vector<P> const &v, std::vector<P> &fv)
        -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < v.size(); i++)
        fv[i] = -nu * v[i];
    };

  term_1d<P> I = term_identity{};

  term_1d<P> divv_nuv = term_div<P>{vnu, flux_type::upwind, boundary_type::bothsides};

  term_1d<P> div = term_div<P>{1, flux_type::central, boundary_type::bothsides};

  term_1d<P> div_grad = term_1d<P>({term_div<P>{-1, flux_type::upwind, boundary_type::bothsides},
                                    term_grad<P>{1, flux_type::upwind, boundary_type::bothsides}});

  int const num_pos = domain_.num_pos();

  if constexpr (std::is_same_v<opmode, source<P>>) {
    if (num_pos == 1) // separable case has no weight
      return;
  }

  switch(domain_.num_vel())
  {
  case 1:
    if (num_pos == 1)
    {
      *this += term_md<P>({I, divv_nuv});

      *this += term_md<P>({term_moment_over_density{lbc.nu, moment{1}}, div});
      *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, div_grad});
    }
    else // interpolation case
    {
      moment_id const m0 = this->register_moment(moment{0});
      moment_id const m1 = this->register_moment(moment{1});
      moment_id const m2 = this->register_moment(moment{2});

      #ifdef ASGARD_USE_GPU
      auto m1over0 = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                         P const f[], P vals[]) -> void
        {
          gpu::moment_ratio(nu, moments[m1], moments[m0], f, vals);
        };

      auto theta = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                       P const f[], P vals[]) -> void
        {
          gpu::lbc_vel1(nu, moments[m0], moments[m1], moments[m2], f, vals);
        };
      #else
      auto m1over0 = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                         std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0 = moments[m0];
          std::vector<P> const &mom1 = moments[m1];
          assert(static_cast<size_t>(x.num_strips()) == mom0.size());
          assert(static_cast<size_t>(x.num_strips()) == mom1.size());
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = (nu * mom1[i] * f[i]) / mom0[i];
        };

      auto theta = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                       std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0 = moments[m0];
          std::vector<P> const &mom1 = moments[m1];
          std::vector<P> const &mom2 = moments[m2];
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = nu * (mom2[i] / mom0[i] - (mom1[i] * mom1[i]) / (mom0[i] * mom0[i])) * f[i];
        };
      #endif

      if constexpr (std::is_same_v<opmode, source<P>>) {
        this->set_adapt_weight(theta, {m0, m1, m2});
      } else {
        if (num_pos == 2) {
          *this += term_md<P>({I, I, divv_nuv});
          *this += term_md<P>{term_md<P>{I, I, div}, term_interp<P>{m1over0, {m0, m1}}};
          *this += term_md<P>{term_md<P>{I, I, div_grad}, term_interp<P>{theta, {m0, m1, m2}}};
        } else {
          *this += term_md<P>({I, I, I, divv_nuv});
          *this += term_md<P>{term_md<P>{I, I, I, div}, term_interp<P>{m1over0, {m0, m1}}};
          *this += term_md<P>{term_md<P>{I, I, I, div_grad}, term_interp<P>{theta, {m0, m1, m2}}};
        }
      }
    }
    break;

  case 2:
    if (num_pos == 1)
    {
      *this += term_md<P>({I, divv_nuv, I});
      *this += term_md<P>({I, I, divv_nuv});

      *this += term_md<P>({term_moment_over_density{lbc.nu, moment{1, 0}}, div, I});
      *this += term_md<P>({term_moment_over_density{lbc.nu, moment{0, 1}}, I, div});

      *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, div_grad, I});
      *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, I, div_grad});
    }
    else // interpolation case
    {
      moment_id const m0  = this->register_moment(moment{0, 0});
      moment_id const m10 = this->register_moment(moment{1, 0});
      moment_id const m01 = this->register_moment(moment{0, 1});
      moment_id const m20 = this->register_moment(moment{2, 0});
      moment_id const m02 = this->register_moment(moment{0, 2});

      #ifdef ASGARD_USE_GPU
      auto m10over0 = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                          P const f[], P vals[]) -> void
        {
          gpu::moment_ratio(nu, moments[m10], moments[m0], f, vals);
        };
      auto m01over0 = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                          P const f[], P vals[]) -> void
        {
          gpu::moment_ratio(nu, moments[m01], moments[m0], f, vals);
        };

      auto theta = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                       P const f[], P vals[]) -> void
        {
          gpu::lbc_vel2(nu, moments[m0], moments[m10], moments[m01], moments[m20],
                        moments[m02], f, vals);
        };
      #else
      auto m10over0 = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                          std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0  = moments[m0];
          std::vector<P> const &mom10 = moments[m10];
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = (nu * mom10[i] * f[i]) / mom0[i];
        };
      auto m01over0 = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                          std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0  = moments[m0];
          std::vector<P> const &mom01 = moments[m01];
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = (nu * mom01[i] * f[i]) / mom0[i];
        };
      auto theta = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                       std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0  = moments[m0];
          std::vector<P> const &mom10 = moments[m10];
          std::vector<P> const &mom01 = moments[m01];
          std::vector<P> const &mom20 = moments[m20];
          std::vector<P> const &mom02 = moments[m02];
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = 0.5 *  nu * f[i] *
                      ((mom20[i] + mom02[i]) / mom0[i] -
                       (mom10[i] * mom10[i] + mom01[i] * mom01[i]) / (mom0[i] * mom0[i]));
        };
      #endif

      if constexpr (std::is_same_v<opmode, source<P>>) {
        this->set_adapt_weight(theta, {m0, m10, m01, m20, m02});
      } else {
        if (num_pos == 2) {
          *this += term_md<P>({I, I, divv_nuv, I});
          *this += term_md<P>({I, I, I, divv_nuv});

          *this += term_md<P>{term_md<P>{I, I, div, I}, term_interp<P>{m10over0, {m0, m10}}};
          *this += term_md<P>{term_md<P>{I, I, I, div}, term_interp<P>{m01over0, {m0, m01}}};

          *this += term_md<P>{term_md<P>{I, I, div_grad, I}, term_interp<P>{theta, {m0, m10, m01, m20, m02}}};
          *this += term_md<P>{term_md<P>{I, I, I, div_grad}, term_interp<P>{theta, {m0, m10, m01, m20, m02}}};
        } else {
          *this += term_md<P>({I, I, I, divv_nuv, I});
          *this += term_md<P>({I, I, I, I, divv_nuv});

          *this += term_md<P>{term_md<P>{I, I, I, div, I}, term_interp<P>{m10over0, {m0, m10}}};
          *this += term_md<P>{term_md<P>{I, I, I, I, div}, term_interp<P>{m01over0, {m0, m01}}};

          *this += term_md<P>{term_md<P>{I, I, I, div_grad, I}, term_interp<P>{theta, {m0, m10, m01, m20, m02}}};
          *this += term_md<P>{term_md<P>{I, I, I, I, div_grad}, term_interp<P>{theta, {m0, m10, m01, m20, m02}}};
        }
      }
    }
    break;

  case 3:
    if (num_pos == 1)
    {
      *this += term_md<P>({I, divv_nuv, I, I});
      *this += term_md<P>({I, I, divv_nuv, I});
      *this += term_md<P>({I, I, I, divv_nuv});

      *this += term_md<P>({term_moment_over_density{lbc.nu, moment{1, 0, 0}}, div, I, I});
      *this += term_md<P>({term_moment_over_density{lbc.nu, moment{0, 1, 0}}, I, div, I});
      *this += term_md<P>({term_moment_over_density{lbc.nu, moment{0, 0, 1}}, I, I, div});

      *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, div_grad, I, I});
      *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, I, div_grad, I});
      *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, I, I, div_grad});
    }
    else // interpolation case
    {
      moment_id const m0   = this->register_moment(moment{0, 0, 0});
      moment_id const m100 = this->register_moment(moment{1, 0, 0});
      moment_id const m010 = this->register_moment(moment{0, 1, 0});
      moment_id const m001 = this->register_moment(moment{0, 0, 1});
      moment_id const m200 = this->register_moment(moment{2, 0, 0});
      moment_id const m020 = this->register_moment(moment{0, 2, 0});
      moment_id const m002 = this->register_moment(moment{0, 0, 2});

      #ifdef ASGARD_USE_GPU
      auto m100over0 = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                           P const f[], P vals[]) -> void
        {
          gpu::moment_ratio(nu, moments[m100], moments[m0], f, vals);
        };
      auto m010over0 = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                           P const f[], P vals[]) -> void
        {
          gpu::moment_ratio(nu, moments[m010], moments[m0], f, vals);
        };
      auto m001over0 = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                           P const f[], P vals[]) -> void
        {
          gpu::moment_ratio(nu, moments[m001], moments[m0], f, vals);
        };
      auto theta = [=](int64_t, P, P const[], momentset_gpu<P> const &moments,
                       P const f[], P vals[]) -> void
        {
          gpu::lbc_vel3(nu, moments[m0], moments[m100], moments[m010], moments[m001],
                        moments[m200], moments[m020], moments[m002], f, vals);
        };
      #else
      auto m100over0 = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                           std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0   = moments[m0];
          std::vector<P> const &mom100 = moments[m100];
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = (nu * mom100[i] * f[i]) / mom0[i];
        };
      auto m010over0 = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                           std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0   = moments[m0];
          std::vector<P> const &mom010 = moments[m010];
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = (nu * mom010[i] * f[i]) / mom0[i];
        };
      auto m001over0 = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                           std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0   = moments[m0];
          std::vector<P> const &mom001 = moments[m001];
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = (nu * mom001[i] * f[i]) / mom0[i];
        };
      auto theta = [=](P, vector2d<P> const &x, momentset<P> const &moments,
                       std::vector<P> const &f, std::vector<P> &vals) -> void
        {
          std::vector<P> const &mom0  = moments[m0];
          std::vector<P> const &mom100 = moments[m100];
          std::vector<P> const &mom010 = moments[m010];
          std::vector<P> const &mom001 = moments[m001];
          std::vector<P> const &mom200 = moments[m200];
          std::vector<P> const &mom020 = moments[m020];
          std::vector<P> const &mom002 = moments[m002];
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < x.num_strips(); i++)
            vals[i] = (P{1} / P{3}) *  nu * f[i] *
                      ((mom200[i] + mom020[i] + mom002[i]) / mom0[i] -
                       (mom100[i] * mom100[i] + mom010[i] * mom010[i] + mom001[i] * mom001[i]) / (mom0[i] * mom0[i]));
        };
      #endif

      std::vector<moment_id> theta_deps = {m0, m100, m010, m001, m200, m020, m002};

      if constexpr (std::is_same_v<opmode, source<P>>) {
        this->set_adapt_weight(theta, theta_deps);
      } else {
        if (num_pos == 2) {
          *this += term_md<P>({I, I, divv_nuv, I, I});
          *this += term_md<P>({I, I, I, divv_nuv, I});
          *this += term_md<P>({I, I, I, I, divv_nuv});

          *this += term_md<P>{term_md<P>{I, I, div, I, I}, term_interp<P>{m100over0, {m0, m100}}};
          *this += term_md<P>{term_md<P>{I, I, I, div, I}, term_interp<P>{m010over0, {m0, m010}}};
          *this += term_md<P>{term_md<P>{I, I, I, I, div}, term_interp<P>{m001over0, {m0, m001}}};

          *this += term_md<P>{term_md<P>{I, I, div_grad, I, I}, term_interp<P>{theta, theta_deps}};
          *this += term_md<P>{term_md<P>{I, I, I, div_grad, I}, term_interp<P>{theta, theta_deps}};
          *this += term_md<P>{term_md<P>{I, I, I, I, div_grad}, term_interp<P>{theta, theta_deps}};
        } else {
          *this += term_md<P>({I, I, I, divv_nuv, I, I});
          *this += term_md<P>({I, I, I, I, divv_nuv, I});
          *this += term_md<P>({I, I, I, I, I, divv_nuv});

          *this += term_md<P>{term_md<P>{I, I, I, div, I, I}, term_interp<P>{m100over0, {m0, m100}}};
          *this += term_md<P>{term_md<P>{I, I, I, I, div, I}, term_interp<P>{m010over0, {m0, m010}}};
          *this += term_md<P>{term_md<P>{I, I, I, I, I, div}, term_interp<P>{m001over0, {m0, m001}}};

          *this += term_md<P>{term_md<P>{I, I, I, div_grad, I, I}, term_interp<P>{theta, theta_deps}};
          *this += term_md<P>{term_md<P>{I, I, I, I, div_grad, I}, term_interp<P>{theta, theta_deps}};
          *this += term_md<P>{term_md<P>{I, I, I, I, I, div_grad}, term_interp<P>{theta, theta_deps}};
        }
      }
    }
    break;
  default:
    // unreachable
    break;
  };
}

template<typename P>
template<typename opmode>
void pde_scheme<P>::process(operators::simple_bgk_collisions bgkc)
{
  static_assert(std::is_same_v<opmode, source<P>> or std::is_same_v<opmode, term_md<P>>);

  int const pdof = degree() + 1;

  rassert(domain_.num_pos() > 0, "cannot set simple_bgk_collisions operator for a pde_domain with no position dimensions");
  rassert(domain_.num_vel() > 0, "cannot set simple_bgk_collisions operator for a pde_domain with no velocity dimensions");
  rassert(domain_.num_pos() <= 3, "cannot set simple_bgk_collisions operator for a pde_domain with more than 3 position dimensions");
  rassert(domain_.num_vel() <= 3, "cannot set simple_bgk_collisions operator for a pde_domain with more than 3 velocity dimensions");
  rassert(2 <= pdof and pdof <= 4, "simple_bgk_collisions only valid on polynomial degrees 1, 2, and 3.");
  rassert(bgkc.nu > 0, "the collision frequency has to be positive");

  P const nu = static_cast<P>(bgkc.nu);

  int const num_dims = domain_.num_dims();

  if constexpr (std::is_same_v<opmode, term_md<P>>) {
    std::vector<term_1d<P>> nuI(domain_.num_dims(), term_identity{});
    nuI[0] = term_volume<P>{nu};
    *this += term_md<P>(nuI);
  }

  // copy the wavelet transformation matrices G0/G1 into a single vector
  std::vector<P> gbasis = [&, this]() -> std::vector<P>
    {
      std::array<std::vector<double>, 4> basis_mats = legendre::generate_multi_wavelets(degree());
      std::vector<P> result(2 * pdof * pdof);
      std::copy_n(basis_mats[2].data(), pdof * pdof, result.data());
      std::copy_n(basis_mats[3].data(), pdof * pdof, result.data() + pdof * pdof);
      return result;
    }();

  auto wavelet_maxwell = [gbasis, pdof](
      int idx, P const a, P const b, P const u, P const th, P mulin[], P mulout[])
  {
    if (idx == 0)
    {
      // Integrate from -infty to infty to preserve collision invariants
      P mid = 0.5*(b+a);

      // Integral of 1/√(2πt)(1,v,v^2)exp(-(v-u)^2/2t) for v=\pm\infty
      P I0 = 1.0;
      P I1 = u;
      P I2 = th + u*u;
      P I3 = 3.0*u*th + u*u*u;

      // Integral of 1/√(2πt)(phi_0,phi_1,phi_2)exp(-(v-u)^2/2t) where
      //   phi_i are the orthonormal Legendre polynomials on each element
      P jv2 = 2.0 / (b - a);
      P jv  = std::sqrt(jv2); // inverse root jacobian

      mulout[0] = std::sqrt(1.0/2.0)*jv*I0;
      if (pdof > 1)
        mulout[1] = std::sqrt(3.0/2.0)*jv*jv2*( -I0*mid + I1 );
      if (pdof > 2)
        mulout[2] = std::sqrt(5.0/8.0)*jv*( (3.0*jv2*jv2*mid*mid-1.0)*I0 - 6.0*jv2*jv2*mid*I1 + 3.0*jv2*jv2*I2);
      if (pdof > 3)
        mulout[3] = std::sqrt(7.0/8.0)*jv*(  (3.0*jv2*mid - 5.0*jv2*jv2*jv2*mid*mid*mid)*I0
                                         + (15.0*jv2*jv2*jv2*mid*mid - 3.0*jv2)*I1
                                         -  15.0*jv2*jv2*jv2*mid*I2
                                         +  5.0*jv2*jv2*jv2*I3
                                         );
    }
    else
    {
      // given idx > 0, this computes 2^( floor( log_2(idx) ) ) using integer bit-wise operations
      int const num_idx_above = fm::ipow2_log2(idx); // number of points on the level above idx

      P const dv = (b-a) / num_idx_above; // (b-a)/2^(lev-1)
      // Endpoints of wavelet element
      P const loca = a + (idx - num_idx_above) * dv;

      // For v_lev > 0 wavlets are piecewise polynomials
      // Integrate legendre polynomials the map to wavelets
      for (int l=0; l<2; l++)
      {
        P left  = loca + l*dv/2;
        P right = loca + (l+1)*dv/2;
        P mid = 0.5*(left+right);

        // v -> (v-u)/sqrt(t)
        P z_l = (left-u)/std::sqrt(th);
        P z_r = (right-u)/std::sqrt(th);

        // Integrals of 1/√(2π)(1,z,z^2)exp(-z^2/2) from z=a..b
        P T0_l = 0.5*std::erf(z_l/std::sqrt(2.0));
        P T0_r = 0.5*std::erf(z_r/std::sqrt(2.0));
        P T1_l = 1.0/std::sqrt(2.0*PI)*std::exp(-0.5*z_l*z_l);
        P T1_r = 1.0/std::sqrt(2.0*PI)*std::exp(-0.5*z_r*z_r);
        P C0   =   T0_r - T0_l;
        P C1   = -(T1_r - T1_l);
        P C2   = -(z_r*T1_r - z_l*T1_l) + C0;
        P C3   = (z_l*z_l+2.0)*T1_l - (z_r*z_r+2.0)*T1_r;

        // Integral of 1/√(2πt)(1,v,v^2)exp(-(v-u)^2/2t) dv using z = (v-u)/√t
        P I0 =       C0;
        P I1 =     u*C0 +         std::sqrt(th)*C1;
        P I2 =   u*u*C0 +   2.0*u*std::sqrt(th)*C1 +       th*C2;
        P I3 = u*u*u*C0 + 3.0*u*u*std::sqrt(th)*C1 + 3.0*u*th*C2 + std::sqrt(th)*th*C3;

        // Integral of 1/√(2πt)(phi_0,phi_1,phi_2)exp(-(v-u)^2/2t) where
        //   phi_i are the orthonormal Legendre polynomials on each element
        P jv = std::sqrt(4.0/dv); // inverse root jacobian
        P jv2= jv*jv;

        mulin[0] = std::sqrt(1.0/2.0)*jv*I0;
        if (pdof > 1)
          mulin[1] = std::sqrt(3.0/2.0)*jv*jv*jv*( -I0*mid + I1 );
        if (pdof > 2)
          mulin[2] = std::sqrt(5.0/8.0)*jv*( (3.0*jv*jv*jv*jv*mid*mid-1.0)*I0 - 6.0*jv*jv*jv*jv*mid*I1 + 3.0*jv*jv*jv*jv*I2);
        if (pdof > 3)
          mulin[3] = std::sqrt(7.0/8.0)*jv*(  (3.0*jv2*mid - 5.0*jv2*jv2*jv2*mid*mid*mid)*I0
                                          + (15.0*jv2*jv2*jv2*mid*mid - 3.0*jv2)*I1
                                          -  15.0*jv2*jv2*jv2*mid*I2
                                          +   5.0*jv2*jv2*jv2*I3
                                        );

        if (l == 0)
          asgard::smmat::gemv(pdof, pdof, gbasis.data(), mulin, mulout);
        else
          asgard::smmat::gemv1(pdof, pdof, gbasis.data() + pdof * pdof, mulin, mulout);
      }

    }

  };

  switch(domain_.num_vel())
  {
  case 1: {
    moment_id im0 = this->register_moment(moment(0));
    moment_id im1 = this->register_moment(moment(1));
    moment_id im2 = this->register_moment(moment(2));

    #ifdef ASGARD_USE_GPU
    auto fbgk = [=](int64_t, P, P const nodes[], momentset_gpu<P> const &moments, P vals[])
    {
      gpu::bgk_vel1(nu, num_pos, nodes, moments[im0], moments[im1], moments[im2], vals);
    };
    auto wbgk = [=](int64_t num, P time, P const nodes[], momentset_gpu<P> const &moments, P const[], P vals[])
    {
      fbgk(num, time, nodes, moments, vals);
    };
    #else
    P const domain_left1  = domain_.xleft(1);
    P const domain_right1 = domain_.xright(1);

    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &,
                    asgard::momentset<P> const &moments,
                    std::vector<int> const &indexes,
                    std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m1 = moments[im1];
      std::vector<P> const &m2 = moments[im2];

      int64_t const num_indexes = static_cast<int64_t>(indexes.size()) / num_dims;

      #pragma omp parallel
      {
        std::vector<P> wspace(2 * pdof, 0.0);

        P *mulout1 = wspace.data();
        P *mulin1  = wspace.data() + pdof;

        #pragma omp for
        for (int64_t i = 0; i < num_indexes; i++)
        {
          // Get index starting point of cell
          int64_t j = i * pdof * pdof;

          // Loop over polynomial x dof in element
          for (int ix1 = 0; ix1 < pdof; ix1++)
          {
            // Get fluid variables that given on points in (x,v).  The v coordinate doesnt matter
            P const n  = m0[j];
            P const u1 = m1[j] / m0[j];
            P const t  = m2[j] / m0[j] - u1 * u1;

            // Calculate 1D analytic maxwellian
            wavelet_maxwell(indexes[2*i+1], domain_left1, domain_right1, u1, t, mulin1, mulout1);

            // Take kroneckor product and store
            for (int iv1 = 0; iv1 < pdof; iv1++)
              vals[j + iv1] = nu * n * mulout1[iv1];

            j += pdof;
          }
        }
      }
    };
    #endif

    if constexpr (std::is_same_v<opmode, term_md<P>>) {
      this->set_source(fbgk, {im0, im1, im2});
    } else {
      this->set_adapt_hybrid(fbgk, {im0, im1, im2});
    }
  }
  break;
  case 2: {
    moment_id im0 = this->register_moment(moment(0, 0));
    moment_id im10 = this->register_moment(moment(1, 0));
    moment_id im01 = this->register_moment(moment(0, 1));
    moment_id im20 = this->register_moment(moment(2, 0));
    moment_id im02 = this->register_moment(moment(0, 2));

    std::vector<moment_id> const mids = {im0, im10, im01, im20, im02};

    #ifdef ASGARD_USE_GPU
    auto fbgk = [=](int64_t, P, P const nodes[], momentset_gpu<P> const &moments, P vals[])
    {
      gpu::bgk_vel2(nu, num_pos, nodes, moments[im0], moments[im10], moments[im01],
                    moments[im20], moments[im02], vals);
    };
    auto wbgk = [=](int64_t num, P time, P const nodes[], momentset_gpu<P> const &moments, P const[], P vals[])
    {
      fbgk(num, time, nodes, moments, vals);
    };
    #else
    P const domain_left2  = domain_.xleft(2);
    P const domain_right2 = domain_.xright(2);
    P const domain_left3  = domain_.xleft(3);
    P const domain_right3 = domain_.xright(3);

    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &,
                    asgard::momentset<P> const &moments,
                    std::vector<int> const &indexes,
                    std::vector<P> &vals)
    {
      std::vector<P> const &m0  = moments[im0];
      std::vector<P> const &m10 = moments[im10];
      std::vector<P> const &m01 = moments[im01];
      std::vector<P> const &m20 = moments[im20];
      std::vector<P> const &m02 = moments[im02];

      int64_t const num_indexes = static_cast<int64_t>(indexes.size()) / num_dims;

      #pragma omp parallel
      {
        int const pdof2 = pdof * pdof;

        std::vector<P> workspace(4 * pdof, 0.0);

        P* mulout1 = workspace.data();
        P* mulin1  = workspace.data() + pdof;
        P* mulout2 = workspace.data() + 2 * pdof;
        P* mulin2  = workspace.data() + 3 * pdof;

        #pragma omp for
        for (int64_t i = 0; i < num_indexes; i++)
        {
          // Get index starting point of cell in this coordinate for x1,x2
          int64_t j = i * pdof2 * pdof2;

          // Loop over polynomial x dof in element
          for (int ix1 = 0; ix1 < pdof; ix1++)
            for (int ix2 = 0; ix2 < pdof; ix2++)
            {
              // Get fluid variables that given on points in (x,v).  The v coordinate doesnt matter
              P const n  = m0[j];
              P const u1 = m10[j] / m0[j];
              P const u2 = m01[j] / m0[j];
              P const t  = P{0.5} * ((m20[j] + m02[j]) / m0[j] - u1 * u1 - u2 * u2);

              // Calculate 1D analytic maxwellian
              wavelet_maxwell(indexes[4*i+2], domain_left2, domain_right2, u1, t, mulin1, mulout1);
              wavelet_maxwell(indexes[4*i+3], domain_left3, domain_right3, u2, t, mulin2, mulout2);

              // Take Kronecker product and store
              for (int iv1 = 0; iv1 < pdof; iv1++)
                for (int iv2 = 0; iv2 < pdof; iv2++)
                  vals[j + iv1 * pdof + iv2] = nu * n * mulout1[iv1] * mulout2[iv2];

              j += pdof2;
            }
        }
      }
    };
    #endif

    if constexpr (std::is_same_v<opmode, term_md<P>>) {
      this->set_source(fbgk, mids);
    } else {
      this->set_adapt_hybrid(fbgk, mids);
    }
  }
  break;
  case 3: {
    moment_id im0 = this->register_moment(moment(0, 0, 0));
    moment_id im100 = this->register_moment(moment(1, 0, 0));
    moment_id im010 = this->register_moment(moment(0, 1, 0));
    moment_id im001 = this->register_moment(moment(0, 0, 1));
    moment_id im200 = this->register_moment(moment(2, 0, 0));
    moment_id im020 = this->register_moment(moment(0, 2, 0));
    moment_id im002 = this->register_moment(moment(0, 0, 2));

    std::vector<moment_id> const mids = {im0, im100, im010, im001, im200, im020, im002};

    #ifdef ASGARD_USE_GPU
    auto fbgk = [=](int64_t, P, P const nodes[], momentset_gpu<P> const &moments, P vals[])
    {
      gpu::bgk_vel3(nu, num_pos, nodes, moments[im0], moments[im100], moments[im010],
                    moments[im001], moments[im200], moments[im020], moments[im002], vals);
    };
    auto wbgk = [=](int64_t num, P time, P const nodes[], momentset_gpu<P> const &moments, P const[], P vals[])
    {
      fbgk(num, time, nodes, moments, vals);
    };
    #else
    P const domain_left3  = domain_.xleft(3);
    P const domain_right3 = domain_.xright(3);
    P const domain_left4  = domain_.xleft(4);
    P const domain_right4 = domain_.xright(4);
    P const domain_left5  = domain_.xleft(5);
    P const domain_right5 = domain_.xright(5);

    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &,
                    asgard::momentset<P> const &moments,
                    std::vector<int> const &indexes,
                    std::vector<P> &vals)
    {
      std::vector<P> const &m0   = moments[im0];
      std::vector<P> const &m100 = moments[im100];
      std::vector<P> const &m010 = moments[im010];
      std::vector<P> const &m001 = moments[im001];
      std::vector<P> const &m200 = moments[im200];
      std::vector<P> const &m020 = moments[im020];
      std::vector<P> const &m002 = moments[im002];

      int64_t const num_indexes = static_cast<int64_t>(indexes.size()) / num_dims;

      #pragma omp parallel
      {
        int const pdof3 = pdof * pdof * pdof;

        std::vector<P> workspace(6 * pdof, 0.0);

        P* mulout1 = workspace.data();
        P* mulin1  = workspace.data() + pdof;
        P* mulout2 = workspace.data() + 2 * pdof;
        P* mulin2  = workspace.data() + 3 * pdof;
        P* mulout3 = workspace.data() + 4 * pdof;
        P* mulin3  = workspace.data() + 5 * pdof;

        #pragma omp for
        for (int64_t i = 0; i < num_indexes; i++)
        {
          // Get index starting point of cell
          int64_t j = i * pdof3 * pdof3;

          // Loop over polynomial x dof in element
          for (int64_t ix1 = 0; ix1 < pdof; ix1++)
            for (int64_t ix2 = 0; ix2 < pdof; ix2++)
              for (int64_t ix3 = 0; ix3 < pdof; ix3++)
              {
                // Get fluid variables that given on points in (x,v).  The v coordinate doesnt matter
                P const n  = m0[j];
                P const u1 = m100[j] / m0[j];
                P const u2 = m010[j] / m0[j];
                P const u3 = m001[j] / m0[j];
                P const t  =  (1.0/3.0) * ((m200[j] + m020[j] + m002[j]) / m0[j] - u1 * u1 - u2 * u2 - u3 * u3);

                // Calculate 1D analytic maxwellian
                wavelet_maxwell(indexes[6*i+3], domain_left3, domain_right3, u1, t, mulin1, mulout1);
                wavelet_maxwell(indexes[6*i+4], domain_left4, domain_right4, u2, t, mulin2, mulout2);
                wavelet_maxwell(indexes[6*i+5], domain_left5, domain_right5, u3, t, mulin3, mulout3);

                // Take kroneckor product and store
                for (int iv1 = 0; iv1 < pdof; iv1++)
                  for (int iv2 = 0; iv2 < pdof; iv2++)
                    for (int iv3 = 0; iv3 < pdof; iv3++)
                      vals[j + iv1 * pdof * pdof + iv2 * pdof + iv3] = nu * n * mulout1[iv1] * mulout2[iv2] * mulout3[iv3];

                j += pdof3;
              }
        }
      }
    };
    #endif

    if constexpr (std::is_same_v<opmode, term_md<P>>) {
      this->set_source(fbgk, mids);
    } else {
      this->set_adapt_hybrid(fbgk, mids);
    }
  }
  break;
  default:
    // unreachable
    break;
  };
}

template<typename P>
void pde_scheme<P>:: update_deps(term_md<P> &tmd) {
  if (tmd.is_separable()) {
    for (int d = 0; d < domain_.num_dims(); d++) {
      term_1d<P> &t1d = tmd.dim(d);
      term_dependence const dep = t1d.depends();
      switch (dep) {
      case term_dependence::electric_field:
      case term_dependence::electric_field_only:
        rassert(1 <= domain_.num_vel() and domain_.num_vel() <= 3,
                "electric field dependence requires moments which in turn require 1 - 3 velocity dimensions");
        t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel())), this->register_electric_moment(dimension_id(d))};
        break;
      case term_dependence::moment_divided_by_density:
        rassert(1 <= domain_.num_vel() and domain_.num_vel() <= 3,
                "moment-over-density requires defined velocity dimensions");
        rassert(domain_.num_pos() == 1,
                "moment-over-density work only for one position dimension");
        rassert(t1d.moment_over().num_dims() == domain_.num_vel(),
                "moment-over-density requires moment with dimension matching the number of velocity dimensions");
        t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel())),
                     this->register_moment(t1d.moment_over())};
        break;
      case term_dependence::lenard_bernstein_coll_theta:
        rassert(1 <= domain_.num_vel() and domain_.num_vel() <= 3,
                "Lenard-Bernstein-theta requires defined velocity dimensions");
        rassert(domain_.num_pos() == 1,
                "Lenard-Bernstein-theta work only for one position dimension");
        // the zero-th moment is always needed, the others are set based on the dimensions
        switch (domain_.num_vel()) {
        case 1:
          t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel())),
                       this->register_moment(moment(1)),
                       this->register_moment(moment(2)), };
          break;
        case 2:
          t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel())),
                       this->register_moment(moment(1, 0)),
                       this->register_moment(moment(0, 1)),
                       this->register_moment(moment(2, 0)),
                       this->register_moment(moment(0, 2)), };
          break;
        case 3:
          t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel())),
                       this->register_moment(moment(1, 0, 0)),
                       this->register_moment(moment(0, 1, 0)),
                       this->register_moment(moment(0, 0, 1)),
                       this->register_moment(moment(2, 0, 0)),
                       this->register_moment(moment(0, 2, 0)),
                       this->register_moment(moment(0, 0, 2)), };
          break;
        default:
          // unreachable due to the assertion above
          break;
        };
        break;
      default:
        // nothing to do for term_dependence::none
        break;
      };
    }
  } else if (tmd.is_interpolatory() and tmd.is_electric(mlist)) {
    moment_id m0 = this->register_moment(moment::zero(domain_.num_vel()));
    tmd.mids_.push_back(m0);
  } else if (tmd.is_chain()) {
    // recursively process the chain
    for (int i = 0; i < tmd.num_chain(); i++)
      update_deps(tmd.chain(i));
  }

  // check the boundary conditions, interpolation BC require at least 3D
  int const nd = domain_.num_dims();
  for (boundary_flux<P> const &bc : tmd.get_bc_flux()) {
    if (not bc.is_separable()) {
      has_interp_funcs = true;
      has_interp_bc    = true;
      rassert(nd >= 3,
              "non-separable boundary conditions set for a 1D or 2D problem, "
              "but those can always be expressed in a separable form");
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template void pde_scheme<double>::process<source<double>>(operators::lenard_bernstein_collisions);
template void pde_scheme<double>::process<term_md<double>>(operators::lenard_bernstein_collisions);
template void pde_scheme<double>::process<source<double>>(operators::simple_bgk_collisions);
template void pde_scheme<double>::process<term_md<double>>(operators::simple_bgk_collisions);
template void pde_scheme<double>::update_deps(term_md<double> &tmd);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void pde_scheme<float>::process<source<float>>(operators::lenard_bernstein_collisions);
template void pde_scheme<float>::process<term_md<float>>(operators::lenard_bernstein_collisions);
template void pde_scheme<float>::process<source<float>>(operators::simple_bgk_collisions);
template void pde_scheme<float>::process<term_md<float>>(operators::simple_bgk_collisions);
template void pde_scheme<float>::update_deps(term_md<float> &tmd);
#endif
} // namespace asgard
