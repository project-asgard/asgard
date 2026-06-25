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

  int const dp1 = options_.degree.value()+1;

  rassert(domain_.num_pos() > 0, "cannot set simple_bgk_collisions operator for a pde_domain with no position dimensions");
  rassert(domain_.num_vel() > 0, "cannot set simple_bgk_collisions operator for a pde_domain with no velocity dimensions");
  rassert(domain_.num_pos() <= 3, "cannot set simple_bgk_collisions operator for a pde_domain with more than 3 position dimensions");
  rassert(domain_.num_vel() <= 3, "cannot set simple_bgk_collisions operator for a pde_domain with more than 3 velocity dimensions");
  rassert((dp1 < 2) || (dp1 > 4), "simple_bgk_collisions only valid on polynomial degrees 1, 2, and 3.");
  rassert(bgkc.nu > 0, "the collision frequency has to be positive");

  P const nu = static_cast<P>(bgkc.nu);

  if constexpr (std::is_same_v<opmode, term_md<P>>) {
    std::vector<asgard::term_1d<P>> nuI(domain_.num_dims(), asgard::term_identity{});
    nuI[0] = asgard::term_volume<P>{nu};
    *this += asgard::term_md<P>(nuI);
  }

  int const num_pos = domain_.num_pos();

  std::array<std::vector<double>,4>basis_mats_ = asgard::legendre::generate_multi_wavelets(dp1-1); // H0, H1, G0, G1
  // Convert to type P
  std::array<std::vector<P>,4>basis_mats; 
  for (std::size_t i = 0; i < 4; ++i) 
  {
    basis_mats[i].reserve(basis_mats_[i].size());
    for (double x : basis_mats_[i]) basis_mats[i].push_back(static_cast<P>(x));
  }


  auto wavelet_maxwell = [&](
      int64_t const v_lev, int64_t const v_pos, int64_t const dp1,
      P const a, P const b,
      P const u, P const th,
      std::vector<P> &mulin,
      std::vector<P> &mulout
    )
  {

    P dv = (v_lev == 0) ? b-a : (b-a)/(1 << (v_lev-1)); // (b-a)/2^(lev-1)
    // Endpoints of wavelet element
    P loca = a + v_pos*dv;
    P locb = a + (v_pos+1)*dv;

    if (v_lev == 0)
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
      P jv = std::sqrt(2.0/dv); // inverse root jacobian
      P jv2= jv*jv;
      P L0 = std::sqrt(1.0/2.0)*jv*I0; 
      P L1 = std::sqrt(3.0/2.0)*jv*jv2*( -I0*mid + I1 );
      P L2 = std::sqrt(5.0/8.0)*jv*( (3.0*jv2*jv2*mid*mid-1.0)*I0 - 6.0*jv2*jv2*mid*I1 + 3.0*jv2*jv2*I2);
      P L3 = std::sqrt(7.0/8.0)*jv*(  (3.0*jv2*mid - 5.0*jv2*jv2*jv2*mid*mid*mid)*I0 
                                  + (15.0*jv2*jv2*jv2*mid*mid - 3.0*jv2)*I1 
                                  -  15.0*jv2*jv2*jv2*mid*I2
                                  +  5.0*jv2*jv2*jv2*I3
                                  );

      // Populate to vals
      mulout[0] = L0;
      if (dp1 > 1) mulout[1] = L1;
      if (dp1 > 2) mulout[2] = L2;
      if (dp1 > 3) mulout[3] = L3;

    }
    else 
    {
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
        P L0 = std::sqrt(1.0/2.0)*jv*I0; 
        P L1 = std::sqrt(3.0/2.0)*jv*jv*jv*( -I0*mid + I1 );
        P L2 = std::sqrt(5.0/8.0)*jv*( (3.0*jv*jv*jv*jv*mid*mid-1.0)*I0 - 6.0*jv*jv*jv*jv*mid*I1 + 3.0*jv*jv*jv*jv*I2);
        P L3 = std::sqrt(7.0/8.0)*jv*(  (3.0*jv2*mid - 5.0*jv2*jv2*jv2*mid*mid*mid)*I0 
                                    + (15.0*jv2*jv2*jv2*mid*mid - 3.0*jv2)*I1 
                                    -  15.0*jv2*jv2*jv2*mid*I2
                                    +   5.0*jv2*jv2*jv2*I3
                                  );

        mulin[0] = L0;
        if (dp1 > 1) mulin[1] = L1;
        if (dp1 > 2) mulin[2] = L2;
        if (dp1 > 3) mulin[3] = L3;

        // Multiply by G0/G1
        if (l == 0)
        {
          asgard::smmat::gemv(dp1,dp1,basis_mats[2].data(),mulin.data(),mulout.data());
        }
        else
        {
          asgard::smmat::gemv1(dp1,dp1,basis_mats[3].data(),mulin.data(),mulout.data());
        }
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
    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, 
                    std::vector<int> const &indexes,
                    std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m1 = moments[im1];
      std::vector<P> const &m2 = moments[im2];

      #pragma omp parallel
      {

        std::vector<P> mulout1(dp1,0.0);
        std::vector<P>  mulin1(dp1,0.0);

        #pragma omp for
        for (int64_t i = 0; i < indexes.size()/(2*domain_.num_vel()); i++)
        {
          // Loop over polynomial x dof in element
          for (int64_t poly_x1 = 0; poly_x1 < dp1; poly_x1++)
          {
            // Get index starting point of cell in this coordinate for x1,x2
            int64_t const idx_start = (i*dp1 + poly_x1)*dp1;

            // Get fluid variables that given on points in (x,v).  The v coordinate doesnt matter
            P const n  = m0[idx_start];
            P const u1 = m1[idx_start] / m0[idx_start];
            P const t  = m2[idx_start] / m0[idx_start] - u1 * u1;

            // Need v index
            int64_t const v1_idx = indexes[2*i+1];

            // Get level and position of current index
            int64_t const v1_lev = int64_t(std::ceil(std::log2(v1_idx+1)));
            int64_t const v1_pos = (v1_lev == 0) ? 0 : v1_idx - (1 << (v1_lev-1)); // v_idx - 2^(v_lev-1)

            // Calculate 1D analytic maxwellian
            wavelet_maxwell(v1_lev,v1_pos,dp1,domain_.xleft(1),domain_.xright(1),u1,t,mulin1,mulout1);

            // Take kroneckor product and store
            for (int poly_v1 = 0; poly_v1 < dp1; poly_v1++)
            {
              vals[idx_start + poly_v1] = nu*n*mulout1[poly_v1];
            }
          }
        }
      }
    };

    auto wbgk = [=](P time, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> const &f,
                    std::vector<P> &vals)
    {
      fbgk(time, nodes, moments, asgard::global_grid->iset().indexes(), vals);
    };
    #endif

    if constexpr (std::is_same_v<opmode, term_md<P>>) {
      this->set_source(fbgk, {im0, im1, im2});
    } else {
      this->set_adapt_weight(wbgk, {im0, im1, im2});
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
    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, 
                    std::vector<int> const &indexes,
                    std::vector<P> &vals)
    {
      std::vector<P> const &m0  = moments[im0];
      std::vector<P> const &m10 = moments[im10];
      std::vector<P> const &m01 = moments[im01];
      std::vector<P> const &m20 = moments[im20];
      std::vector<P> const &m02 = moments[im02];

      #pragma omp parallel
      {

        std::vector<P> mulout1(dp1,0.0);
        std::vector<P>  mulin1(dp1,0.0);
        std::vector<P> mulout2(dp1,0.0);
        std::vector<P>  mulin2(dp1,0.0);

        #pragma omp for
        for (int64_t i = 0; i < indexes.size()/(2*domain_.num_vel()); i++)
        {
          // Loop over polynomial x dof in element
          for (int64_t poly_x1 = 0; poly_x1 < dp1; poly_x1++)
            for (int64_t poly_x2 = 0; poly_x2 < dp1; poly_x2++)
            {
              // Get index starting point of cell in this coordinate for x1,x2
              int64_t const idx_start = (i*dp1*dp1 + poly_x1*dp1 + poly_x2)*dp1*dp1;

              // Get fluid variables that given on points in (x,v).  The v coordinate doesnt matter
              P const n  = m0[idx_start];
              P const u1 = m10[idx_start] / m0[idx_start];
              P const u2 = m01[idx_start] / m0[idx_start];
              P const t  =  (1.0/2.0) * ((m20[idx_start] + m02[idx_start]) / m0[idx_start] - u1 * u1 - u2 * u2);

              // Need v index
              int64_t const v1_idx = indexes[4*i+2];
              int64_t const v2_idx = indexes[4*i+3];

              // Get level and position of current index
              int64_t const v1_lev = int64_t(std::ceil(std::log2(v1_idx+1)));
              int64_t const v1_pos = (v1_lev == 0) ? 0 : v1_idx - (1 << (v1_lev-1)); // v_idx - 2^(v_lev-1)

              int64_t const v2_lev = int64_t(std::ceil(std::log2(v2_idx+1)));
              int64_t const v2_pos = (v2_lev == 0) ? 0 : v2_idx - (1 << (v2_lev-1)); // v_idx - 2^(v_lev-1)

              // Calculate 1D analytic maxwellian
              wavelet_maxwell(v1_lev,v1_pos,dp1,domain_.xleft(2),domain_.xright(2),u1,t,mulin1,mulout1);
              wavelet_maxwell(v2_lev,v2_pos,dp1,domain_.xleft(3),domain_.xright(3),u2,t,mulin2,mulout2);

              // Take kroneckor product and store
              for (int poly_v1 = 0; poly_v1 < dp1; poly_v1++)
                for (int poly_v2 = 0; poly_v2 < dp1; poly_v2++)
                {
                  vals[idx_start + poly_v1*dp1 + poly_v2] = nu*n*mulout1[poly_v1]*mulout2[poly_v2];
                }
            }
        }
      }
    };

    auto wbgk = [=](P time, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> const &f,
                    std::vector<P> &vals)
    {
      fbgk(time, nodes, moments, asgard::global_grid->iset().indexes(), vals);
    };
    #endif

    if constexpr (std::is_same_v<opmode, term_md<P>>) {
      this->set_source(fbgk, mids);
    } else {
      this->set_adapt_weight(wbgk, mids);
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
    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &nodes,
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

      #pragma omp parallel
      {

        std::vector<P> mulout1(dp1,0.0);
        std::vector<P>  mulin1(dp1,0.0);
        std::vector<P> mulout2(dp1,0.0);
        std::vector<P>  mulin2(dp1,0.0);
        std::vector<P> mulout3(dp1,0.0);
        std::vector<P>  mulin3(dp1,0.0);

        #pragma omp for
        for (int64_t i = 0; i < indexes.size()/(2*domain_.num_vel()); i++)
        {
          // Loop over polynomial x dof in element
          for (int64_t poly_x1 = 0; poly_x1 < dp1; poly_x1++)
            for (int64_t poly_x2 = 0; poly_x2 < dp1; poly_x2++)
              for (int64_t poly_x3 = 0; poly_x3 < dp1; poly_x3++)
              {
                // Get index starting point of cell in this coordinate for x1,x2
                int64_t const idx_start = (i*dp1*dp1*dp1 + poly_x1*dp1*dp1 + poly_x2*dp1 + poly_x3)*dp1*dp1*dp1;

                // Get fluid variables that given on points in (x,v).  The v coordinate doesnt matter
                P const n  = m0[idx_start];
                P const u1 = m100[idx_start] / m0[idx_start];
                P const u2 = m010[idx_start] / m0[idx_start];
                P const u3 = m001[idx_start] / m0[idx_start];
                P const t  =  (1.0/3.0) * ((m200[idx_start] + m020[idx_start] + m002[idx_start]) / m0[idx_start] - u1 * u1 - u2 * u2 - u3 * u3);

                // Need v index
                int64_t const v1_idx = indexes[6*i+3];
                int64_t const v2_idx = indexes[6*i+4];
                int64_t const v3_idx = indexes[6*i+5];

                // Get level and position of current index
                int64_t const v1_lev = int64_t(std::ceil(std::log2(v1_idx+1)));
                int64_t const v1_pos = (v1_lev == 0) ? 0 : v1_idx - (1 << (v1_lev-1)); // v_idx - 2^(v_lev-1)

                int64_t const v2_lev = int64_t(std::ceil(std::log2(v2_idx+1)));
                int64_t const v2_pos = (v2_lev == 0) ? 0 : v2_idx - (1 << (v2_lev-1)); // v_idx - 2^(v_lev-1)

                int64_t const v3_lev = int64_t(std::ceil(std::log2(v3_idx+1)));
                int64_t const v3_pos = (v3_lev == 0) ? 0 : v3_idx - (1 << (v3_lev-1)); // v_idx - 2^(v_lev-1)

                // Calculate 1D analytic maxwellian
                wavelet_maxwell(v1_lev,v1_pos,dp1,domain_.xleft(3),domain_.xright(3),u1,t,mulin1,mulout1);
                wavelet_maxwell(v2_lev,v2_pos,dp1,domain_.xleft(4),domain_.xright(4),u2,t,mulin2,mulout2);
                wavelet_maxwell(v3_lev,v3_pos,dp1,domain_.xleft(5),domain_.xright(5),u3,t,mulin3,mulout3);

                // Take kroneckor product and store
                for (int poly_v1 = 0; poly_v1 < dp1; poly_v1++)
                  for (int poly_v2 = 0; poly_v2 < dp1; poly_v2++)
                    for (int poly_v3 = 0; poly_v3 < dp1; poly_v3++)
                    {
                      vals[idx_start + poly_v1*dp1*dp1 + poly_v2*dp1 + poly_v3] = nu*n*mulout1[poly_v1]*mulout2[poly_v2]*mulout3[poly_v3];
                    }
              }
        }
      }
    };

    auto wbgk = [=](P time, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> const &f,
                    std::vector<P> &vals)
    {
      fbgk(time, nodes, moments, asgard::global_grid->iset().indexes(), vals);
    };
    #endif

    if constexpr (std::is_same_v<opmode, term_md<P>>) {
      this->set_source(fbgk, mids);
    } else {
      this->set_adapt_weight(wbgk, mids);
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
        t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel())), };
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
