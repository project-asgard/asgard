#include "asgard_pde_functions.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_pde.hpp"
#endif

namespace asgard
{

template<typename P>
pde_scheme<P> &pde_scheme<P>::operator += (operators::lenard_bernstein_collisions lbc)
{
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
          expect(static_cast<size_t>(x.num_strips()) == mom0.size());
          expect(static_cast<size_t>(x.num_strips()) == mom1.size());
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
    break;
  default:
    // unreachable
    break;
  };

  return *this;
}

template<typename P>
pde_scheme<P> &pde_scheme<P>::operator += (operators::simple_bgk_collisions bgkc)
{
  rassert(domain_.num_pos() > 0, "cannot set simple_bgk_collisions operator for a pde_domain with no position dimensions");
  rassert(domain_.num_vel() > 0, "cannot set simple_bgk_collisions operator for a pde_domain with no velocity dimensions");
  rassert(domain_.num_pos() <= 3, "cannot set simple_bgk_collisions operator for a pde_domain with more than 3 position dimensions");
  rassert(domain_.num_vel() <= 3, "cannot set simple_bgk_collisions operator for a pde_domain with more than 3 velocity dimensions");
  rassert(bgkc.nu > 0, "the collision frequency has to be positive");

  P const nu = static_cast<P>(bgkc.nu);

  {
    std::vector<asgard::term_1d<P>> nuI(domain_.num_dims(), asgard::term_identity{});
    nuI[0] = asgard::term_volume<P>{nu};
    *this += asgard::term_md<P>(nuI);
  }

  int const num_pos = domain_.num_pos();

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
    #else
    auto fbgk = [=](P /* time */, vector2d<P> const &nodes,
                    momentset<P> const &moments, std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m1 = moments[im1];
      std::vector<P> const &m2 = moments[im2];

      int64_t const num_nodes = nodes.num_strips();
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < num_nodes; i++) {
        P const v = nodes[i][num_pos];

        P const n = m0[i];
        P const u = m1[i] / m0[i];
        P const t = m2[i] / m0[i] - u * u;

        vals[i] = nu * n / std::sqrt(2 * PI * t);
        P const d = v - u;
        vals[i] *= std::exp(- P{0.5} * d * d / t);
      }
    };
    #endif

    this->set_source(fbgk, {im0, im1, im2});
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
    #else
    auto fbgk = [=](P /* time */, vector2d<P> const &nodes,
                    momentset<P> const &moments, std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m10 = moments[im10];
      std::vector<P> const &m01 = moments[im01];
      std::vector<P> const &m20 = moments[im20];
      std::vector<P> const &m02 = moments[im02];

      int64_t const num_nodes = nodes.num_strips();
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < num_nodes; i++) {
        P const n = m0[i];
        P const u0 = m10[i] / m0[i];
        P const u1 = m01[i] / m0[i];
        P const t = 0.5 * ((m20[i] + m02[i]) / m0[i] - u0 * u0 - u1 * u1);

        vals[i] = nu * n / (2 * PI * t);
        P const vu0 = nodes[i][num_pos] - u0;
        P const vu1 = nodes[i][num_pos + 1] - u1;
        P const d = vu0 * vu0 + vu1 * vu1;
        vals[i] *= std::exp(- P{0.5} * d / t);
      }
    };
    #endif

    this->set_source(fbgk, mids);
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
    #else
    auto fbgk = [=](P /* time */, vector2d<P> const &nodes,
                    momentset<P> const &moments, std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m100 = moments[im100];
      std::vector<P> const &m010 = moments[im010];
      std::vector<P> const &m001 = moments[im001];
      std::vector<P> const &m200 = moments[im200];
      std::vector<P> const &m020 = moments[im020];
      std::vector<P> const &m002 = moments[im002];

      int64_t const num_nodes = nodes.num_strips();
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < num_nodes; i++) {
        P const n = m0[i];
        P const u0 = m100[i] / m0[i];
        P const u1 = m010[i] / m0[i];
        P const u2 = m001[i] / m0[i];
        P const t = ((m200[i] + m020[i] + m002[i]) / m0[i] - u0 * u0 - u1 * u1 - u2 * u2) / P{3};

        P const pit = 2 * PI * t;
        vals[i] = nu * n / (pit * std::sqrt(pit));
        P const vu0 = nodes[i][num_pos] - u0;
        P const vu1 = nodes[i][num_pos + 1] - u1;
        P const vu2 = nodes[i][num_pos + 2] - u2;
        P const d = vu0 * vu0 + vu1 * vu1 + vu2 * vu2;
        vals[i] *= std::exp(- P{0.5} * d / t);
      }
    };
    #endif

    this->set_source(fbgk, mids);
  }
  break;
  default:
    // unreachable
    break;
  };

  return *this;
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
}

#ifdef ASGARD_ENABLE_DOUBLE
template class pde_scheme<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class pde_scheme<float>;
#endif
} // namespace asgard
