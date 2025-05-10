#include "asgard_pde_functions.hpp"

#include "asgard_kronmult_common.hpp"

namespace asgard
{

template<typename P>
pde_scheme<P> &pde_scheme<P>::operator += (operators::lenard_bernstein_collisions lbc)
{
  rassert(domain_.num_vel() > 0, "cannot set collision operator for a pde_domain with velocity dimensions");
  rassert(domain_.num_pos() == 1, "currently lenard-bernstein collisions work for only 1 position dimension");
  rassert(lbc.nu > 0, "the collision frequency has to be positive");

  auto vnu = [nu=lbc.nu](std::vector<P> const &v, std::vector<P> &fv)
        -> void {
      for (size_t i = 0; i < v.size(); i++)
        fv[i] = -nu * v[i];
    };

  term_1d<P> I = term_identity{};

  term_1d<P> divv_nuv = term_div<P>{vnu, flux_type::upwind, boundary_type::bothsides};

  term_1d<P> div_nu = term_div<P>{static_cast<P>(lbc.nu), flux_type::central, boundary_type::bothsides};

  P const snu = std::sqrt(lbc.nu);
  term_1d<P> nu_div_grad = term_1d<P>({term_div<P>{-snu, flux_type::upwind, boundary_type::bothsides},
                                       term_grad<P>{snu, flux_type::upwind, boundary_type::bothsides}});

  if (domain_.num_vel() == 1) {
    *this += term_md<P>({I, divv_nuv});
    *this += term_md<P>({term_moment_over_density{1}, div_nu});

    term_1d<P> vol_theta(term_dependence::lenard_bernstein_coll_theta_1x1v);
    *this += term_md<P>({vol_theta, nu_div_grad});

  } else if (domain_.num_vel() == 2) {
    *this += term_md<P>({I, divv_nuv, I});
    *this += term_md<P>({I, I, divv_nuv});

    *this += term_md<P>({term_moment_over_density{1}, div_nu, I});
    *this += term_md<P>({term_moment_over_density{2}, I, div_nu});

    term_1d<P> vol_theta(term_dependence::lenard_bernstein_coll_theta_1x2v);
    *this += term_md<P>({vol_theta, nu_div_grad, I});
    *this += term_md<P>({vol_theta, I, nu_div_grad});

  } else {
    *this += term_md<P>({I, divv_nuv, I, I});
    *this += term_md<P>({I, I, divv_nuv, I});
    *this += term_md<P>({I, I, I, divv_nuv});

    *this += term_md<P>({term_moment_over_density{1}, div_nu, I, I});
    *this += term_md<P>({term_moment_over_density{2}, I, div_nu, I});
    *this += term_md<P>({term_moment_over_density{3}, I, I, div_nu});

    term_1d<P> vol_theta(term_dependence::lenard_bernstein_coll_theta_1x3v);
    *this += term_md<P>({vol_theta, nu_div_grad, I, I});
    *this += term_md<P>({vol_theta, I, nu_div_grad, I});
    *this += term_md<P>({vol_theta, I, I, nu_div_grad});
  }

  return *this;
}

template<typename P>
void builtin_v<P>::positive(std::vector<P> const &x, std::vector<P> &y)
{
#pragma omp parallel for
  for (size_t i = 0; i < x.size(); i++)
    y[i] = std::max(P{0}, x[i]);
}
template<typename P>
void builtin_v<P>::negative(std::vector<P> const &x, std::vector<P> &y)
{
#pragma omp parallel for
  for (size_t i = 0; i < x.size(); i++)
    y[i] = std::min(P{0}, x[i]);
}

template<typename P>
void builtin_v<P>::sin(std::vector<P> const &x, std::vector<P> &y) {
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = std::sin(x[i]);
}
template<typename P>
void builtin_v<P>::cos(std::vector<P> const &x, std::vector<P> &y) {
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = std::cos(x[i]);
}
template<typename P>
void builtin_v<P>::dcos(std::vector<P> const &x, std::vector<P> &y) {
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = -std::sin(x[i]);
}

template<typename P>
void builtin_v<P>::expneg(std::vector<P> const &x, std::vector<P> &y) {
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = std::exp(-x[i]);
}
template<typename P>
void builtin_v<P>::dexpneg(std::vector<P> const &x, std::vector<P> &y) {
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = -std::exp(-x[i]);
}
template<typename P>
void builtin_v<P>::expneg2(std::vector<P> const &x, std::vector<P> &y) {
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = std::exp(-x[i] * x[i]);
}

namespace functions
{
sfixed_func1d<float> negate(sfixed_func1d<float> f) {
  return [=](std::vector<float> const &x, std::vector<float> &fx) -> void {
      f(x, fx);
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < fx.size(); i++)
        fx[i] = -fx[i];
    };
}
sfixed_func1d<double> negate(sfixed_func1d<double> f) {
  return [=](std::vector<double> const &x, std::vector<double> &fx) -> void {
      f(x, fx);
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < fx.size(); i++)
        fx[i] = -fx[i];
    };
}
sfixed_func1d<float> take_positive_float() {
  return [=](std::vector<float> const &x, std::vector<float> &fx) -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < fx.size(); i++)
        fx[i] = std::min(0.0f, x[i]);;
    };
}
sfixed_func1d<double> take_positive_double() {
  return [=](std::vector<double> const &x, std::vector<double> &fx) -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < fx.size(); i++)
        fx[i] = std::min(0.0, x[i]);;
    };
}
sfixed_func1d<float> take_negative_float() {
  return [=](std::vector<float> const &x, std::vector<float> &fx) -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < fx.size(); i++)
        fx[i] = std::max(0.0f, x[i]);;
    };
}
sfixed_func1d<double> take_negative_double() {
  return [=](std::vector<double> const &x, std::vector<double> &fx) -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < fx.size(); i++)
        fx[i] = std::max(0.0, x[i]);;
    };
}

} // namespace functions

#ifdef ASGARD_ENABLE_DOUBLE
  template class pde_scheme<double>;

  template struct builtin_v<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
  template class pde_scheme<float>;

  template struct builtin_v<float>;
#endif
} // namespace asgard
