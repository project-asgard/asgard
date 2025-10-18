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
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < v.size(); i++)
        fv[i] = -nu * v[i];
    };

  term_1d<P> I = term_identity{};

  term_1d<P> divv_nuv = term_div<P>{vnu, flux_type::upwind, boundary_type::bothsides};

  term_1d<P> div = term_div<P>{1, flux_type::central, boundary_type::bothsides};

  term_1d<P> div_grad = term_1d<P>({term_div<P>{-1, flux_type::upwind, boundary_type::bothsides},
                                    term_grad<P>{1, flux_type::upwind, boundary_type::bothsides}});

  switch(domain_.num_vel())
  {
  case 1:
    *this += term_md<P>({I, divv_nuv});
    *this += term_md<P>({term_moment_over_density{lbc.nu, moment{1}}, div});

    *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, div_grad});
    break;

  case 2:
    *this += term_md<P>({I, divv_nuv, I});
    *this += term_md<P>({I, I, divv_nuv});

    *this += term_md<P>({term_moment_over_density{lbc.nu, moment{1, 0}}, div, I});
    *this += term_md<P>({term_moment_over_density{lbc.nu, moment{0, 1}}, I, div});

    *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, div_grad, I});
    *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, I, div_grad});
    break;

  case 3:
    *this += term_md<P>({I, divv_nuv, I, I});
    *this += term_md<P>({I, I, divv_nuv, I});
    *this += term_md<P>({I, I, I, divv_nuv});

    *this += term_md<P>({term_moment_over_density{lbc.nu, moment{1, 0, 0}}, div, I, I});
    *this += term_md<P>({term_moment_over_density{lbc.nu, moment{0, 1, 0}}, I, div, I});
    *this += term_md<P>({term_moment_over_density{lbc.nu, moment{0, 0, 1}}, I, I, div});

    *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, div_grad, I, I});
    *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, I, div_grad, I});
    *this += term_md<P>({term_lenard_bernstein_coll_theta{lbc.nu}, I, I, div_grad});
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
        t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel(), moment::regular)), };
        break;
      case term_dependence::moment_divided_by_density:
        rassert(1 <= domain_.num_vel() and domain_.num_vel() <= 3,
                "moment-over-density requires defined velocity dimensions");
        rassert(domain_.num_pos() == 1,
                "moment-over-density work only for one position dimension");
        rassert(t1d.moment_over().num_dims() == domain_.num_vel(),
                "moment-over-density requires moment with dimension matching the number of velocity dimensions");
        t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel(), moment::regular)),
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
          t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel(), moment::regular)),
                       this->register_moment(moment(1, moment::regular)),
                       this->register_moment(moment(2, moment::regular)), };
          break;
        case 2:
          t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel(), moment::regular)),
                       this->register_moment(moment(1, 0, moment::regular)),
                       this->register_moment(moment(0, 1, moment::regular)),
                       this->register_moment(moment(2, 0, moment::regular)),
                       this->register_moment(moment(0, 2, moment::regular)), };
          break;
        case 3:
          t1d.mids_ = {this->register_moment(moment::zero(domain_.num_vel(), moment::regular)),
                       this->register_moment(moment(1, 0, 0, moment::regular)),
                       this->register_moment(moment(0, 1, 0, moment::regular)),
                       this->register_moment(moment(0, 0, 1, moment::regular)),
                       this->register_moment(moment(2, 0, 0, moment::regular)),
                       this->register_moment(moment(0, 2, 0, moment::regular)),
                       this->register_moment(moment(0, 0, 2, moment::regular)), };
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
  }
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
