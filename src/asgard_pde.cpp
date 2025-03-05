#include "asgard_pde.hpp"

#include "device/asgard_kronmult_common.hpp"

namespace asgard
{

template<typename P>
PDEv2<P> & PDEv2<P>::operator += (operators::lenard_bernstein_collisions lbc)
{
  rassert(domain_.num_vel() > 0, "cannot set collision operator for a pde_domain with velocity dimensions");
  rassert(domain_.num_pos() == 1, "currently lenard-bernstein collisions work for only 1 position dimension");

  auto vnu = [nu=lbc.nu](std::vector<P> const &x, std::vector<P> &y)
        -> void {
      for (size_t i = 0; i < x.size(); i++)
        y[i] = nu * x[i];
    };

  term_1d<P> I = term_identity{};

  if (domain_.num_vel() == 1) {
    *this += term_md<P>({I, term_div<P>{vnu, flux_type::upwind, boundary_type::bothsides}});
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

#ifdef ASGARD_ENABLE_DOUBLE
  template class PDEv2<double>;

  template struct builtin_v<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
  template class PDEv2<float>;

  template struct builtin_v<float>;
#endif
} // namespace asgard
