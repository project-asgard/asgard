#include "asgard_pde.hpp"

#include "device/asgard_kronmult_common.hpp"

namespace asgard
{

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
  template struct builtin_v<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
  template struct builtin_v<float>;
#endif
} // namespace asgard
