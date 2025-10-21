#include "asgard_pde_functions.hpp"

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
template struct builtin_v<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct builtin_v<float>;
#endif
} // namespace asgard
