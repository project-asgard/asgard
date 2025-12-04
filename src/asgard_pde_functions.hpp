#pragma once
#include "asgard_pde.hpp"

namespace asgard
{
/*!
 * \internal
 * \brief Wraps a scalar function into a vector one
 *
 * \endinternal
 */
template<typename P, typename scalar_callable>
auto vectorize(scalar_callable scal) {
  if constexpr (std::is_same_v<P, double>) {
    static_assert(std::is_convertible_v<scalar_callable, std::function<double(double)>>,
                  "vectorize<double> must be called with a function with signature double(double)");
    sfixed_func1d<double> res = [=](std::vector<double> const &x, std::vector<double> &fx) -> void
    {
      for (size_t i = 0; i < x.size(); i++) fx[i] = scal(x[i]);
    };
    return res;
  } else {
    static_assert(std::is_convertible_v<scalar_callable, std::function<float(float)>>,
                  "vectorize<float> must be called with a function with signature float(float)");
    sfixed_func1d<float> res = [=](std::vector<float> const &x, std::vector<float> &fx) -> void
    {
      for (size_t i = 0; i < x.size(); i++) fx[i] = scal(x[i]);
    };
    return res;
  }
}
/*!
 * \internal
 * \brief Wraps a scalar function into a vector one
 *
 * \endinternal
 */
template<typename P, typename scalar_callable>
auto vectorize_t(scalar_callable scal) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  if constexpr (std::is_same_v<P, double>) {
    static_assert(std::is_convertible_v<scalar_callable, std::function<double(double)>> or
                  std::is_convertible_v<scalar_callable, std::function<double(double, double)>>,
    "vectorize_t<double> must be called with signature double(double) or double(double, double)");
    if constexpr (std::is_convertible_v<scalar_callable, std::function<double(double)>>) {
      svector_func1d<double> res = [=](std::vector<double> const &x, double, std::vector<double> &fx) -> void
      {
        for (size_t i = 0; i < x.size(); i++) fx[i] = scal(x[i]);
      };
      return res;
    } else {
      svector_func1d<double> res = [=](std::vector<double> const &x, double t, std::vector<double> &fx) -> void
      {
        for (size_t i = 0; i < x.size(); i++) fx[i] = scal(x[i], t);
      };
      return res;
    }
  } else {
    static_assert(std::is_convertible_v<scalar_callable, std::function<float(float)>> or
                  std::is_convertible_v<scalar_callable, std::function<float(float, float)>>,
    "vectorize_t<float> must be called with signature float(float) or float(float, float)");
    if constexpr (std::is_convertible_v<scalar_callable, std::function<float(float)>>) {
      svector_func1d<float> res = [=](std::vector<float> const &x, float, std::vector<float> &fx) -> void
      {
        for (size_t i = 0; i < x.size(); i++) fx[i] = scal(x[i]);
      };
      return res;
    } else {
      svector_func1d<float> res = [=](std::vector<float> const &x, float t, std::vector<float> &fx) -> void
      {
        for (size_t i = 0; i < x.size(); i++) fx[i] = scal(x[i], t);
      };
      return res;
    }
  }
}

} // namespace asgard
