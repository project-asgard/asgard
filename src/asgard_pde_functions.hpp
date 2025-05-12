#pragma once
#include "asgard_pde.hpp"

namespace asgard
{

/*!
 * \internal
 * \brief Wraps around commonly used vector functions
 *
 * \endinternal
 */
template<typename P>
struct builtin_v {
  //! y is equal to x with all negative values replaced by zero
  static void positive(std::vector<P> const &x, std::vector<P> &y);
  //! y is equal to x with all positive values replaced by zero
  static void negative(std::vector<P> const &x, std::vector<P> &y);

  //! vector version of std::sin()
  static void sin(std::vector<P> const &x, std::vector<P> &y);
  //! vector version of std::cos()
  static void cos(std::vector<P> const &x, std::vector<P> &y);
  //! vector version of derivative of std::cos(), i.e., -std::sin()
  static void dcos(std::vector<P> const &x, std::vector<P> &y);

  //! vector version of std::exp(-x)
  static void expneg(std::vector<P> const &x, std::vector<P> &y);
  //! vector version of derivative of std::exp(-x), i.e., -std::exp(-x)
  static void dexpneg(std::vector<P> const &x, std::vector<P> &y);
  //! vector version of std::exp(-x^2)
  static void expneg2(std::vector<P> const &x, std::vector<P> &y);
};

/*!
 * \internal
 * \brief Wraps around commonly used functions, with time parameter
 *
 * \endinternal
 */
template<typename P>
struct builtin_t {
  //! overloads with dummy time parameter
  static void sin(std::vector<P> const &x, P, std::vector<P> &y) {
    builtin_v<P>::sin(x, y);
  }
  //! overloads with dummy time parameter
  static void cos(std::vector<P> const &x, P, std::vector<P> &y) {
    builtin_v<P>::cos(x, y);
  }
  //! overloads with dummy time parameter
  static void dcos(std::vector<P> const &x, P, std::vector<P> &y) {
    builtin_v<P>::dcos(x, y);
  }
  static void expneg(std::vector<P> const &x, std::vector<P> &y) {
    builtin_v<P>::expneg(x, y);
  }
  static void dexpneg(std::vector<P> const &x, std::vector<P> &y) {
    builtin_v<P>::dexpneg(x, y);
  }
  static void expneg2(std::vector<P> const &x, std::vector<P> &y) {
    builtin_v<P>::expneg2(x, y);
  }
};

/*!
 * \internal
 * \brief Wraps around commonly used functions, scalar variant
 *
 * \endinternal
 */
template<typename P>
struct builtin_s {
  //! std::sin(x)
  static P sin(P x) { return std::sin(x); }
  //! std::sin(x)
  static P cos(P x) { return std::cos(x); }
  //! d/dx std::cos(x) = -std::sin(x)
  static P dcos(P x) { return -std::sin(x); }
  //! std::exp(-x)
  static P expneg(P x) { return std::exp(-x); }
  //! d/dx std::exp(-x) = - std::exp(-x)
  static P dexpneg(P x) { return -std::exp(-x); }
  //! std::exp(-x * x)
  static P expneg2(P x) { return std::exp(-x * x); }
};

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

/*!
 * \ingroup asgard_pde_definition
 * \brief Function transformation utilities
 */
namespace functions
{
//! create a new function that calls f and negates the output
sfixed_func1d<float> negate(sfixed_func1d<float> f);
//! create a new function that calls f and negates the output
sfixed_func1d<double> negate(sfixed_func1d<double> f);
//! selects the positive values of x
sfixed_func1d<float> take_positive_float();
//! selects the positive values of x
sfixed_func1d<double> take_positive_double();
//! selects the negative values of x
sfixed_func1d<float> take_negative_float();
//! selects the negative values of x
sfixed_func1d<double> take_negative_double();
//! select the positive function for the right template parameter
template<typename P>
sfixed_func1d<P> take_positive() {
  if constexpr (std::is_same_v<P, double>)
    return take_positive_float();
  else
    return take_positive_double();
}
//! select the negative function for the right template parameter
template<typename P>
sfixed_func1d<P> take_negative() {
  if constexpr (std::is_same_v<P, double>)
    return take_negative_float();
  else
    return take_negative_double();
}

} // namespace functions

} // namespace asgard
