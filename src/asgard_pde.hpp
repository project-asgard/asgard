#pragma once
#include "pde/asgard_pde_collisional_landau.hpp"
#include "pde/asgard_pde_collisional_landau_1x2v.hpp"
#include "pde/asgard_pde_collisional_landau_1x3v.hpp"
#include "pde/asgard_pde_fokkerplanck1_4p3.hpp"
#include "pde/asgard_pde_fokkerplanck1_4p4.hpp"
#include "pde/asgard_pde_fokkerplanck1_4p5.hpp"
#include "pde/asgard_pde_fokkerplanck1_pitch_C.hpp"
#include "pde/asgard_pde_fokkerplanck1_pitch_E.hpp"
#include "pde/asgard_pde_fokkerplanck2_complete.hpp"
#include "pde/asgard_pde_relaxation_1x1v.hpp"
#include "pde/asgard_pde_relaxation_1x2v.hpp"
#include "pde/asgard_pde_relaxation_1x3v.hpp"
#include "pde/asgard_pde_riemann_1x2v.hpp"
#include "pde/asgard_pde_riemann_1x3v.hpp"
#include "pde/asgard_pde_vlasov_lb_full_f.hpp"

namespace asgard
{
//
// this file contains the PDE factory and the utilities to
// select the PDEs being made available by the included
// implementations
//

// ---------------------------------------------------------------------------
//
// A free function factory for making pdes. eventually will want to change the
// return for some of these once we implement them...
//
// ---------------------------------------------------------------------------

template<typename pde_class>
auto make_custom_pde(prog_opts const &cli_input)
{
  static_assert(std::is_base_of_v<PDE<float>, pde_class> or std::is_base_of_v<PDE<double>, pde_class>,
                "the requested PDE class must inherit from the asgard::PDE base-class");

  using precision = typename pde_class::precision_mode;

  return std::unique_ptr<PDE<precision>>(std::make_unique<pde_class>(cli_input));
}

template<typename P>
std::unique_ptr<PDE<P>> make_PDE(prog_opts const &cli_input)
{
  rassert(cli_input.pde_choice, "cannot create an unspecified PDE");
  switch (cli_input.pde_choice.value())
  {
  case PDE_opts::fokkerplanck_1d_pitch_E_case1:
    return std::make_unique<
        PDE_fokkerplanck_1d_pitch_E<P, PDE_case_opts::case0>>(cli_input);
  case PDE_opts::fokkerplanck_1d_pitch_E_case2:
    return std::make_unique<
        PDE_fokkerplanck_1d_pitch_E<P, PDE_case_opts::case1>>(cli_input);
  case PDE_opts::fokkerplanck_1d_pitch_C:
    return std::make_unique<PDE_fokkerplanck_1d_pitch_C<P>>(cli_input);
  case PDE_opts::fokkerplanck_1d_4p3:
    return std::make_unique<PDE_fokkerplanck_1d_4p3<P>>(cli_input);
  case PDE_opts::fokkerplanck_1d_4p4:
    return std::make_unique<PDE_fokkerplanck_1d_4p4<P>>(cli_input);
  case PDE_opts::fokkerplanck_1d_4p5:
    return std::make_unique<PDE_fokkerplanck_1d_4p5<P>>(cli_input);
  case PDE_opts::fokkerplanck_2d_complete_case1:
    return std::make_unique<
        PDE_fokkerplanck_2d_complete<P, PDE_case_opts::case1>>(cli_input);
  case PDE_opts::fokkerplanck_2d_complete_case2:
    return std::make_unique<
        PDE_fokkerplanck_2d_complete<P, PDE_case_opts::case2>>(cli_input);
  case PDE_opts::fokkerplanck_2d_complete_case3:
    return std::make_unique<
        PDE_fokkerplanck_2d_complete<P, PDE_case_opts::case3>>(cli_input);
  case PDE_opts::fokkerplanck_2d_complete_case4:
    return std::make_unique<
        PDE_fokkerplanck_2d_complete<P, PDE_case_opts::case4>>(cli_input);
  case PDE_opts::vlasov_lb_full_f:
    return std::make_unique<PDE_vlasov_lb<P>>(cli_input);
  case PDE_opts::relaxation_1x1v:
    return std::make_unique<PDE_relaxation_1x1v<P>>(cli_input);
  case PDE_opts::relaxation_1x2v:
    return std::make_unique<PDE_relaxation_1x2v<P>>(cli_input);
  case PDE_opts::relaxation_1x3v:
    return std::make_unique<PDE_relaxation_1x3v<P>>(cli_input);
  case PDE_opts::riemann_1x2v:
    return std::make_unique<PDE_riemann_1x2v<P>>(cli_input);
  case PDE_opts::riemann_1x3v:
    return std::make_unique<PDE_riemann_1x3v<P>>(cli_input);
  case PDE_opts::collisional_landau:
    return std::make_unique<PDE_collisional_landau<P>>(cli_input);
  case PDE_opts::collisional_landau_1x2v:
    return std::make_unique<PDE_collisional_landau_1x2v<P>>(cli_input);
  case PDE_opts::collisional_landau_1x3v:
    return std::make_unique<PDE_collisional_landau_1x3v<P>>(cli_input);
  default:
    std::cout << "Invalid pde choice" << std::endl;
    exit(-1);
  }
}

template<typename P>
std::unique_ptr<PDE<P>> make_PDE(std::string const &opts)
{
  return make_PDE<P>(make_opts(opts));
}

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

} // namespace asgard
