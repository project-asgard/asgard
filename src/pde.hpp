#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "program_options.hpp"
#include "quadrature.hpp"

#include "pde/pde_base.hpp"

#include "asgard_vector.hpp"
#include "pde/pde_advection1.hpp"
#include "pde/pde_collisional_landau.hpp"
#include "pde/pde_collisional_landau_1x2v.hpp"
#include "pde/pde_collisional_landau_1x3v.hpp"
#include "pde/pde_continuity1.hpp"
#include "pde/pde_continuity2.hpp"
#include "pde/pde_continuity3.hpp"
#include "pde/pde_continuity6.hpp"
#include "pde/pde_diffusion1.hpp"
#include "pde/pde_diffusion2.hpp"
#include "pde/pde_fokkerplanck1_4p3.hpp"
#include "pde/pde_fokkerplanck1_4p4.hpp"
#include "pde/pde_fokkerplanck1_4p5.hpp"
#include "pde/pde_fokkerplanck1_pitch_C.hpp"
#include "pde/pde_fokkerplanck1_pitch_E.hpp"
#include "pde/pde_fokkerplanck2_complete.hpp"
#include "pde/pde_relaxation_1x1v.hpp"
#include "pde/pde_relaxation_1x2v.hpp"
#include "pde/pde_relaxation_1x3v.hpp"
#include "pde/pde_riemann_1x2v.hpp"
#include "pde/pde_riemann_1x3v.hpp"
#include "pde/pde_two_stream.hpp"
#include "pde/pde_vlasov_lb_full_f.hpp"

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
auto make_custom_pde(parser const &cli_input)
{
  static_assert(std::is_base_of_v<PDE<float>, pde_class> or std::is_base_of_v<PDE<double>, pde_class>,
                "the requested PDE class must inherit from the asgard::PDE base-class");

  using precision = typename pde_class::precision_mode;

  return std::unique_ptr<PDE<precision>>(std::make_unique<pde_class>(cli_input));
}

template<typename P>
std::unique_ptr<PDE<P>> make_PDE(parser const &cli_input)
{
  switch (cli_input.get_selected_pde())
  {
  case PDE_opts::continuity_1:
    return std::make_unique<PDE_continuity_1d<P>>(cli_input);
  case PDE_opts::continuity_2:
    return std::make_unique<PDE_continuity_2d<P>>(cli_input);
  case PDE_opts::continuity_3:
    return std::make_unique<PDE_continuity_3d<P>>(cli_input);
  case PDE_opts::continuity_6:
    return std::make_unique<PDE_continuity_6d<P>>(cli_input);
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
  case PDE_opts::diffusion_1:
    return std::make_unique<PDE_diffusion_1d<P>>(cli_input);
  case PDE_opts::diffusion_2:
    return std::make_unique<PDE_diffusion_2d<P>>(cli_input);
  case PDE_opts::advection_1:
    return std::make_unique<PDE_advection_1d<P>>(cli_input);
  case PDE_opts::vlasov_lb_full_f:
    return std::make_unique<PDE_vlasov_lb<P>>(cli_input);
  case PDE_opts::vlasov_two_stream:
    return std::make_unique<PDE_vlasov_two_stream<P>>(cli_input);
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

// WARNING for tests only!
// features rely on options, parser, and PDE constructed w/ same arguments
// shim for easy PDE creation in tests
template<typename P>
std::unique_ptr<PDE<P>>
make_PDE(PDE_opts const pde_choice, fk::vector<int> levels,
         int const degree = parser::NO_USER_VALUE,
         double const cfl = parser::DEFAULT_CFL)
{
  return make_PDE<P>(parser(pde_choice, levels, degree, cfl));
}

// old tests based on uniform level need conversion
template<typename P>
std::unique_ptr<PDE<P>>
make_PDE(PDE_opts const pde_choice, int const level = parser::NO_USER_VALUE,
         int const degree = parser::NO_USER_VALUE,
         double const cfl = parser::DEFAULT_CFL)
{
  auto const levels = [level, pde_choice]() {
    if (level == parser::NO_USER_VALUE)
    {
      return fk::vector<int>();
    }

    switch (pde_choice)
    {
    case PDE_opts::continuity_1:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::continuity_2:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::continuity_3:
      return fk::vector<int>(std::vector<int>(3, level));

    case PDE_opts::continuity_6:
      return fk::vector<int>(std::vector<int>(6, level));

    case PDE_opts::fokkerplanck_1d_pitch_E_case1:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_pitch_E_case2:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_pitch_C:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_4p3:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_4p4:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_4p5:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_2d_complete_case1:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::fokkerplanck_2d_complete_case2:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::fokkerplanck_2d_complete_case3:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::fokkerplanck_2d_complete_case4:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::diffusion_1:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::diffusion_2:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::advection_1:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::vlasov_lb_full_f:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::vlasov_two_stream:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::relaxation_1x1v:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::relaxation_1x2v:
      return fk::vector<int>(std::vector<int>(3, level));

    case PDE_opts::relaxation_1x3v:
      return fk::vector<int>(std::vector<int>(4, level));

    case PDE_opts::riemann_1x2v:
      return fk::vector<int>(std::vector<int>(3, level));

    case PDE_opts::riemann_1x3v:
      return fk::vector<int>(std::vector<int>(4, level));

    case PDE_opts::collisional_landau:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::collisional_landau_1x2v:
      return fk::vector<int>(std::vector<int>(3, level));

    case PDE_opts::collisional_landau_1x3v:
      return fk::vector<int>(std::vector<int>(4, level));

    default:
      std::cout << "Invalid pde choice" << std::endl;
      exit(-1);
    }
  }();

  return make_PDE<P>(parser(pde_choice, levels, degree, cfl));
}
} // namespace asgard
