#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "program_options.hpp"

#include "pde/pde_base.hpp"
#include "pde/pde_continuity1.hpp"
#include "pde/pde_continuity2.hpp"
#include "pde/pde_continuity3.hpp"
#include "pde/pde_continuity6.hpp"
#include "pde/pde_diffusion1.hpp"
#include "pde/pde_diffusion2.hpp"
#include "pde/pde_fokkerplanck1_4p1a.hpp"
#include "pde/pde_fokkerplanck1_4p2.hpp"
#include "pde/pde_fokkerplanck1_4p3.hpp"
#include "pde/pde_fokkerplanck1_4p4.hpp"
#include "pde/pde_fokkerplanck1_4p5.hpp"
#include "pde/pde_fokkerplanck2_complete.hpp"
#include "tensors.hpp"

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

template<typename P>
std::unique_ptr<PDE<P>> make_PDE(options const &opts)
{
  switch (opts.get_selected_pde())
  {
  case PDE_opts::continuity_1:
    return std::make_unique<PDE_continuity_1d<P>>(opts);
  case PDE_opts::continuity_2:
    return std::make_unique<PDE_continuity_2d<P>>(opts);
  case PDE_opts::continuity_3:
    return std::make_unique<PDE_continuity_3d<P>>(opts);
  case PDE_opts::continuity_6:
    return std::make_unique<PDE_continuity_6d<P>>(opts);
  case PDE_opts::fokkerplanck_1d_4p1a:
    return std::make_unique<PDE_fokkerplanck_1d_4p1a<P>>(opts);
  case PDE_opts::fokkerplanck_1d_4p2:
    return std::make_unique<PDE_fokkerplanck_1d_4p2<P>>(opts);
  case PDE_opts::fokkerplanck_1d_4p3:
    return std::make_unique<PDE_fokkerplanck_1d_4p3<P>>(opts);
  case PDE_opts::fokkerplanck_1d_4p4:
    return std::make_unique<PDE_fokkerplanck_1d_4p4<P>>(opts);
  case PDE_opts::fokkerplanck_1d_4p5:
    return std::make_unique<PDE_fokkerplanck_1d_4p5<P>>(opts);
  case PDE_opts::fokkerplanck_2d_complete:
    return std::make_unique<PDE_fokkerplanck_2d_complete<P>>(opts);
  case PDE_opts::diffusion_1:
    return std::make_unique<PDE_diffusion_1d<P>>(opts);
  case PDE_opts::diffusion_2:
    return std::make_unique<PDE_diffusion_2d<P>>(opts);
  // TODO not yet implemented, replace return with appropriate types
  case PDE_opts::vlasov4:
    return std::make_unique<PDE_continuity_1d<P>>(opts);
  case PDE_opts::vlasov43:
    return std::make_unique<PDE_continuity_1d<P>>(opts);
  case PDE_opts::vlasov5:
    return std::make_unique<PDE_continuity_1d<P>>(opts);
  case PDE_opts::vlasov7:
    return std::make_unique<PDE_continuity_1d<P>>(opts);
  case PDE_opts::vlasov8:
    return std::make_unique<PDE_continuity_1d<P>>(opts);
  case PDE_opts::pde_user:
    return std::make_unique<PDE_continuity_1d<P>>(opts);
  default:
    std::cout << "Invalid pde choice" << std::endl;
    exit(-1);
  }
}
// shim for easy PDE creation in tests
template<typename P>
std::unique_ptr<PDE<P>>
make_PDE(PDE_opts const pde_choice, int const level = options::NO_USER_VALUE,
         int const degree = options::NO_USER_VALUE)
{
  return make_PDE<P>(options(pde_choice, level, degree));
}
