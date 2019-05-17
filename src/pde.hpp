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

#include "pde/pde_base.hpp"
#include "pde/pde_continuity1.hpp"
#include "pde/pde_continuity2.hpp"
#include "pde/pde_continuity3.hpp"
#include "pde/pde_continuity6.hpp"
#include "tensors.hpp"

//
// this file contains the PDE factory and the utilities to
// select the PDEs being made available by the included
// implementations
//

//
// the choices for supported PDE types
//
enum class PDE_opts
{
  continuity_1,
  continuity_2,
  continuity_3,
  continuity_6,
  // FIXME the below have not been implemented according to the
  // new specification. david is working on that in the matlab
  vlasov4,  // PDE corresponding to Fig. 4 in FIXME
  vlasov43, // PDE corresponding to Fig. 4.3 in FIXME
  vlasov5,  // PDE corresponding to Fig. 5 in FIXME
  vlasov7,  // PDE corresponding to Fig. 7 in FIXME
  vlasov8,  // PDE corresponding to Fig. 8 in FIXME
  pde_user  // FIXME will need to add the user supplied PDE choice
};

//
// map those choices to selection strings
//
using pde_map_t                    = std::map<std::string, PDE_opts>;
static pde_map_t const pde_mapping = {{"continuity_1", PDE_opts::continuity_1},
                                      {"continuity_2", PDE_opts::continuity_2},
                                      {"continuity_3", PDE_opts::continuity_3},
                                      {"continuity_6", PDE_opts::continuity_6},
                                      {"pde_user", PDE_opts::pde_user},
                                      {"vlasov4", PDE_opts::vlasov4},
                                      {"vlasov7", PDE_opts::vlasov7},
                                      {"vlasov8", PDE_opts::vlasov8},
                                      {"vlasov5", PDE_opts::vlasov5},
                                      {"vlasov43", PDE_opts::vlasov43}};

// ---------------------------------------------------------------------------
//
// A free function factory for making pdes. eventually will want to change the
// return for some of these once we implement them...
//
// ---------------------------------------------------------------------------

template<typename P>
std::unique_ptr<PDE<P>>
make_PDE(PDE_opts choice, int const level = -1, int const degree = -1)
{
  switch (choice)
  {
  case PDE_opts::continuity_1:
    return std::make_unique<PDE_continuity_1d<P>>(level, degree);
  case PDE_opts::continuity_2:
    return std::make_unique<PDE_continuity_2d<P>>(level, degree);
  case PDE_opts::continuity_3:
    return std::make_unique<PDE_continuity_3d<P>>(level, degree);
  case PDE_opts::continuity_6:
    return std::make_unique<PDE_continuity_6d<P>>(level, degree);
  // TODO not yet implemented, replace return with appropriate types
  case PDE_opts::vlasov4:
    return std::make_unique<PDE_continuity_1d<P>>(level, degree);
  case PDE_opts::vlasov43:
    return std::make_unique<PDE_continuity_1d<P>>(level, degree);
  case PDE_opts::vlasov5:
    return std::make_unique<PDE_continuity_1d<P>>(level, degree);
  case PDE_opts::vlasov7:
    return std::make_unique<PDE_continuity_1d<P>>(level, degree);
  case PDE_opts::vlasov8:
    return std::make_unique<PDE_continuity_1d<P>>(level, degree);
  case PDE_opts::pde_user:
    return std::make_unique<PDE_continuity_1d<P>>(level, degree);
  default:
    std::cout << "Invalid pde choice" << std::endl;
    exit(-1);
  }
}
