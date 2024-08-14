#pragma once
#include "asgard_discretization.hpp"

namespace asgard::time_advance
{
#ifdef ASGARD_USE_CUDA
static constexpr resource imex_resrc = resource::device;
#else
static constexpr resource imex_resrc = resource::host;
#endif

// technically advance_time belongs here but it is declared in asgard_discretization
// so it can be "friend" with the manager

} // namespace asgard::time_advance
