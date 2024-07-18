#pragma once

#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "tools.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#ifdef ASGARD_USE_MATLAB
#include "matlab_plot.hpp"
#endif

#include "asgard_vector.hpp"
#include "pde.hpp"
#include "program_options.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

namespace asgard
{

template<typename precision>
void simulate(parser const &cli_input,
              std::unique_ptr<PDE<precision>> &pde);

template<typename precision>
void simulate(parser const &cli_input)
{
  std::unique_ptr<PDE<precision>> pde;
  simulate(cli_input, pde);
}

}
