#pragma once

#include "coefficients.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#ifdef ASGARD_USE_MATLAB
#include "matlab_plot.hpp"
#endif

#include "time_advance.hpp"

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

void print_info(std::ostream &os = std::cout);

}
