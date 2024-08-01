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
void simulate(std::unique_ptr<PDE<precision>> &pde);

template<typename precision>
void simulate(prog_opts const &options)
{
  if (options.show_help)
  {
    if (get_local_rank() == 0)
      options.print_help();
    return;
  }
  if (options.show_version)
  {
    if (get_local_rank() == 0)
      options.print_version_help();
    return;
  }
  if (options.show_pde_help)
  {
    if (get_local_rank() == 0)
      options.print_pde_help();
    return;
  }

  rassert(options.pde_choice and options.pde_choice.value() != PDE_opts::custom,
          "must provide a valid PDE choice");

  auto pde = make_PDE<precision>(options);
  simulate(pde);
}

void print_info(std::ostream &os = std::cout);

}
