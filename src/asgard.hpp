#pragma once

#include "asgard_time_advance.hpp"

/*!
 * \file asgard.hpp
 * \brief Main include file for all ASGarD methods and classes
 * \author The ASGarD Team
 * \ingroup asgard_discretization
 */

namespace asgard
{
/*!
 * \defgroup asgard_discretization ASGarD Discretization Tools
 *
 * Given a PDE specification, these tools create a sparse grid discretization
 * of the state, source and operator terms, and performs time-stepping
 * integration in time.
 */

/*!
 * \ingroup asgard_discretization
 * \brief Simulates one of the builtin PDEs
 *
 * The options.pde_choice must contain a valid PDE_opts that is not
 * PDE_opts::custom.
 * This method is equivalent to calling the asgard executable.
 */
template<typename precision>
void simulate_builtin(prog_opts const &options)
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

  discretization_manager discretization(make_PDE<precision>(options),
                                        verbosity_level::high);

  advance_time(discretization);

  discretization.save_final_snapshot();

  // collects the state across MPI ranks but ignores the result
  discretization.current_mpistate();

  if (tools::timer.enabled())
    node_out() << tools::timer.report() << '\n';
}

/*!
 * \ingroup asgard_discretization
 * \brief Creates a discretization for the given pde and options
 *
 * Creates a new asgard::discretization_manager, unlike the one-shot
 * method asgard::simulate(), the discretization allows for finer control of
 * the time steeping and output frequency.
 *
 * \tparam pde_class is a user provided PDE specification that is derived from
 *                   the base asgard::PDE class
 *
 * \param options is a set of options, either provided from the command line,
 *                a file or manually set by the user
 *
 * \param verbosity indicates how much output ASGarD should provide,
 *                  verbosity_level::high is useful for development and debugging
 *                  while quiet is suitable for long simulations
 *
 * \returns an instance of asgard::discretization_manager
 *
 * Note: options.pde_choice must be either empty (does not contain a value)
 *       or set to PDE_opts::custom
 */
template<typename pde_class>
auto discretize(prog_opts const &options, verbosity_level verbosity = verbosity_level::quiet)
{
  static_assert(std::is_base_of_v<PDE<float>, pde_class> or std::is_base_of_v<PDE<double>, pde_class>,
                "the requested PDE class must inherit from the asgard::PDE base-class");

  return discretization_manager(make_custom_pde<pde_class>(options), verbosity);
}

/*!
 * \ingroup asgard_discretization
 * \brief One shot method, simulate the pde with the given options
 *
 * Creates a new PDE, discretizes it and advances in time according to the
 * options specified by the prog_opts instance and the PDE specification.
 *
 * \tparam pde_class is a user provided PDE specification that is derived from
 *                   the base asgard::PDE class
 *
 * \param options is a set of options, either provided from the command line,
 *                a file or manually set by the user
 *
 * \param verbosity indicates how much output ASGarD should provide,
 *                  verbosity_level::high is useful for development and debugging
 *                  while quiet is suitable for long simulations
 *
 * Note: options.pde_choice must be either empty (does not contain a value)
 *       or set to PDE_opts::custom
 */
template<typename pde_class>
void simulate(prog_opts const &options, verbosity_level verbosity = verbosity_level::quiet)
{
  auto discretization = discretize<pde_class>(options, verbosity);

  advance_time(discretization);

  discretization.save_final_snapshot();
}

//! \brief Print core library information, e.g., build options and such
void print_info(std::ostream &os = std::cout);

}
