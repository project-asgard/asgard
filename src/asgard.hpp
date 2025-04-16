#pragma once

#include "asgard_discretization.hpp"

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
 * \brief One shot method, simulate the pde with the given options
 *
 * Performs time integration until the provided stop-time or number of steps.
 * - if output name is provided, saves a final snapshot
 * - if verbosity is not set to silent, prints a timing report
 *
 * \tparam precision is double or float
 *
 * \param disc is an existing discretization manager
 */
template<typename precision>
void simulate(discretization_manager<precision> &disc) {
  disc.advance_time(); // integrate until num-steps or stop-time

  if (not disc.stop_verbosity())
    disc.progress_report();

  disc.save_final_snapshot(); // only if output filename is provided

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';
}

}
