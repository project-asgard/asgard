#pragma once
// #include "asgard_discretization.hpp"

#include "asgard_reconstruct.hpp"
#include "asgard_boundary_conditions.hpp"
#include "asgard_coefficients.hpp"
#include "asgard_moment.hpp"
#include "asgard_solver.hpp"

/*!
 * \internal
 * \file asgard_time_advance.hpp
 * \brief Defines the time advance methods
 * \author The ASGarD Team
 * \ingroup asgard_discretization
 *
 * \endinternal
 */

/*!
 * \internal
 * \defgroup asgard_time_advance ASGarD Time Advance Methods
 *
 * Defines the time-advance methods. The header asgard_time_advance.hpp defines
 * the data-structures and methods. The file is included in asgard_discretization.hpp
 * so the structs can be included in the discretization_manager.
 * The implementation in asgard_time_advance.cpp circles around and includes
 * the asgard_discretization.hpp, so the time advance can operate on the manager
 * and the internal data-structures.
 *
 * \endinternal
 */

// forward declare so we can declare the fiend time-advance
template<typename precision>
class discretization_manager;

/*!
 * \ingroup asgard_discretization
 * \brief Integrates in time until the final time or number of steps
 *
 * This method manipulates the problems internal state, applying adaptivity,
 * checkpointing and other related operations.
 * The method is decalred as a friend to simplify the implementation is external
 * to simplify the discretization_manager class, which will primarily focus on
 * data storage.
 *
 * The optional variable num_steps indicates the number of time steps to take:
 * - if zero, the method will return immediately,
 * - if negative, integration will continue until the final time step
 */
template<typename P> // implemented in time-advance
void advance_time(discretization_manager<P> &manager, int64_t num_steps = -1);

#ifndef __ASGARD_DOXYGEN_SKIP
// placeholder for the new api
template<typename P> // implemented in time-advance
void advance_time_v2(discretization_manager<P> &manager, int64_t num_steps = -1);

namespace asgard::time_advance
{
#ifdef ASGARD_USE_CUDA
static constexpr resource imex_resrc = resource::device;
#else
static constexpr resource imex_resrc = resource::host;
#endif

} // namespace asgard::time_advance

#endif

/*!
 * \internal
 * \ingroup asgard_time_advance
 * \brief Contains the different time-advance methods
 *
 * \endinternal
 */
namespace asgard::time_advance
{

/*!
 * \internal
 * \ingroup asgard_time_advance
 * \brief Runge Kutta 3-stage method, 4th order accuracy in step-size
 *
 * Simple 3-stage explicit method, stability region is 0.1.
 * \endinternal
 */
template<typename P>
struct rungekutta3
{
  //! Default empty stepper
  rungekutta3() = default;
  //! Performs RK3 step forward in time, uses the current and next step
  void next_step(discretization_manager<P> const &dist, std::vector<P> const &current,
                 std::vector<P> &next) const;

private:
  // workspace vectors
  mutable std::vector<P> k1, k2, k3, s1;
};

}

/*!
 * \internal
 * \ingroup asgard_time_advance
 * \brief Wrapper class for different time-advance methods
 *
 * Simple 3-stage explicit method, stability region is 0.1.
 * \endinternal
 */
struct time_advance_manager
{
  //! default constructor, makes an empty manager
  time_advance_manager() = default;

  time_advance::method mode = time_advance::method::rk3;

  std::variant<time_advance::rungekutta3<P>> data;
};
