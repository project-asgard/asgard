#pragma once
#include "asgard_time_data.hpp"

/*!
 * \internal
 * \file asgard_time_advance.hpp
 * \brief Defines the time advance methods
 * \author The ASGarD Team
 * \ingroup asgard_discretization
 *
 * \endinternal
 */

namespace asgard
{

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
void advance_in_time(discretization_manager<P> &manager, int64_t num_steps = -1);

} // namespace asgard

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
 * \brief Steady state solver, advances to the final time and assumes d/dt = 0
 *
 * The two methods are simple variants of each other, this class will read the correct
 * one from the options and make the adjustments.
 * \endinternal
 */
template<typename P>
struct steady_state
{
  //! Default empty stepper
  steady_state() = default;
  //! Initialize the stepper and
  steady_state(prog_opts const &options)
    : solver(options)
  {
    expect(options.step_method.value() == method);
  }
  //! Solves for the final step
  void next_step(discretization_manager<P> const &disc, std::vector<P> const &current,
                 std::vector<P> &endstep) const;

  //! requires a solver
  static bool constexpr needs_solver = true;
  //! needed precondtioner, if using an iterative solver
  precon_method needed_precon() const { return solver.precon; }
  //! returns the number of matrix-vector products, if using an iterative solver
  int64_t num_apply_calls() const { return solver.num_apply; }

  //! prints options for the solver
  void print_solver_opts(std::ostream &os = std::cout) const {
    os << solver;
  }

private:
  static time_method constexpr method = time_method::steady;
  // the solver used
  mutable solver_manager<P> solver;
  // workspace (rhs)
  mutable std::vector<P> work;
};

/*!
 * \internal
 * \ingroup asgard_time_advance
 * \brief Runge Kutta 3-stage method, 4th order accuracy in step-size
 *
 * Simple 3-stage explicit method, stability region is 0.1.
 * \endinternal
 */
template<typename P>
struct rungekutta
{
  //! Default empty stepper
  rungekutta() = default;
  //! Default empty stepper
  rungekutta(time_method rk) : rktype(rk)
  {
    expect(rktype == time_method::forward_euler or rktype == time_method::rk2
           or rktype == time_method::rk3 or rktype == time_method::rk4);
  }
  //! Performs RK3 step forward in time, uses the current and next step
  void next_step(discretization_manager<P> const &disc, std::vector<P> const &current,
                 std::vector<P> &next) const;
  //! explicit solver and does not require a solver
  static bool constexpr needs_solver = false;

private:
  time_method rktype = time_method::rk3;

  // workspace vectors
  mutable std::vector<P> k1, k2, k3, k4, s1;
};

/*!
 * \internal
 * \ingroup asgard_time_advance
 * \brief Crank-Nicolson or Backward-Euler 1-stage method, 2nd or 1st order accuracy in step-size
 *
 * The two methods are simple variants of each other, this class will read the correct
 * one from the options and make the adjustments.
 * \endinternal
 */
template<typename P>
struct crank_nicolson
{
  //! Default empty stepper
  crank_nicolson() = default;
  //! Initialize the stepper and
  crank_nicolson(prog_opts const &options)
      : method(options.step_method.value()), solver(options)
  {
    expect(method == time_method::cn or
           method == time_method::back_euler);
  }
    //! computes the rhs of the implicit solver using single MPI operation
  void mpi_rhs(discretization_manager<P> const &dist, P time, P substep, P dt,
               std::vector<P> const &current, std::vector<P> &next) const;

  //! Performs Crank-Nicolson step forward in time, uses the current and next step
  void next_step(discretization_manager<P> const &dist, std::vector<P> const &current,
                 std::vector<P> &next) const;

  //! requires a solver
  static bool constexpr needs_solver = true;
  //! needed precondtioner, if using an iterative solver
  precon_method needed_precon() const { return solver.precon; }
  //! returns the number of matrix-vector products, if using an iterative solver
  int64_t num_apply_calls() const { return solver.num_apply; }

  //! prints options for the solver
  void print_solver_opts(std::ostream &os = std::cout) const {
    os << solver;
  }

private:
  time_method method = time_method::cn;
  // the solver used
  mutable solver_manager<P> solver;
  // workspace
  mutable std::vector<P> work;
};

/*!
 * \internal
 * \ingroup asgard_time_advance
 * \brief Implicit-Explicit time-stepper
 *
 * Variations combining implicit and explicit time-steppers.
 * \endinternal
 */
template<typename P>
struct imex_stepper
{
  //! Default empty stepper
  imex_stepper() = default;
  //! Initialize the stepper and
  imex_stepper(prog_opts const &options, imex_implicit_group im, imex_explicit_group ex)
      : method(options.step_method.value()), solver(options),
        imex_implicit(im), imex_explicit(ex)
  {
    expect(is_imex(method));
  }
  //! Performs Crank-Nicolson step forward in time, uses the current and next step
  void next_step(discretization_manager<P> const &disc, std::vector<P> const &current,
                 std::vector<P> &next) const;

  //! rebuilds the operator matrix
  //void rebuild_matrix(discretization_manager<P> const &dist) const;
  //! requires a solver
  static bool constexpr needs_solver = true;
  //! needed precondtioner, if using an iterative solver
  precon_method needed_precon() const { return solver.precon; }
  //! returns the number of matrix-vector products, if using an iterative solver
  int64_t num_apply_calls() const { return solver.num_apply; }

  //! prints options for the solver
  void print_solver_opts(std::ostream &os = std::cout) const {
    os << solver;
  }

private:
  //! fills into R the ode_rhs for the explicit part
  void explicit_ode_rhs(discretization_manager<P> const &disc, P time,
                        std::vector<P> const &current, std::vector<P> &R) const;
  //! fills into R the ode_rhs for the explicit part
  void implicit_solve(discretization_manager<P> const &disc, P time,
                      std::vector<P> &current, std::vector<P> &R) const;

  time_method method = time_method::imex2;
  // the solver used
  mutable solver_manager<P> solver;
  // implicit and explicit groups
  imex_implicit_group imex_implicit;
  imex_explicit_group imex_explicit;
  // workspace
  mutable std::vector<P> fs, f;
};

}

namespace asgard
{

/*!
 * \internal
 * \ingroup asgard_time_advance
 * \brief Wrapper class for different time-advance methods
 *
 * Simple 3-stage explicit method, stability region is 0.1.
 * \endinternal
 */
template<typename P>
struct time_advance_manager
{
  //! default constructor, makes an empty manager
  time_advance_manager() = default;
  //! creates a new time-stepping manager for the given method
  time_advance_manager(time_data const &tdata, prog_opts const &options);
  //! creates a new time-stepping manager for the given imex method
  time_advance_manager(time_data const &tdata, prog_opts const &options,
                       imex_implicit_group im, imex_explicit_group ex);
  //! advance to the next time-step
  void next_step(discretization_manager<P> const &dist, std::vector<P> const &current,
                 std::vector<P> &next) const;
  //! returns whether the manager requires a solver
  bool needs_solver() const {
    switch (method.index()) {
      case 0:
        return time_advance::steady_state<P>::needs_solver;
      case 1:
        return time_advance::rungekutta<P>::needs_solver;
      case 2:
        return time_advance::crank_nicolson<P>::needs_solver;
      case 3:
        return time_advance::imex_stepper<P>::needs_solver;
      default:
        return false; // unreachable
    };
  }
  //! returns the precondtioner required by the solver, if any
  precon_method needed_precon() const {
    switch (method.index()) {
      case 0: // steady state
        return std::get<0>(method).needed_precon();
      case 2: // implicit stepper
        return std::get<2>(method).needed_precon();
      case 3: // implicit stepper
        return std::get<3>(method).needed_precon();
      default:
        return precon_method::none;
    };
  }

  //! prints the time-advance stats
  void print_time(std::ostream &os = std::cout) const {
    if (method.index() == 0) {
      os << "steady state solver:\n";
      os << "  stop-time (T)   " << data.stop_time() << '\n';
      std::get<0>(method).print_solver_opts(os);
      return;
    }
    os << "time stepping:\n  method          " << prog_opts::get_name(data.step_method()) << "\n"
       << data;
    if (needs_solver()) { // show solver data
      switch (method.index()) {
        case 2: // crank_nicolson
          std::get<2>(method).print_solver_opts(os);
          break;
        case 3: // imex
          std::get<3>(method).print_solver_opts(os);
        default: // implicit method or steady-state already done above, nothing to do
          break;
      };
    }
  }
  //! returns the count the iterations of the iterative solver, -1 if using a direct solver
  int64_t solver_iterations() const {
    switch (method.index()) {
      case 0:
        return std::get<0>(method).num_apply_calls();
      case 2:
        return std::get<2>(method).num_apply_calls();
      case 3:
        return std::get<3>(method).num_apply_calls();
      default:
        return -1;
    };
  }
  //! returns true of the stepper is set to steady-state
  bool is_steady_state() const { return (method.index() == 0); }

  //! holds the common time-stepping parameters
  time_data data;
  //! wrapper around the specific method being used
  std::variant<time_advance::steady_state<P>, time_advance::rungekutta<P>,
               time_advance::crank_nicolson<P>, time_advance::imex_stepper<P>> method;
};

/*!
 * \internal
 * \ingroup asgard_time_advance
 * \brief Allows writing time-data to a stream
 *
 * \endinternal
 */
template<typename P>
inline std::ostream &operator<<(std::ostream &os, time_advance_manager<P> const &manger)
{
  manger.print_time(os);
  return os;
}

}

