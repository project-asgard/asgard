#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file sinwav.cpp
 * \brief Moving sine wave
 * \author The ASGarD Team
 *
 * Simple example of boundary conditions.
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_sinwav Example 6, Moving sine wave
 *
 * \par Example 6
 * Creates a simple hyperbolic 1D PDE where the solution starts from zero and
 * grows to a moving sine wave, namely
 * \f[ \frac{\partial}{\partial t} f +  \frac{\partial}{\partial x} f = 0 \f]
 * the domain is (0, 1) and initial and right boundary conditions are
 * \f[ f(x, 0) = 0, \qquad f(0, t) = \sin(-2 \pi t) \f]
 * The right boundary condition is free.
 *
 * \par
 * The wave starts from the left corner and moves to the right with speed 1,
 * after t = 1 the domain holds the entire cycle of a sine wave but shifted.
 *
 * \par
 * A second version of the problem has the wave traveling in the opposite
 * direction, i.e., the sign of the divergence term is flipped to negative.
 * The initial condition remains zero but the boundary condition is defined
 * on the right.
 *
 * \par
 * This example has two PDEs that have similarities but also significant differences,
 * which is in contrast to the previous problems.
 * Part of the goal here is to demonstrate how multiple similar PDEs can be
 * coded in the same file and easily controlled,
 * e.g., using custom command line parameters.
 * The main goal, is to show how to set separable boundary conditions.
 */

/*!
 * \ingroup asgard_examples_sinwav
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

/*!
 * \ingroup asgard_examples_sinwav
 * \brief Indicates the direction of the traveling wave
 */
enum class from_direction {
  //! from left or from the bottom in the 2D case
  left,
  //! from right or from the top in the 2D case
  right
};

/*!
 * \ingroup asgard_examples_sinwav
 * \brief Make single sine-wave PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam dir indicates the direction from which the wave comes (left/right)
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \snippet sinwav.cpp sinwav make
 */
template<from_direction dir, typename P = asgard::default_precision>
asgard::PDEv2<P> make_sinwav_pde(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [sinwav make]
#endif
  options.title = "Sine Wave PDE";
  // put into the subtitle and indication of the direction
  options.subtitle = (dir == from_direction::left) ? "(from-left)" : "(from-right)";

  asgard::pde_domain<P> domain({{0, 1}, }); // 1D domain

  options.default_degree = 2;
  options.default_start_levels = {6, };

  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);

  options.default_dt = 0.5 * 0.1 * dx;

  // do not stop right at 1, due to rounding integration should go until slightly more
  options.default_stop_time   = 1.25;
  options.default_step_method = asgard::time_method::rk2;

  asgard::PDEv2<P> pde(options, std::move(domain));

  // setting the components of the operators
  if constexpr (dir == from_direction::left) {
    auto sine = [=](P t)-> P { return std::sin(-2 * PI * t); };

    asgard::term_1d<P> div1 = asgard::term_div<P>(
        1, asgard::flux_type::upwind, asgard::boundary_type::right_free,
        asgard::dirichelt_boundary1d<P>{sine, 0});

    pde += div1;
  }
  else
  {
    auto sine = [=](P t)-> P { return std::sin(2 * PI * t); };

    asgard::term_1d<P> div1 = asgard::term_div<P>(
        -1, asgard::flux_type::upwind, asgard::boundary_type::left_free,
        asgard::dirichelt_boundary1d<P>{0, sine});

    pde += div1;
  }

  // no sources, initial condition is zero

  return pde;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [sinwav make]
#endif
}

/*!
 * \ingroup asgard_examples_sinwav
 * \brief Computes the L^2 error for the given example
 *
 * The provided discretization_manager should hold a PDE made with
 * make_sinwav_pde() and the current time should be at least 1.
 * This will compute the L^2 error, but if time is less than 1
 * it will return 0.
 *
 * \tparam P is double or float, the precision of the manager
 *
 * \param disc is the discretization of a PDE
 *
 * \returns the L^2 error between the known exact solution and
 *          the current state in the \b disc manager
 *
 * \snippet sinwav.cpp sinwav get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [sinwav get-err]
#endif

  double const t = disc.time_params().time();

  if (t < 1) {
    if (not disc.stop_verbosity())
      std::cerr << " -- warning: cannot compute l2 error for t < 1\n";
    return 0;
  }

  // get the direction from the subtitle
  bool const use_left = disc.subtitle_contains("(from-left)");

  // exact solution for t >= 1
  // here the time and use_left are captured by copy
  // avoid capturing by copy when dealing with large objects, e.g., disc
  // capture by reference can be used, but make sure not to make calls to sine
  // after any of the captured-by-reference objects have been destroyed
  auto sine = [=](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      if (use_left) {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = std::sin(2 * PI * (x[i] - t));
      } else {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = std::sin(2 * PI * (x[i] + t));
      }
    };

  // see the example 3 that explains the orthogonality trick
  // to compute the L^2 error
  std::vector<P> const eref = disc.project_function(asgard::separable_func<P>{{sine, }});

  double constexpr enorm = 0.5;

  std::vector<P> const &state = disc.current_state();

  double nself = 0;
  double ndiff = 0;
  for (size_t i = 0; i < state.size(); i++)
  {
    double const e = eref[i] - state[i];
    ndiff += e * e;
    double const r = eref[i];
    nself += r * r;
  }

  return std::sqrt((ndiff + enorm - nself) / enorm);
#ifndef __ASGARD_DOXYGEN_SKIP
//! [sinwav get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_sinwav
 * \brief main() for the sine-wave example
 *
 * The main() processes the command line arguments and calls both
 * make_sinwav_pde() and get_error_l2().
 *
 * \snippet sinwav.cpp sinwav main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [sinwav main]
#endif
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves a messy testing pde:\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout <<
R"help(<< additional options for this file >>
-left                    -          wave comes from the left
-right                   -          wave comes from the right

-test                               perform self-testing
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", "--test", "-left", "-right"}, {});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    self_test();
    return 0;
  }

  // find the direction selected by the user
  // from_direction::left is default, throw if using conflicting options
  from_direction dir = from_direction::left;
  if (options.has_cli_entry("-left")) {
    if (options.has_cli_entry("-right"))
      throw std::runtime_error("cannot provide both -left and -right arguments");
  } else if (options.has_cli_entry("-right")) {
    dir = from_direction::right;
  } else {
    std::cout << "using default direction: from left\n";
  }

  auto pde = (dir == from_direction::left)
             ? make_sinwav_pde<from_direction::left, P>(options)
             : make_sinwav_pde<from_direction::right, P>(options);

  asgard::discretization_manager<P> disc(std::move(pde),
                                         asgard::verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  if (not disc.stop_verbosity() and disc.time_params().time() >= 1) {
    P const err = get_error_l2(disc);
    std::cout << " -- final error: " << err << '\n';
  }

  return 0;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [sinwav main]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////

// just for convenience to avoid using asgard:: all over the place
// normally, one should only include what is needed
using namespace asgard;

template<typename P>
void dotest(double tol, std::string const &opts) {
  current_test<P> test_(opts, 1);

  auto options = make_opts(opts);

  bool const left = options.has_cli_entry("-left");

  bool const final_only = options.has_cli_entry("-test-final-only");

  auto pde = (left) ? make_sinwav_pde<from_direction::left, P>(options)
                    : make_sinwav_pde<from_direction::right, P>(options);

  discretization_manager<P> disc(std::move(pde), verbosity_level::quiet);

  if (final_only) {
    disc.advance_time();
    double const err = get_error_l2(disc);
    // std::cout << err << '\n';
    tcheckless(disc.time_params().step(), err, tol);
  } else {
    while (disc.time_params().num_remain() > 0)
    {
      disc.advance_time(1);

      if (not final_only) {
        double const err = get_error_l2(disc);
        // std::cout << err << '\n';
        tcheckless(disc.time_params().step(), err, tol);
      }
    }
  }
}

void self_test() {
  #ifdef ASGARD_ENABLE_DOUBLE
  // the solution starts as constant zero and turns into sine wave
  // this creates a kink (discontinuity in the first derivative)
  // thus, at time t = 1 the convergence is only 1-st order in dx
  // after the initial kink leaves the domain, the error goes down
  dotest<double>(1.E-2, "-l 4 -t 1.1 -left");
  dotest<double>(1.E-2, "-l 4 -t 1.1 -right");
  dotest<double>(5.E-3, "-l 5 -t 1.1 -left");
  dotest<double>(5.E-3, "-l 5 -t 1.1 -right");
  dotest<double>(1.E-3, "-l 6 -t 1.1 -left");
  dotest<double>(1.E-3, "-l 6 -t 1.1 -right");
  dotest<double>(5.E-4, "-l 7 -t 1.1 -left");
  dotest<double>(5.E-4, "-l 7 -t 1.1 -right");

  dotest<double>(5.E-4, "-l 4 -t 1.4 -left -test-final-only");
  dotest<double>(5.E-4, "-l 4 -t 1.4 -right -test-final-only");
  dotest<double>(1.E-4, "-l 5 -t 1.4 -left -test-final-only");
  dotest<double>(1.E-4, "-l 5 -t 1.4 -right -test-final-only");
  dotest<double>(5.E-5, "-l 6 -t 1.4 -left -test-final-only");
  dotest<double>(5.E-5, "-l 6 -t 1.4 -right -test-final-only");
  dotest<double>(7.E-6, "-l 7 -t 1.4 -left -test-final-only");
  dotest<double>(7.E-6, "-l 7 -t 1.4 -right -test-final-only");

  dotest<double>(3.E-5, "-m 8 -a 1.E-5 -t 1.4 -test-final-only");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(1.E-2, "-l 4 -t 1.1 -left");
  dotest<float>(1.E-2, "-l 4 -t 1.1 -right");
  dotest<float>(5.E-3, "-l 5 -t 1.1 -left");
  dotest<float>(3.E-3, "-l 6 -t 1.1 -right");

  dotest<float>(5.E-2, "-l 2 -t 1.4 -left -test-final-only");
  dotest<float>(5.E-2, "-l 2 -t 1.4 -right -test-final-only");
  dotest<float>(5.E-3, "-l 3 -t 1.4 -left -test-final-only");
  dotest<float>(5.E-3, "-l 3 -t 1.4 -right -test-final-only");
  dotest<float>(1.E-3, "-l 4 -t 1.4 -left -test-final-only");
  dotest<float>(1.E-3, "-l 4 -t 1.4 -right -test-final-only");

  dotest<float>(3.E-3, "-m 6 -a 1.E-3 -t 1.4 -test-final-only");
  #endif
}

#endif
