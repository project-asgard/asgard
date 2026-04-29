#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file elliptic_nonsep.cpp
 * \brief Non-separable Elliptic equation
 * \author The ASGarD Team
 *
 * Steady-state partial differential equation with non-separable coefficient and boundary conditions.
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_ellipticns Example: Elliptic equation
 *
 * \par Elliptic equation
 * Creates a simple elliptic PDE that multiplies across the dimensions
 * the same one-dimensional boundary value problem
 * \f[ -\frac{d^2}{d x^2} f = 2 \f]
 * the domain is (0, 1) and the exact solution is
 * \f[ f(x) = 2 x - x^2 \f]
 * The solution can be obtained by assigning homogeneous boundary conditions,
 * Dirichlet on the left and Neumann on the right,
 * or alternatively we can assign inhomogeneous conditions
 * \f[ \frac{d}{dx} f(0) = 2, \qquad f(1) = 1 \f]
 * Since the solution is a quadratic function, using degree of 2 or more
 * should resolve the exact solution regardless of the grid
 * (up to rounding error due to conditioning and precision).
 *
 * \par
 * This examples shows how to set different types of boundary conditions
 * and how to solve a steady state problem.
 */

/*!
 * \ingroup asgard_examples_ellipticns
 * \brief Make an elliptic PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam boudnary indicates the type of boundary to use
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param num_dims number of dimensions
 * \param options is the set of options
 *
 * \returns the asgard::pde_scheme description
 *
 * \b Note: The asgard namespace includes the name \b boundary_type,
 * it a natural name but it is possible to create a conflict if the entire namespace
 * is included.
 *
 * \snippet elliptic_nonsep.cpp ellipticns make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_elliptic_pde(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [ellipticns make]
#endif
  options.title = "Non-separable elliptic PDE";

  asgard::pde_domain<P> domain({{0, 1}, {0, 1}, {0, 1}});

  options.default_degree = 2;
  options.default_start_levels = {4, };

  // previous examples were setting a default stepping method
  // which allows the cli options to overwrite the selection
  // here, we are overwriting the cli selection, if another
  // method was requested then a warning will be generated
  // (this should probably be an error instead of a warning)
  options.force_step_method(asgard::time_method::steady);

  // OK for small problems, larger one should switch to gmres or bicgstab
  options.default_solver = asgard::solver_method::direct;

  // defaults for iterative solvers, not necessarily optimal
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;

  asgard::pde_scheme<P> pde(options, std::move(domain));


  // if an initial condition is specified, it will be used as the initial guess
  // of an iterative solver, other zeros is used as the initial guess
  // the direct solver does not use an initial guess

  return pde;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [ellipticns make]
#endif
}

/*!
 * \ingroup asgard_examples_ellipticns
 * \brief Computes the L^2 error for the given example
 *
 * The provided discretization_manager should hold a PDE made with
 * make_elliptic_pde() and the solution should be set.
 *
 * \tparam P is double or float, the precision of the manager
 *
 * \param disc is the discretization of a PDE
 *
 * \returns the L^2 error between the known exact solution and
 *          the current state in the \b disc manager
 *
 * \snippet elliptic_nonsep.cpp ellipticns get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [ellipticns get-err]
#endif

  int const num_dims = disc.num_dims();

  // see the continuity example for the orthogonality trick

  // construct the exact solution, since there is no initial condition
  auto s1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = x[i] * (P{2} - x[i]);
    };

  // set the right-hand-side for each dimension
  std::vector<asgard::svector_func1d<P>> func(num_dims, s1d);

  std::vector<P> const eref = disc.project_function(asgard::separable_func<P>(func));

  double constexpr space1d = 8.0 / 15.0; // integral of (2x - x^2)^2 over (0, 1)

  // this is the L^2 norm-squared of the exact solution
  double const enorm = asgard::fm::powi(space1d, num_dims);

  disc.sync_mpi_state(); // is using multiple ranks, sync across the ranks
  std::vector<P> const &state = disc.current_state();
  assert(eref.size() == state.size());

  double nself = 0;
  double ndiff = 0;
  for (size_t i = 0; i < state.size(); i++)
  {
    double const e = eref[i] - state[i];
    ndiff += e * e;
    double const r = eref[i];
    nself += r * r;
  }

  return std::sqrt((ndiff + std::abs(enorm - nself)) / enorm);

#ifndef __ASGARD_DOXYGEN_SKIP
//! [ellipticns get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_ellipticns
 * \brief main() for the sine-wave example
 *
 * The main() processes the command line arguments and calls both
 * make_elliptic_pde() and get_error_l2().
 *
 * \snippet elliptic_nonsep.cpp elliptic main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic main]
#endif
  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves an non-separable elliptic PDE\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout <<
R"help(<< additional options for this file >>
-test                               perform self-testing
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", }, {});

  if (options.has_cli_entry("-test")) {
    self_test();
    return 0;
  }

  auto pde = make_elliptic_pde(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  P const err = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- steady state error: " << err << '\n';

  return 0;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic main]
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
void dotest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  auto pde = make_elliptic_pde<P>(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);
  // std::cout << err << '\n';
  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("elliptic steady state problem", " div.grad f = sources");

  #ifdef ASGARD_ENABLE_DOUBLE
  dotest<double>(5.E-3, 1, "-d 1 -l 3");
  dotest<double>(1.E-3, 1, "-d 1 -l 4");
  dotest<double>(5.E-4, 1, "-d 1 -l 5");
  dotest<double>(5.E-4, 1, "-d 1 -l 5 -bc 1");
  dotest<double>(5.E-4, 1, "-d 1 -l 5 -bc 1");
  dotest<double>(5.E-4, 1, "-d 1 -l 5 -bc 1");

  dotest<double>(1.E-3, 2, "-d 1 -l 4");
  dotest<double>(1.E-3, 3, "-d 1 -l 5 -sv bicgstab");
  dotest<double>(1.E-3, 4, "-d 1 -l 5 -sv bicgstab");

  dotest<double>(1.E-7, 1, "-d 2 -l 3");
  dotest<double>(5.E-7, 2, "-d 2 -l 3");
  dotest<double>(5.E-7, 3, "-d 2 -l 3");

  dotest<double>(1.E-3, 1, "-d 1 -l 4");
  dotest<double>(1.E-3, 2, "-d 1 -l 5");
  dotest<double>(1.E-3, 3, "-d 1 -l 6  -sv bicgstab");

  dotest<double>(1.E-7, 1, "-d 2 -l 3 -bc 1");
  dotest<double>(5.E-7, 2, "-d 2 -l 3 -bc 1");
  dotest<double>(5.E-7, 3, "-d 2 -l 3 -bc 1");

  dotest<double>(1.E-3, 1, "-d 1 -l 4 -bc 1");
  dotest<double>(1.E-3, 2, "-d 1 -l 5 -bc 1");
  dotest<double>(1.E-3, 3, "-d 1 -l 6 -bc 1 -sv bicgstab");

  dotest<double>(1.E-3, 2, "-bc 1 -l 3 -m 8 -a 1.E-5");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(5.E-3, 1, "-d 1 -l 5");
  dotest<float>(5.E-3, 1, "-d 2 -l 3");
  dotest<float>(5.E-3, 1, "-d 2 -l 3 -bc 1");

  dotest<float>(5.E-3, 2, "-d 1 -l 5");
  #endif
}

#endif
