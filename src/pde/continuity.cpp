#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file continuity.cpp
 * \brief Simple continuity example
 * \author The ASGarD Team
 * \ingroup asgard_examples_continuity_md
 *
 * \endinternal
 */

/*!
 * \defgroup asgard_examples ASGarD Example Problems
 *
 * \par
 * Several examples are included that demonstrate the usage
 * of the ASGarD library.
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_continuity_md Example 3, xD continuity equation
 *
 * \par Example 3
 * Solves the continuity partial differential equation in arbitrary dimension \b d
 * \f[ \frac{\partial}{\partial t} f + \nabla \cdot f = s \f]
 * where the right-hand-side source \b s is chosen so the exact solution
 * is the d-dimensional separable function
 * \f[ f(t, x_1, x_2, \cdots, x_d) = \cos(t) \prod_{j=1}^d \sin(x_j) \f]
 *
 * \par
 * This example provides a flexibility in the choice of the dimension
 * which can be controlled from the command line.
 * The range in dimension is (-2 PI, 2 PI)
 *
 * \par
 * The interesting part of this example is setup of a PDE in arbitrary dimension
 * (between 1 and 6) and the use of the asgard::prog_opts to handle custom
 * project options.
 */

/*!
 * \ingroup asgard_examples_continuity_md
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

/*!
 * \ingroup asgard_examples_continuity_md
 * \brief Make single continuity PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param num_dims is the number of dimensions, currently between 1 and 6
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \snippet continuity.cpp continuity_md make
 */
template<typename P = asgard::default_precision>
asgard::PDEv2<P> make_continuity_pde(int num_dims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_md make]
#endif

  options.title = "Continuity " + std::to_string(num_dims) + "D";

  // the domain will have range -2 * PI to 2 * PI in each direction
  std::vector<asgard::domain_range<P>> ranges(num_dims, {-2 * PI, 2 * PI});

  asgard::pde_domain<P> domain(ranges); // can use move here, but copy is cheap enough

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  options.default_degree = 2;
  options.default_start_levels = {4, };

  // find the max-level that the grid can have
  // that is the max between the defaults, start_levels and max_levels
  // the default is set above, the other may be provided from the command line
  int const max_level = options.max_level();

  // compute the smallest a cell can be
  P const dx = domain.min_cell_size(max_level);

  // the cfl condition is that dt < stability-region * dx
  // RK3 stability region is 0.1, dx is domain-length / number of cells or 4 PI / 2^max_level
  // for good measure, we take half of that value
  options.default_dt = 0.5 * 0.1 * dx;

  options.default_stop_time = 1.0; // integrate until T = 1

  // the exact solution vanishes when any dimension is at the origin
  // setting an off-center default view will yield a better plots
  // this is just the default and it does not limit any other options
  if (num_dims > 2 and options.default_plotter_view == "") {
    options.default_plotter_view = " * : * ";
    for (int d = 2; d < num_dims; d++)
      options.default_plotter_view += " : 1.57";
  }

  // create a pde from the given options and domain
  // we can read the variables using pde.options() and pde.domain() (both return const-refs)
  // the option entries may have been populated or updated with default values
  asgard::PDEv2<P> pde(options, std::move(domain));

  // one dimensional divergence term using upwind flux
  // multiple terms can be chained to obtain higher order derivatives
  asgard::term_1d<P> div = asgard::term_div<P>(1, asgard::flux_type::upwind,
                                               asgard::boundary_type::periodic);

  // the multi-dimensional divergence, initially set to identity in md
  std::vector<asgard::term_1d<P>> ops(num_dims);
  for (int d = 0; d < num_dims; d++)
  {
    ops[d] = div; // using derivative in the d-direction
    pde += asgard::term_md<P>(ops);
    ops[d] = asgard::term_identity{}; // reset back to identity
  }

  // defining the separable known solution
  // sin(x_1) * sin(x_2) * ... * sin(x_d) * cos(t)

  // creating a vector function corresponding to sin(x)
  // the function is called for batch of points in the domain (-2 PI, 2 PI)
  // corresponding to the 1d cells and the quadrature points in each cell
  auto sin_1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      // given values in x, must populate fx with the corresponding values
      assert(fx.size() == x.size()); // this is guaranteed, do NOT resize fx
      // OpenMP and SIMD directives can be used here
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::sin(x[i]);
    };

  // time functions are not called in batch
  // hence the signature takes a single entry
  auto cos_t = [](P t) -> P { return std::cos(t); };

  // the derivatives, d/dx sin(x) = cos(x) and d/dx cos(t) = -sin(t)
  auto cos_1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::cos(x[i]);
    };

  // negative sin(t)
  auto nsin_t = [](P t) -> P { return -std::sin(t); };

  // multidimensional product of functions, initializing to just sin(x)
  std::vector<asgard::svector_func1d<P>> sign_md(num_dims, sin_1d);

  // this is the exact solution
  asgard::separable_func<P> exact(sign_md, cos_t);

  // setting the exact solution as the initial condition
  // in general, the PDE can have multiple functions as initial conditions
  pde.add_initial(exact);

  // setting up the sources
  pde.add_source({sign_md, nsin_t}); // derivative in time

  // compute the spacial derivatives
  for (int d = 0; d < num_dims; d++)
  {
    sign_md[d] = cos_1d; // set derivative in x for direction d
    pde.add_source({sign_md, cos_t});
    sign_md[d] = sin_1d; // revert to the original value
  }

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_md make]
#endif
}

/*!
 * \ingroup asgard_examples_continuity_md
 * \brief Computes the L^2 error for the given example
 *
 * The provided discretization_manager should hold a PDE made with
 * make_continuity_pde(). This will compute the L^2 error.
 *
 * \tparam P is double or float, the precision of the manager
 *
 * \param disc is the discretization of a PDE
 *
 * \returns the L^2 error between the known exact solution and
 *          the current state in the \b disc manager
 *
 * \snippet continuity.cpp continuity_md get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_md get-err]
#endif

  // using the orthogonality of the basis and ignoring quadrature error
  // in the projection of the exact solution onto the current basis
  // the error has two components:
  // - difference between the current state and the projection
  // - the L^2 norm of the exact solution minus the projection

  int const num_dims = disc.num_dims();

  // using the fact that the initial condition is the exact solution
  // disc.get_pde2().ic_sep() returns the separable initial conditions
  // disc.project_function() projects a set of separable functions
  // onto the current sparse grid basis and returns the coefficients
  std::vector<P> const eref = disc.project_function(disc.get_pde2().ic_sep());

  double constexpr space1d = 2 * PI; // integral of sin(x)^2 over (-2 * PI, 2 * PI)
  double const time_val    = std::cos(disc.time_params().time());

  // this is the L^2 norm-squared of the exact solution
  // powi works the same as std::pow but the second input is an integer
  double const enorm = asgard::fm::powi(space1d, num_dims) * time_val * time_val;

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

  // when cos(t) vanishes, so does the exact solution and enorm -> 0
  // for small values of enorm, the relative error is artificially magnified
  // switch between relative and absolute error
  if (enorm < 1.E-3)
    return std::sqrt(ndiff + enorm - nself);
  else
    return std::sqrt((ndiff + enorm - nself) / enorm);
#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_md get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_continuity_md
 * \brief main() for the continuity example
 *
 * The main() processes the command line arguments and calls both
 * make_continuity_pde() and get_error_l2().
 *
 * \snippet continuity.cpp continuity_md main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_md main]
#endif

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this file and the two additional options accepted for this problem
  if (options.show_help) {
    std::cout << "\n solves the continuity equation:\n";
    std::cout << "    f_t + div f = s(t, x)\n";
    std::cout << " with periodic boundary conditions \n"
                 " and source term that generates a known artificial solution\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-dims            -dm     int        accepts: 1 - 6\n";
    std::cout << "                                    the number of dimensions\n\n";
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  // this is an optional step, check if there are misspelled or incorrect cli entries
  // the first set/vector of entries are those that can appear by themselves
  // the second set/vector requires extra parameters
  options.throw_if_argv_not_in({"-test", "--test"}, {"-dims", "-dm"});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  // check if either -dims or -dm was provided with an int value
  std::optional<int> opt_dims = options.extra_cli_value<int>("-dims");
  if (not opt_dims)
    opt_dims = options.extra_cli_value<int>("-dm");

  int const num_dims = opt_dims.value_or(2);

  if (not opt_dims)
    std::cout << "no -dims provided, setting a default 2D problem\n";
  else
    std::cout << "setting a " << num_dims << "D problem\n";

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(make_continuity_pde(num_dims, options),
                                         asgard::verbosity_level::high);

  // time-integration is performed using the advance_time() method
  // advance_time(disc, n); will integrate for n time-steps
  // skipping n (or using a negative) will integrate until the end

  if (not disc.stop_verbosity())
    std::cout << " -- error in the initial conditions: " << get_error_l2(disc) << "\n";

  disc.advance_time(); // integrate until num-steps or stop-time

  disc.progress_report();

  if (not disc.stop_verbosity())
    std::cout << " -- final error: " << get_error_l2(disc) << "\n";

  disc.save_final_snapshot(); // only if output filename is provided

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_md main]
#endif
};

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

  discretization_manager<P> disc(make_continuity_pde<P>(num_dims, options),
                                 verbosity_level::quiet);

  while (disc.time_params().num_remain() > 0)
  {
    disc.advance_time(1);

    double const err = get_error_l2(disc);

    tcheckless(disc.time_params().step(), err, tol);
  }
}

template<typename P>
void dolongtest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  discretization_manager<P> disc(make_continuity_pde<P>(num_dims, options),
                                 verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);

  tcheckless(disc.time_params().step(), err, tol);
}

template<typename P>
void dotest(double tol, int num_dims, std::string const &opts, int np) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  discretization_manager<P> disc(make_continuity_pde<P>(num_dims, options),
                                 verbosity_level::quiet);

  // makes a dense grid over the domain using np points each direction
  vector2d<double> const mesh = make_grid<double>(disc.get_pde2().domain(), np);

  // the reconstruction is always done in double-precision even if the data
  // coming from the discretization_manager is in floats
  // thus, use the double-precision version of the exact solution
  auto sin_1d = [](std::vector<double> const &x, double, std::vector<double> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::sin(x[i]);
    };

  auto cos_t = [](double t) -> double { return std::cos(t); };

  separable_func<double> exact(
      std::vector<svector_func1d<double>>(num_dims, sin_1d), cos_t);

  std::vector<double> ref(mesh.num_strips());
  std::vector<double> com(mesh.num_strips());

  while (disc.time_params().num_remain() > 0)
  {
    disc.advance_time(1);

    double const time = disc.time_params().time();
#pragma omp parallel for
    for (int64_t i = 0; i < mesh.num_strips(); i++)
      ref[i] = exact.eval(mesh[i], time);

    auto shot = disc.get_snapshot();

    shot.reconstruct(mesh[0], mesh.num_strips(), com.data());

    double err = 0;
    for (size_t i = 0; i < ref.size(); i++)
      err = std::max(err, std::abs(com[i] - ref[i]));

    tcheckless(disc.time_params().step(), err, tol);
  }
}

void self_test() {
  all_tests testing_("continuity equation:", " f_t + div f = sources");

  // continuity is a simple pde and tests are cheap
  // thus, we can use to indirectly test multiple aspects of ASGarD
  // we still want tests to run fast but also show the important dynamics

#ifdef ASGARD_ENABLE_DOUBLE
  // test convergence with respect to sparse grid level
  dotest<double>(0.05,   2, "-l 4 -n 20");
  dotest<double>(0.02,   2, "-l 5 -n 20");
  dotest<double>(0.005,  2, "-l 6 -n 20");
  dotest<double>(0.0005, 2, "-l 7 -n 20");

  // test convergence with respect to polynomial degree
  // the target function is osculatory and degree 0 and 1 yield high error
  dotest<double>(0.5,   2, "-l 6 -d 0 -n 20");
  dotest<double>(0.3,   2, "-l 5 -d 1 -n 20");
  dotest<double>(0.02,  2, "-l 5 -d 2 -n 20");
  dotest<double>(0.002, 2, "-l 5 -d 3 -n 20");

  // test adaptivity, should remain within the error tolerance
  dotest<double>(1.E-2, 2, "-l 4 -m 8 -d 2 -n 20 -a 1.E-2");
  dotest<double>(1.E-3, 2, "-l 4 -m 8 -d 2 -n 20 -a 1.E-3");
  dotest<double>(1.E-4, 2, "-l 4 -m 8 -d 2 -n 20 -a 1.E-4");

  // test different number of dimensions
  dotest<double>(1.E-1, 1, "-l 4 -m 8 -d 2 -n 20 -a 1.E-1");
  dotest<double>(1.E-1, 3, "-l 4 -m 8 -d 2 -n 10 -a 1.E-1");
  dotest<double>(5.E-2, 4, "-l 5 -d 3 -n 10");

  // test against point-wise values of the exact solution
  // using 30 or 10 as point density in each direction
  dotest<double>(0.05, 2, "-l 4 -n 20", 30);
  dotest<double>(0.02, 2, "-l 5 -n 20", 10);

  // run long time integration, saves on computing the error on each time-step
  // also, at t of pi/2 the solution vanishes and the relative error explodes
  dolongtest<double>(0.02, 2, "-l 5 -t 10");

  // adaptivity is tricky near the time-period when the solution vanishes
  dolongtest<double>(0.03, 2, "-l 4 -m 8 -t 10 -a 5.E-3");

  // different explicit time-stepping
  dotest<double>(0.05, 2, "-s rk2 -l 5 -n 20");
  dotest<double>(0.01, 2, "-s rk2 -l 6 -n 10");

  // implicit stepping is fast, test some of the implicit methods
  dotest<double>(0.05, 1, "-l 7 -n 20 -sv direct -s be -dt 0.05");
  dotest<double>(0.025, 1, "-l 7 -n 20 -sv direct -s be -dt 0.025");
  dotest<double>(0.01, 1, "-l 7 -n 20 -sv direct -s be -dt 0.01");

  dotest<double>(0.001, 1, "-l 7 -n 20 -sv direct -s cn -dt 0.05");
  dotest<double>(0.00025, 1, "-l 7 -n 20 -sv direct -s cn -dt 0.025");
  dotest<double>(1.E-5, 1, "-l 7 -n 20 -sv direct -s cn -dt 0.01");
#endif

#ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(0.05,  2, "-l 4 -n 20");
  dotest<float>(0.02,  2, "-l 5 -n 20");
  dotest<float>(0.005, 2, "-l 6 -n 20");
  dotest<float>(0.001, 2, "-l 7 -n 20");

  dotest<float>(0.8,   2, "-l 5 -d 0 -n 20");
  dotest<float>(0.3,   2, "-l 5 -d 1 -n 20");
  dotest<float>(0.02,  2, "-l 5 -d 2 -n 20");
  dotest<float>(0.002, 2, "-l 5 -d 3 -n 20");

  dotest<float>(5.E-2, 2, "-l 4 -m 8 -d 2 -n 20 -a 1.E-2");
  dotest<float>(1.E-2, 2, "-l 4 -m 8 -d 2 -n 20 -a 1.E-3");
  dotest<float>(3.E-3, 2, "-l 4 -m 8 -d 2 -n 20 -a 1.E-3");

  dotest<float>(1.E-1, 1, "-l 4 -m 8 -d 2 -n 20 -a 1.E-1");
  dotest<float>(1.E-1, 3, "-l 4 -m 8 -d 2 -n 10 -a 1.E-1");
  dotest<float>(5.E-2, 4, "-l 5 -d 3 -n 10");

  dotest<float>(0.05, 2, "-l 4 -n 20", 30);
  dotest<float>(0.02, 2, "-l 5 -n 20", 10);

  dolongtest<float>(0.02, 2, "-l 5 -t 10");

  dotest<float>(0.05, 1, "-l 7 -n 20 -sv direct -s be -dt 0.05");
  dotest<float>(0.025, 1, "-l 7 -n 20 -sv direct -s be -dt 0.025");

  dotest<float>(0.02, 1, "-l 7 -n 20 -sv direct -s cn -dt 0.06");
  dotest<float>(0.005, 1, "-l 7 -n 20 -sv direct -s cn -dt 0.03");
#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
