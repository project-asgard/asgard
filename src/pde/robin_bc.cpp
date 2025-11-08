#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file robin_bc.cpp
 * \brief Robin boundary conditions
 * \author The ASGarD Team
 *
 * Simple example of steady-state partial differential equation
 * with Robin boundary conditions.
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_robin Example: Robin boundary condition
 *
 * \par Elliptic equation
 * Creates a simple elliptic PDE in one dimension
 * \f[ -\frac{d^2}{d x^2} f = \cos(x) \f]
 * the domain is (-0.5, 2) and the exact solution is
 * \f[ f(x) = \cos(x) \f]
 * The solution is obtained by setting Robin boundary conditions that link
 * the value of the function to the derivative at the two end-points of the domain.
 * \f[ \frac{d}{dx} f + R f = 0 \f]
 * Unlike other boundary conditions that have the effect of sources,
 * the Robin condition is represented as an additional term in the PDE scheme.
 * Currently, ASGarD supports only constant condition for R.
 * Taking the exact solution, the values for R are
 * \f[ R_{left} = \frac{ \sin(-0.5) }{ \cos(-0.5) } \qquad R_{right} = \frac{ \sin(2) }{ \cos(2) } \f]
 *
 *
 * \par
 * This examples shows how to set Robin boundary conditions.
 */

/*!
 * \ingroup asgard_examples_robin
 * \brief Indicate the way the terms are constructed
 */
enum class pde_mode {
  //! put the penalty and Robin conditions into a single term
  coalesced,
  //! split the terms individually
  split
};

/*!
 * \ingroup asgard_examples_robin
 * \brief Make an elliptic PDE with Robin boundary conditions
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam pde_mode indicates the mode for the terms
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param options is the set of options
 *
 * \returns the asgard::pde_scheme definition
 *
 * \snippet robin_bc.cpp elliptic-robin make
 */
template<pde_mode mode, typename P = asgard::default_precision>
asgard::pde_scheme<P> make_robin_pde(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic-robin make]
#endif
  options.title = "Elliptic PDE with Robin boundary conditions";

  asgard::pde_domain<P> domain({{-0.5, 2}, });

  options.default_degree = 2;
  options.default_start_levels = {4, };

  // forces the problem to a steady-state mode
  options.force_step_method(asgard::time_method::steady);

  // OK for small problems, larger one should switch to gmres or bicgstab
  options.default_solver = asgard::solver_method::direct;

  // defaults for iterative solvers, not necessarily optimal
  options.default_precon = asgard::precon_method::jacobi;
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 2000;

  asgard::pde_scheme<P> pde(options, std::move(domain));

  auto source_1d = [](std::vector<P> const &x, P, std::vector<P> &f)
      -> void
    {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < x.size(); i++)
        f[i] = std::cos(x[i]);
    };

  pde.add_source(std::vector<asgard::svector_func1d<P>>{source_1d, });

  // setting Robin boundary conditions is done in two-stages
  // first, we set the terms for Neumann boundary conditions
  // second, in place of setting left/right-flux, we use a term that sets the flux
  //         as a function of the field itself

  // the condition is df/dx + gamma * f = 0, if f = cos(x) then
  // gamma = sin(x) / cos(x) and x is at the left and right boundary
  double const left  = std::sin(-0.5) / std::cos(-0.5);
  double const right = std::sin(2.0) / std::cos(2.0);

  // the first step is to set Neumann condition on both sides, in a standard setting
  // two Neumann conditions are not well-posed, the solution is unique up to a constant
  // the Robin term will ensure the problem has a unique solution
  asgard::term_1d<P> div  = asgard::term_div{-1, asgard::boundary_type::bothsides};
  asgard::term_1d<P> grad = asgard::term_grad{1, asgard::boundary_type::none};
  asgard::term_1d<P> dxx{asgard::term_chain{}, {div, grad}};

  if constexpr (mode == pde_mode::coalesced)
  {
    dxx.set_left_robin(left);
    dxx.set_right_robin(right);

    double const inv_dx = 1.0 / pde.cell_size(0);
    dxx.set_penalty(inv_dx);

    pde += asgard::term_md<P>{{dxx, }};
  }
  else
  {
    pde += asgard::term_md<P>{{dxx, }};

    asgard::term_1d<P> robin = asgard::term_robin{left, right};
    pde += asgard::term_md<P>{{robin, }};

    // we also need a penalty term
    P const inv_dx = 1.0 / pde.cell_size(0);
    asgard::term_1d<P> pen = asgard::term_penalty<P>{inv_dx, asgard::boundary_type::none};
    pde += asgard::term_md<P>{{pen, }};
  }

  return pde;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic-robin make]
#endif
}

/*!
 * \ingroup asgard_examples_robin
 * \brief Computes the L^2 error for the given example
 *
 * The provided discretization_manager should hold a PDE made with
 * make_robin_pde() and the solution should be set.
 *
 * \tparam P is double or float, the precision of the manager
 *
 * \param disc is the discretization of a PDE
 *
 * \returns the L^2 error between the known exact solution and
 *          the current state in the \b disc manager
 *
 * \snippet robin_bc.cpp elliptic-robin get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic-robin get-err]
#endif

  // construct the exact solution, since there is no initial condition
  auto s1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::cos(x[i]);
    };

  std::vector<P> const eref = disc.project_function(asgard::separable_func<P>({s1d, }));

  double constexpr enorm = 1.271167122374992; // integral of cos(x)^2 over (-0.5, 2)

  disc.sync_mpi_state(); // is using multiple ranks, sync across the ranks
  std::vector<P> const &state = disc.current_state();

  // see the continuity example for the trick used here
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
//! [elliptic-robin get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_robin
 * \brief main() for the sine-wave example
 *
 * The main() processes the command line arguments and calls both
 * make_robin_pde() and get_error_l2().
 *
 * \snippet robin_bc.cpp elliptic-robin main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic-robin main]
#endif
  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves an elliptic PDE with Robin boundary conditions\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout <<
R"help(<< additional options for this file >>
-bound           -bc     int        accepts: 0 or 1
                                    0 - use coalesced conditions
                                    1 - separate conditions

-test                               perform self-testing
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", }, {"-bound", "-bc"});

  if (options.has_cli_entry("-test")) {
    self_test();
    return 0;
  }

  // using a method similar to extra_cli_value that will loop for multiple entries
  // returns the first entry found
  std::optional<int> cli_btype = options.extra_cli_value_group<int>({"-bound", "-bc"});
  int const btype = cli_btype.value_or(0);

  if (options.is_mpi_rank_zero()) {
    if (not cli_btype) {
      std::cout << "using default coalesced mode\n";
    } else if (btype == 0) {
      std::cout << "using coalesced mode\n";
    } else if (btype == 1) {
      std::cout << "using separate mode\n";
    } else {
      std::cerr << "incorrect value for -bound, must use 0 or 1\n";
      return 1;
    }
  }

  auto pde = (btype == 0)
             ? make_robin_pde<pde_mode::coalesced, P>(options)
             : make_robin_pde<pde_mode::split, P>(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  P const err = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- steady state error: " << err << '\n';

  return 0;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic-robin main]
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
  current_test<P> test_(opts);

  auto options = make_opts(opts);

  // using a method similar to extra_cli_value that will loop for multiple entries
  // returns the first entry found
  int const btype = options.extra_cli_value<int>("-bc").value_or(0);

  auto pde = (btype == 0)
             ? make_robin_pde<pde_mode::coalesced, P>(options)
             : make_robin_pde<pde_mode::split, P>(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);
  // std::cout << err << '\n';
  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("Robin boundary conditions", " div.grad f = sources");

  #ifdef ASGARD_ENABLE_DOUBLE
  dotest<double>(5.E-1, "-d 1 -l 1");
  dotest<double>(5.E-1, "-d 1 -l 1 -bc 1");
  dotest<double>(4.E-2, "-d 1 -l 2");
  dotest<double>(1.E-2, "-d 1 -l 3");
  dotest<double>(3.E-3, "-d 1 -l 4");
  dotest<double>(1.E-4, "-d 1 -l 6");
  dotest<double>(1.E-4, "-d 1 -l 6 -bc 1");
  dotest<double>(3.E-6, "-d 1 -l 9");
  dotest<double>(1.E-2, "-d 2 -l 1");
  dotest<double>(2.E-3, "-d 2 -l 2");
  dotest<double>(2.E-4, "-d 2 -l 3");
  dotest<double>(2.E-4, "-d 2 -l 3 -bc 1");
  dotest<double>(4.E-7, "-d 2 -l 6");
  dotest<double>(1.E-3, "-d 3 -l 1");
  dotest<double>(5.E-6, "-d 3 -l 3");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(5.E-1, "-d 1 -l 1");
  dotest<float>(2.E-3, "-d 1 -l 6");
  dotest<float>(2.E-3, "-d 1 -l 6 -bc 1");
  dotest<float>(5.E-3, "-d 2 -l 3");
  dotest<float>(5.E-3, "-d 2 -l 3 -bc 1");
  dotest<float>(1.E-3, "-d 2 -l 6");
  dotest<float>(8.E-4, "-d 3 -l 3");
  #endif
}

#endif
