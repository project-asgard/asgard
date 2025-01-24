#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file two_stream.cpp
 * \brief Simple continuity example
 * \author The ASGarD Team
 * \ingroup asgard_examples_diffusion
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_two_stream Example 5, Two stream instability
 *
 * \par Example 5
 * Solves the Vlasov-Poisson equation in a common example
 * often called the two stream instability problem
 * \f[ \frac{\partial}{\partial t} f(x, v) + v \nabla_x f(x, v, t) + E(x, t) \cdot \nabla_v f(x, v, t) = 0 \f]
 * where the electric field term depends on the Poisson equation
 * \f[ - \nabla_x \cdot \nabla_x \Phi(x, t) = \int_v f(x, v, t) dv \f]
 * The equation represents the evolution of a charged particle field under the effects
 * of self-induced electric field.
 * The right-hand integral represents the density of the particles and creates
 * non-linear coupling between the fields.
 *
 *
 */

/*!
 * \ingroup asgard_examples_two_stream
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_two_stream
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls both
 * make_diffusion_pde() and get_error_l2().
 * The interesting part is how to add custom command line parameters
 * to the default ones provided by ASGarD.
 *
 * \snippet diffusion.cpp diffusion_md main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md main]
#endif

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves the two stream Vlasov-Poisson in 1x-1v dimensions\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  // this is an optional step, check if there are misspelled or incorrect cli entries
  // the first set/vector of entries are those that can appear by themselves
  // the second set/vector requires extra parameters
  options.throw_if_argv_not_in({"-test", "--test"}, {});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  options.title = "Two Stream Instability";

  asgard::pde_domain<P> domain(asgard::position_dims{1}, asgard::velocity_dims{1},
                               {{-2 * PI, 2 * PI}, {-2 * PI, 2 * PI}});

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  int const default_degree = 2;

  options.default_degree = default_degree;
  options.default_start_levels = {7, 7};

  // the CFL is more complicated, it depends both on the polynomial degree
  // and on the maximum number of cells (TODO: add more here)
  int const k = options.degree.value_or(default_degree);
  int const n = (1 << options.max_level());
  options.default_dt = 3.0 / (2 * (2 * k + 1) * n);

  options.default_stop_time = 1.0;

  // using explicit RK3
  options.default_step_method = asgard::time_advance::method::rk2;

  // create a pde from the given options and domain
  asgard::PDEv2<P> pde(options, domain);

  // terms are split into positive and negative
  auto positive = [](std::vector<P> const &x, std::vector<P> &y)
      -> void
    {
#pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        y[i] = std::max(P{0}, x[i]);
    };

  // terms are split into positive and negative
  auto negative = [](std::vector<P> const &x, std::vector<P> &y)
      -> void
    {
#pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        y[i] = std::min(P{0}, x[i]);
    };

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
      asgard::term_div<P>(asgard::flux_type::upwind, asgard::boundary_type::periodic, P{1}),
      asgard::term_mass<P>(positive)
    });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
    asgard::term_div<P>(asgard::flux_type::downwind, asgard::boundary_type::periodic, P{1}),
    asgard::term_mass<P>(negative),
    });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
      asgard::mass_electric<P>(positive),
      asgard::term_div<P>(asgard::flux_type::upwind, asgard::boundary_type::dirichlet, P{1})
    });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
      asgard::mass_electric<P>(negative),
      asgard::term_div<P>(asgard::flux_type::downwind, asgard::boundary_type::dirichlet, P{1})
    });

  // initial conditions in x and y
  auto ic_x = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = 1.0 - 0.5 * std::cos(0.5 * x[i]);
    };

  auto ic_v = [](std::vector<P> const &v, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{1} / std::sqrt(PI);

      for (size_t i = 0; i < v.size(); i++)
        fv[i] = c * v[i] * v[i] * std::exp(-v[i] * v[i]);
    };

  pde.add_initial(asgard::separable_func<P>({ic_x, ic_v}));

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::high);


  asgard::advance_time(disc); // integrate until num-steps or stop-time

  disc.progress_report();

  // PUT ENERGY CONVERSATION HERE
  //if (not disc.stop_verbosity())
  //  std::cout << " -- final error: " << get_error_l2(disc) << "\n";

  disc.save_final_snapshot(); // only if output filename is provided

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md main]
#endif
};

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////

// just for convenience to avoid using asgard:: all over the place
// normally, should only include what is needed
using namespace asgard;

void self_test() {
  all_tests testing_("two-stream instability");

#ifdef ASGARD_ENABLE_DOUBLE

#endif

#ifdef ASGARD_ENABLE_FLOAT

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
