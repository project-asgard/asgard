#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file relaxation_internal.cpp
 * \brief Simple collision problem
 * \author The ASGarD Team
 *
 * Provides a stress-test with a contrived PDE that contains only
 * Lenard-Bernstein collision terms.
 *
 * \endinternal
 */

using namespace asgard;

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

template<typename P = asgard::default_precision>
asgard::PDEv2<P> make_relaxation(int vdims, asgard::prog_opts options) {
  rassert(1 <= vdims and vdims <= 3, "problem is set for 1, 2 or 3 velocity dimensions")

  options.title = "Relaxation 1x" + std::to_string(vdims) + "v";

  std::vector<domain_range<P>> ranges;
  ranges.reserve(vdims + 1);
  ranges.emplace_back(-0.5, +0.5);
  for (int v : iindexof(vdims))
    ranges.emplace_back(-8.0, 12.0);

  // the domain has one position and one velocity dimension: 1x1v
  pde_domain<P> domain(position_dims{1}, velocity_dims{vdims}, ranges);

  options.default_degree = 2;
  options.default_start_levels = {7, };

  options.default_dt = 0.05;

  options.default_stop_time = 1.0;

  options.default_solver = solver_method::gmres;
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;
  options.default_isolver_inner_iterations = 50;

  options.default_precon = precon_method::jacobi;

  // using implicit backward Euler
  options.default_step_method = asgard::time_method::back_euler;

  // get the collision frequency
  P const nu = options.extra_cli_value_group<P>({"-nu", "-collision_freq"}).value_or(1000);
  options.subtitle = "collision frequency: " + std::to_string(nu);

  // create a pde from the given options and domain
  PDEv2<P> pde(options, domain);

  pde += operators::lenard_bernstein_collisions{nu};

  if (vdims == 1) {
    separable_func<P> ic({0.5, 0.5}); // separable initial conditions

    ic.set_fdomain(1, [](std::vector<P> const &v, P, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr ux    = -1.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - ux) * (v[i] - ux));
      });
    pde.add_initial(ic);

    ic.set_fdomain(1, [](std::vector<P> const &v, P, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr ux    = 2.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - ux) * (v[i] - ux));
      });
    pde.add_initial(ic);
  }

  // initial conditions in x and v
  // auto ic_x = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
  //   void {
  //     for (size_t i = 0; i < x.size(); i++)
  //       fx[i] = 1.0 - 0.5 * std::cos(0.5 * x[i]);
  //   };
  //
  // auto ic_v = [](std::vector<P> const &v, P /* time */, std::vector<P> &fv) ->
  //   void {
  //     P const c = P{1} / std::sqrt(PI);
  //
  //     for (size_t i = 0; i < v.size(); i++)
  //       fv[i] = c * v[i] * v[i] * std::exp(-v[i] * v[i]);
  //   };
  //
  // pde.add_initial(asgard::separable_func<P>({ic_x, ic_v}));

  return pde;
}

int main(int argc, char** argv)
{
  using P = default_precision;

  prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves the two stream Vlasov-Poisson in 1x-1v dimensions\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-vdims                              velocity dimensions (1 - 3)\n";
    std::cout << "-nu                                 collision frequency\n";
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

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  discretization_manager<P> disc(make_relaxation(1, options),
                                 asgard::verbosity_level::high);

  disc.advance_time(); // integrate until num-steps or stop-time

  disc.final_output();

  return 0;
};

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////

// just for convenience to avoid using asgard:: all over the place
// normally, should only include what is needed
using namespace asgard;

template<typename P>
void test_energy(std::string const &opt_str) {
  current_test<P> test_(opt_str, 2);

}

void self_test() {
  // all_tests testing_("two-stream instability");

  std::cout << " no tests, yet!\n";

#ifdef ASGARD_ENABLE_DOUBLE

  // test_energy<double>("-l 5 -d 2 -g dense -dt 6.25e-3 -n 20");
  // test_energy<double>("-l 5 -d 2 -n 10 -dt 6.25e-3 -a 1.0e-6");

#endif

#ifdef ASGARD_ENABLE_FLOAT

  std::cout << "no tests for single precision only builds\n";

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
