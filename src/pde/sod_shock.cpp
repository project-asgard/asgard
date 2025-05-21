#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file sod_shock.cpp
 * \brief Sod shock problem
 * \author The ASGarD Team
 * \ingroup asgard_examples_sodshock
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_sodshock Example: Sod shock tube problem
 *
 * \par Sod shock tube problem
 * Solves the Sod shock problem
 *
 * \f[ \frac{\partial}{\partial t} f(x, v) + v \nabla_x f(x, v, t) =
 *  \mathcal{C}_{LB}[f](x, v, t) \f]
 * where the Lenard Bernstein collision operator is the same as defined in equations
 * (2.1) - (2.6) in <a href="https://arxiv.org/pdf/2402.06493">Schnake, et al.</a>
 * This equation is the Vlasov part only, i.e., no Poisson-electric field feedback.
 *
 * \par
 * The focus of this example is the adaptivity and the ability to save multiple
 * fields in one file to later create slide shows.
 *
 * \par
 * <i>This is still work-in-progress, the documentation needs more work.</i>
 */

/*!
 * \ingroup asgard_examples_sodshock
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_sodshock
 * \brief Make single Sod shock PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param vdims is the number of velocity dimensions, 1-3
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \snippet sod_shock.cpp asgard_examples_sodshock make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_sod(int vdims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_sodshock make]
#endif

  rassert(1 <= vdims and vdims <= 3, "problem is set for 1, 2 or 3 velocity dimensions")

  options.title = "Sod shock 1x" + std::to_string(vdims) + "v";

  // get the collision frequency
  P const nu = options.extra_cli_value_group<P>({"-nu", "-collision_freq"}).value_or(2.0);
  options.subtitle = "collision frequency: " + std::to_string(nu);

  std::vector<asgard::domain_range> ranges;
  ranges.reserve(vdims + 1);
  ranges.emplace_back(-1.0, 1.0);
  for (int v = 0; v < vdims; v++)
    ranges.emplace_back(-6.0, 6.0);

  // the domain has one position and multiple velocity dimensions
  asgard::pde_domain<P> domain(asgard::position_dims{1}, asgard::velocity_dims{vdims}, ranges);

  // setting some default options
  options.default_degree = 2;
  options.default_start_levels = {5,};

  // if no adaptivity is set and adaptivity is not explicitly disabled
  // then enable adaptivity to relative tolerance 0.1%
  if (not options.adapt_ralative and not options.set_no_adapt)
    options.adapt_ralative = 1.E-3;

  // using implicit-explicit stepper
  options.default_step_method = asgard::time_method::imex2;
  options.throw_if_not_imex_stepper();

  // cfl condition for the explicit component
  options.default_dt = 0.01 * domain.min_cell_size(options.max_level());

  options.default_stop_time = 1.0;

  // default solver parameters for the implicit component
  options.default_solver = asgard::solver_method::gmres;
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 400;
  options.default_isolver_inner_iterations = 50;

  options.default_precon = asgard::precon_method::jacobi;

  // create a pde from the given options and domain
  asgard::pde_scheme<P> pde(options, domain);

  // adding the Vlasov terms
  // the vlasov_id will persist until new_term_group() is called again
  int const vlasov_id = pde.new_term_group();

  // see the two-stream instability example for details
  auto positive = [](std::vector<P> const &x, std::vector<P> &y)
      -> void
    {
#pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        y[i] = std::max(P{0}, x[i]);
    };

  auto negative = [](std::vector<P> const &x, std::vector<P> &y)
      -> void
    {
#pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        y[i] = std::min(P{0}, x[i]);
    };

  std::vector<asgard::term_1d<P>> dx_positive = {
      asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::periodic),
      asgard::term_volume<P>(positive)
    };

  std::vector<asgard::term_1d<P>> dx_negative = {
      asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::periodic),
      asgard::term_volume<P>(negative),
    };

  // pad with identity term for dimensions after v1
  for (int v = 1; v < vdims; v++) {
    dx_positive.emplace_back(asgard::term_identity{});
    dx_negative.emplace_back(asgard::term_identity{});
  }

  pde += dx_positive;
  pde += dx_negative;

  // here, the Vlasov-Poisson group will be finalized
  // moving over to the lenard-bernstein group
  int const lb_group_id = pde.new_term_group();

  pde += asgard::operators::lenard_bernstein_collisions{nu};

  // adding penalty
  double const pen = 10.0 / pde.min_cell_size(1);
  asgard::term_1d<P> const term_pen = asgard::term_penalty<P>(
      pen, asgard::flux_type::upwind, asgard::boundary_type::none);

  std::vector<asgard::term_1d<P>> penop(1 + vdims);
  for (int v = 0; v < vdims; v++) {
    penop[1 + v] = term_pen;
    pde += penop;
    penop[1 + v] = asgard::term_identity{};
  }

  // finished with the terms

  // set the implicit and explicit operator groups
  pde.set(asgard::imex_implicit_group{lb_group_id},
          asgard::imex_explicit_group{vlasov_id});

  // separable initial conditions in x and v
  auto ic_x1 = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = (std::abs(x[i]) > 0.5) ? 1.0 : 0.0;;
    };
  auto ic_x2 = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = (std::abs(x[i]) <= 0.5) ? 1.0 : 0.0;;
    };

  auto ic_v1 = [](std::vector<P> const &v, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{1} / std::sqrt(2 * PI);

      for (size_t i = 0; i < v.size(); i++)
        fv[i] = c * std::exp(-0.5 * v[i] * v[i]);
    };
  auto ic_v2 = [](std::vector<P> const &v, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = (1.0 / 8.0) / std::sqrt(1.6 * PI);

      for (size_t i = 0; i < v.size(); i++)
        fv[i] = c * std::exp(-0.625 * v[i] * v[i]);
    };

  // setting the initial conditions based on the number of dimensions
  switch (vdims) {
    case 1:
      pde.add_initial(asgard::separable_func<P>({ic_x1, ic_v1}));
      pde.add_initial(asgard::separable_func<P>({ic_x2, ic_v2}));
      break;
    case 2:
      pde.add_initial(asgard::separable_func<P>({ic_x1, ic_v1, ic_v1}));
      pde.add_initial(asgard::separable_func<P>({ic_x2, ic_v2, ic_v2}));
      break;
    case 3:
      pde.add_initial(asgard::separable_func<P>({ic_x1, ic_v1, ic_v1, ic_v1}));
      pde.add_initial(asgard::separable_func<P>({ic_x2, ic_v2, ic_v2, ic_v2}));
      break;
    default:
      break;
  }

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_sodshock make]
#endif
}

/*!
 * \ingroup asgard_examples_sodshock
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls make_two_stream().
 *
 * \snippet sod_shock.cpp asgard_examples_sodshock main
 *
 * The example saves multiple aux fields in the output file, and those can be
 * presented as a slideshow using the following python script included in the examples
 * folder as \b slideshow.py
 *
 * \snippet slideshow.py slideshow python
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_sodshock main]
#endif

  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves the two stream Vlasov-Poisson in 1x-(1v-3v) dimensions\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << R"help(<< additional options for this file >>
-vdims           -dv     int        accepts: 1, 2 or 3
                                    number of velocity dimensions
-nu                      double     accepts: a positive number
                                    collision frequency

-test                               perform self-testing
)help";
    return 0;
  }

  // check for misspelled cli entries
  options.throw_if_argv_not_in({"-test", "--test"}, {"-nu", "-vdims", "-dv" });

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  // get the number of velocity dimensions, defaults to 1
  int const vdims = options.extra_cli_value_group<P>({"-dv", "-vdims"}).value_or(1);

  // build the discretization, high verbosity shows details about the setup
  asgard::discretization_manager<P> disc(make_sod(vdims, options),
                                         asgard::verbosity_level::high);

  // disable the built in status report during time integration
  disc.set_verbosity(asgard::verbosity_level::low);

  // save the initial condition
  disc.add_aux_field({"initial condition", disc.current_state()});

  // save snapshots for every interval of time equal to 0.1
  // the stride is approximately the number of time-steps that make up 0.1
  int const stride = static_cast<int>(0.1 / disc.dt());

  // look over the entries and save multiple snapshots
  while (disc.remaining_steps() > 0)
  {
    disc.advance_time(stride);
    disc.progress_report();
    disc.add_aux_field({"smapshot time = " + std::to_string(disc.time()),
                        disc.current_state()});
  }

  // save final state
  disc.add_aux_field({"final state", disc.current_state()});

  // re-enable the output to show final stats
  disc.set_verbosity(asgard::verbosity_level::high);

  // write everything to a file
  disc.final_output();

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_sodshock main]
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
  all_tests testing_("Sod shock problem");

#ifdef ASGARD_ENABLE_DOUBLE

  {
    using P = double;
    current_test<P> name_("basic adaptivity");

    prog_opts options;
    options.stop_time = 0.1;
    discretization_manager<P> disc(make_sod(1, options), verbosity_level::quiet);

    // initial grid, 2D, level 5, degree 2
    tassert(disc.current_state().size() == 900u);

    disc.advance_time();

    size_t const dofs = disc.current_state().size();
    tassert(dofs == 2511u); // maybe too exact?
  }

#endif

#ifdef ASGARD_ENABLE_FLOAT

  std::cout << "(float) no tests due to bad conditioning\n";

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
