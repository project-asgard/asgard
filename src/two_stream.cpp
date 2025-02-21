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
 * \addtogroup asgard_examples_two_stream Example 7, Two stream instability
 *
 * \par Example 7
 * Solves the Vlasov-Poisson equation in a common example
 * often called the two stream instability problem
 * \f[ \frac{\partial}{\partial t} f(x, v) + v \nabla_x f(x, v, t) + E(x, t) \cdot \nabla_v f(x, v, t) = 0 \f]
 * where the electric field term depends on the Poisson equation
 * \f[ E(x,t) = -\nabla_x \Phi(x, t), \qquad - \nabla_x \cdot \nabla_x \Phi(x, t) = \int_v f(x, v, t) dv \f]
 * The equation represents the evolution of a charged particle field under the effects
 * of self-induced electric field.
 * The right-hand integral represents the density of the particles and creates
 * non-linear coupling between the fields.
 *
 * \par
 * The focus of this example is the coupling with the electric field and Poisson
 * solver.
 *
 * \par
 * <i>This is still work-in-progress, the documentation needs more work.</i>
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
 * \brief Make single continuity PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \snippet two_stream.cpp two_stream make
 */
template<typename P = asgard::default_precision>
asgard::PDEv2<P> make_two_stream(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [two_stream make]
#endif

  options.title = "Two Stream Instability";

  // the domain has one position and one velocity dimension: 1x1v
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
  options.default_step_method = asgard::time_method::rk2;

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
      asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::periodic),
      asgard::term_mass<P>(positive)
    });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
    asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::periodic),
    asgard::term_mass<P>(negative),
    });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
      asgard::mass_electric<P>(positive),
      asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::dirichlet)
    });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
      asgard::mass_electric<P>(negative),
      asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::dirichlet)
    });

  // initial conditions in x and v
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

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [two_stream make]
#endif
}

/*!
 * \ingroup asgard_examples_two_stream
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls make_two_stream().
 *
 * \snippet two_stream.cpp two_stream main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [two_stream main]
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

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(make_two_stream(options),
                                         asgard::verbosity_level::high);

  disc.advance_time(); // integrate until num-steps or stop-time

  disc.final_output();

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [two_stream main]
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

template<typename P>
void test_energy(std::string const &opt_str) {
  current_test<P> test_(opt_str, 2);
  // analytic solution is not available, hence we use energy conservation for
  // the test quantity in place of an L^2 error

  prog_opts const options = make_opts(opt_str);

  discretization_manager disc(make_two_stream(options), verbosity_level::quiet);

  P E0 = 0; // initial total energy (potential + kinetic), will initialize on first iteration

  // the pde needs only the zeroth moment and computes that internally
  // we are using the other moments to check conservation properties
  int const num_moms = 3;
  int const pdof     = disc.degree() + 1;
  moments1d moms(num_moms, pdof - 1, disc.get_pde2().max_level(),
                 disc.get_pde2().domain());
  std::vector<P> mom_vec;

  int const n = disc.time_params().num_remain();

  for (int i = 0; i < n; i++)
  {
    disc.advance_time(1);

    int const level0   = disc.get_sgrid().current_level(0);
    int const num_cell = fm::ipow2(level0);
    P const dx         = disc.get_pde2().domain().length(0) / num_cell;

    moms.project_moments(disc.get_sgrid(), disc.current_state(), mom_vec);

    disc.do_poisson_update(disc.current_state()); // update the electric field

    auto const &efield = disc.get_terms().cdata.electric_field;

    P Ep = 0;
    for (auto e : efield)
      Ep += e * e;
    Ep *= dx;

    span2d<P> moments(num_moms * pdof, num_cell, mom_vec.data());

    P Ek = 0;
    for (int j : iindexof(num_cell))
      Ek += moments[j][2 * pdof]; // integrating the third moment
    Ek *= std::sqrt(disc.get_pde2().domain().length(0));

    if (disc.time_params().step() == 1) // first time-step
      E0 = Ep + Ek;

    tcheckless(i, std::abs(Ep + Ek - E0), 1.E-6);

    P mv = 0;
    for (auto j : indexof(num_cell))
      for (auto k : indexof(pdof))
        mv += moments[j][k] * moments[j][k + pdof];
    tcheckless(i, std::abs(mv), 1.0e-14);

    // check the initial slight energy decay before it stabilizes
    if (i > 0)
      tassert(std::abs(Ep + Ek - E0) > 1.E-9);
  }
}

void self_test() {
  all_tests testing_("two-stream instability");

#ifdef ASGARD_ENABLE_DOUBLE

  test_energy<double>("-l 5 -d 2 -g dense -dt 6.25e-3 -n 20");
  test_energy<double>("-l 5 -d 2 -n 10 -dt 6.25e-3 -a 1.0e-6");

#endif

#ifdef ASGARD_ENABLE_FLOAT

  std::cout << "no tests for single precision only builds\n";

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
