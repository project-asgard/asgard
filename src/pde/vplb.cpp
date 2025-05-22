#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file vplb.cpp
 * \brief Vlasov-Poisson-Lenard-Bernstein
 * \author The ASGarD Team
 * \ingroup asgard_examples_vplb
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_vplb Example: Vlasov-Poisson-Lenard-Bernstein
 *
 * \par Vlasov-Poisson-Lenard-Bernstein
 * Solves the Vlasov-Poisson equation with Lenard-Bernstein collisions
 *
 * \f[ \frac{\partial}{\partial t} f(x, v) + v \nabla_x f(x, v, t) + E(x, t) \cdot \nabla_v f(x, v, t) =
 *  \mathcal{C}_{LB}[f](x, v, t) \f]
 * where the electric field term depends on the Poisson equation
 * \f[ E(x,t) = -\nabla_x \Phi(x, t), \qquad - \nabla_x \cdot \nabla_x \Phi(x, t) = \int_v f(x, v, t) dv \f]
 * and the Lenard Bernstein collision operator is the same as defined in equations
 * (2.1) - (2.6) in <a href="https://arxiv.org/pdf/2402.06493">Schnake, et al.</a>
 *
 * \par
 * The focus of this example is the term groups needed for the IMEX time-stepping,
 * the builtin LB collision operator and the functionality to store and plot additional
 * (auxiliary) fields for the problem.
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
 * \ingroup asgard_examples_vplb
 * \brief Make single VPLB PDE
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
 * \snippet vplb.cpp asgard_examples_vplb make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_vplb(int vdims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_vplb make]
#endif

  rassert(1 <= vdims and vdims <= 3, "problem is set for 1, 2 or 3 velocity dimensions")

  options.title = "Vlasov-Poisson-Lenard-Bernstein 1x" + std::to_string(vdims) + "v";

  // get the collision frequency
  P const nu = options.extra_cli_value_group<P>({"-nu", "-collision_freq"}).value_or(1.0);
  options.subtitle = "collision frequency: " + std::to_string(nu);

  std::vector<asgard::domain_range> ranges;
  ranges.reserve(vdims + 1);
  ranges.emplace_back(-2 * PI, 2 * PI);
  for (int v = 0; v < vdims; v++)
    ranges.emplace_back(-6.0, 6.0);

  // the domain has one position and multiple velocity dimensions
  asgard::pde_domain<P> domain(asgard::position_dims{1}, asgard::velocity_dims{vdims}, ranges);

  // setting some default options
  options.default_degree = 2;
  options.default_start_levels = {6, 7};
  for (int v = 1; v < vdims; v++)
    options.default_start_levels.emplace_back(7);

  // using implicit-explicit stepper
  options.default_step_method = asgard::time_method::imex2;
  options.throw_if_not_imex_stepper();

  // cfl condition for the explicit component
  options.default_dt = 0.01 * domain.min_cell_size(options.max_level());

  options.default_stop_time = 1.0;

  // default solver parameters for the implicit component
  options.default_solver = asgard::solver_method::bicgstab;
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 400;
  options.default_isolver_inner_iterations = 50;

  options.default_precon = asgard::precon_method::jacobi;

  // create a pde from the given options and domain
  asgard::pde_scheme<P> pde(options, domain);

  // adding the terms for the pde
  // the terms are split into two groups
  // explicit Vlasov-Poisson, implicit Lenard-Bernstein
  // each group corresponds to a set of terms that has been added constitutively
  // 1. initialize a new term group, get the group-id
  // 2. add the term from the group
  // 3. move to the next group, or stop adding terms

  // adding the Vlasov-Poisson terms
  // the vp_group_id will persist until new_term_group() is called again
  int const vp_group_id = pde.new_term_group();

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

  std::vector<asgard::term_1d<P>> dv_Epositive = {
      asgard::volume_electric<P>(positive),
      asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides)
    };

  std::vector<asgard::term_1d<P>> dv_Enegative = {
      asgard::volume_electric<P>(negative),
      asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides)
    };

  // pad with identity term for dimensions after v1
  for (int v = 1; v < vdims; v++) {
    dx_positive.emplace_back(asgard::term_identity{});
    dx_negative.emplace_back(asgard::term_identity{});
    dv_Epositive.emplace_back(asgard::term_identity{});
    dv_Enegative.emplace_back(asgard::term_identity{});
  }

  pde += dx_positive;
  pde += dx_negative;
  pde += dv_Epositive;
  pde += dv_Enegative;

  // here, the Vlasov-Poisson group will be finalized
  // moving over to the lenard-bernstein group
  int const lb_group_id = pde.new_term_group();

  pde += asgard::operators::lenard_bernstein_collisions{nu};

  // adding penalty
  double const pen = 10.0 / pde.min_cell_size(1);
  std::vector<asgard::term_1d<P>> penop = {
      asgard::term_identity{},
      asgard::term_penalty<P>(pen, asgard::flux_type::upwind, asgard::boundary_type::none)
    };

  for (int v = 1; v < vdims; v++)
    penop.emplace_back(asgard::term_identity{});

  pde += penop;

  // finished with the terms

  // using IMEX stepper requires that the pde "knows" the corresponding groups
  // the technique used here is called "strong-types" for C++,
  // e.g., see here: https://www.fluentcpp.com/2016/12/08/strong-types-for-strong-interfaces/

  // set the implicit and explicit operator groups
  pde.set(asgard::imex_implicit_group{lb_group_id},
          asgard::imex_explicit_group{vp_group_id});

  // separable initial conditions in x and v
  auto ic_x = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = 1.0 + 1.E-4 * std::cos(0.5 * x[i]);
    };

  auto ic_v = [](std::vector<P> const &v, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{1} / std::sqrt(2 * PI);

      for (size_t i = 0; i < v.size(); i++)
        fv[i] = c * std::exp(-0.5 * v[i] * v[i]);
    };

  // setting the initial conditions based on the number of dimensions
  switch (vdims) {
    case 1:
      pde.add_initial(asgard::separable_func<P>({ic_x, ic_v}));
      break;
    case 2:
      pde.add_initial(asgard::separable_func<P>({ic_x, ic_v, ic_v}));
      break;
    case 3:
      pde.add_initial(asgard::separable_func<P>({ic_x, ic_v, ic_v, ic_v}));
      break;
    default:
      break;
  }

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_vplb make]
#endif
}

/*!
 * \ingroup asgard_examples_vplb
 * \brief Computes the perturbation between the Maxwellian and the current state
 *
 * The initial condition is a small perturbation of a Maxwellian, which is not
 * visible on a regular plot of the state.
 * This method computes the difference between the current state of the
 * discretization and the final Maxwellian distribution.
 *
 * \tparam P is the precision to use, float or double
 *
 * \param disc is a discretization of a PDE created with make_vplb()
 *
 * \returns the difference between the current state and the Maxwellian
 *          projected on the current sparse grid
 *
 * \snippet vplb.cpp asgard_examples_vplb compute_perturbation
 */
template<typename P = asgard::default_precision>
std::vector<P> compute_perturbation(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_vplb compute_perturbation]
#endif

  // The Maxwellian is the initial condition
  // but with constant value of 1.0 set in dimension 0 (the position dimension)
  asgard::separable_func<P> maxw = disc.initial_cond_sep().front();

  // set dimension 0 to be a constant function with value 1
  maxw.set(0, P{1});

  // project the Maxwellian onto the current grid
  std::vector<P> proj_max = disc.project_function(maxw);

  // subtract the current state
  std::vector<P> const &state = disc.current_state();

  // the projected size will always match the size of the current state
  if (proj_max.size() != state.size())
    throw std::runtime_error("this will never happen");

  size_t n = state.size();
  for (size_t i = 0; i < n; i++)
    proj_max[i] -= state[i];

  return proj_max;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_vplb compute_perturbation]
#endif
}

/*!
 * \ingroup asgard_examples_vplb
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls make_two_stream().
 *
 * \snippet vplb.cpp asgard_examples_vplb main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_vplb main]
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

  // this is an optional step, check if there are misspelled or incorrect cli entries
  // the first set/vector of entries are those that can appear by themselves
  // the second set/vector requires extra parameters
  options.throw_if_argv_not_in({"-test", "--test"}, {"-nu", "-vdims", "-dv" });

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  // get the number of velocity dimensions, defaults to 1
  int const vdims = options.extra_cli_value_group<P>({"-dv", "-vdims"}).value_or(1);

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(make_vplb<P>(vdims, options),
                                         asgard::verbosity_level::high);

  // save the perturbation as an auxiliary field, for plotting
  disc.add_aux_field({"initial perturbation", compute_perturbation(disc)});

  disc.advance_time(); // integrate until num-steps or stop-time

  // save the final perturbation
  disc.add_aux_field({"final perturbation", compute_perturbation(disc)});

  disc.final_output();

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_vplb main]
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
void test_energy(int const vdims, std::string const &opt_str) {
  current_test<P> test_(opt_str, 1 + vdims);
  // analytic solution is not available, hence we use energy conservation for
  // the test quantity in place of an L^2 error

  prog_opts const options = make_opts(opt_str);

  discretization_manager disc(make_vplb<P>(vdims, options), verbosity_level::quiet);

  double E0 = 0; // initial total energy (potential + kinetic), will initialize on first iteration

  // the pde needs only the zeroth moment and computes that internally
  // we are using the other moments to check conservation properties
  int const num_moms = 3;
  int const pdof     = disc.degree() + 1;
  moments1d moms(num_moms, pdof - 1, disc.options().max_level(), disc.domain());
  std::vector<P> mom_vec;

  int64_t const n = disc.remaining_steps();

  P constexpr tol = (std::is_same_v<P, double>) ? 5.E-7 : 5.E-3;

  for (int64_t i = 0; i < n; i++)
  {
    disc.advance_time(1);

    int const level0   = disc.get_grid().current_level(0);
    int const num_cell = fm::ipow2(level0);
    P const dx         = disc.domain().length(0) / num_cell;

    moms.project_moments(disc.get_grid(), disc.current_state(), mom_vec);

    disc.do_poisson_update(disc.current_state()); // update the electric field

    auto const &efield = disc.get_terms().cdata.electric_field;

    double Ep = 0;
    for (auto e : efield)
      Ep += e * e;
    Ep *= dx;

    span2d<P> moments(num_moms * pdof, num_cell, mom_vec.data());

    double Ek = 0;
    for (int j : iindexof(num_cell))
      Ek += moments[j][2 * pdof]; // integrating the third moment
    Ek *= std::sqrt(disc.domain().length(0));

    if (disc.current_step() == 1) // first time-step
      E0 = Ep + Ek;

    // check the initial slight energy decay before it stabilizes
    tcheckless(i, std::abs(Ep + Ek - E0), tol);
  }
}

void self_test() {
  all_tests testing_("Vlasov-Poisson-Lenard-Bernstein");

#ifdef ASGARD_ENABLE_DOUBLE

  test_energy<double>(1, "-l 5 -t 0.5 -s imex1");

  test_energy<double>(1, "-l 5 -t 0.5 -s imex2");
  test_energy<double>(1, "-l 6 -t 0.25 -s imex2");

#endif

#ifdef ASGARD_ENABLE_FLOAT

  test_energy<float>(1, "-l 5");

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
