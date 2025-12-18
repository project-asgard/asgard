#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file bgk.cpp
 * \brief Bhatnagar-Gross-Krook
 * \author The ASGarD Team
 * \ingroup asgard_examples_bgk
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_bgk Example: Bhatnagar-Gross-Krook (BGK)
 *
 * \par Bhatnagar-Gross-Krook
 * Solves the BGK model
 *
 * \f[ \frac{\partial}{\partial t} f(x, v, t) + v \nabla_x f(x, v, t) =
 *  \nu ( M(f) - f) \f]
 * where the collision operator has a source term that depends on the moments
 * of the field
 * \f[ M(f)(x, v) = \frac{n(x)}{\sqrt{2\pi \theta(x)}} \exp \left( - \frac{|v - u(x)|^2}{2 \theta(x)} \right) \f]
 * where
 * \f[ n(x) = \int_v f dv, \qquad u(x) = (u_1, u_2, u_3), \quad u_i(x) = \frac{1}{n(x)} \int_v v_i f dv \f]
 * and
 * \f[ \theta(x) = \frac{1}{3 n(x)} \int_v |v|^2 f dv - \frac{1}{3} \| u(x) \|^2  \f]
 *
 * This file implements 2 examples, a 1x1v (2D) example with simple initial conditions
 * that is just a perturbation of a Maxwellian and a more complex "explosion" problem
 * borrowed from
 * <a href="https://link.springer.com/book/10.1007/b79761">
 * E. F. Toro. "Riemann Solvers and Numerical Methods for Fluid Dynamics" </a>,
 * page 586, section 17.1.
 *
 * In the above equation, M has only implicit dependence on f through the moments,
 * which means that it appears as a moment source term in the equation.
 *
 * \par
 * The focus of this example is to show the usage of asgard::moment_source and
 * the specialized solver asgard::solver_method::scaled_identity that is designed
 * for problems where the operator is a scaled identity and therefore trivial to invert.
 *
 * \par
 * <i>This is still work-in-progress, the documentation needs more work.</i>
 */

/*!
 * \ingroup asgard_examples_bgk
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_bgk
 * \brief Make single BGK PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param dims is the number of spatial velocity dimensions, 1-3
 * \param options is the set of options
 *
 * \returns the asgard::pde_scheme definition
 *
 * \snippet bgk.cpp asgard_examples_bgk make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_bgk(int dims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk make]
#endif

  rassert(1 <= dims and dims <= 3, "problem is set for 1, 2 or 3 position dimensions");

  options.title = "Bhatnagar-Gross-Krook "
                 + std::to_string(dims) + "x" + std::to_string(dims) + "v";

  // get the collision frequency
  P const nu = options.extra_cli_value_group<P>({"-nu", "-collision_freq"}).value_or(1.0);
  options.subtitle = "collision frequency: " + std::to_string(nu);

  std::vector<asgard::domain_range> ranges;
  ranges.reserve(2 * dims);
  for (int x = 0; x < dims; x++)
    ranges.emplace_back(-1.0, 1.0);
  for (int v = 0; v < dims; v++)
    ranges.emplace_back(-6.0, 6.0);

  // the domain has one position and multiple velocity dimensions
  asgard::pde_domain<P> domain(asgard::position_dims{dims}, asgard::velocity_dims{dims}, ranges);

  // setting some default options
  options.default_degree = 2;
  options.default_start_levels = {6, };

  // default: using implicit-explicit stepper
  // since both the implicit and explicit components are linear,
  // this PDE can use fully implicit time-stepping
  options.default_step_method = asgard::time_method::imex2;

  // cfl condition for the explicit component
  options.default_dt = 0.01 * domain.min_cell_size(options.max_level());

  options.default_stop_time = 1.0;

  // if the user has selected a time-stepping method that is not imex
  if (options.step_method and not asgard::is_imex(options.step_method.value())) {
    // then we default to the GMRES solver
    options.default_solver = asgard::solver_method::gmres;

    options.default_isolver_tolerance  = 1.E-8;
    options.default_isolver_iterations = 400;
    options.default_isolver_inner_iterations = 50;
  } else {
    // if IMEX is selected or default, using specialized solver
    options.default_solver = asgard::solver_method::scaled_identity;
  }

  // create a pde from the given options and domain
  asgard::pde_scheme<P> pde(options, domain);

  // adding the advection terms in the explicit group
  // the vp_group_id will persist until new_term_group() is called again
  int const explicit_id = pde.new_term_group();

  {
    // creating intermediate variables and adding terms to the PDE
    // using the curly braces {} creates a local scope and intermediates
    // will be cleaned at the close of the braces limiting the scope
    // and lowering the chance of accidental reuse or error

    // see the two-stream instability example for details
    asgard::term_1d<P> positive = asgard::term_volume<P>(
        [](std::vector<P> const &x, std::vector<P> &y)
          -> void {
          #pragma omp parallel for
          for (size_t i = 0; i < x.size(); i++)
            y[i] = std::max(P{0}, x[i]);
        });

    asgard::term_1d<P> negative = asgard::term_volume<P>(
        [](std::vector<P> const &x, std::vector<P> &y)
          -> void {
          #pragma omp parallel for
          for (size_t i = 0; i < x.size(); i++)
            y[i] = std::min(P{0}, x[i]);
        });

    asgard::term_1d<P> div_up = asgard::term_div<P>(1, asgard::flux_type::upwind,
                                                    asgard::boundary_type::periodic);

    asgard::term_1d<P> div_do = asgard::term_div<P>(1, asgard::flux_type::downwind,
                                                    asgard::boundary_type::periodic);

    asgard::term_1d<P> I = asgard::term_identity{};

    switch (dims) {
      case 1:
        pde += asgard::term_md{div_up, positive};
        pde += asgard::term_md{div_do, negative};
        break;
      case 2:
        pde += asgard::term_md{div_up, I, positive, I};
        pde += asgard::term_md{div_do, I, negative, I};
        pde += asgard::term_md{I, div_up, I, positive};
        pde += asgard::term_md{I, div_do, I, negative};
        break;
      case 3:
        pde += asgard::term_md{div_up, I, I, positive, I, I};
        pde += asgard::term_md{div_do, I, I, negative, I, I};
        pde += asgard::term_md{I, div_up, I, I, positive, I};
        pde += asgard::term_md{I, div_do, I, I, negative, I};
        pde += asgard::term_md{I, I, div_up, I, I, positive};
        pde += asgard::term_md{I, I, div_do, I, I, negative};
        break;
    }
  }

  // the right-hand-side of the equation is set for the implicit group
  int const implicit_id = pde.new_term_group();

  { // setting the nu * f term
    std::vector<asgard::term_1d<P>> nuI(2 * dims, asgard::term_identity{});
    nuI[0] = asgard::term_volume<P>{nu};
    pde += asgard::term_md<P>(nuI);
  }

  if (dims == 1) {
    asgard::moment_id im0 = pde.register_moment(asgard::moment(0));
    asgard::moment_id im1 = pde.register_moment(asgard::moment(1));
    asgard::moment_id im2 = pde.register_moment(asgard::moment(2));

    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m1 = moments[im1];
      std::vector<P> const &m2 = moments[im2];

      int64_t const num_nodes = nodes.num_strips();
      #pragma omp parallel for
      for (int64_t i = 0; i < num_nodes; i++) {
        // P const x = nodes[i][0]; // no explicit spatial dependence
        P const v = nodes[i][1];

        P const n = m0[i];
        P const u = m1[i] / m0[i];
        P const t = m2[i] / m0[i] - u * u;

        vals[i] = nu * n / std::sqrt(2 * PI * t);
        P const d = v - u;
        vals[i] *= std::exp(- P{0.5} * d * d / t);
      }
    };

    pde.set_source(asgard::moment_source<P>(fbgk, {im0, im1, im2}));

    auto abgk = [=](P time, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> const &,
                    std::vector<P> &vals)
    {
      fbgk(time, nodes, moments, vals);
    };

    pde.set_adapt_weight(abgk, {im0, im1, im2});

  } else if (dims == 2){

    asgard::moment_id im0 = pde.register_moment(asgard::moment(0, 0));
    asgard::moment_id im10 = pde.register_moment(asgard::moment(1, 0));
    asgard::moment_id im01 = pde.register_moment(asgard::moment(0, 1));
    asgard::moment_id im20 = pde.register_moment(asgard::moment(2, 0));
    asgard::moment_id im02 = pde.register_moment(asgard::moment(0, 2));

    std::vector<asgard::moment_id> const mids = {im0, im10, im01, im20, im02};

    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m10 = moments[im10];
      std::vector<P> const &m01 = moments[im01];
      std::vector<P> const &m20 = moments[im20];
      std::vector<P> const &m02 = moments[im02];

      int64_t const num_nodes = nodes.num_strips();
      #pragma omp parallel for
      for (int64_t i = 0; i < num_nodes; i++) {
        P const n = m0[i];
        P const u0 = m10[i] / m0[i];
        P const u1 = m01[i] / m0[i];
        P const t = 0.5 * ((m20[i] + m02[i]) / m0[i] - u0 * u0 - u1 * u1);

        vals[i] = nu * n / std::sqrt(2 * PI * t);
        P const vu0 = nodes[i][2] - u0;
        P const vu1 = nodes[i][3] - u1;
        P const d = vu0 * vu0 + vu1 * vu1;
        vals[i] *= std::exp(- P{0.5} * d / t);
      }
    };

    pde.set_source(asgard::moment_source<P>(fbgk, mids));

    auto abgk = [=](P time, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> const &,
                    std::vector<P> &vals)
    {
      fbgk(time, nodes, moments, vals);
    };

    pde.set_adapt_weight(abgk, mids);

  } else /* if (dims == 3) */ {
    //
  }

  // set the implicit and explicit operator groups
  pde.set(asgard::imex_implicit_group{implicit_id},
          asgard::imex_explicit_group{explicit_id});

  // separable initial conditions in x and v
  auto ic_x = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = 1.0 + 1.E-4 * std::cos(PI * x[i]);
    };

  auto ic_v = [](std::vector<P> const &v, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{1} / std::sqrt(2 * PI);

      for (size_t i = 0; i < v.size(); i++)
        fv[i] = c * std::exp(-0.5 * v[i] * v[i]);
    };

  pde.add_initial(asgard::separable_func<P>({ic_x, ic_v}));

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk make]
#endif
}

/*!
 * \ingroup asgard_examples_bgk
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
 * \snippet bgk.cpp asgard_examples_bgk compute_perturbation
 */
template<typename P = asgard::default_precision>
std::vector<P> compute_perturbation(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk compute_perturbation]
#endif

  // The Maxwellian is the initial condition
  // but with constant value of 1.0 set in dimension 0 (the position dimension)
  asgard::separable_func<P> maxw = disc.initial_cond_sep().front();

  // set dimension 0 to be a constant function with value 1
  maxw.set(asgard::dimension_id{0}, P{1});

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
//! [asgard_examples_bgk compute_perturbation]
#endif
}

/*!
 * \ingroup asgard_examples_bgk
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls make_two_stream().
 *
 * \snippet bgk.cpp asgard_examples_bgk main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk main]
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
  asgard::discretization_manager<P> disc(make_bgk<P>(vdims, options),
                                         asgard::verbosity_level::high);

  // save the perturbation as an auxiliary field, for plotting
  disc.add_aux_field({"initial perturbation", compute_perturbation(disc)});

  disc.advance_time(); // integrate until num-steps or stop-time

  // save the final perturbation
  disc.add_aux_field({"final perturbation", compute_perturbation(disc)});

  disc.final_output();

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk main]
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
void test_energy(int const dims, std::string const &opt_str) {
  current_test<P> test_(opt_str, 2 * dims);
  // analytic solution is not available, hence we use energy conservation for
  // the test quantity in place of an L^2 error

  prog_opts const options = make_opts(opt_str);

  auto pde = make_bgk<P>(dims, options);
  moment_id const m0 = pde.register_moment({0, moment::inactive});
  moment_id const m1 = pde.register_moment({1, moment::inactive});
  moment_id const m2 = pde.register_moment({2, moment::inactive});
  discretization_manager disc(std::move(pde), verbosity_level::quiet);

  double mass0   = 0; // initial total mass
  double energy0 = 0; // initial total energy

  int64_t const n = disc.remaining_steps();

  P constexpr tol = (std::is_same_v<P, double>) ? 5.E-7 : 5.E-3;

  moment_manager<P> const &moms = disc.get_moment_manager();

  std::cout.precision(8);
  std::cout << std::scientific;

  for (int64_t i = 0; i < n; i++)
  {
    disc.advance_time(1);

    disc.compute_moments();

    double const mass = moms.get_cached_raws()[m0][0];
    if (i == 0)
      mass0 = mass;

    //std::cout << " delta-mass = " << std::abs(mass - mass0) << '\n';
    //tassert(std::abs(mass - mass0) < 1.E-11);

    double const energy = moms.get_cached_raws()[m2][0];
    if (i == 0)
      energy0 = energy;

    //std::cout << " delta-energy = " << std::abs(energy - energy0) << '\n';
    //tassert(std::abs(energy - energy0) < 1.E-11);

    std::cout << " delta-mass: " << std::abs(mass - mass0) << "    " << std::abs(energy - energy0) << '\n';


  //   int const level0   = disc.get_grid().current_level(0);
  //   int const num_cell = fm::ipow2(level0);
  //   P const dx         = disc.domain().length(0) / num_cell;
  //
  //   auto efield = disc.get_electric();
  //
  //   double Ep = 0;
  //   for (auto e : efield)
  //     Ep += e * e;
  //   Ep *= dx;
  //
  //   std::vector<P> mom2 = disc.get_moment(m2);
  //
  //   P const Ek = mom2[0] * std::sqrt(disc.domain().length(0));
  //
  //   if (disc.current_step() == 1) // first time-step
  //     E0 = Ep + Ek;
  //
  //   // check the initial slight energy decay before it stabilizes
  //   tcheckless(i, std::abs(Ep + Ek - E0), tol);
  }
}

void self_test() {
  all_tests testing_("Bhatnagar-Gross-Krook");

#ifdef ASGARD_ENABLE_DOUBLE

  // test_energy<double>(1, "-l 6 -n 100 -s imex1");
  // test_energy<double>(1, "-l 6 -n 100 -s imex2");

  // test_energy<double>(1, "-l 5 -t 0.5 -s imex2");
  // test_energy<double>(1, "-l 6 -t 0.25 -s imex2");

#endif

#ifdef ASGARD_ENABLE_FLOAT

  // test_energy<float>(1, "-l 5");

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
