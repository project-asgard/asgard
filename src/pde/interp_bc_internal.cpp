#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

using namespace asgard;

template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_3d_pde(asgard::prog_opts options)
{
  options.title = "Non-separable elliptic PDE";

  asgard::pde_domain<P> domain({{0, 1}, {0, 1}, {0, 1}});

  options.default_degree = 2;
  options.default_start_levels = {4, };

  options.force_step_method(asgard::time_method::steady);

  options.default_solver = asgard::solver_method::bicgstab;

  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;

  asgard::pde_scheme<P> pde(options, std::move(domain));

  asgard::term_1d<P> I = asgard::term_identity{};

  asgard::term_md<P> divx  = { asgard::term_div<P>{-1, asgard::boundary_type::right}, I, I};

  auto fx1 = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &f) ->
    void {
      // value of the solution at x = 0, the nodes are at the wall corresponding to x = 0
      assert(nodes.stride() == 2);
      for (int64_t i = 0; i < nodes.num_strips(); i++)
        f[i] = 1;
    };

  divx += asgard::right_boundary_flux<P>(fx1);

  pde += divx;

  P const dx = pde.cell_size(asgard::dimension_id{0});

  pde += { asgard::term_penalty<P>{P{1} / dx}, I, I};

  auto source = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &s) ->
    void {
      for (int64_t i = 0; i < nodes.num_strips(); i++)
        s[i] = 1;
    };

  pde += asgard::source<P>(source);

  return pde;
}

template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc)
{
  int const num_dims = disc.num_dims();

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

  auto pde = make_3d_pde<P>(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  P const err = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- steady state error: " << err << '\n';

  return 0;
}

#ifndef __ASGARD_DOXYGEN_SKIP
template<typename P>
void dotest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  auto pde = make_3d_pde<P>(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);
  // std::cout << err << '\n';
  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("elliptic steady state problem", " div.grad f = sources");

  #ifdef ASGARD_ENABLE_DOUBLE
  // dotest<double>(5.E-3, 1, "-d 1 -l 3");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  // dotest<float>(5.E-3, 1, "-d 1 -l 5");
  #endif
}

#endif
