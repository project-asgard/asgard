#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file misc_internal.cpp
 * \brief Internal stress-test
 * \author The ASGarD Team
 *
 * Provides coverage for edge cases, solution that consists of multiple separable
 * terms, chains of volume and div/grad terms (with boundary conditions), etc.
 *
 * \endinternal
 */

using namespace asgard;

template<typename P>
double run_chain_test(prog_opts options) {
  // the test here uses an exact solution that is the sum of two separable functions
  // and the terms are build from chains of volume and derivative terms

  options.title = "Deep Chains - 2D diffusion";

  pde_domain<P> domain({{0.25, 2}, {1, 4}});

  options.default_degree = 1;
  options.default_start_levels = {4, };

  options.default_step_method = time_method::steady;
  options.default_solver = solver_method::direct;

  options.default_dt = 0.01;
  options.default_stop_time = 1.0;

  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 2000;
  options.default_isolver_inner_iterations = 200;

  pde_scheme<P> pde(options, domain);

  P const cellx = pde.min_cell_size(0);
  P const celly = pde.min_cell_size(1);

  // exact solution is the sum of two separable functions
  separable_func<P> exact1({
      vectorize_t<P>([](P x) -> P { return std::exp(x); }),
      vectorize_t<P>([](P y) -> P { return std::cos(y); }),
  }, ignores_time);
  separable_func<P> exact2({
      vectorize_t<P>([](P x) -> P { return std::exp(-x); }),
      vectorize_t<P>([](P y) -> P { return std::sin(y); }),
  }, ignores_time);

  {
    // first operator, mix of volume and derivative terms
    auto c1 = vectorize<P>([](P x) -> P { return std::sin(x); });
    term_1d<P> dxx1 = std::vector<term_1d<P>>{
                        term_div<P>{-1, boundary_type::left},
                        term_volume<P>{c1},
                        term_grad<P>{1, boundary_type::right}};

    dxx1.set_penalty(P{1} / cellx);
    term_md<P> dxx({dxx1, term_identity{}});

    separable_func<P> bc1 = exact1;
    separable_func<P> bc2 = exact2;

    bc1.set(0, std::exp(P{2}));
    bc2.set(0, std::exp(-P{2}));

    dxx += right_boundary_flux<P>{bc1};
    dxx += right_boundary_flux<P>{bc2};

    // the Neumann boundary condition is applied to c1 * f (f is the solution)
    // this would not be necessary if c1 was included in the div term
    bc1.set(0, std::exp(0.25) * std::sin(P{0.25}));
    bc2.set(0, -std::exp(-0.25) * std::sin(P{0.25}));
    boundary_flux<P> f_bc1 = left_boundary_flux<P>{bc1};
    boundary_flux<P> f_bc2 = left_boundary_flux<P>{bc2};

    f_bc1.chain_level(0) = 0;
    f_bc2.chain_level(0) = 0;

    dxx += f_bc1;
    dxx += f_bc2;

    pde += dxx;
  }

  {
    auto c2 = vectorize<P>([](P y) -> P { return y; });
    term_1d<P> dyy1 = std::vector<term_1d<P>>{
                        term_volume<P>{2},
                        term_div<P>{-1, boundary_type::left},
                        term_volume<P>{c2},
                        term_grad<P>{1, boundary_type::right}};

    dyy1.set_penalty(P{1} / celly);
    term_md<P> dyy({term_identity{}, dyy1});

    separable_func<P> bc1 = exact1;
    separable_func<P> bc2 = exact2;

    bc1.set(1, std::cos(P{4}));
    bc2.set(1, std::sin(P{4}));

    dyy += right_boundary_flux<P>{bc1};
    dyy += right_boundary_flux<P>{bc2};

    bc1.set(1, -std::sin(P{1}));
    bc2.set(1, std::cos(P{1}));
    boundary_flux<P> f_bc1 = left_boundary_flux<P>{bc1};
    boundary_flux<P> f_bc2 = left_boundary_flux<P>{bc2};

    f_bc1.chain_level(1) = 1;
    f_bc2.chain_level(1) = 1;

    dyy += f_bc1;
    dyy += f_bc2;

    pde += dyy;
  }

  // derivatives in x
  pde.add_source(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return -std::cos(x) * std::exp(x); }),
      vectorize_t<P>([](P y) -> P { return std::cos(y); }),
  }, ignores_time));

  pde.add_source(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return std::cos(x) * std::exp(-x); }),
      vectorize_t<P>([](P y) -> P { return std::sin(y); }),
  }, ignores_time));

  pde.add_source(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return -std::sin(x) * std::exp(x); }),
      vectorize_t<P>([](P y) -> P { return std::cos(y); }),
  }, ignores_time));

  pde.add_source(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return -std::sin(x) * std::exp(-x); }),
      vectorize_t<P>([](P y) -> P { return std::sin(y); }),
  }, ignores_time));

  // derivatives in y
  pde.add_source(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return std::exp(x); }),
      vectorize_t<P>([](P y) -> P { return 2 * std::sin(y); }),
  }, ignores_time));

  pde.add_source(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return std::exp(-x); }),
      vectorize_t<P>([](P y) -> P { return -2 * std::cos(y); }),
  }, ignores_time));

  pde.add_source(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return std::exp(x); }),
      vectorize_t<P>([](P y) -> P { return 2 * y * std::cos(y); }),
  }, ignores_time));

  pde.add_source(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return std::exp(-x); }),
      vectorize_t<P>([](P y) -> P { return 2 * y * std::sin(y); }),
  }, ignores_time));

  discretization_manager<P> disc(std::move(pde), verbosity_level::low);

  disc.advance_time();

  disc.save_final_snapshot();

  std::vector<P> const eref   = disc.project_function({exact1, exact2});
  std::vector<P> const &state = disc.current_state();

  // L^2 norm squared of the exact solution for the given domain
  double const enorm = 40.44042709727439;

  double nself = 0, ndiff = 0;
  for (size_t i = 0; i < state.size(); i++)
  {
    double const e = eref[i] - state[i];
    ndiff += e * e;
    double const r = eref[i];
    nself += r * r;
  }

  return std::sqrt(ndiff + std::abs(enorm - nself)) / std::sqrt(enorm);
}

template<typename P>
double run_volume_test(prog_opts options) {
  // the test here uses a chain of only volume terms
  // the problem has no spatial derivatives

  options.title = "Volume Chains - 1D ode";

  pde_domain<P> domain({{0.5, 1}, });

  options.default_degree = 1;
  options.default_start_levels = {4, };

  options.default_step_method = time_method::rk2;

  options.default_dt = 0.005;
  options.default_stop_time = 1.0;

  pde_scheme<P> pde(options, domain);

  pde.add_initial(separable_func<P>({
      vectorize_t<P>([](P x) -> P { return std::cos(x); }),
  }, ignores_time));

  {
    // first operator, mix of volume and derivative terms
    auto c1 = vectorize<P>([](P x) -> P { return std::sin(x); });
    auto c2 = vectorize<P>([](P x) -> P { return (1 + x); });
    term_1d<P> dv = std::vector<term_1d<P>>{
                        term_volume<P>{3},
                        term_volume<P>{c1},
                        term_volume<P>{c2}};

    pde += {dv, };
  }

  discretization_manager<P> disc(std::move(pde), verbosity_level::low);

  disc.advance_time();

  disc.save_final_snapshot();

  P const t = disc.time();

  separable_func<P> exact({
      vectorize_t<P>(
          [=](P x) -> P { return std::exp(- P{3} * (1 + x) * std::sin(x) * t) * std::cos(x); }),
  }, ignores_time);

  std::vector<P> const eref   = disc.project_function(exact);
  std::vector<P> const &state = disc.current_state();

  double nself = 0, ndiff = 0;
  for (size_t i = 0; i < state.size(); i++)
  {
    double const e = eref[i] - state[i];
    ndiff += e * e;
    double const r = eref[i];
    nself += r * r;
  }

  return std::sqrt(ndiff) / std::sqrt(nself);
}

void self_test();

int main(int argc, char** argv)
{
  libasgard_runtime running_(argc, argv);

  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves couple of messy testing pde:\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout <<
R"help(<< additional options for this file >>
-chains                             test messy chains example in 2D
-volumes                            test chain of volume terms in 1D
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", "-chains", "-volumes"}, {});

  if (options.has_cli_entry("-test")) {
    self_test();
    return 0;
  }

  if (options.has_cli_entry("-chains")) {
    double const err = run_chain_test<P>(options);
    std::cout << " L^2 error = " << err << '\n';
  } else if (options.has_cli_entry("-volumes")) {
    double const err = run_volume_test<P>(options);
    std::cout << " L^2 error = " << err << '\n';
  } else {
    std::cout << " missing PDE selection, e.g., -chains, -volumes\n";
  }

  return 0;
}

template<typename P>
void test_chains(double tol, std::string const &opts) {
  std::string const name = ((is_double<P>) ? "(chains 2d) " : " (chains 2d) ") + opts;
  current_test<P> test_(name);

  auto options = make_opts(opts);
  options.verbosity = verbosity_level::quiet;

  double const err = run_chain_test<P>(options);

  // std::cout << opts << "   " << err << "   " << tol << '\n';
  tcheckless(1, err, tol);
}

template<typename P>
void test_volumes(double tol, std::string const &opts) {
  std::string const name = ((is_double<P>) ? "(volumes 1d) " : " (volumes 1d) ") + opts;
  current_test<P> test_(name);

  auto options = make_opts(opts);
  options.verbosity = verbosity_level::quiet;

  double const err = run_volume_test<P>(options);

  // std::cout << opts << "   " << err << "   " << tol << '\n';
  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("misc edge cases");

#ifdef ASGARD_ENABLE_DOUBLE
  test_chains<double>(5.E-3, "");
  test_chains<double>(1.E-3, "-l 5");
  test_chains<double>(5.E-4, "-l 6");
  test_chains<double>(1.E-3, "-d 2 -l 3");
  test_chains<double>(1.E-4, "-d 2 -l 4");

  #ifdef ASGARD_USE_GPU
  // large test for the direct solver
  test_chains<double>(5.E-5, "-l 8 -sv direct");
  #endif

  test_volumes<double>(5.E-4, "");
  test_volumes<double>(5.E-6, "-dt 0.001 -d 2 -l 5");
#endif

#ifdef ASGARD_ENABLE_FLOAT
  test_chains<float>(5.E-3, "");
  test_chains<float>(2.E-3, "-l 5");
  test_chains<float>(1.E-3, "-d 2 -l 3");

  #ifdef ASGARD_USE_GPU
  // large test for the direct solver
  test_chains<float>(5.E-3, "-l 8 -sv direct");
  #endif

  test_volumes<float>(5.E-4, "");
#endif
}
