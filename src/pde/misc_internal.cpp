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

template<typename P = default_precision>
double run_chain_test(prog_opts options) {

  options.title = "Deep Chains 1D";

  pde_domain<P> domain({{0, 2}, {1, 4}});

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

  separable_func<P> exact1({
      vectorize_t<P>([](P x) -> P { return std::exp(x); }),
      vectorize_t<P>([](P y) -> P { return std::cos(y); }),
  }, ignores_time);
  separable_func<P> exact2({
      vectorize_t<P>([](P x) -> P { return std::exp(-x); }),
      vectorize_t<P>([](P y) -> P { return std::sin(y); }),
  }, ignores_time);

  {
    auto c1 = vectorize<P>([](P x) -> P { return std::sin(x); });
    term_1d<P> dxx1 = std::vector<term_1d<P>>{
                        term_div<P>{-1, boundary_type::left},
                        term_volume{c1},
                        term_grad{1, boundary_type::right}};

    dxx1.set_penalty(P{1} / cellx);
    term_md<P> dxx({dxx1, term_identity{}});

    separable_func<P> bc1 = exact1;
    separable_func<P> bc2 = exact2;

    bc1.set(0, std::exp(P{2}));
    bc2.set(0, std::exp(-P{2}));

    dxx += right_boundary_flux<P>{bc1};
    dxx += right_boundary_flux<P>{bc2};

    bc2.set(0, std::exp(0));
    bc2.set(0, -std::exp(0));
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
                        term_volume{2},
                        term_div<P>{-1, boundary_type::left},
                        term_volume{c2},
                        term_grad{1, boundary_type::right}};

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

  discretization_manager<P> disc(std::move(pde), verbosity_level::high);

  disc.advance_time();

  disc.save_final_snapshot();

  std::vector<P> const eref   = disc.project_function({exact1, exact2});
  std::vector<P> const &state = disc.current_state();

  double const enorm = 41.19079366502316;

  double nself = 0, ndiff = 0;
  for (size_t i = 0; i < state.size(); i++)
  {
    double const e = eref[i] - state[i];
    ndiff += e * e;
    double const r = eref[i];
    nself += r * r;
  }

  return std::sqrt(ndiff + std::abs(enorm - nself));
}

template<typename P>
double get_error_l2(discretization_manager<P> const &disc)
{
  if (disc.title_contains("quadratic")) {
    // int const num_dims = disc.num_dims();
    //
    // double constexpr n1d = 25.0 / 3000.0;
    // double const enorm   = fm::ipow(n1d, disc.num_dims());
    //
    // auto ex1d = [=](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    // void {
    //   for (size_t i = 0; i < x.size(); i++)
    //     fx[i] = 0.5 * x[i] * (1 - x[i]);
    // };

    // std::vector<P> const eref = disc.project_function({std::vector<svector_func1d<P>>(num_dims, ex1d),
    //                                                    ignores_time});
    //
    // std::vector<P> const &state = disc.current_state();
    // assert(eref.size() == state.size());
    //
    // double nself = 0;
    // double ndiff = 0;
    // double nnn = 0;
    // for (size_t i = 0; i < state.size(); i++)
    // {
    //   double const e = eref[i] - state[i];
    //   ndiff += e * e;
    //   double const r = eref[i];
    //   nself += r * r;
    //   nnn += state[i] * state[i];
    // }
    //
    // return std::sqrt(ndiff + std::abs(enorm - nself));
    return 0;
  }

  return 0;
}

void self_test();

int main(int argc, char** argv)
{
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
-chains                             test messy chains example
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", "-chains"}, {});

  if (options.has_cli_entry("-test")) {
    self_test();
    return 0;
  }

  if (options.has_cli_entry("-chains")) {
    double const err = run_chain_test<P>(options);
    std::cout << " L^2 error = " << err << '\n';
  }

  return 0;
}

template<typename P>
void test_chains(double tol, std::string const &opts) {
  current_test<P> test_(opts);

  auto options = make_opts(opts);

  double const err = run_chain_test(options);

  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("boundary conditions");

#ifdef ASGARD_ENABLE_DOUBLE
  test_chains<double>(1.E-7, "-v 0");
#endif

#ifdef ASGARD_ENABLE_FLOAT
  //
#endif
}
