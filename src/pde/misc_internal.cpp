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

  separable_func<P> exact1({
      vectorize_t([](P x) -> P { return std::exp(x); }),
      vectorize_t([](P y) -> P { return std::cos(y); }),
  }, ignores_time);
  separable_func<P> exact2({
      vectorize_t([](P x) -> P { return std::exp(-x); }),
      vectorize_t([](P y) -> P { return std::sin(y); }),
  }, ignores_time);

  {
    auto c1 = vectorize<P>([](P x) -> P { return std::sin(x); });
    term_1d<P> dxx1 = {term_div<P>{-1}, term_volume{c1},
                       term_grad{1, boundary_type::bothsides}};

    term_md<P> dxx({dxx1, term_identity{}});
  }

  // term_1d<P> div = term_div<P>(1, boundary_type::bothsides);
  //
  // div.set_penalty(P{1} / pde.min_cell_size());
  //
  // if constexpr (std::is_same_v<btype, type_left>) {
  //   // the multi-dimensional divergence, initially set to identity in md
  //   std::vector<term_1d<P>> ops(num_dims);
  //   ops[dim] = div;
  //
  //   term_md<P> div_md(ops);
  //
  //   separable_func<P> lbc(std::vector<P>(num_dims, 1));
  //   separable_func<P> rbc(std::vector<P>(num_dims, 1));
  //   rbc.set(dim, 2);
  //
  //   div_md += left_boundary_flux{lbc};
  //   div_md += right_boundary_flux{rbc};
  //
  //   pde += div_md;
  //
  // } else {
  //   std::vector<term_1d<P>> ops(num_dims);
  //   ops[dim] = div;
  //
  //   term_md<P> div_md(ops);
  //
  //   separable_func<P> bc(std::vector<P>(num_dims, 1));
  //
  //   div_md += right_boundary_flux{bc};
  //
  //   pde += div_md;
  // }
  //
  // auto one = [=](std::vector<P> const &, P /* time */, std::vector<P> &fx) ->
  //   void {
  //     std::fill(fx.begin(), fx.end(), P{1});
  //   };
  //
  // pde.add_source({std::vector<svector_func1d<P>>(num_dims, one),
  //                 ignores_time});
  //
  // std::vector<svector_func1d<P>> one_md(num_dims, one);
  // one_md[dim] = [=](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
  //   void {
  //     if constexpr (std::is_same_v<btype, type_left>) {
  //       for (size_t i = 0; i < x.size(); i++)
  //         fx[i] = x[i] + P{1};
  //     } else {
  //       std::copy(x.begin(), x.end(), fx.begin());
  //     }
  //   };
  //
  // pde.add_initial({one_md, ignores_time});
  //
  // return pde;
  return 0;
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

  discretization_manager<P> disc; // delay initialization

  if (options.has_cli_entry("-chains")) {

    disc = discretization_manager<P>(make_quad_pde<P>(num_dims, options), verbosity_level::low);

  }

  disc.advance_time();

  disc.final_output();

  if (not disc.stop_verbosity())
    std::cout << " -- final error: " << get_error_l2(disc) << "\n";

  return 0;
}

template<typename P>
void dotest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  int const dv = options.extra_cli_value<int>("-dv").value();

  bool const left = options.has_cli_entry("-left");

  auto pde = (left) ? make_side_pde<P, type_left>(num_dims, dv, options)
                    : make_side_pde<P, type_right>(num_dims, dv, options);

  discretization_manager<P> disc(std::move(pde), verbosity_level::quiet);

  // make sure there's something to solve
  disc.set_current_state(std::vector<P>(disc.current_state().size(), P{0}));

  while (disc.remaining_steps() > 0)
  {
    disc.advance_time(1);

    double const err = get_error_l2(disc);

    tcheckless(disc.current_step(), err, tol);
  }
}

template<typename P>
void test_chains(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  auto pde = make_quad_pde<P>(num_dims, options);

  discretization_manager<P> disc(std::move(pde), verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);

  // std::cout << err << "\n";

  tcheckless(disc.current_step(), err, tol);
}

void self_test() {
  all_tests testing_("boundary conditions");

#ifdef ASGARD_ENABLE_DOUBLE
  test_chains<double>(1.E-7,  1, "");
#endif

#ifdef ASGARD_ENABLE_FLOAT
  //
#endif
}
