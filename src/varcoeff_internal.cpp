#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file mass_internal.cpp
 * \brief Internal stress-test
 * \author The ASGarD Team
 *
 * Providing an stress test using a very contrived pde,
 * no-logic or physics here, just messy components that
 * must yield a specific solution.
 *
 * Focusing on variable coefficients.
 * \endinternal
 */

// There are two PDEs, 1D and 2D versions
// f = cos(x) cos(t)
// f_t + div exp(-x) f = sources


using namespace asgard;

template<typename P = default_precision>
PDEv2<P> make_var_pde(int num_dims, asgard::prog_opts options) {
  rassert(1 <= num_dims and num_dims <= 2, "make_var_pde() sets 1D and 2D problems");

  options.title = "PDE with Variable Coeffs " + std::to_string(num_dims) + "D";

  asgard::pde_domain<P> domain(std::vector<domain_range<P>>(num_dims, {-1, 1}));

  options.default_degree = 2;
  options.default_start_levels = {4, };

  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);

  options.default_dt = 0.5 * 0.1 * dx;
  options.default_stop_time = 1.0;

  PDEv2<P> pde(options, std::move(domain));

  if (num_dims == 1) {
    term_1d<P> div = term_div<P>(builtin_v<P>::expneg, flux_type::upwind, boundary_type::right_free);
    pde += {div, };

    auto cospi2 = vectorize_t<P>([](P x)->P{ return std::cos(0.5 * PI * x); });

    separable_func<P> exact({cospi2, }, builtin_s<P>::cos);
    pde.add_initial(exact);

    // add the time derivative
    pde.add_source({{cospi2, }, builtin_s<P>::dcos});

    pde.add_source({{vectorize_t<P>([](P x)->P{ return -std::exp(-x) * std::cos(0.5 * PI * x); }), },
                   builtin_s<P>::cos});
    pde.add_source({{vectorize_t<P>([](P x)->P{ return -0.5 * PI * std::exp(-x) * std::sin(0.5 * PI * x); }), },
                   builtin_s<P>::cos});
  }

  return pde;
}

template<typename P>
double get_error_l2(discretization_manager<P> const &disc)
{
  std::vector<P> const eref = disc.project_function(disc.get_pde2().ic_sep());

  double const space = (disc.num_dims() == 1) ? 1 : 1;
  double const time_val  = std::cos(disc.time_params().time());

  double const enorm = space * time_val * time_val;

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

  if (enorm < 1.0)
    return std::sqrt(ndiff + enorm - nself);
  else
    return std::sqrt((ndiff + enorm - nself) / enorm);
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
    std::cout << "\n solves a messy testing pde:\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-dims            -dm     int        accepts: 1 - 2\n";
    std::cout << "                                    the number of dimensions\n\n";
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", "--test"}, {"-dims", "-dm"});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    self_test();
    return 0;
  }

  std::optional<int> opt_dims = options.extra_cli_value<int>("-dims");
  if (not opt_dims)
    opt_dims = options.extra_cli_value<int>("-dm");

  int const num_dims = opt_dims.value_or(1);

  discretization_manager<P> disc(make_var_pde<P>(num_dims, options), verbosity_level::low);

  if (not disc.stop_verbosity())
    std::cout << " -- error in the initial conditions: " << get_error_l2(disc) << "\n";

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

  discretization_manager<P> disc(make_var_pde<P>(num_dims, options),
                                 verbosity_level::quiet);

  while (disc.time_params().num_remain() > 0)
  {
    disc.advance_time(1);

    double const err = get_error_l2(disc);

    tcheckless(disc.time_params().step(), err, tol);
  }
}

void self_test() {
  all_tests testing_("variable coefficient equation:", " f_t + div exp(x) f = sources");

#ifdef ASGARD_ENABLE_DOUBLE
  dotest<double>(0.05,  1, "-l 7 -d 0 -t 1.57");
  dotest<double>(0.005, 1, "-l 4 -d 1 -t 1.57");
  dotest<double>(0.005, 1, "-l 4 -d 1 -t 3.14");
  dotest<double>(1.E-4, 1, "-l 4 -d 2 -t 3.14");

#endif

#ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(0.05,  1, "-l 7 -d 0 -t 1.57");
  dotest<float>(0.005, 1, "-l 4 -d 1 -t 1.57");
  dotest<float>(0.005, 1, "-l 4 -d 1 -t 3.14");
  dotest<float>(0.001, 1, "-l 4 -d 2 -t 3.14");

#endif
}
