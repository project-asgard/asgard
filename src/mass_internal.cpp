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
 * \endinternal
 */

// The PDE is:
// exp(-x^2) cos(y) df/dt + cos(x) df/dx + y df/dy = sources
// f = sin(x) cos(y) exp(-t)
// domain is (x, y) in [-1, 1] x [0.5 1]
//   - the ranges are chosen so that cos(x) and y are both positive
//   - key here is that the time-derivative has a mass term

using namespace asgard;

template<typename P = default_precision>
PDEv2<P> make_mass_pde(asgard::prog_opts options) {
  options.title = "PDE with Mass 2D";

  // the domain will have range (-1, 1) in each direction
  std::vector<asgard::domain_range<P>> ranges({{-1, 1}, {0.5, 1}});

  asgard::pde_domain<P> domain({{-1, 1}, {0.5, 1}}); // can use move here, but copy is cheap enough

  options.default_degree = 2;
  options.default_start_levels = {4, };

  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);

  options.default_dt = 0.5 * 0.1 * dx;
  options.default_stop_time = 1.0;

  PDEv2<P> pde(options, std::move(domain));

  pde.set_mass({term_mass<P>{builtin_v<P>::expneg2}, term_mass<P>{builtin_v<P>::cos}});

  term_1d<P> cdiv = term_div<P>(builtin_v<P>::cos, flux_type::upwind, boundary_type::free);
  term_1d<P> ydiv = term_div<P>(builtin_v<P>::positive, flux_type::upwind, boundary_type::free);

  pde += {cdiv, term_mass<P>{builtin_v<P>::cos}};
  pde += {term_mass<P>{builtin_v<P>::expneg2}, ydiv};

  // exact solution
  auto exact_x = [](std::vector<P> const &x, P, std::vector<P> &fx) -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::exp(-x[i] * x[i]) * std::sin(x[i]);
    };
  auto exact_y = [](std::vector<P> const &x, P, std::vector<P> &fx) -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < x.size(); i++) {
        P const c = std::cos(x[i]);
        fx[i] = c * c;
      }
    };
  separable_func<P> exact({exact_x, exact_y}, builtin_s<P>::expneg);
  pde.add_initial(exact);

  // time-derivative source
  pde.add_source({{exact_x, exact_y}, builtin_s<P>::dexpneg});

  // source derivative in x
  auto exact_dx = [](std::vector<P> const &x, P, std::vector<P> &fx) -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < x.size(); i++) {
        fx[i] = std::exp(-x[i] * x[i]) * std::cos(x[i]) * std::cos(x[i]);
      }
    };
  pde.add_source({{exact_dx, exact_y}, builtin_s<P>::expneg});

  // source derivative in y
  auto exact_dy = [](std::vector<P> const &y, P, std::vector<P> &fy) -> void {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < y.size(); i++) {
        fy[i] = - y[i] * std::sin(y[i]) * std::cos(y[i]);
      }
    };

  pde.add_source({{exact_x, exact_dy}, builtin_s<P>::expneg});

  return pde;
}

template<typename P>
double get_error_l2(discretization_manager<P> const &disc)
{
  std::vector<P> const eref = disc.project_function(disc.get_pde2().ic_sep());

  //double constexpr space = 0.545351286587159 * 0.266956610504446;
  double constexpr space = 0.317726350677342 * 0.200169503107421;
  double const time_val  = std::exp(-disc.time_params().time());

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

  // std::cout << ndiff << "  " << enorm << "  " << nself << "\n";

  return std::sqrt(ndiff);

  if (enorm < 1.0)
    return std::sqrt(ndiff + enorm - nself);
  else
    return std::sqrt((ndiff + enorm - nself) / enorm);
}


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
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", "--test"}, {});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // self_test();
    return 0;
  }

  discretization_manager<P> disc(make_mass_pde<P>(options), verbosity_level::low);

  if (not disc.stop_verbosity())
    std::cout << " -- error in the initial conditions: " << get_error_l2(disc) << "\n";

  disc.advance_time();

  disc.final_output();

  if (not disc.stop_verbosity())
    std::cout << " -- final error: " << get_error_l2(disc) << "\n";

  return 0;
}
