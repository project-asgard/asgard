#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file relaxation_internal.cpp
 * \brief Simple collision problem
 * \author The ASGarD Team
 *
 * Provides a stress-test with a contrived PDE that contains only
 * Lenard-Bernstein collision terms.
 *
 * \endinternal
 */

using namespace asgard;

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_relaxation(int xdims, int vdims, asgard::prog_opts options) {
  rassert(1 <= xdims and xdims <= 3, "problem is set for 1, 2 or 3 position dimensions")
  rassert(1 <= vdims and vdims <= 3, "problem is set for 1, 2 or 3 velocity dimensions")

  options.title = "Relaxation " + std::to_string(xdims) + "x" + std::to_string(vdims) + "v";

  std::vector<domain_range> ranges;
  ranges.reserve(xdims + vdims);
  for (int x = 0; x < xdims; x++)
    ranges.emplace_back(-0.5, +0.5);
  for (int v = 0; v < vdims; v++)
    ranges.emplace_back(-8.0, 12.0);

  // the domain has one position and multiple velocity dimensions
  pde_domain<P> domain(position_dims{xdims}, velocity_dims{vdims}, ranges);

  options.default_degree = 2;
  options.default_start_levels = {7, };

  options.default_dt = 0.05;

  options.default_stop_time = 1.0;

  options.default_solver = solver_method::gmres;
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;
  options.default_isolver_inner_iterations = 50;

  options.default_precon = (xdims == 1) ? precon_method::jacobi : precon_method::none;
  options.default_precon = precon_method::none;

  // using implicit backward Euler
  options.default_step_method = asgard::time_method::back_euler;

  // get the collision frequency
  P const nu = options.extra_cli_value_group<P>({"-nu", "-collision_freq"}).value_or(1000);
  options.subtitle = "collision frequency: " + std::to_string(nu);

  // create a pde from the given options and domain
  pde_scheme<P> pde(options, domain);

  pde += operators::lenard_bernstein_collisions{nu};

  if (vdims == 1)
  {
    std::vector<P> ic_vec(xdims + vdims, 1);
    ic_vec[0] = 0.5;
    separable_func<P> ic(ic_vec); // separable initial conditions

    dimension_id const v1{xdims};
    ic.set(v1, [](std::vector<P> const &v, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr ux    = -1.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - ux) * (v[i] - ux));
      });
    pde.add_initial(ic);

    ic.set(v1, [](std::vector<P> const &v, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr ux    = 2.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - ux) * (v[i] - ux));
      });
    pde.add_initial(ic);
  }
  else if (vdims == 2)
  {
    std::vector<P> ic_vec(xdims + vdims, 1);
    ic_vec[0] = 0.5;
    separable_func<P> ic(ic_vec);

    dimension_id const v1{xdims};
    dimension_id const v2{xdims + 1};
    ic.set(v1, [](std::vector<P> const &v, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr u     = 3.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      });
    ic.set(v2, [](std::vector<P> const &v, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr u     = 0.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      });
    pde.add_initial(ic);

    ic.set(v1, [](std::vector<P> const &v, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr u     = 0.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      });
    ic.set(v2, [](std::vector<P> const &v, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr u     = 3.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      });
    pde.add_initial(ic);
  }
  else if (vdims == 3)
  {
    P constexpr xc = 1.0 / 3.0;

    auto max3 = [](std::vector<P> const &v, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr u     = 3.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      };
    auto max0 = [](std::vector<P> const &v, std::vector<P> &fv) -> void {
        P constexpr theta = 0.5;
        P constexpr u     = 0.0;
        P const c         = 1.0 / std::sqrt(2.0 * PI * theta);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      };

    dimension_id const v1{xdims};
    dimension_id const v2{xdims + 1};
    dimension_id const v3{xdims + 2};

    std::vector<P> ic_vec(xdims + vdims, 1);
    ic_vec[0] = xc;
    separable_func<P> ic;

    ic = separable_func<P>(ic_vec);
    ic.set(v1, max3);
    ic.set(v2, max0);
    ic.set(v3, max0);
    pde.add_initial(ic);

    ic = separable_func<P>(ic_vec);
    ic.set(v1, max0);
    ic.set(v2, max3);
    ic.set(v3, max0);
    pde.add_initial(ic);

    ic = separable_func<P>(ic_vec);
    ic.set(v1, max0);
    ic.set(v2, max0);
    ic.set(v3, max3);
    pde.add_initial(ic);
  }

  return pde;
}

template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc) {
  // there is no analytic solution in time, only the final state
  // in a "short" time, the solution will converge to a steady state
  // effective time-scale is collision-frequency (nu) * final-time

  int const num_dims = disc.num_dims();

  int const xdims = disc.domain().num_pos();
  int const vdims = disc.domain().num_vel();

  std::vector<P> eref;
  P enorm = fm::powi(0.170109559932217, vdims);

  if (vdims == 1) { // 1x1v, 2x1v and 3x1v
    dimension_id const v1{xdims}; // due to zero indexing, the index of v1 is the number of x-dimensions
    separable_func<P> exact(std::vector<P>(num_dims, 1.0));
    exact.set(v1, [&](std::vector<P> const &v, P, std::vector<P> &fv)
          -> void {
        P constexpr theta = 2.75;
        P constexpr u     = 0.5;

        P const c = P{1} / std::sqrt(2.0 * PI * theta);
        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      });

    eref = disc.project_function({exact, });
  }
  else if (vdims == 2) // 1x2v, 2x2v, 3x2v
  {
    separable_func<P> exact(std::vector<P>(num_dims, 1.0));

    dimension_id const v1{xdims};
    dimension_id const v2{xdims + 1};
    exact.set(v1, [&](std::vector<P> const &v, P, std::vector<P> &fv)
          -> void {
        P constexpr theta = 2.75;
        P constexpr u     = 1.5;

        P const c = 1.0 / std::sqrt(2.0 * PI * theta);
        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      });
    exact.set(v2, [&](std::vector<P> const &v, P, std::vector<P> &fv)
          -> void {
        P constexpr theta = 2.75;
        P constexpr u     = 1.5;

        P const c = 1.0 / std::sqrt(2.0 * PI * theta);
        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      });

    eref = disc.project_function({exact, });
  }
  else if (vdims == 3) // 1x3v, 2x3v, 3x3v
  {
    separable_func<P> exact(std::vector<P>(num_dims, 1.0));

    dimension_id const v1{xdims};
    dimension_id const v2{xdims + 1};
    dimension_id const v3{xdims + 2};
    auto max1 = [](std::vector<P> const &v, P, std::vector<P> &fv)
          -> void {
        P constexpr theta = 2.5;
        P constexpr u     = 1.0;

        P const c = 1.0 / std::sqrt(2.0 * PI * theta);
        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-(0.5 / theta) * (v[i] - u) * (v[i] - u));
      };

    exact.set(v1, max1);
    exact.set(v2, max1);
    exact.set(v3, max1);

    eref = disc.project_function({exact, });

    enorm = 5.679043443503443e-03;
  }

  std::vector<P> const &state = disc.current_state_mpi();
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

  return std::sqrt(ndiff + std::abs(enorm - nself));
}

int main(int argc, char** argv)
{
  libasgard_runtime running_(argc, argv);

  using P = default_precision;

  prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves the two stream Vlasov-Poisson in 1x-1v dimensions\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-xdims                              position dimensions (1 - 3)\n";
    std::cout << "-vdims                              velocity dimensions (1 - 3)\n";
    std::cout << "-nu                                 collision frequency\n";
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  // this is an optional step, check if there are misspelled or incorrect cli entries
  // the first set/vector of entries are those that can appear by themselves
  // the second set/vector requires extra parameters
  options.throw_if_argv_not_in({"-test", "--test"}, {"-nu", "-xdims", "-xd", "-vdims", "-vd"});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  int const xdims = options.extra_cli_value_group<int>({"-xdims", "-xd"}).value_or(1);
  int const vdims = options.extra_cli_value_group<int>({"-vdims", "-vd"}).value_or(1);

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  discretization_manager<P> disc(make_relaxation<P>(xdims, vdims, options),
                                 asgard::verbosity_level::high);

  disc.advance_time(); // integrate until num-steps or stop-time

  double const err = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- final error: " << err << "\n";

  disc.final_output();

  return 0;
};

#ifndef __ASGARD_DOXYGEN_SKIP
template<typename P>
void test_final(double tol, int xdims, int vdims, std::string const &opt_str)
{
  assert(1 <= xdims and xdims <= 3);
  assert(1 <= vdims and vdims <= 3);
  int const num_dims = xdims + vdims;
  current_test<P> test_(opt_str, num_dims);

  auto options = make_opts(opt_str);

  discretization_manager<P> disc(make_relaxation<P>(xdims, vdims, options),
                                 verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);

  // std::cout << " err = " << err << '\n';

  tcheckless(disc.current_step(), err, tol);
}

template<typename P>
void test_aniso(double tol, int xdims, int vdims, std::vector<int> const &levels,
                std::string const &opt_str)
{
  assert(1 <= xdims and xdims <= 3);
  assert(1 <= vdims and vdims <= 3);
  int const num_dims = xdims + vdims;
  assert(static_cast<size_t>(num_dims) == levels.size());
  std::string rstr = "aniso {";
  for (size_t i = 0; i < levels.size() - 1; i++)
    rstr += std::to_string(levels[i]) + ", ";
  rstr += std::to_string(levels.back()) + "} ";
  current_test<P> test_(rstr + opt_str, num_dims);

  auto options = make_opts(opt_str);

  options.start_levels = levels;

  discretization_manager<P> disc(make_relaxation<P>(xdims, vdims, options),
                                 verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);

  // std::cout << " err = " << err << '\n';

  tcheckless(disc.current_step(), err, tol);
}

void self_test() {
  all_tests testing_("simple relaxation problem");

#ifdef ASGARD_ENABLE_DOUBLE

  // 1x1v
  test_final<double>(5.E-3, 1, 1, "-l 5 -t 2 -d 2 -nu 1000");
  test_final<double>(1.E-4, 1, 1, "-l 6 -t 2 -d 2 -nu 1000");
  test_final<double>(1.E-4, 1, 1, "-l 6 -t 1 -d 2 -nu 2000");
  test_final<double>(1.E-3, 1, 1, "-l 2 -m 5 -nu 1000 -a 1.E-3");
  // *x2v
  test_final<double>(5.E-3, 1, 2, "-l 5 -t 0.25 -d 2 -nu 2000");

  #ifdef ASGARD_LONG_TESTS
  // *x2v
  test_final<double>(5.E-3, 2, 2, "-l 5 -t 0.25 -d 2 -nu 2000");
  // *x3v
  test_final<double>(5.E-2, 1, 3, "-l 4 -t 0.125 -d 2 -nu 200");
  test_final<double>(5.E-2, 2, 3, "-l 4 -t 0.125 -d 2 -nu 200");
  #endif

  // test ansitropic sparse grid with one level restricted to zero
  test_aniso<double>(5.E-3, 1, 1, {1, 4}, "-nu 1000");
  test_aniso<double>(5.E-3, 2, 1, {0, 0, 5}, "-nu 1000");

  #ifdef ASGARD_LONG_TESTS
  test_aniso<double>(5.E-3, 3, 1, {0, 0, 0, 5}, "-nu 1000");
  test_aniso<double>(5.E-3, 3, 2, {0, 0, 0, 5, 5}, "-t 0.125 -nu 200");
  test_aniso<double>(5.E-2, 3, 3, {0, 0, 0, 4, 4, 4}, "-t 0.125 -nu 200");
  #endif

#endif

#ifdef ASGARD_ENABLE_FLOAT

  test_final<float>(5.E-3, 1, 1, "-l 5 -t 2 -d 2 -nu 1000 -ist 1.E-4");
  test_final<float>(8.E-3, 1, 2, "-l 4 -t 2 -d 2 -nu 100 -ist 1.E-4");

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
