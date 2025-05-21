#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file bound_internal.cpp
 * \brief Internal stress-test
 * \author The ASGarD Team
 *
 * Providing an stress test using a very contrived pde,
 * no-logic or physics here, just messy components that
 * must yield a specific solution.
 *
 * Focusing on boundary conditions.
 * \endinternal
 */

using namespace asgard;

struct type_left {};
struct type_right {};

template<typename P = default_precision, typename btype = type_right>
pde_scheme<P> make_side_pde(int num_dims, int dim, prog_opts options) {
  // df / dt + df / dx_i = 1, exact solution is f = x_i, i = dim
  static_assert(std::is_same_v<btype, type_left> or std::is_same_v<btype, type_right>);

  rassert(dim < num_dims, "cannot set boundary conditions for pde");
  if constexpr (std::is_same_v<btype, type_left>) {
    options.title = "PDE with Fixed BC " + std::to_string(num_dims) + "D (left)";
  } else {
    options.title = "PDE with Fixed BC " + std::to_string(num_dims) + "D (right)";
  }

  pde_domain<P> domain(std::vector<domain_range>(num_dims, {0, 1}));

  options.default_degree = 1;
  options.default_start_levels = {4, };

  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);

  options.default_step_method = time_method::steady;
  options.default_solver = solver_method::direct;

  options.default_dt = 0.5 * 0.1 * dx;
  options.default_stop_time = 1.0;

  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 2000;

  options.default_isolver_inner_iterations = 200;

  pde_scheme<P> pde(options, std::move(domain));

  term_1d<P> div = term_div<P>(1, boundary_type::bothsides);

  div.set_penalty(P{1} / pde.min_cell_size());

  if constexpr (std::is_same_v<btype, type_left>) {
    // the multi-dimensional divergence, initially set to identity in md
    std::vector<term_1d<P>> ops(num_dims);
    ops[dim] = div;

    term_md<P> div_md(ops);

    separable_func<P> lbc(std::vector<P>(num_dims, 1));
    separable_func<P> rbc(std::vector<P>(num_dims, 1));
    rbc.set(dim, 2);

    div_md += left_boundary_flux{lbc};
    div_md += right_boundary_flux{rbc};

    pde += div_md;

  } else {
    std::vector<term_1d<P>> ops(num_dims);
    ops[dim] = div;

    term_md<P> div_md(ops);

    separable_func<P> bc(std::vector<P>(num_dims, 1));

    div_md += right_boundary_flux{bc};

    pde += div_md;
  }

  auto one = [=](std::vector<P> const &, P /* time */, std::vector<P> &fx) ->
    void {
      std::fill(fx.begin(), fx.end(), P{1});
    };

  pde.add_source({std::vector<svector_func1d<P>>(num_dims, one),
                  ignores_time});

  std::vector<svector_func1d<P>> one_md(num_dims, one);
  one_md[dim] = [=](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      if constexpr (std::is_same_v<btype, type_left>) {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = x[i] + P{1};
      } else {
        std::copy(x.begin(), x.end(), fx.begin());
      }
    };

  pde.add_initial({one_md, ignores_time});

  return pde;
}

template<typename P = default_precision>
pde_scheme<P> make_quad_pde(int num_dims, prog_opts options) {
  // -u_xx = 1 u(0) = u(1) = 0 -> u = 0.5 * x * (1 - x)
  options.title = "PDE quadratic solution " + std::to_string(num_dims) + "D";

  pde_domain<P> domain(std::vector<domain_range>(num_dims, {0, 1}));

  options.default_degree = 1;
  options.default_start_levels = {4, };

  options.default_stop_time = 0.0;

  options.default_step_method = time_method::steady;

  options.default_solver = solver_method::direct;

  // for when we use bicgstab or gmres
  options.default_isolver_tolerance  = 1.E-6;
  options.default_isolver_iterations = 4000;

  pde_scheme<P> pde(options, std::move(domain));

  term_1d<P> div  = term_div<P>(-1, boundary_type::none);
  term_1d<P> grad = term_grad<P>(1, boundary_type::bothsides);

  term_1d<P> diffusion({div, grad});

  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);

  term_1d<P> penalty = term_penalty<P>(P{1} / dx, boundary_type::bothsides);

  std::vector<term_1d<P>> ops(num_dims);
  for (int d = 0; d < num_dims; d++)
  {
    ops[d] = diffusion; // using operator in the d-direction
    pde += term_md<P>(ops);

    if (num_dims == 1 and pde.options().step_method.value_or(time_method::rk3)
          == time_method::steady) {
      ops[d] = penalty;
      pde += term_md<P>(ops);
    }

    ops[d] = term_identity{}; // reset back to identity
  }

  auto one = [=](std::vector<P> const &, P /* time */, std::vector<P> &fx) ->
    void {
      std::fill(fx.begin(), fx.end(), P{1});
    };
  auto s1d = [=](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = 0.5 * x[i] * (P{1} - x[i]);
    };

  std::vector<svector_func1d<P>> func(num_dims, s1d);

  for (int d : iindexof(num_dims)) {
    func[d] = one;
    pde.add_source({func, ignores_time});
    func[d] = s1d;
  }

  // no initial state
  return pde;
}

template<typename P>
double get_error_l2(discretization_manager<P> const &disc)
{
  if (disc.title_contains("quadratic")) {
    int const num_dims = disc.num_dims();

    double constexpr n1d = 25.0 / 3000.0;
    double const enorm   = fm::ipow(n1d, disc.num_dims());

    auto ex1d = [=](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
      void {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = 0.5 * x[i] * (1 - x[i]);
      };

    std::vector<P> const eref = disc.project_function({std::vector<svector_func1d<P>>(num_dims, ex1d),
                                                       ignores_time});

    std::vector<P> const &state = disc.current_state_mpi();
    assert(eref.size() == state.size());

    double nself = 0;
    double ndiff = 0;
    double nnn = 0;
    for (size_t i = 0; i < state.size(); i++)
    {
      double const e = eref[i] - state[i];
      ndiff += e * e;
      double const r = eref[i];
      nself += r * r;
      nnn += state[i] * state[i];
    }

    return std::sqrt(ndiff + std::abs(enorm - nself));
  }

  std::vector<P> const eref = disc.project_function(disc.initial_cond_sep());

  bool const left    = disc.title_contains("(left)");
  double const enorm = (left) ? P{7} / P{3} : P{1} / P{3};

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

  if (enorm < 1.0)
    return std::sqrt(ndiff + std::abs(enorm - nself));
  else
    return std::sqrt((ndiff + std::abs(enorm - nself)) / enorm);
}

void self_test();

int main(int argc, char** argv)
{
  #ifdef ASGARD_USE_MPI
  // if MPI is not enabled, make sure the PDE works fine without this line
  libasgard_runtime running_(argc, argv);
  #endif

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
-dims            -dm     int        accepts: 1 - 6
                                    the number of dimensions
-div             -dv     int        accepts: 0 - 5, must be less than dims
                                    the dimensions with div term

-left                    -          put inhomogeneity on the left
-right                   -          put inhomogeneity on the right
-quad                    -          switch to Poisson equation with quadratic solution

-test                               perform self-testing
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", "--test", "-left", "-right", "-quad"},
                               {"-dims", "-dm", "-div", "-dv"});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    self_test();
    return 0;
  }

  discretization_manager<P> disc; // delay initialization

  int const num_dims = options.extra_cli_value_group<int>({"-dims", "-dm"}).value_or(1);

  if (options.has_cli_entry("-quad")) {

    disc = discretization_manager<P>(make_quad_pde<P>(num_dims, options), verbosity_level::low);

  } else {

    int const num_div = options.extra_cli_value_group<int>({"-div", "-dv"}).value_or(0);

    bool const left = options.has_cli_entry("-left");

    auto pde = (left) ? make_side_pde<P, type_left>(num_dims, num_div, options)
                      : make_side_pde<P, type_right>(num_dims, num_div, options);

    disc = discretization_manager<P>(std::move(pde), verbosity_level::low);

    disc.set_current_state(std::vector<P>(disc.current_state().size(), P{0}));
  }

  disc.advance_time();

  disc.final_output();

  P const err = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- final error: " << err << "\n";

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
void dotest_quad(double tol, int num_dims, std::string const &opts) {
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
  dotest<double>(1.E-7,  1, "-dv 0 -right");
  dotest<double>(1.E-7,  2, "-dv 0 -right");
  dotest<double>(1.E-7,  2, "-dv 1 -right");
  dotest<double>(1.E-7,  1, "-dv 0 -left");
  dotest<double>(1.E-7,  2, "-dv 0 -left");
  dotest<double>(1.E-7,  2, "-dv 1 -left");
  dotest<double>(1.E-7,  3, "-dv 0 -right");
  dotest<double>(1.E-7,  3, "-dv 1 -right");
  dotest<double>(1.E-7,  3, "-dv 2 -right");
  dotest<double>(2.E-7,  3, "-dv 0 -left");
  dotest<double>(2.E-7,  3, "-dv 1 -left");
  dotest<double>(2.E-7,  3, "-dv 2 -left");
  dotest<double>(2.E-7,  4, "-dv 0 -left");
  dotest<double>(1.E-7,  4, "-dv 2 -right");
  dotest<double>(2.E-7,  5, "-dv 1 -left -sv gmres");
  dotest<double>(1.E-7,  5, "-dv 3 -right -sv gmres");
  dotest<double>(2.E-7,  6, "-dv 1 -left -sv gmres");
  dotest<double>(1.E-7,  6, "-dv 2 -right -sv gmres");

  dotest_quad<double>(1.E-3, 1, "-quad -l 5 -s cn -d 1 -dt 0.1 -t 10");
  dotest_quad<double>(1.E-3, 1, "-quad -l 5 -s cn -d 2 -dt 0.1 -t 10");
  dotest_quad<double>(5.E-5, 1, "-quad -l 6 -s steady");
  dotest_quad<double>(1.E-8, 1, "-quad -l 6 -s steady -d 2");
  dotest_quad<double>(1.E-4, 1, "-quad -l 5 -s steady -sv bicgstab");
  dotest_quad<double>(1.E-4, 2, "-quad -l 5 -s steady");
  dotest_quad<double>(1.E-5, 3, "-quad -l 5 -s steady");
  dotest_quad<double>(1.E-6, 4, "-quad -l 5 -sv bicgstab");
#endif

#ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(1.E-3,  1, "-dv 0 -left");
  dotest<float>(1.E-3,  1, "-dv 0 -right");
  dotest<float>(1.E-3,  2, "-dv 0");
  dotest<float>(1.E-3,  2, "-dv 1");
  dotest<float>(1.E-3,  3, "-dv 0 -right");
  dotest<float>(1.E-3,  3, "-dv 1 -right");
  dotest<float>(1.E-3,  3, "-dv 2 -right");
  dotest<float>(2.E-3,  3, "-dv 0 -left");
  dotest<float>(2.E-3,  3, "-dv 1 -left");
  dotest<float>(2.E-3,  3, "-dv 2 -left");
  dotest<float>(1.E-3,  4, "-dv 0");
  dotest<float>(1.E-3,  4, "-dv 2");
  dotest<float>(5.E-3,  5, "-dv 1 -sv gmres");
  dotest<float>(5.E-3,  5, "-dv 3 -sv gmres");
  dotest<float>(5.E-3,  6, "-dv 1 -sv gmres");
  dotest<float>(5.E-3,  6, "-dv 2 -sv gmres");

  dotest_quad<float>(5.E-4, 1, "-quad -l 5");
  dotest_quad<float>(1.E-4, 1, "-quad -l 6");
  dotest_quad<float>(1.E-4, 1, "-quad -l 2 -d 2");
#endif
}
