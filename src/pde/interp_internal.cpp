#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

template<typename P>
void test_ode1d(double const tol, std::string const &opts)
{
  prog_opts options = make_opts(opts);

  pde_domain<P> domain({{0, 2}, });

  options.default_degree = 1;
  options.default_stop_time = 1;
  options.default_start_levels = {5, };

  options.default_step_method = time_method::forward_euler;
  options.default_dt = P{1} / P{64};

  // separable and interpolation odes
  pde_scheme<P> sode(options, domain);
  pde_scheme<P> iode(options, domain);

  sode += {term_volume<P>{2}, };
  iode += term_interp<P>([](P, vector2d<P> const &, std::vector<P> const &f,
                            std::vector<P> &vals)
        -> void {
      for (auto i : indexof(f))
        vals[i] = 2 * f[i];
    });

  separable_func<P> ic(std::vector<P>{1, });
  sode.add_initial(ic);
  iode.add_initial(ic);

  discretization_manager<P> sdisc(sode, verbosity_level::quiet);
  discretization_manager<P> idisc(iode, verbosity_level::quiet);

  sdisc.advance_time();
  idisc.advance_time();

  auto const &sstate = sdisc.current_state_mpi();
  auto const &istate = idisc.current_state_mpi();

  double err = 0;
  for (auto i : indexof(sstate)) {
    P const e = sstate[i] - istate[i];
    err += e * e;
  }

  tcheckless(0, std::sqrt(err), tol);
}

void test_ode1d() {
  current_test name_("ode 1d");
#ifdef ASGARD_ENABLE_DOUBLE
  test_ode1d<double>(1.E-11, "-l 4 -d 1");
  test_ode1d<double>(1.E-11, "-l 5 -d 1");
  test_ode1d<double>(1.E-11, "-l 5 -d 2");
  test_ode1d<double>(1.E-11, "-l 4 -d 3");
  test_ode1d<double>(1.E-11, "-l 5 -d 3");
  test_ode1d<double>(1.E-11, "-l 6 -d 3");
  test_ode1d<double>(1.E-11, "-l 7 -d 3");
#endif
#ifdef ASGARD_ENABLE_FLOAT
  test_ode1d<float>(1.E-5, "-l 5 -d 1");
  test_ode1d<float>(1.E-5, "-l 5 -d 2");
  test_ode1d<float>(1.E-5, "-l 5 -d 3");
#endif
}

template<typename P>
void test_ic(double const tol, std::string const &opts)
{
  prog_opts options = make_opts(opts);

  pde_domain<P> domain({{0, 2}, {-1, 3}});

  options.default_degree = 1;
  options.default_stop_time = 1;
  options.default_start_levels = {5, };

  options.default_step_method = time_method::rk2;
  options.default_dt = P{1} / P{64};

  // separable and interpolation odes
  pde_scheme<P> sode(options, domain);
  pde_scheme<P> iode(options, domain);

  auto ic0x = [](P x) -> P { return std::sin(x); };
  auto ic0y = [](P y) -> P { return std::exp(y); };
  auto ic1x = [](P x) -> P { return std::cos(x); };
  auto ic1y = [](P y) -> P { return std::exp(2 * y); };

  sode.add_initial(std::vector<svector_func1d<P>>{vectorize_t<P>(ic0x), vectorize_t<P>(ic0y)});
  sode.add_initial(std::vector<svector_func1d<P>>{vectorize_t<P>(ic1x), vectorize_t<P>(ic1y)});

  auto icmd = [=](P t, vector2d<P> const &nodes, std::vector<P> &vals) ->
    void {
      expect(t == 0);
      for (auto i : indexof(vals)) {
        P const x = nodes[i][0];
        P const y = nodes[i][1];
        vals[i] = ic0x(x) * ic0y(y) + ic1x(x) * ic1y(y);
      }
    };

  iode.set_initial(icmd);

  discretization_manager<P> sdisc(sode, verbosity_level::quiet);
  discretization_manager<P> idisc(iode, verbosity_level::quiet);

  auto const &sstate = sdisc.current_state_mpi();
  auto const &istate = idisc.current_state_mpi();

  double err = 0;
  double total = 0;
  for (auto i : indexof(sstate)) {
    P const e = sstate[i] - istate[i];
    err += e * e;
    total += sstate[i] * sstate[i];
  }

  tcheckless(0, std::sqrt(err) / std::sqrt(total), tol);
}

void test_ic() {
  current_test name_("initial conditions");
#ifdef ASGARD_ENABLE_DOUBLE
  test_ic<double>(1.E-2, "-l 4 -d 1");
  test_ic<double>(5.E-3, "-l 5 -d 1");
  test_ic<double>(1.E-3, "-l 6 -d 1");
  test_ic<double>(5.E-2, "-l 4 -d 2");
  test_ic<double>(5.E-2, "-l 5 -d 2");
  test_ic<double>(2.E-2, "-l 6 -d 2");
  test_ic<double>(5.E-5, "-l 4 -d 3");
  test_ic<double>(5.E-6, "-l 5 -d 3");
  test_ic<double>(5.E-7, "-l 6 -d 3");
#endif
#ifdef ASGARD_ENABLE_FLOAT
  test_ic<float>(1.E-3, "-l 6 -d 1");
  test_ic<float>(5.E-2, "-l 5 -d 2");
  test_ic<float>(1.E-4, "-l 5 -d 3");
#endif
}

template<typename P>
void test_sources(double const tol, std::string const &opts)
{
  prog_opts options = make_opts(opts);

  pde_domain<P> domain({{-0.5 * PI, 0.5 * PI}, {0, 1}});

  options.default_degree = 1;
  options.default_stop_time = 0.5;
  options.default_start_levels = {5, };

  options.default_step_method = time_method::rk2;
  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);

  options.default_dt = 0.1 * 0.5 * dx;

  // function that will build the sources and initial conditions
  auto ft  = [](P t) -> P { return  std::exp(-t); };
  auto fdt = [](P t) -> P { return -std::exp(-t); };
  auto fx  = [](P x) -> P { return std::sin(x); };
  auto fdx = [](P x) -> P { return std::cos(x); };
  auto fy  = [](P y) -> P { return std::cos(y); };

  separable_func<P> exact({vectorize_t<P>(fx), vectorize_t<P>(fy)}, ft);
  separable_func<P> s0({vectorize_t<P>(fx), vectorize_t<P>(fy)}, fdt);
  separable_func<P> s1({vectorize_t<P>(fdx), vectorize_t<P>(fy)}, ft);

  separable_func<P> bc = exact;
  bc.set(0, P{-1});

  auto smd = [=](P t, vector2d<P> const &nodes, std::vector<P> &vals) ->
    void {
      for (auto i : indexof(vals)) {
        P const x = nodes[i][0];
        P const y = nodes[i][1];
        vals[i] = fdt(t) * fx(x) * fy(y) + ft(t) * fdx(x) * fy(y);
      }
    };

  // separable and interpolation odes
  pde_scheme<P> spde(options, domain);
  pde_scheme<P> ipde(options, domain);

  term_md<P> div = {term_div<P>(1, flux_type::upwind, boundary_type::left), term_identity{}};
  div += left_boundary_flux<P>(bc);

  spde += div;
  ipde += div;

  spde.add_initial(exact);
  ipde.add_initial(exact);

  spde.add_source(s0);
  spde.add_source(s1);
  ipde.set_source(smd);

  discretization_manager<P> sdisc(spde, verbosity_level::quiet);
  discretization_manager<P> idisc(ipde, verbosity_level::quiet);

  sdisc.advance_time();
  idisc.advance_time();

  auto const &sstate = sdisc.current_state_mpi();
  auto const &istate = idisc.current_state_mpi();

  double err = 0;
  double total = 0;
  for (auto i : indexof(sstate)) {
    P const e = sstate[i] - istate[i];
    err += e * e;
    total += sstate[i] * sstate[i];
  }

  // std::cout << " err = " << std::sqrt(err) / std::sqrt(total) << "\n";
  tcheckless(0, std::sqrt(err) / std::sqrt(total), tol);
}

void test_sources() {
  current_test name_("source terms");
#ifdef ASGARD_ENABLE_DOUBLE
  test_sources<double>(5.E-3, "-l 4 -d 1");
  test_sources<double>(1.E-3, "-l 5 -d 1");
  test_sources<double>(5.E-4, "-l 6 -d 1");
  test_sources<double>(5.E-3, "-l 4 -d 2");
  test_sources<double>(1.E-3, "-l 5 -d 2");
  test_sources<double>(1.E-7, "-l 5 -d 3");
  test_sources<double>(5.E-8, "-l 6 -d 3");
#endif
#ifdef ASGARD_ENABLE_FLOAT
  test_sources<float>(1.E-3, "-l 5 -d 1");
  test_sources<float>(1.E-4, "-l 5 -d 3");
#endif
}

template<typename P>
void test_pde(double const tol, std::string const &opts)
{
  prog_opts options = make_opts(opts);

  pde_domain<P> domain({{-0.125 * PI, 0.125 * PI}, {0, 0.25 * PI}});

  options.default_degree = 1;
  options.default_stop_time = 0.1;
  options.default_start_levels = {5, };

  options.default_step_method = time_method::rk2;
  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);

  options.default_dt = 0.1 * 0.5 * dx;

  options.outfile = "itest.h5";

  // function that will build the sources and initial conditions
  auto ft  = [](P t) -> P { return  std::exp(-t); };
  auto fdt = [](P t) -> P { return -std::exp(-t); };
  auto fx  = [](P x) -> P { return std::sin(x); };
  auto fy  = [](P y) -> P { return  std::cos(y); };
  auto fdy = [](P y) -> P { return -std::sin(y); };

  auto flbc = [](P x) -> P { return std::sin(x) * std::cos(x); };

  separable_func<P> exact({vectorize_t<P>(fx), vectorize_t<P>(fy)}, ft);
  separable_func<P> s0({vectorize_t<P>(fx), vectorize_t<P>(fy)}, fdt);

  separable_func<P> bcL = exact;
  bcL.set(1, P{1});
  bcL.set(0, vectorize_t<P>(flbc));

  // non-separable source
  auto smd = [=](P t, vector2d<P> const &nodes, std::vector<P> &vals) ->
    void {
      for (auto i : indexof(vals)) {
        P const x = nodes[i][0];
        P const y = nodes[i][1];
        vals[i] = -std::sin(x + y) * ft(t) * fx(x) * fy(y)
                  +std::cos(x + y) * ft(t) * fx(x) * fdy(y);
      }
    };

  // non-separable coefficient
  auto cmd = [=](P, vector2d<P> const &nodes, std::vector<P> const &f, std::vector<P> &vals) ->
    void {
      for (auto i : indexof(vals)) {
        P const x = nodes[i][0];
        P const y = nodes[i][1];
        vals[i] = std::cos(x + y) * f[i];
      }
    };

  // separable and interpolation odes
  pde_scheme<P> pde(options, domain);

  term_md<P> div = {term_identity{}, term_div<P>(1, flux_type::upwind, boundary_type::left)};
  div += left_boundary_flux<P>(bcL);

  term_md<P> coeff = term_interp<P>(cmd);

  pde += {div, coeff};

  pde.add_initial(exact);

  pde.add_source(s0);  // separable
  pde.set_source(smd); // non-separable

  discretization_manager<P> disc(pde, verbosity_level::quiet);

  disc.advance_time();

  auto const &state = disc.current_state_mpi();

  std::vector<P> eref = disc.project_function(disc.initial_cond_sep());

  double const tval  = ft(disc.time());
  double const enorm = 3.914569110545039e-02 * 0.642699081698724 * tval * tval;

  double nself = 0;
  double ndiff = 0;
  for (size_t i = 0; i < state.size(); i++)
  {
    double const e = eref[i] - state[i];
    ndiff += e * e;
    double const r = eref[i];
    nself += r * r;
  }

  double err = std::sqrt((ndiff + std::abs(enorm - nself)) / enorm);

  // for (auto i : indexof(state))
  //   eref[i] -= state[i];
  // disc.add_aux_field({"diff", eref});
  // disc.save_final_snapshot();

  tcheckless(0, err, tol);
}

void test_pde() {
  current_test name_("non-separable pde");
#ifdef ASGARD_ENABLE_DOUBLE
  test_pde<double>(1.E-3, "-l 4 -d 1");
  test_pde<double>(5.E-4, "-l 5 -d 1");
  test_pde<double>(5.E-6, "-l 5 -d 3");
#endif
#ifdef ASGARD_ENABLE_FLOAT
  test_pde<float>(5.E-3, "-l 5 -d 1");
  test_pde<float>(5.E-3, "-l 5 -d 3");
#endif
}

template<typename P>
void test_interp_mass(double const tol, std::string const &opts)
{
  prog_opts options = make_opts(opts);
  // builds a separable PDE is both separable and interpolatory way
  // compares the results from the two simulations
  pde_domain<P> domain({{0, 1}, {0, PI}});

  auto jac_x = [](P x) -> P { return std::cos(x - P{0.5}); };
  auto jac_y = [](P y) -> P { return std::exp(2*y); };
  auto eta_x = [](P x) -> P { return x * (P{1} - x) + P{1}; };

  auto massx = [=](std::vector<P> const &x, std::vector<P> &fx)
        -> void {
      #pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = jac_x(x[i]);
    };
  auto massy = [=](std::vector<P> const &y, std::vector<P> &fy)
        -> void {
      #pragma omp parallel for
      for (size_t i = 0; i < y.size(); i++)
        fy[i] = jac_y(y[i]);
    };

  // non-separable coefficient
  auto coeff_md = [=](P, vector2d<P> const &nodes, std::vector<P> const &f, std::vector<P> &vals)
      -> void {
      #pragma omp parallel for
      for (size_t i = 0; i < f.size(); i++) {
        P const x = nodes[i][0];
        vals[i] = eta_x(x) * jac_x(x) * f[i];
      }
    };
  // same but separable coefficient
  auto coeffx = [=](std::vector<P> const &x, std::vector<P> &fx)
        -> void {
      #pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = eta_x(x[i]) * jac_x(x[i]);
    };

  auto coeffy = [=](std::vector<P> const &y, std::vector<P> &fy)
        -> void {
      #pragma omp parallel for
      for (size_t i = 0; i < y.size(); i++)
        fy[i] = jac_y(y[i]);
    };

  options.default_degree = 2;
  options.default_start_levels = {5, };
  options.default_stop_time = 0.25;
  options.default_step_method = asgard::time_method::rk2;

  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);
  options.default_dt = 0.5 * 0.1 * dx;

  pde_scheme<P> ipde(options, domain); // interpolatory PDE
  pde_scheme<P> spde(options, domain); // separable PDE

  ipde.set_mass({term_volume<P>{massx}, term_volume<P>{massy}});
  spde.set_mass({term_volume<P>{massx}, term_volume<P>{massy}});

  term_md<P> divx = {term_div<P>(1, asgard::boundary_type::periodic), term_identity{}};
  term_md<P> divy = {term_identity{}, term_div<P>(coeffy, asgard::boundary_type::periodic)};
  term_md<P> divx_coeff = {term_div<P>(coeffx, asgard::boundary_type::periodic), term_identity{}};
  term_md<P> coeff = term_interp<P>(coeff_md);

  ipde += term_md<P>{divx, coeff};
  ipde += divy;

  spde += divx_coeff;
  spde += divy;

  auto exact_x = [=](std::vector<P> const &x, P /*time*/, std::vector<P> &fx) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++)
      fx[i] = jac_x(x[i]) * std::sin(PI * x[i]);
  };
  auto exact_y = [=](std::vector<P> const &y, P /*time*/, std::vector<P> &fy) {
    #pragma omp parallel for
    for (size_t i = 0; i < y.size(); i++)
      fy[i] = jac_y(y[i]) * std::sin(y[i]);
  };

  asgard::separable_func<P> exact({exact_x, exact_y});

  ipde.add_initial(exact);
  spde.add_initial(exact);

  discretization_manager<P> idisc(ipde, verbosity_level::quiet);
  discretization_manager<P> sdisc(spde, verbosity_level::quiet);

  idisc.advance_time();
  sdisc.advance_time();

  // idisc.save_final_snapshot();
  // sdisc.save_final_snapshot();

  tassert(sdisc.current_state().size() == idisc.current_state().size());

  std::vector<P> diff = sdisc.current_state();
  for (size_t i = 0; i < diff.size(); i++)
    diff[i] -= idisc.current_state()[i];

  double err = sdisc.normL2(diff);
  // std::cout << err << "\n";
  tcheckless(0, err, tol);
}

void test_interp_mass() {
  current_test name_("compare separable and non-separable pdes");
#ifdef ASGARD_ENABLE_DOUBLE
  test_interp_mass<double>(5.E-2, "-l 4 -d 1");
  test_interp_mass<double>(1.E-2, "-l 5 -d 1");
  test_interp_mass<double>(5.E-3, "-l 4 -d 2");
  test_interp_mass<double>(1.E-3, "-l 5 -d 2");
  test_interp_mass<double>(1.E-3, "-l 4 -d 3");
  test_interp_mass<double>(5.E-4, "-l 5 -d 3");
#endif
#ifdef ASGARD_ENABLE_FLOAT
  test_interp_mass<float>(5.E-2, "-l 4 -d 1");
  test_interp_mass<float>(1.E-3, "-l 5 -d 2");
  test_interp_mass<float>(5.E-4, "-l 5 -d 3");
#endif
}

int main(int argc, char **argv)
{
  libasgard_runtime running_(argc, argv);

  all_tests global_("interpolation operators");

  test_ode1d();
  test_ic();
  test_sources();
  test_pde();
  test_interp_mass();

  return 0;
}
