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

  auto const &sstate = sdisc.current_state();
  auto const &istate = idisc.current_state();

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

  auto const &sstate = sdisc.current_state();
  auto const &istate = idisc.current_state();

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

int main(int, char**)
{
  all_tests global_("interpolation operators");

  // test_ode1d();
  test_ic();
}
