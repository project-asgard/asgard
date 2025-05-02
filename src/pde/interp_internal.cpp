#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

template<typename P>
void test_ode1d(double const tol, std::string const &opts)
{
  prog_opts options = make_opts(opts);

  pde_domain<P> domain({{0, 2}, });

  options.default_degree = 1;
  options.default_stop_time = 0.01;
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
      std::cout << " ----------------------- \n";
      for (size_t i = 0; i < f.size(); i++) {
        vals[i] = 2 * f[i];
        std::cout << " vals = " << vals[i] << "  " << f[i] << "\n";
      }
    });

  separable_func<P> ic(std::vector<P>{1, });
  sode.add_initial(ic);
  iode.add_initial(ic);

  discretization_manager<P> sdisc(sode, verbosity_level::quiet);
  discretization_manager<P> idisc(iode, verbosity_level::quiet);

  for (auto i : indexof(sdisc.current_state())) {
    std::cout << sdisc.current_state()[i] << "   " << idisc.current_state()[i] << "\n";
  }
  std::cout << " ========================== \n";

  sdisc.advance_time();
  idisc.advance_time();

  auto const &sstate = sdisc.current_state();
  auto const &istate = idisc.current_state();

  double err = 0;
  for (auto i : indexof(sstate)) {
    P const e = sstate[i] - istate[i];
    std::cout << sstate[i] << "   " << istate[i] << "\n";
    err += e * e;
  }

  std::cout << " err = " << std::sqrt(err) << "\n";

  //tcheckless(0, std::sqrt(err), tol);
}

void test_ode1d() {
#ifdef ASGARD_ENABLE_DOUBLE
  test_ode1d<double>(1.E-3, "-l 1 -d 3");
  // test_ode1d<double>(1.E-3, "-l 6 -d 3");
  // test_ode1d<double>(1.E-3, "-l 7 -d 3");
  // test_ode1d<double>(1.E-3, "-l 8 -d 3");
#endif
#ifdef ASGARD_ENABLE_FLOAT
  //test_ode1d<float>(1.E-3, "-l 5");
#endif
}

int main(int, char**)
{
  test_ode1d();
}
