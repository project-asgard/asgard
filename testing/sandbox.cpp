#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

using prec = asgard::default_precision;

template<typename P>
double test_interp_mass(prog_opts options)
{
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
        vals[i] = eta_x(x) * f[i];
      }
    };
  // same but separable coefficient
  auto coeffx = [=](std::vector<P> const &x, std::vector<P> &fx)
        -> void {
      #pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = eta_x(x[i]);
    };

  options.default_degree = 2;
  options.default_start_levels = {5, };
  options.default_stop_time = 0.5;
  options.default_step_method = asgard::time_method::rk2;

  int const max_level = options.max_level();
  P const dx = domain.min_cell_size(max_level);
  options.default_dt = 0.5 * 0.1 * dx;

  pde_scheme<P> ipde(options, domain); // interpolatory PDE
  pde_scheme<P> spde(options, domain); // separable PDE

  ipde.set_mass({term_volume<P>{massx}, term_volume<P>{massy}});
  spde.set_mass({term_volume<P>{massx}, term_volume<P>{massy}});

  term_md<P> divx = {term_div<P>(1, asgard::boundary_type::periodic), term_identity{}};
  term_md<P> divy = {term_identity{}, term_div<P>(1, asgard::boundary_type::periodic)};
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

  double err = 0;
  for (size_t i = 0; i < idisc.current_state().size(); i++) {
    P const e = (idisc.current_state()[i] - sdisc.current_state()[i]);
    err += e * e;
  }
  return std::sqrt(err);
}

int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  libasgard_runtime running_(argc, argv);

  prog_opts options(argc, argv);

  std::cout << " error = " << test_interp_mass<double>(options) << '\n';


  return 0;
}
