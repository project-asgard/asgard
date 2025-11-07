#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

using P = asgard::default_precision;

template<typename P>
pde_scheme<P> make_robin_pde(prog_opts options)
{
  pde_domain<P> domain({{-1.0, 1.0}, });

  options.default_degree = 2;
  options.default_start_levels = {4, };

  options.default_solver = solver_method::bicgstab;
  options.default_solver = solver_method::direct;
  options.default_isolver_tolerance = 1.E-7;
  options.default_isolver_iterations = 1000;

  options.default_precon = precon_method::jacobi;
  options.default_step_method = time_method::steady;

  pde_scheme<P> pde(options, domain);

  auto exact_1d = [](std::vector<P> const &x, P, std::vector<P> &f)
      -> void
    {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < x.size(); i++)
        f[i] = std::cos(x[i]);
    };

  auto source_1d = [](std::vector<P> const &x, P, std::vector<P> &f)
      -> void
    {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < x.size(); i++)
        f[i] = std::cos(x[i]);
    };

  separable_func<P> exact({exact_1d,});
  separable_func<P> source({source_1d,});

  pde.add_source(source);

  term_1d<P> div = term_div{-1, boundary_type::bothsides};
  // term_1d<P> grad = term_grad{1};
  term_1d<P> grad = term_grad{1, boundary_type::none};
  term_1d<P> dxx(std::vector<term_1d<P>>{div, grad});
  // dxx.set_penalty(P{1} / P{pde.cell_size(0)});

  // term_md<P> laplacian({dxx, });
  // laplacian += left_boundary_flux<P>{std::vector<P>{std::cos(-1.0), }};
  // laplacian += right_boundary_flux<P>{std::vector<P>{std::cos(1.0), }};
  // pde += laplacian;

  //term_1d<P> robin = term_robin{-std::sin(-1.0) / std::cos(-1.0), std::sin(1.0) / std::cos(1.0)};
  // term_1d<P> robin = term_robin{-100* std::sin(-1.0) / std::cos(-1.0), 0.0};
  // term_1d<P> robin = term_robin{-0.1 * 0.25 * std::cos(-1.0) / std::sin(-1.0), 0.0};
  // term_1d<P> robin = term_robin{-std::sin(-1.0) / std::cos(-1.0), 0.0};
  // term_1d<P> robin = term_robin{0.0, 1.0};

  dxx.set_left_robin(std::sin(-1.0) / std::cos(-1.0));
  dxx.set_right_robin(std::sin(1.0) / std::cos(1.0));

  pde += term_md{{dxx, }};

  // du/dx + gamma * u = 0,  u = cos(x), du/dx = -sin(x) -> gamma = sin(x) / cos(x)

  term_1d<P> pen = term_penalty<P>{1.0 / pde.cell_size(0), boundary_type::none};
  pde += term_md{{pen, }};

  //term_1d<P> robin = term_robin{std::sin(-1.0) / std::cos(-1.0), std::sin(1.0) / std::cos(1.0)};
  //pde += term_md{{robin, }};

  // using 0 as the initial conditions

  return pde;
};

int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);

  prog_opts options(argc, argv);

  auto pde = make_robin_pde<P>(options);

  asgard::discretization_manager<P> disc(pde, asgard::verbosity_level::high);

  disc.advance_time();

  disc.final_output();

  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
