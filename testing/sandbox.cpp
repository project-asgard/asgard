#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

using P = asgard::default_precision;

template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_elliptic_pde(int num_dims, asgard::prog_opts options) {
  rassert(1 <= num_dims and num_dims <= 6, "invalid number of dimensions");

  options.title = "Elliptic PDE " + std::to_string(num_dims) + "D";

  asgard::pde_domain<P> domain(std::vector<asgard::domain_range>(num_dims, {0, 1}));

  options.default_degree = 1;
  options.default_start_levels = {4, };

  // previous examples were setting a default stepping method
  // which allows the cli options to overwrite the selection
  // here, we are overwriting the cli selection, if another
  // method was requested then a warning will be generated
  // (this should probably be an error instead of a warning)
  options.force_step_method(asgard::time_method::steady);

  // OK for small problems, larger one should switch to gmres or bicgstab
  options.default_solver = asgard::solver_method::gmres;

  // defaults for iterative solvers, not necessarily optimal
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 100;

  options.default_isolver_inner_iterations = 100;

  options.default_precon = asgard::precon_method::jacobi;

  asgard::pde_scheme<P> pde(options, std::move(domain));

  // s1d is the exact solution in 1d
  auto s1d = [](std::vector<P> const &x, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = x[i] * (P{2} - x[i]);
    };

  // "exact" is the solution in multiple dimensions
  asgard::separable_func<P> exact(std::vector<asgard::sfixed_func1d<P>>(num_dims, s1d));


    // fixed boundary set to the div term corresponds to Neumann boundary
    asgard::term_1d<P> div = asgard::term_div<P>(-1, asgard::flux_type::upwind,
                                                 asgard::boundary_type::right);
    // fixed boundary set to the grad term corresponds to Dirichlet boundary
    asgard::term_1d<P> grad = asgard::term_grad<P>(1, asgard::flux_type::upwind,
                                                   asgard::boundary_type::left);

  auto coeff = [=](P, asgard::vector2d<P> const &,
                 std::vector<P> const &f, std::vector<P> &vals) ->
    void {
      vals = f;
      // ignore the first input, it is time but it is not implemented yet
      // for (size_t i = 0; i < f.size(); i++) {
      //   P const x = nodes[i][0];
      //   P const y = nodes[i][1];
      //   P const z = nodes[i][2];
      //
      //   vals[i] = f[i];
      // }
    };

  enum class mode { merged, chained, interp };

  mode constexpr mm = mode::interp;

  if constexpr (mm == mode::merged)
  {
    // the multi-dimensional operator, initially set to identity in md
    std::vector<asgard::term_1d<P>> ops(num_dims);
    for (int d = 0; d < num_dims; d++)
    {
      // combine the div and grad into a single chain term
      asgard::term_1d<P> fxx({div, grad});

      // based on the domain and max-level, get the cell-size in direction d
      P const dx = pde.cell_size(asgard::dimension_id{d});

      // adding penalty to stabilize the steady state equation
      // the penalty is applied only to discontinuities, if the solution is continuous
      // then the penalty will not alter the result, this only improves the conditioning
      fxx.set_penalty(P{1} / dx);

      // add the second order operator in dimension dim
      ops[d] = fxx;
      pde += asgard::term_md<P>(ops);
      ops[d] = asgard::term_identity{};
    }

  } else if constexpr (mm == mode::chained) {

    for (int d = 0; d < num_dims; d++)
    {
      std::vector<asgard::term_1d<P>> div_md(num_dims);
      std::vector<asgard::term_1d<P>> grad_md(num_dims);
      std::vector<asgard::term_1d<P>> pen_md(num_dims);

      P const dx = pde.cell_size(asgard::dimension_id{d});

      div_md[d]  = div;
      grad_md[d] = grad;
      pen_md[d]  = asgard::term_penalty(P{1} / dx);

      //asgard::term_md<P> div_grad = {div_md, grad_md};
      pde += asgard::term_md<P>(std::vector<asgard::term_md<P>>{div_md, grad_md});
      pde += pen_md;
    }

  } else if constexpr (mm == mode::interp) {

    for (int d = 0; d < num_dims; d++)
    {
      std::vector<asgard::term_1d<P>> div_md(num_dims);
      std::vector<asgard::term_1d<P>> grad_md(num_dims);
      std::vector<asgard::term_1d<P>> pen_md(num_dims);

      P const dx = pde.cell_size(asgard::dimension_id{d});

      div_md[d]  = div;
      grad_md[d] = grad;
      pen_md[d]  = asgard::term_penalty(P{1} / dx);

      //asgard::term_md<P> div_grad = {div_md, grad_md};
      pde += asgard::term_md<P>(std::vector<asgard::term_md<P>>{
              div_md, asgard::term_interp<P>{coeff}, grad_md
      });
      pde += pen_md;
    }

  }


  for (int d = 0; d < num_dims; d++) {
    // using separability properties, copy over the exact solution
    asgard::separable_func<P> src = exact;
    // differentiate in the d-th direction, i.e., replace the function
    // with a constant 2
    src.set(asgard::dimension_id{d}, 2);

    pde.add_source(std::move(src));
  }

  // if an initial condition is specified, it will be used as the initial guess
  // of an iterative solver, other zeros is used as the initial guess
  // the direct solver does not use an initial guess

  return pde;
}

template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc)
{
  int const num_dims = disc.num_dims();

  // see the continuity example for the orthogonality trick

  // construct the exact solution, since there is no initial condition
  auto s1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = x[i] * (P{2} - x[i]);
    };

  // set the right-hand-side for each dimension
  std::vector<asgard::svector_func1d<P>> func(num_dims, s1d);

  std::vector<P> const eref = disc.project_function(asgard::separable_func<P>(func));

  double constexpr space1d = 8.0 / 15.0; // integral of (2x - x^2)^2 over (0, 1)

  // this is the L^2 norm-squared of the exact solution
  double const enorm = asgard::fm::powi(space1d, num_dims);

  disc.sync_mpi_state(); // is using multiple ranks, sync across the ranks
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

  return std::sqrt((ndiff + std::abs(enorm - nself)) / enorm);
}

int main(int argc, char **argv)
{
  std::ignore = argc;
  std::ignore = argv;

  asgard::prog_opts options(argc, argv);

  options.throw_if_argv_not_in({"-test", }, {"-dims", "-dm", "-bound", "-bc"});

  if (options.has_cli_entry("-test")) {
    std::cerr << " NO-TEST\n";
    return 0;
  }

  // setting the dimensions
  std::optional<int> cli_dims = options.extra_cli_value_group<int>({"-dims", "-dm"});
  int const num_dims = cli_dims.value_or(2);

  if (options.is_mpi_rank_zero()) {
    if (not cli_dims) {
      std::cout << "setting default 2D problem\n";
    } else {
      std::cout << "setting " << num_dims << "D problem\n";
    }
  }

  auto pde = make_elliptic_pde<P>(num_dims, options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  P const err = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- steady state error: " << err << '\n';

  return 0;


  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
