#include "asgard.h" // alias for asgard.hpp

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file burgers.cpp
 * \brief Burgers' equation
 * \author The ASGarD Team
 *
 * Simple example of non-linear Burgers equation.
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_elliptic Example: Burgers' non-linear equation
 *
 * \par Burgers' equation
 * The Burger's equation is generally defined as
 * \f[ \frac{d}{d t} f + f \cdot \nabla f = \nu \Delta f \f]
 * the formulation used here is the equivalent
 * \f[ \frac{d}{d t} f + \nabla \cdot f^2 - \nu \nabla \cdot \nabla f = 0 \f]
 * the domain is arbitrarily chosen as [-8, 8] in all directions.
 *
 * \par
 * This examples shows how to set a PDE with non-linear and non-separable coefficients.
 */

/*!
 * \ingroup asgard_examples_burgers
 * \brief Make an elliptic PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam boudnary indicates the type of boundary to use
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param num_dims number of dimensions
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \b Note: The asgard namespace includes the name \b boundary_type,
 * it a natural name but it is possible to create a conflict if the entire namespace
 * is included.
 *
 * \snippet elliptic.cpp elliptic make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_burgers_pde(int num_dims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic make]
#endif
  rassert(1 <= num_dims and num_dims <= 3, "invalid number of dimensions, use 1 - 3");

  options.title = "Burgers PDE " + std::to_string(num_dims) + "D";

  std::optional<P> const cli_nu = options.extra_cli_value_group<P>({"-nu", });
  if (not cli_nu and options.is_mpi_rank_zero())
    std::cout << "no '-nu' provided, defaulting to inviscit Burgers '-nu 0'\n";

  P const nu = cli_nu.value_or(0);

  if (nu < 0)
    throw std::runtime_error("the viscosity coefficient '-nu' should be non-negative");

  // the 1D case is set on (-8, 8), the higher dimensions use (-1, 1)^d
  asgard::pde_domain<P> domain = (num_dims == 1)
    ? asgard::pde_domain<P>(std::vector<asgard::domain_range>(1, {-8.0, 8.0}))
    : asgard::pde_domain<P>(std::vector<asgard::domain_range>(num_dims, {-1.0, 1.0}));

  options.default_degree = 3;
  options.default_start_levels = {4, };

  // the inviscit equation can be done with an explicit time stepper
  if (nu == 0)
    options.default_step_method = asgard::time_method::rk2;
  else
    options.default_step_method = asgard::time_method::imex1;

  P const dx = domain.min_cell_size(options.max_level());
  options.default_dt = 0.1 * dx;
  options.default_stop_time = 0.5;

  if (options.max_level() > 5)
    options.default_solver = asgard::solver_method::bicgstab;
  else
    options.default_solver = asgard::solver_method::direct;

  // defaults for iterative solvers, not necessarily optimal
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;

  asgard::pde_scheme<P> pde(options, std::move(domain));

  // non-separable coefficient
  auto f2 = [=](P, asgard::vector2d<P> const &,
                std::vector<P> const &f, std::vector<P> &vals) ->
    void {
      // ignore the first input, it is time but it is not implemented yet
      // the coefficient function must return values at specific points
      // the number of points is f.size() and f contains the values
      // of the current solution at the corresponding points
      // in the case Burger's equation, the coefficient values depend only
      // on f, but in a general case the nodes can be needed too
      // asgard::vector2d<P> const &nodes provides a 2D organization of data,
      // so that the nodes of i-th point are
      // nodes[i][0], ..., nodes[i][num_dims - 1] corresponding to x1, x2, ..., xd
      // e.g., x1 = nodes[i][0], x2 = nodes[i][1] ...
      for (size_t i = 0; i < f.size(); i++) {
        vals[i] = f[i] * f[i];
      }
    };

  asgard::term_1d<P> div_grad;

  if (nu > 0) {
    div_grad = std::vector<asgard::term_1d<P>>{
        asgard::term_div{-std::sqrt(nu), asgard::boundary_type::none},
        asgard::term_grad{std::sqrt(nu), asgard::boundary_type::bothsides},
      };

    div_grad.set_penalty(1 / dx);
  }

  asgard::term_md<P> term_f2 = asgard::term_interp<P>{f2};

  if (num_dims == 1)
  {
    // example from Wikipedia https://en.wikipedia.org/wiki/Burgers%27_equation
    // the 1D inviscit case uses initial condition std::exp(- 0.5 * x * x)
    // the viscous case uses two exponentials

    // the derivative term for d/dx f^2
    asgard::term_md<P> div = {asgard::term_div{0.5, asgard::boundary_type::bothsides}, };

    // set the initial conditions, will be used to set the boundary conditions too
    auto ic = (nu > 0) ? [](P x)
        -> P {
        return std::exp(-P{0.5} * (x - 1) * (x - 1)) - std::exp(-P{0.5} * (x + 1) * (x + 1));
      }
      : [](P x)
        -> P {
        return std::exp(-P{0.5} * x * x);
      };

    // the boundary conditions for d/dx are set on f^2, thus the value is square that of f
    {
      P const val = ic(pde.domain().xleft(0));
      asgard::separable_func<P> fl(std::vector<P>{val * val, });
      div += asgard::left_boundary_flux{fl};
    }{
      P const val = ic(pde.domain().xright(0));
      asgard::separable_func<P> fr(std::vector<P>{val * val, });
      div += asgard::right_boundary_flux{fr};
    }

    // the group ids are needed for IMEX scheme in the viscous way
    int const non_linear_group_id = pde.new_term_group();

    pde += asgard::term_md<P>{div, term_f2};

    if (nu > 0) {
      // the viscous mode for the right-hand-side second derivative
      asgard::term_md<P> dg = {div_grad, };

      asgard::separable_func<P> fl(std::vector<P>{ic(pde.domain().xleft(0)), });
      asgard::separable_func<P> fr(std::vector<P>{ic(pde.domain().xright(0)), });

      dg += asgard::left_boundary_flux{fl};
      dg += asgard::right_boundary_flux{fr};

      int const laplacian_group_id = pde.new_term_group();
      pde += dg;

      pde.set(asgard::imex_implicit_group{laplacian_group_id},
              asgard::imex_explicit_group{non_linear_group_id});
    }

    // the vector version of the initial conditions
    auto ic_vec = [=](std::vector<P> const &x, P, std::vector<P> &fx)
      -> void {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = ic(x[i]);
      };

    pde.add_initial(asgard::separable_func({ic_vec, }));

    return pde;
  }

  if (num_dims == 2) {
    std::cout << " num_dims = " << num_dims << '\n';
    // derivative terms
    auto icx   = [](P x) -> P { return P{1} + P{0.75} * x - P{0.25} * x * x; };
    auto icdx  = [](P x) -> P { return P{0.75} - P{0.5} * x; };
    auto icdxx = [](P x) -> P { return - P{0.5} * x; };
    auto icy   = [](P y) -> P { return (P{1} - y * y); };
    auto icdy  = [](P y) -> P { return -2 * y; };
    auto icdyy = [](P) -> P { return -2; };

    asgard::term_md<P> divx = {asgard::term_div{0.5, asgard::boundary_type::left},
                               asgard::term_identity{}};
    asgard::term_md<P> divy = {asgard::term_identity{},
                               asgard::term_div{0.5, asgard::boundary_type::bothsides}, };

    // the group ids are needed for IMEX scheme in the viscous way
    int const non_linear_group_id = pde.new_term_group();

    pde += asgard::term_md<P>{divx, term_f2};
    pde += asgard::term_md<P>{divy, term_f2};

    if (nu > 0) {

      auto smd = [=](P t, asgard::vector2d<P> const &nodes, std::vector<P> &vals) ->
        void {
          for (int64_t i = 0; i < nodes.num_strips(); i++) {
            P const x = nodes[i][0];
            P const y = nodes[i][1];
            vals[i] = std::exp(-t) * (-icx(x) * icy(y) + icx(x) * icdx(x) * icy(y) * icy(y)
                                      + icx(x) * icx(x) * icy(y) * icdy(y)
                                      - nu * icdxx(x) * icy(y) - nu * icx(x) * icdyy(y));
          }
        };

      pde.set_source(smd);

      int const laplacian_group_id = pde.new_term_group();
      pde += {div_grad, asgard::term_identity{}};
      pde += {asgard::term_identity{}, div_grad};

      pde.set(asgard::imex_implicit_group{laplacian_group_id},
              asgard::imex_explicit_group{non_linear_group_id});

    } else {

      auto smd = [=](P t, asgard::vector2d<P> const &nodes, std::vector<P> &vals) ->
        void {
          for (int64_t i = 0; i < nodes.num_strips(); i++) {
            P const x = nodes[i][0];
            P const y = nodes[i][1];
            vals[i] = std::exp(-t) * (-icx(x) * icy(y) + icx(x) * icdx(x) * icy(y) * icy(y)
                                      + icx(x) * icx(x) * icy(y) * icdy(y));
          }
        };

      pde.set_source(smd);
    }

    return pde;
  }

  return asgard::pde_scheme<P>();
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic make]
#endif
}

void self_test();

int main(int argc, char **argv) {

  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this file and the two additional options accepted for this problem
  if (options.show_help) {
    std::cout << "\n solves the continuity equation:\n";
    std::cout << "    f_t + div f^2 = f_xx + s(t, x)\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-dims            -dm     int        accepts: 1 - 3\n";
    std::cout << "                                    the number of dimensions\n\n";
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", }, {"-dims", "-dm", "-nu"});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  int const num_dims = options.extra_cli_value_group<int>({"-dims", "-dm"}).value_or(1);

  // creating a discretization manager
  asgard::discretization_manager<P> disc(make_burgers_pde<P>(num_dims, options),
                                         asgard::verbosity_level::high);

  disc.advance_time();

  if (not disc.stop_verbosity())
    disc.progress_report();

  disc.save_final_snapshot();

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';

  return 0;
}

void self_test() {
  //
}
