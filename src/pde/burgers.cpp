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
 * \addtogroup asgard_examples_burgers Example: Burgers' non-linear equation
 *
 * \par Burgers' equation
 * The Burger's equation is generally defined as
 * \f[ \frac{d}{d t} f + f \cdot \nabla f = \nu \Delta f \f]
 * the formulation used here is the equivalent
 * \f[ \frac{d}{d t} f + \frac{1}{2} \nabla \cdot f^2 - \nu \nabla \cdot \nabla f = 0 \f]
 * This file implements several different versions of this equation.
 *
 * \par
 * - 1D inviscit (nu == 0) case, with bell shaped curve as initial conditions
 * - 1D viscous case (nu > 0) borrowed from the [Wikipedia page](https://en.wikipedia.org/wiki/Burgers%27_equation)
 * - 2D viscous and inviscit cases with known exact solution
 *
 * \par
 * Adding viscosity leads to a more stable problem,
 * but the higher condition number of the second order derivative
 * requires an implicit solver.
 * Thus, the viscous case uses an IMEX stepping scheme while the inviscit case
 * is done explicitly.
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
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param num_dims number of dimensions
 * \param options is the set of options
 *
 * \returns a pde_scheme<P> set for the Burgers equation
 *
 * \snippet burgers.cpp burgers make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_burgers_pde(int num_dims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers make]
#endif
  rassert(1 <= num_dims and num_dims <= 2, "invalid number of dimensions, use 1 or 2");

  options.title = "Burgers PDE " + std::to_string(num_dims) + "D";

  std::optional<P> const cli_nu = options.extra_cli_value_group<P>({"-nu", });
  if (not cli_nu and options.is_mpi_rank_zero())
    std::cout << "no '-nu' provided, defaulting to inviscit Burgers '-nu 0'\n";

  P const nu = cli_nu.value_or(0);

  if (nu < 0)
    throw std::runtime_error("the viscosity coefficient '-nu' should be non-negative");

  if (nu == 0)
    options.title += " (inviscit)";
  else
    options.title += " (viscosity nu = " + std::to_string(nu) + ")";

  // the 1D case is set on (-8, 8), the higher dimensions use (-1, 1)^d
  asgard::pde_domain<P> domain = (num_dims == 1)
    ? asgard::pde_domain<P>(std::vector<asgard::domain_range>(1, {-8.0, 8.0}))
    : asgard::pde_domain<P>(std::vector<asgard::domain_range>(num_dims, {-1.0, 1.0}));

  options.default_degree = 3;
  options.default_start_levels = {6, };

  // the inviscit equation can be done with an explicit time stepper
  if (nu == 0)
    options.default_step_method = asgard::time_method::rk2;
  else
    options.default_step_method = asgard::time_method::imex1;

  P const dx = domain.min_cell_size(options.max_level());
  options.default_dt = 0.05 * dx;
  options.default_stop_time = 0.5;

  options.default_solver = asgard::solver_method::bicgstab;

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
      // see also the source term of the 2D case

      for (size_t i = 0; i < f.size(); i++) {
        vals[i] = f[i] * f[i];
      }
    };

  // setting up multidimensional volume term that uses interpolated coefficient
  asgard::term_md<P> term_f2 = asgard::term_interp<P>{f2};

  if (num_dims == 1)
  {
    // example from Wikipedia https://en.wikipedia.org/wiki/Burgers%27_equation
    // the 1D inviscit case uses initial condition std::exp(- 0.5 * x * x)
    // the viscous case uses two exponentials

    // the derivative term for d/dx f^2
    asgard::term_md<P> div = {asgard::term_div<P>{0.5, asgard::boundary_type::bothsides}, };

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
      asgard::term_1d<P> div_grad = std::vector<asgard::term_1d<P>>{
          asgard::term_div<P>{-std::sqrt(nu), asgard::boundary_type::none},
          asgard::term_grad<P>{std::sqrt(nu), asgard::boundary_type::bothsides},
        };

      div_grad.set_penalty(1 / dx);

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

    pde.add_initial(asgard::separable_func<P>({ic_vec, }));

    return pde;
  }

  if (num_dims == 2) {
    // initial conditions and derivatives in x and y, also the exact solution in time
    auto icx   = [](P x) -> P { return P{1} + P{0.75} * x - P{0.25} * x * x; };
    auto icdx  = [](P x) -> P { return P{0.75} - P{0.5} * x; };
    auto icdxx = [](P) -> P { return - P{0.5}; };

    auto icy   = [](P y) -> P { return (P{1} - y * y); };
    auto icdy  = [](P y) -> P { return -2 * y; };
    auto icdyy = [](P) -> P { return -2; };

    auto exact_t = [](P t) -> P { return std::exp(-t); };

    if (nu == 0) {
      // inviscit mode, using explicit time-stepping and no second order terms
      asgard::term_md<P> divx = {asgard::term_div<P>{0.5, asgard::boundary_type::left},
                                 asgard::term_identity{}};
      asgard::term_md<P> divy = {asgard::term_identity{},
                                 asgard::term_div<P>{0.5, asgard::boundary_type::bothsides}};

      pde += asgard::term_md<P>{divx, term_f2};
      pde += asgard::term_md<P>{divy, term_f2};

      // setting up the non-separable source
      // the term can be split into separable and non-separable components
      // splitting may improve stability but will increase the overall cost
      auto smd = [=](P t, asgard::vector2d<P> const &nodes, std::vector<P> &vals) ->
        void {
          for (int64_t i = 0; i < nodes.num_strips(); i++) {
            P const x = nodes[i][0];
            P const y = nodes[i][1];
            // linear contribution
            vals[i] = -std::exp(-t) * icx(x) * icy(y);
            // non-linear contribution
            vals[i] += std::exp(-t) * std::exp(-t)
                      * (icx(x) * icdx(x) * icy(y) * icy(y) + icx(x) * icx(x) * icy(y) * icdy(y));
          }
        };

      // a term-group can have at most one non-separable source
      // thus we use the "set" method, as opposed to "add"
      pde.set_source(smd);

    } else {
      // boundary conditions in y are homogeneous and simple to impose to all terms
      // boundary conditions in x are imposed only on the second order term
      asgard::term_md<P> divx = {asgard::term_div<P>{0.5, asgard::boundary_type::none},
                                asgard::term_identity{}};
      asgard::term_md<P> divy = {asgard::term_identity{},
                                asgard::term_div<P>{0.5, asgard::boundary_type::bothsides}, };

      // the group ids are needed for IMEX scheme in the viscous way
      int const non_linear_group_id = pde.new_term_group();

      pde += asgard::term_md<P>{divx, term_f2};
      pde += asgard::term_md<P>{divy, term_f2};

      // setting up the non-separable source
      auto smd = [=](P t, asgard::vector2d<P> const &nodes, std::vector<P> &vals) ->
        void {
          for (int64_t i = 0; i < nodes.num_strips(); i++) {
            P const x = nodes[i][0];
            P const y = nodes[i][1];
            // linear contribution
            vals[i] = std::exp(-t)
                     * (-icx(x) * icy(y) - nu * icdxx(x) * icy(y) - nu * icx(x) * icdyy(y));
            // non-linear contribution
            vals[i] += std::exp(-t) * std::exp(-t)
                      * (icx(x) * icdx(x) * icy(y) * icy(y) + icx(x) * icx(x) * icy(y) * icdy(y));
          }
        };

      // setting the non-separable source into the pde_scheme
      pde.set_source(smd);

      // second order term in x
      asgard::term_1d<P> div_grad_x = std::vector<asgard::term_1d<P>>{
          asgard::term_div<P>{-std::sqrt(nu), asgard::boundary_type::none},
          asgard::term_grad<P>{std::sqrt(nu), asgard::boundary_type::bothsides},
        };

      div_grad_x.set_penalty(1 / dx);

      asgard::term_md<P> dgx = {div_grad_x, asgard::term_identity{}};

      // adding inhomogeneous boundary condition on the right
      asgard::separable_func<P> fr(std::vector<P>{icx(pde.domain().xright(0)), 1}, exact_t);
      fr.set(1, [=](std::vector<P> const &y, P, std::vector<P> &fy) ->
                void {
                  for (size_t i = 0; i < y.size(); i++)
                    fy[i] = icy(y[i]);
                });
      dgx += asgard::right_boundary_flux{fr};

      asgard::term_1d<P> div_grad_y = std::vector<asgard::term_1d<P>>{
          asgard::term_div<P>{-std::sqrt(nu), asgard::boundary_type::none},
          asgard::term_grad<P>{std::sqrt(nu), asgard::boundary_type::bothsides},
        };

      div_grad_y.set_penalty(1 / dx);

      asgard::term_md<P> dgy = {asgard::term_identity{}, div_grad_y};

      // adding the second order terms to a new term-group
      int const laplacian_group_id = pde.new_term_group();
      pde += dgx;
      pde += dgy;

      pde.set(asgard::imex_implicit_group{laplacian_group_id},
              asgard::imex_explicit_group{non_linear_group_id});
    }

    // the vector version of the initial conditions
    auto icx_vec = [=](std::vector<P> const &x, P, std::vector<P> &fx)
      -> void {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = icx(x[i]);
      };
    auto icy_vec = [=](std::vector<P> const &y, P, std::vector<P> &fy)
      -> void {
        for (size_t i = 0; i < y.size(); i++)
          fy[i] = icy(y[i]);
      };

    pde.add_initial(asgard::separable_func<P>({icx_vec, icy_vec}, exact_t));

    return pde;
  }

  return asgard::pde_scheme<P>();
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers make]
#endif
}

/*!
 * \ingroup asgard_examples_burgers
 * \brief Computes the L^2 error for the given example
 *
 * The provided discretization_manager should hold a PDE made with
 * make_burgers_pde(). This will compute the L^2 error.
 *
 * \tparam P is double or float, the precision of the manager
 *
 * \param disc is the discretization of a PDE
 *
 * \returns the L^2 error between the known exact solution and
 *          the current state in the \b disc manager
 *
 * \snippet burgers.cpp burgers get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers get-err]
#endif

  int const num_dims = disc.num_dims();

  // l-2 error checking is set for 2D problem only
  if (num_dims != 2)
    return 0;

  // using the fact that the initial condition is the exact solution
  std::vector<P> const eref = disc.project_function(disc.initial_cond_sep());

  double constexpr space = 1984.0 / 900.0;
  double const time_val  = std::exp(-disc.time());

  // this is the L^2 norm-squared of the exact solution
  double const enorm = space * time_val * time_val;

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

  return std::sqrt((ndiff + std::abs(enorm - nself)) / enorm);
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// internal testing, not part of the example
void self_test();
#endif

/*!
 * \ingroup asgard_examples_burgers
 * \brief main() for the continuity example
 *
 * The main() processes the command line arguments and calls both
 * make_burgers_pde() and get_error_l2().
 *
 * \snippet burgers.cpp burgers main
 */
int main(int argc, char **argv) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers main]
#endif

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
    std::cout << "    f_t + div f^2 = nu * f_xx + s(t, x)\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-dims            -dm     int        accepts: 1 - 2\n";
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

  P const err_init = get_error_l2(disc);
  if (num_dims > 1 and not disc.stop_verbosity())
    std::cout << " -- error in the initial conditions: " << err_init << "\n";

  disc.advance_time();

  if (not disc.stop_verbosity())
    disc.progress_report();

  disc.save_final_snapshot();

  P const err_final = get_error_l2(disc);
  if (num_dims > 1 and not disc.stop_verbosity()) {
    disc.progress_report();
    std::cout << " -- final error: " << err_final << "\n";
  }

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';

  return 0;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers main]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////
using namespace asgard;

template<typename P>
void dotest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  discretization_manager<P> disc(make_burgers_pde<P>(num_dims, options),
                                 verbosity_level::quiet);

  while (disc.remaining_steps() > 0)
  {
    disc.advance_time(1);

    double const err = get_error_l2(disc);

    tcheckless(disc.current_step(), err, tol);
  }
}

void self_test() {
  all_tests testing_("Burgers' equation:", " f_t + div f^2 = nu * Laplacian * f + sources");

#ifdef ASGARD_ENABLE_DOUBLE
  dotest<double>(4.E-4, 2, "-l 6 -nu 0.1");
  dotest<double>(8.E-4, 2, "-l 6 -t 0.25 -nu 1");
#endif

#ifndef ASGARD_ENABLE_DOUBLE
  std::cout << "no tests for single precision due to conditioning\n";
#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
