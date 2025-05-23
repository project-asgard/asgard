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

  asgard::pde_domain<P> domain(std::vector<asgard::domain_range>(num_dims, {-8.0, 8.0}));

  options.default_degree = 2;
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

    asgard::term_md<P> div = {asgard::term_div{1, asgard::boundary_type::bothsides}, };

    auto ic = [](P x)
        -> P {
        return std::exp(-P{0.5} * (x - 1) * (x - 1)) - std::exp(-P{0.5} * (x + 1) * (x + 1));
      };


    P val = ic(pde.domain().xleft(0));
    asgard::separable_func<P> fl(std::vector<P>{val * val, });
    div += asgard::left_boundary_flux{fl};

    val = ic(pde.domain().xright(0));
    asgard::separable_func<P> fr(std::vector<P>{val * val, });
    div += asgard::right_boundary_flux{fr};

    pde += asgard::term_md<P>{div, term_f2};

    if (nu > 0) {
      asgard::term_md<P> dg = {div_grad, };

      dg += asgard::left_boundary_flux{fl};
      dg += asgard::right_boundary_flux{fr};
    }

    auto ic_vec = [=](std::vector<P> const &x, P, std::vector<P> &fx)
      -> void {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = ic(x[i]);
      };

    pde.add_initial(asgard::separable_func({ic_vec, }));

    return pde;
  }

  // // s1d is the exact solution in 1d
  // auto s1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
  //   void {
  //     for (size_t i = 0; i < x.size(); i++)
  //       fx[i] = x[i] * (P{2} - x[i]);
  //   };
  //
  // // "exact" is the solution in multiple dimensions
  // asgard::separable_func<P> exact(std::vector<asgard::svector_func1d<P>>(num_dims, s1d),
  //                                 asgard::ignores_time);
  //
  // if constexpr (boundary == boundary_enum::homogeneous)
  // {
  //   // fixed boundary set to the div term corresponds to Neumann boundary
  //   asgard::term_1d<P> div = asgard::term_div<P>(-1, asgard::flux_type::upwind,
  //                                                asgard::boundary_type::right);
  //   // fixed boundary set to the grad term corresponds to Dirichlet boundary
  //   asgard::term_1d<P> grad = asgard::term_grad<P>(1, asgard::flux_type::upwind,
  //                                                  asgard::boundary_type::left);
  //
  //   // the multi-dimensional operator, initially set to identity in md
  //   std::vector<asgard::term_1d<P>> ops(num_dims);
  //   for (int d = 0; d < num_dims; d++)
  //   {
  //     // combine the div and grad into a single chain term
  //     asgard::term_1d<P> fxx({div, grad});
  //
  //     // based on the domain and max-level, get the cell-size in direction d
  //     P const dx = pde.cell_size(d);
  //
  //     // adding penalty to stabilize the steady state equation
  //     // the penalty is applied only to discontinuities, if the solution is continuous
  //     // then the penalty will not alter the result, this only improves the conditioning
  //     fxx.set_penalty(P{1} / dx);
  //
  //     // add the second order operator in dimension dim
  //     ops[d] = fxx;
  //     pde += asgard::term_md<P>(ops);
  //     ops[d] = asgard::term_identity{};
  //   }
  //
  // } else { // inhomogeneous case
  //
  //   // allowing for inhomogeneous boundary, we can use many combinations of
  //   // Dirichlet and Neumann data
  //   // the 1D case is set for Dirichlet boundary
  //   // the mD case is set for mix Dirichlet and Neumann conditions
  //
  //   if (num_dims == 1)
  //   {
  //     // fixed boundary set to the div term corresponds to Neumann boundary
  //     asgard::term_1d<P> div = asgard::term_div<P>(-1, asgard::flux_type::upwind);
  //
  //     // fixed boundary set to the grad term corresponds to Dirichlet boundary
  //     asgard::term_1d<P> grad = asgard::term_grad<P>(1, asgard::flux_type::upwind,
  //                                                    asgard::boundary_type::bothsides);
  //     // merge the div and grad terms
  //     asgard::term_1d<P> fxx({div, grad});
  //
  //     // penalize discontinuities
  //     P const dx = pde.min_cell_size();
  //     fxx.set_penalty(P{1} / dx);
  //
  //     // merge into a multi-dimensional term with one dimension
  //     asgard::term_md<P> fxx_md({fxx, });
  //
  //     // adding inhomogeneous term to the right of the domain
  //     // starting with the exact solution
  //     asgard::separable_func<P> bc = exact;
  //     // the 0-th dimension component is set to constant 1
  //     bc.set(0, P{1});
  //     // add the condition at the right point
  //     fxx_md += asgard::right_boundary_flux(bc);
  //
  //     // add the term with inhomogeneous boundary to the pde
  //     pde += fxx_md;
  //   }
  //   else
  //   {
  //     // setting fixed boundary for the div term in the chain
  //     // results in Neumann conditions imposed on the field
  //     // think of this as imposing Dirichlet condition on the output of the grad term
  //     // and the output of the grad term is the derivative of the field
  //     asgard::term_1d<P> div = asgard::term_div<P>(-1, asgard::flux_type::upwind,
  //                                                  asgard::boundary_type::left);
  //
  //     // Dirichlet boundary set to the grad term corresponds to Dirichlet boundary
  //     asgard::term_1d<P> grad = asgard::term_grad<P>(1, asgard::flux_type::upwind,
  //                                                    asgard::boundary_type::right);
  //     // merge the div and grad terms
  //     asgard::term_1d<P> fxx({div, grad});
  //
  //     // penalize discontinuities
  //     P const dx = pde.min_cell_size();
  //     fxx.set_penalty(P{1} / dx);
  //
  //     for (int d = 0; d < num_dims; d++)
  //     {
  //       // make vector of terms_1d for each dimension
  //       std::vector<asgard::term_1d<P>> terms(num_dims);
  //       terms[d] = fxx;
  //
  //       // merge into a multi-dimensional term with one dimension
  //       asgard::term_md<P> fxx_md(terms);
  //
  //       // setting Dirichlet condition 1 on the right wall of dimension d
  //       // by default, the boundary condition is applied to the field
  //       // that is the input of the term, i.e., the input to the grad term
  //       asgard::separable_func<P> bc = exact;
  //       bc.set(d, P{1});
  //       fxx_md += asgard::right_boundary_flux(bc);
  //
  //       // setting Neumann condition 2 on the left wall of dimension d
  //       bc = exact;
  //       bc.set(d, P{2});
  //       asgard::boundary_flux<P> lbf = asgard::left_boundary_flux(bc);
  //       // at this point we have the boundary flux
  //       // but we also need to apply it to the input of the div-term,
  //       // i.e., set the level of the chain to the index of the div term
  //       lbf.chain_level(d) = 0;
  //       fxx_md += lbf;
  //
  //       // add the term with boundary conditions to the pde
  //       pde += fxx_md;
  //     }
  //   }
  // }
  //
  // for (int d = 0; d < num_dims; d++) {
  //   // using separability properties, copy over the exact solution
  //   asgard::separable_func<P> src = exact;
  //   // differentiate in the d-th direction, i.e., replace the function
  //   // with a constant 2
  //   src.set(d, 2);
  //
  //   pde.add_source(std::move(src));
  // }
  //
  // // if an initial condition is specified, it will be used as the initial guess
  // // of an iterative solver, other zeros is used as the initial guess
  // // the direct solver does not use an initial guess
  //
  // return pde;
  return asgard::pde_scheme<P>();
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic make]
#endif
}

void self_test() {}

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
