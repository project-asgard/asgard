#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file diffusion.cpp
 * \brief Simple continuity example
 * \author The ASGarD Team
 * \ingroup asgard_examples_diffusion
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_diffusion Example 4, Diffusion operator
 *
 * \par Example 4
 * Solves the 2D diffusion partial differential equation
 * \f[ \frac{\partial}{\partial t} f - \nabla \cdot \nabla f = s \f]
 * where the right-hand-side source \b s is chosen so the exact solution is
 * \f[ f(t, x, y) = (1 - \exp(-t)) (\exp(1 - x^2) - 1) \f]
 * The domain is (-1, 1) and the boundary conditions are homogeneous Dirichlet.
 * This example avoids the use of trigonometric functions, since those are eigenfunctions
 * of the operator.
 *
 * \par
 * The second order operator is applied as a chain of two operators, namely
 * \f[ \frac{d}{dt} f - \nabla \cdot g = s, \qquad g = \nabla f \f]
 * Thus, applying Dirichlet boundary conditions on gradient operator results in
 * Dirichlet boundary conditions for the field \b f, while applying Dirichlet
 * conditions on the divergence yields Neumann conditions for the field.
 *
 * \par
 * The condition number of the second order operator is the square of the first order one,
 * and the corresponding CFL condition implies that the time step should be the square
 * of the spacial resolution.
 * Thus, the required number of time-steps is prohibitively high for any explicit
 * time stepping method, and we want to use an implicit stepper.
 * While more stable and less restrictive on the time-step,
 * implicit methods come with the additional challenge of requiring a linear solver.
 *
 * \par
 * The focus of this example is the use of implicit time-stepping and a linear solver.
 */

/*!
 * \ingroup asgard_examples_diffusion
 * \brief Make single diffusion PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \tparam chain1d the chain of div-grad operators can be done with either a chain of
 *         1d or md terms, where the 1d chain allows the use of simpler data-structures
 *         that yield better performance, but is restricted to problems where the
 *         product of 1d matrices results in tri-diagonal matrix.
 *         In this case here, chain1d will change the way the terms are chained but will
 *         have no effect on the final result.
 *
 * \param num_dims is the number of dimensions, currently between 1 and 6
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \snippet diffusion.cpp diffusion_md make
 */
template<typename P = asgard::default_precision, bool chain1d = true>
asgard::PDEv2<P> make_diffusion_pde(int num_dims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md make]
#endif

  options.title = "Diffusion " + std::to_string(num_dims) + "D";

  // the domain will have range (-1, 1) in each direction
  std::vector<asgard::domain_range<P>> ranges(num_dims, {-1, 1});

  asgard::pde_domain<P> domain(ranges); // can use move here, but copy is cheap enough

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  options.default_degree = 2;
  options.default_start_levels = {4, };

  // using implicit time-stepping, thus ignoring any CFL
  options.default_dt = 0.01;

  options.default_stop_time = 3.0; // integrate until T = 3

  // using implicit Crank-Nicolson method, which requires a solver
  options.default_step_method = asgard::time_method::cn;

  if (options.max_level() <= 5) {
    // direct (dense) solver is fast for small problems and works well for prototyping
    // and debugging, since it remove from the problem some additional factors,
    // such as solver tolerance and number of iterations
    options.default_solver = asgard::solver_method::direct;
  } else {
    // when the problem size becomes significant, forming and factorizing the dense
    // operator matrix becomes prohibitively expensive in flops and memory usage
    // iterative solvers are needed and it is good to specify default parameters
    options.default_solver = asgard::solver_method::gmres;
  }

  // only the iterative solvers will use these values
  // jacobi is (currently) the fastest and most stable preconditioner
  options.default_precon = asgard::precon_method::jacobi;

  // the tolerance for the iterative solver should probably be updated
  // based on the time-step and the max-level
  // this is a tight number which removes the solver error from consideration
  options.default_isolver_tolerance = 1.E-7;

  // the number of iterations should depends on the time-step and condition
  // number of the operators, should be kept high to allow for convergence
  options.isolver_iterations = 1000;

  // GMRES uses a two-loop approach (restarted GMRES)
  // the inner iterations explicitly form and manipulate the basis for the Krylov sub-space
  // which requires lots of memory and the number here should be kept moderate
  // (memory usage is dominated by isolver_inner_iterations * degrees-of-freedom)
  // (the bicgstab method ignores this value)
  options.isolver_inner_iterations = 50;

  // create a pde from the given options and domain
  // we can read the variables using pde.options() and pde.domain() (both return const-refs)
  // the option entries may have been populated or updated with default values
  asgard::PDEv2<P> pde(options, std::move(domain));

  // one dimensional divergence term using upwind flux
  // setting Dirichlet condition here will in fact yield Neumann boundary condition
  // for the combined operator and onto the field
  asgard::term_1d<P> div = asgard::term_div<P>(-1, asgard::flux_type::upwind,
                                               asgard::boundary_type::free);

  // Dirichlet conditions applied to the grad term apply Dirichlet condition
  // for the combined operator and respectively the field
  asgard::term_1d<P> grad = asgard::term_grad<P>(1, asgard::flux_type::upwind,
                                                 asgard::boundary_type::dirichlet);

  // different way of operator chaining
  if constexpr (chain1d)
  {
    // the second order operator is a chain of operators
    asgard::term_1d<P> diffusion({div, grad});

    // the multi-dimensional Laplacian, initially set to identity in md
    std::vector<asgard::term_1d<P>> ops(num_dims);
    for (int d = 0; d < num_dims; d++)
    {
      ops[d] = diffusion; // using operator in the d-direction
      pde += asgard::term_md<P>(ops);
      ops[d] = asgard::term_identity{}; // reset back to identity
    }
  }
  else
  {
    // workspace with only identity operators
    std::vector<asgard::term_1d<P>> ops(num_dims);
    for (int d = 0; d < num_dims; d++)
    {
      // creating multi-dimensional div-operator (for dimension d)
      ops[d] = div;
      asgard::term_md<P> div_md(ops);

      // creating multi-dimensional grad-operator (for dimension d)
      ops[d] = grad;
      asgard::term_md<P> grad_md(ops);

      // adding a multidimensional chain operator term
      pde += asgard::term_md<P>({div_md, grad_md});

      ops[d] = asgard::term_identity{}; // reset the workspace back to identity
    }
  }

  // defining the separable known solution
  auto exp_1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      // given values in x, must populate fx with the corresponding values
      assert(fx.size() == x.size()); // this is guaranteed, do NOT resize fx
      // OpenMP and SIMD directives can be used here
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::exp(1 - x[i] * x[i]) - 1;
    };

  // time functions are not called in batch
  // hence the signature takes a single entry
  auto nexp_t = [](P t) -> P { return 1 - std::exp(-t); };

  // the derivatives, d/dx sin(x) = cos(x) and d/dx cos(t) = -sin(t)
  auto ddexp_1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = - (4 * x[i] * x[i] - 2) * std::exp(1 - x[i] * x[i]);
    };

  // negative exp(-t)
  auto exp_t = [](P t) -> P { return std::exp(-t); };

  // multidimensional product of functions, initializing to just cos(x)
  std::vector<asgard::svector_func1d<P>> exp_md(num_dims, exp_1d);

  // this is the exact solution
  asgard::separable_func<P> exact(exp_md, nexp_t);

  // no-initial condition implies zero as the initial condition

  // setting up the sources
  pde.add_source({exp_md, exp_t}); // derivative in time

  // compute the spacial derivatives
  for (int d = 0; d < num_dims; d++)
  {
    exp_md[d] = ddexp_1d; // set derivative in x for direction d
    pde.add_source({exp_md, nexp_t});
    exp_md[d] = exp_1d; // revert to the original value
  }

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md make]
#endif
}

/*!
 * \ingroup asgard_examples_diffusion
 * \brief Computes the L^2 error for the given example
 *
 * The provided discretization_manager should hold a PDE made with
 * make_continuity_pde(). This will compute the L^2 error.
 *
 * \tparam P is double or float, the precision of the manager
 *
 * \param disc is the discretization of a PDE
 *
 * \returns the L^2 error between the known exact solution and
 *          the current state in the \b disc manager
 *
 * \snippet diffusion.cpp diffusion_md get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md get-err]
#endif

  // using the orthogonality of the basis and ignoring quadrature error
  // in the projection of the exact solution onto the current basis
  // the error has two components:
  // - difference between the current state and the projection
  // - the L^2 norm of the exact solution minus the projection

  int const num_dims = disc.num_dims();

  // setting the exact solution so we can project onto the basis
  auto exp_1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < static_cast<int64_t>(x.size()); i++)
        fx[i] = std::exp(1 - x[i] * x[i]) - 1;
    };
  auto nexp_t = [](P t) -> P { return 1 - std::exp(-t); };

  asgard::separable_func<P> exact(
      std::vector<asgard::svector_func1d<P>>(num_dims, exp_1d), nexp_t);

  std::vector<P> const eref = disc.project_function({exact, });

  double const xnorm    = asgard::fm::powi(2.719125363804229, num_dims);
  double const time_val = nexp_t(disc.time_params().time());

  // this is the L^2 norm-squared of the exact solution
  double const enorm = xnorm * time_val * time_val;

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

  // when cos(t) vanishes, so does the exact solution and enorm -> 0
  // for small values of enorm, the relative error is artificially magnified
  // switch between relative and absolute error
  if (enorm < 1)
    return std::sqrt(ndiff + enorm - nself);
  else
    return std::sqrt((ndiff + enorm - nself) / enorm);
#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_diffusion
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls both
 * make_diffusion_pde() and get_error_l2().
 *
 * \snippet diffusion.cpp diffusion_md main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md main]
#endif

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves the diffusion equation:\n";
    std::cout << "    f_t - laplacian f = s(t, x)\n";
    std::cout << " with Dirichlet boundary conditions \n"
                 " and source term that generates a known artificial solution\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  // this is an optional step, check if there are misspelled or incorrect cli entries
  // the first set/vector of entries are those that can appear by themselves
  // the second set/vector requires extra parameters
  options.throw_if_argv_not_in({"-test", "--test"}, {});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(make_diffusion_pde<P, false>(2, options),
                                         asgard::verbosity_level::high);

  // time-integration is performed using the advance_time() method
  // advance_time(disc, n); will integrate for n time-steps
  // skipping n (or using a negative) will integrate until the end

  if (not disc.stop_verbosity())
    std::cout << " -- error in the initial conditions: " << get_error_l2(disc) << "\n";

  disc.advance_time(); // integrate until num-steps or stop-time

  // alternative to the one-shot approach above, integration can be done step-by-step
  // and verbose output can be generated
  // in the code below, first the builtin reporting mechanism is set to quiet,
  // and the error is computed for each time step
  // disc.set_verbosity(asgard::verbosity_level::quiet);
  // while (disc.time_params().num_remain() > 0) {
  //   asgard::advance_time(disc, 1);
  //   disc.progress_report();
  //   std::cout << " -- error: " << get_error_l2(disc) << "\n";
  // }

  disc.progress_report();

  if (not disc.stop_verbosity())
    std::cout << " -- final error: " << get_error_l2(disc) << "\n";

  disc.save_final_snapshot(); // only if output filename is provided

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md main]
#endif
};

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////

// just for convenience to avoid using asgard:: all over the place
// normally, should only include what is needed
using namespace asgard;

template<typename P = double, bool chain1d = true>
void dotest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims, (chain1d) ? "" : "chain term_md");

  auto options = make_opts(opts);

  discretization_manager<P> disc(make_diffusion_pde<P, chain1d>(num_dims, options),
                                 verbosity_level::quiet);

  while (disc.time_params().num_remain() > 0)
  {
    disc.advance_time(1);

    double const err = get_error_l2(disc);
    // std::cout << " err = " << err << "\n";

    tcheckless(disc.time_params().step(), err, tol);
  }
}

template<typename P = double>
void longtest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  discretization_manager<P> disc(make_diffusion_pde<P>(num_dims, options),
                                 verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);
  // std::cout << " err = " << err << "\n";
  tcheckless(disc.time_params().step(), err, tol);
}

void self_test() {
  all_tests testing_("diffusion equation:", " f_t + laplacian f = sources");

  // the diffusion equation is a relatively simple PDE but the condition number
  // of the matrices grows very fast with the level
  // thus, the tests are primarily done in double-precision

#ifdef ASGARD_ENABLE_DOUBLE
  // check convergence w.r.t. level
  dotest(1.E-3, 1, "-l 4 -n 20 -sv direct");
  dotest(1.E-4, 1, "-l 5 -n 20 -sv direct");
  dotest(5.E-5, 1, "-l 6 -n 20 -sv direct");

  dotest<double, false>(1.E-3, 1, "-l 4 -n 20 -sv direct"); // check chaining
  dotest<double, false>(1.E-4, 1, "-l 5 -n 20 -sv direct");
  dotest<double, false>(5.E-5, 1, "-l 6 -n 20 -sv direct");

  dotest<double>(1.E-4, 1, "-l 5 -n 10 -sv bicgstab"); // check the jacobi preconditioner
  dotest<double, false>(1.E-4, 1, "-l 5 -n 10 -sv bicgstab"); // both chain-modes

  dotest(1.E-1, 1, "-l 5 -d 0 -n 20 -sv direct");
  dotest(5.E-3, 1, "-l 5 -d 1 -n 20 -sv direct");
  dotest(5.E-5, 1, "-l 5 -d 2 -n 40 -dt 0.005 -sv direct"); // time error manifests here

  dotest(1.00E-2, 2, "-l 6 -t 0.5 -dt 0.1 -sv direct"); // second order in time
  dotest(2.50E-3, 2, "-l 6 -t 0.5 -dt 0.05 -sv direct");
  dotest(6.25E-4, 2, "-l 6 -t 0.5 -dt 0.025 -sv direct");

  dotest(1.E-2, 3, "-l 4 -t 0.5 -dt 0.1 -sv direct"); // direct solver multi-d

  // in the first few steps here, the grid is very coarse
  // finer refinement and time-step is needed, only the final error is OK
  longtest(2.E-3, 2, "-l 2 -m 8 -a 1.E-3");
  longtest(2.E-3, 2, "-l 2 -m 8 -a 1.E-3 -sv direct"); // check if solver is updated
  dotest(5.E-3, 2, "-l 8 -a 1.E-4");
#endif

#ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(1.E-2, 1, "-l 4 -n 10 -sv direct");
  dotest<float>(1.E-2, 2, "-l 5 -t 0.5 -dt 0.1  -sv direct");
#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
