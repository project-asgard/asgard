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
 * \addtogroup asgard_examples_diffusion Example 4, diffusion operator
 *
 * \par Example 3
 * Solves the continuity partial differential equation in arbitrary dimension \b d
 * \f[ \frac{d}{dt} f - \nabla \cdot \nabla f = s \f]
 * where the right-hand-side source \b s is chosen so the exact solution is
 * \f[ f(t, x, y) = (1 - \exp(-t)) (\exp(1 - x^2) - 1) \f]
 * The domain is (-1, 1) and the boundary conditions are zero-Dirichlet.
 *
 */

/*!
 * \ingroup asgard_examples_diffusion
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

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
 * \param num_dims is the number of dimensions, currently between 1 and 6
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \snippet diffusion.cpp diffusion_md make
 */
template<typename P = asgard::default_precision>
asgard::PDEv2<P> make_diffusion_pde(int num_dims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [diffusion_md make]
#endif

  options.title = "Diffusion " + std::to_string(num_dims) + "D";

  // the domain will have range -2 * PI to 2 * PI in each direction
  std::vector<asgard::domain_range<P>> ranges(num_dims, {-1, 1});

  asgard::pde_domain<P> domain(ranges); // can use move here, but copy is cheap enough

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  options.default_degree = 2;
  options.default_start_levels = {4, };

  // using implicit time-stepping, thus ignoring any CFL
  options.default_dt = 0.01;

  options.default_stop_time = 1.0; // integrate until T = 1

  options.default_solver = asgard::solve_opts::direct; // bad but OK for this example

  // create a pde from the given options and domain
  // we can read the variables using pde.options() and pde.domain() (both return const-refs)
  // the option entries may have been populated or updated with default values
  asgard::PDEv2<P> pde(options, std::move(domain));

  // one dimensional divergence term using upwind flux
  asgard::term_1d<P> div = asgard::term_div(asgard::flux_type::upwind,
                                            asgard::boundary_type::dirichlet,
                                            P{1});

  // Dirichlet conditions applied to the grad term apply Neumann conditions
  // to the second order operator
  asgard::term_1d<P> grad = asgard::term_grad(asgard::flux_type::upwind,
                                              asgard::boundary_type::free,
                                              P{-1});

  // the second order operator is a chain of operators
  asgard::term_1d<P> diffusion({grad, div});

  // the multi-dimensional divergence, initially set to identity in md
  std::vector<asgard::term_1d<P>> ops(num_dims);
  for (int d = 0; d < num_dims; d++)
  {
    ops[d] = diffusion; // using operator in the d-direction
    pde += asgard::term_md<P>(ops);
    ops[d] = asgard::term_identity{}; // reset back to identity
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
  auto cos_1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < static_cast<int64_t>(x.size()); i++)
        fx[i] = std::exp(1 - x[i] * x[i]) - 1;
    };
  auto nexp_t = [](P t) -> P { return 1 - std::exp(-t); };

  asgard::separable_func<P> exact(
      std::vector<asgard::svector_func1d<P>>(num_dims, cos_1d), nexp_t);

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
  if (enorm < 1.E-3)
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
 * The interesting part is how to add custom command line parameters
 * to the default ones provided by ASGarD.
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
    std::cout << "\n solves the continuity equation:\n";
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
    //self_test();
    return 0;
  }

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(make_diffusion_pde(1, options),
                                         asgard::verbosity_level::high);

  // time-integration is performed using the advance_time() method
  // advance_time(disc, n); will integrate for n time-steps
  // skipping n (or using a negative) will integrate until the end

  if (not disc.stop_verbosity())
    std::cout << " -- error in the initial conditions: " << get_error_l2(disc) << "\n";

  asgard::advance_time(disc); // integrate until num-steps or stop-time

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
