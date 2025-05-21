#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file spherical_diffusion.cpp
 * \brief Diffusion using spherical coordinates
 * \author The ASGarD Team
 * \ingroup asgard_spherical_diffusion
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_spherical_diffusion Example: Spherical coordinates
 *
 * \par Spherical diffusion equation
 * Solves the spherical diffusion equation
 * \f[ \frac{\partial}{\partial t} f -
 *    \frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial f }{\partial r} \right)
 *    - \frac{1}{r^2 \sin(\theta)} \frac{\partial}{\partial \theta}
 *    \left( \sin(\theta) \frac{\partial f}{\partial \theta} \right) = s \f]
 * where the operator is the Laplacian in spherical coordinates and the source
 * is chosen so that
 * \f[ f(t, r, \theta) = \exp(-t) r \cos(r) \cos(\theta) \f]
 * Due to the volume Jacobian that needs to be incorporated in all integrals,
 * even though the basis functions are orthonormal, the mass matrix for this
 * problem is no longer identity.
 * The explicit inversion of the matrix is computationally cheap, since it can be done in
 * the non-hierarchical block-diagonal form.
 *
 * \par
 * The purpose of this example is to show how to incorporate volume Jacobian
 * and non-trivial mass matrix into the discretization scheme.
 */

/*!
 * \ingroup asgard_spherical_diffusion
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

/*!
 * \ingroup asgard_spherical_diffusion
 * \brief Make single Fokker-Planck PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \snippet spherical_diffusion.cpp asgard_spherical_diffusion make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_spherical(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_spherical_diffusion make]
#endif

  // selectively pull from the asgard namespace
  using term_volume   = asgard::term_volume<P>;
  using term_div      = asgard::term_div<P>;
  using term_grad     = asgard::term_grad<P>;
  using term_1d       = asgard::term_1d<P>;
  using term_md       = asgard::term_md<P>;
  using boundary_flux = asgard::boundary_flux<P>;

  options.title = "Spherical Diffusion 2D";

  asgard::pde_domain<P> domain({{0.0, 1}, {0.0, PI}});
  domain.set_names({"r", "theta"});

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  options.default_degree = 2;
  options.default_start_levels = {5, };

  options.default_step_method = asgard::time_method::back_euler;

  options.default_dt = 0.01;

  options.default_stop_time = 1.0; // integrate until T = 1

  options.default_solver = asgard::solver_method::direct;

  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;
  options.default_isolver_inner_iterations = 50;

  // create a pde from the given options and domain
  // we can read the variables using pde.options() and pde.domain() (both return const-refs)
  // the option entries may have been populated or updated with default values
  asgard::pde_scheme<P> pde(options, std::move(domain));

  // volume Jacobian in r
  auto dr = [](P r)-> P { return r * r; };
  // vector variants of the single dimensional volume Jacobian
  auto vec_dr = [=](std::vector<P> const &r, std::vector<P> &vol_r)
      -> void {
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(); i++)
      vol_r[i] = dr(r[i]);
  };

  // volume Jacobian in theta
  auto dtheta = [](P theta)-> P { return std::sin(theta); };
  // vector variants of the single dimensional volume Jacobian
  auto vec_dtheta = [&](std::vector<P> const &theta, std::vector<P> &vol_theta)
      -> void {
    #pragma omp parallel for
    for (size_t i = 0; i < theta.size(); i++)
      vol_theta[i] = dtheta(theta[i]);
  };

  // setting up the mass matrix
  pde.set_mass({term_volume{vec_dr}, term_volume{vec_dtheta}});

  // the coefficients of the diffusion equation must contain the volume Jacobian
  // merging the Jacobian and the coefficient allows for cancellation
  // of numerical singularities which in turn improves stability
  {
    // merging -1 and dr()
    auto div_dr = [=](std::vector<P> const &r, std::vector<P> &vol_r)
        -> void {
      #pragma omp parallel for
      for (size_t i = 0; i < r.size(); i++)
        vol_r[i] = -dr(r[i]);
    };
    // merging 1 and dr(), i.e., there is only dr()
    auto grad_dr = vec_dr;

    term_1d div_grad_dr({
        term_div{div_dr, asgard::flux_type::upwind, asgard::boundary_type::bothsides},
        term_grad{grad_dr, asgard::flux_type::upwind}
      });

    // the volume term is exactly the same as the mass matrix
    // applying the inverse of the mass matrix will yield an identity term
    //   -- using an identity term will reduce the computational cost
    //      but that means we have to manually add the volume Jacobian to the theta
    //      component of the boundary conditions
    //      since identity will not have volume Jacobian information
    //   -- using a non-identity term will increase the computational cost
    //      there is no need to worry about volume Jacobian term in theta

    term_md drr({div_grad_dr, term_volume{vec_dtheta}});

    // setting up the inhomogeneous Neumann condition on the right
    // since the components of the boundary function are multiplied by the term coefficients
    // there is no need to add the volume Jacobian

    // setting the fixed condition for r, leaving dummy value of 1 in theta direction
    asgard::separable_func<P> boundary_func({std::cos(P{1}) - std::sin(P{1}), 1},
                                            [](P t)->P{ return std::exp(-t); });
    // setting the boundary condition for theta
    boundary_func.set(1,
        [&](std::vector<P> const &th, P, std::vector<P> &fth)
              -> void {
              for (size_t i = 0; i < th.size(); i++)
                fth[i] = std::cos(th[i]);
            });

    // set the function as right-boundary flux
    boundary_flux bc = asgard::right_boundary_flux{boundary_func};
    // see the example of the elliptic equation regarding the chain levels
    bc.chain_level(0) = 0;

    // add the boundary condition to the term and add the term to the pde
    pde += drr += bc;
  }

  {
    // volume Jacobian dr associated with the theta derivative
    auto sqrt_dr = [&](std::vector<P> const &r, std::vector<P> &vr) -> void { vr = r; };
    // merging -1 and dtheta()
    auto div_dtheta = [&](std::vector<P> const &th, std::vector<P> &vth)
        -> void {
      #pragma omp parallel for
      for (size_t i = 0; i < th.size(); i++)
        vth[i] = -dtheta(th[i]);
    };
    // merging 1 and dtheta(), i.e., there is only dtheta()
    auto grad_dtheta = vec_dtheta;

    term_1d volume_2dr({term_volume{sqrt_dr}, term_volume{sqrt_dr}});
    term_1d div_grad_dtheta({
        term_div{div_dtheta, asgard::flux_type::upwind, asgard::boundary_type::bothsides},
        term_grad{grad_dtheta, asgard::flux_type::upwind}
      });

    pde += {volume_2dr, div_grad_dtheta};
  }

  // source function
  auto source_r_dr = [=](std::vector<P> const &r, P /*time*/, std::vector<P> &fr) {
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(); i++)
      fr[i] = P{4} * r[i] * r[i] * std::sin(r[i]);
  };
  auto source_th_dtheta = [=](std::vector<P> const &th, P /*time*/, std::vector<P> &fth) {
    #pragma omp parallel for
    for (size_t i = 0; i < th.size(); i++)
      fth[i] = std::cos(th[i]) * std::sin(th[i]);
  };
  // used both for the source and the exact solution
  auto exact_time = [](P t) -> P { return std::exp(-t); };

  asgard::separable_func<P> source({source_r_dr, source_th_dtheta}, exact_time);
  pde.add_source(source);

  // exact solution
  auto exact_r = [=](std::vector<P> const &r, P /*time*/, std::vector<P> &fr) {
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(); i++)
      fr[i] = dr(r[i]) * r[i] * std::cos(r[i]);
  };
  auto exact_th = [=](std::vector<P> const &th, P /*time*/, std::vector<P> &fth) {
    #pragma omp parallel for
    for (size_t i = 0; i < th.size(); i++)
      fth[i] = dtheta(th[i]) * std::cos(th[i]);
  };

  asgard::separable_func<P> exact({exact_r, exact_th}, exact_time);

  // technically, the initial condition does not need a time component
  // the functions will be evaluated only once and using t = 0
  // however, here we are using the initial condition as the exact solution
  // when doing error checking
  pde.add_initial(exact);

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_spherical_diffusion make]
#endif
}

/*!
 * \ingroup asgard_spherical_diffusion
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
 * \snippet spherical_diffusion.cpp asgard_spherical_diffusion get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_spherical_diffusion get-err]
#endif

  // the discrete l2 norm and the continuous L2 norm are different but connected
  // the L2 norm is induced by the positive-definite mass-matrix
  // the wavelet basis used by ASGarD is orthonormal
  // therefore the mass-matrix in Cartesian coordinates is the identity
  // and both L2 and l2 norms are the same
  // using spherical coordinates, the mass-matrix is non-trivial

  // ASGarD normL2() method computes the continuous L2 norm
  // but the difference between the error vector must be formed explicitly

  // using the fact that the initial condition is the exact solution
  // form the projection of the exact solution
  std::vector<P> const eref = disc.project_function(disc.initial_cond_sep());

  // this is the L^2 norm-squared of the exact solution
  double constexpr space = 0.245458116975280;
  double const enorm = space * std::exp(-disc.time());

  disc.sync_mpi_state(); // is using multiple ranks, sync across the ranks

  // this is the currently computed solution
  std::vector<P> const &state = disc.current_state();

  // this will always hold true
  assert(eref.size() == state.size());

  // form the difference vector
  std::vector<P> err(state.size());
  for (size_t i = 0; i < state.size(); i++)
    err[i] = eref[i] - state[i];

  // computing the L2 norm of the difference and expected vectors
  double const nself = disc.normL2(eref);
  double const ndiff = disc.normL2(err);

  return std::sqrt((ndiff * ndiff + std::abs(enorm * enorm - nself * nself))) / enorm;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_spherical_diffusion get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_spherical_diffusion
 * \brief main() for the Fokker-Planck example
 *
 * The main() processes the command line arguments and calls both
 * make_spherical() and get_error_l2().
 *
 * \snippet spherical_diffusion.cpp asgard_spherical_diffusion main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_spherical_diffusion main]
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
    std::cout << "\n solves the spherical diffusion equation:\n";
    std::cout << "    f_t - Laplacian(f) = s(t, r, theta)\n";
    std::cout << " using spherical coordinate system \n\n";
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
  asgard::discretization_manager<P> disc(make_spherical(options),
                                         asgard::verbosity_level::high);

  P const err_init = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- error in the initial conditions: " << err_init << "\n";

  disc.advance_time(); // integrate until num-steps or stop-time

  disc.progress_report();

  P const err_final = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- final error: " << err_final << "\n";

  disc.save_final_snapshot(); // only if output filename is provided

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_spherical_diffusion main]
#endif
};

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////

// just for convenience to avoid using asgard:: all over the place
// normally, one should only include what is needed
using namespace asgard;

template<typename P>
void dotest(double tol, std::string const &opts) {
  current_test<P> test_(opts);

  auto options = make_opts(opts);

  discretization_manager<P> disc(make_spherical<P>(options),
                                 verbosity_level::quiet);

  while (disc.remaining_steps() > 0)
  {
    disc.advance_time(1);

    double const err = get_error_l2(disc);

    tcheckless(disc.current_step(), err, tol);
  }
}

void self_test() {
  all_tests testing_("spherical diffusion");

  // the convergence rate is slow due to ill-conditioning

#ifdef ASGARD_ENABLE_DOUBLE
  dotest<double>(5.E-3, "-l 4 -dt 0.01");
  dotest<double>(1.E-3, "-l 4 -dt 0.005");
  dotest<double>(5.E-4, "-l 6 -dt 0.0025");
  dotest<double>(5.E-4, "-l 5 -dt 0.0025 -ar 1.E-5");
#endif

#ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(5.E-3, "-l 4 -dt 0.01");
  dotest<float>(2.E-3, "-l 4 -dt 0.005");
#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
