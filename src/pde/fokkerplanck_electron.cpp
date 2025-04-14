#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file fokkerpanck_electron.cpp
 * \brief Fokker-Planck example from the runaway electron paper
 * \author The ASGarD Team
 * \ingroup asgard_fokkerplanck_el
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_fokkerplanck_el Example: Fokker-Planck runaway electron example
 *
 * \par Fokker-Planck equation
 * Solves the Fokker-Planck partial differential equation in 2d
 * \f[ \frac{\partial}{\partial t} f + \nabla \cdot f = s \f]
 *
 *
 *
 */

/*!
 * \ingroup asgard_fokkerplanck_el
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

/*!
 * \ingroup asgard_fokkerplanck_el
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
 * \snippet fokkerpanck_electron.cpp asgard_fokkerplanck_el make
 */
template<typename P = asgard::default_precision>
asgard::PDEv2<P> make_fokkerplanck(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_fokkerplanck_el make]
#endif

  options.title = "Fokker-Planck 2D (electron example)";

  asgard::pde_domain<P> domain({{0.0, 10}, {-1.0, 1.0}}); // can use move here, but copy is cheap enough
  domain.set_names({"p", "z"});

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  options.default_degree = 2;
  options.default_start_levels = {4, };

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
  asgard::PDEv2<P> pde(options, std::move(domain));

  // using spherical coordinates, the angle dimension has been integrated
  auto dp = [](P p)-> P { return p * p; };
  // vector variants of the single dimensional volume Jacobian
  auto vec_dp = [&](std::vector<P> const &p, std::vector<P> &fp) {
    for (size_t i = 0; i < p.size(); i++)
      fp[i] = dp(p[i]);
  };

  // setting up the mass matrix
  pde.set_mass({asgard::term_volume<P>{vec_dp}, asgard::term_identity{}});


  // constant parameters, problem specific
  auto constexpr phi = [](P x) { return std::erf(x); };

  // auto constexpr psi = [=](P x) {
  //   auto const dphi_dx = 2.0 / std::sqrt(M_PI) * std::exp(-x * x);
  //   auto ret           = 1.0 / (2 * x * x) * (phi(x) - x * dphi_dx);
  //   // DOUBLE CHECK THIS!
  //   if (std::abs(x) < 1e-5)
  //     ret = 0;
  //   return ret;
  // };

  auto constexpr psi_v2 = [=](P x) {
    auto const dphi_dx = 2.0 / std::sqrt(M_PI) * std::exp(-x * x);
    return 0.5 * (phi(x) - x * dphi_dx);
  };

  P constexpr delta  = 0.042;
  P constexpr delta4 = 0.042 * 0.042 * 0.042 * 0.042;

  P constexpr E = []() {
    return 0.0025;
  }();

  auto constexpr gamma = [](P p) {
    return std::sqrt(1 + std::pow(delta * p, 2));
  };
  auto constexpr vx = [=](P p) { return (p / gamma(p)); };

  // auto constexpr Ca = [=](P p) {
  //   if (p < 1.E-6) return P{0}; // lim_{x->0} Ca(x) = 0
  //   return (psi_v2(vx(p)) / vx(p));
  // };

  auto constexpr Ca_v2 = [=](P p) {
    if (p < 1.E-6) return P{0}; // lim_{x->0} Ca(x) = 0
    P const vp = vx(p);
    P const dphi_dx  = 2.0 / std::sqrt(M_PI) * std::exp(-vp * vp);
    P const phi_dphi = 0.5 * (phi(vp) - vp * dphi_dx);
    return (std::pow(gamma(p), 3) / p) * phi_dphi;
  };

  // auto constexpr Cb = [=](P p) {
  //   return 1.0 / 2.0 * 1.0 / vx(p) *
  //          (1 + psi_v2(vx(p)) - psi_v2(vx(p)) +
  //           delta4 * std::pow(vx(p), 2) / 2.0);
  // };

  auto constexpr Cb_v2 = [=](P p) {
    if (p < 1.E-6) return P{0}; // lim_{x->0} Cb(x) = 0
    P const vp    = vx(p);
    P const phivp = phi(vp);
    P const dphi_dx  = 2.0 / std::sqrt(M_PI) * std::exp(-vp * vp);
    P const phi_dphi = 0.5 * (phivp - vp * dphi_dx);
    // return (std::pow(p, 5) / std::pow(gamma(p), 3)) * phi_dphi;
    return (0.5 *  gamma(p) / p) * (p *p + p * p * phivp
      - std::pow(gamma(p), 2) * phi_dphi
      + delta4 * std::pow(vp, 2) / 2.0);
  };

  // auto constexpr Cf = [=](P p) {
  //   if (p < 1.E-6) return P{0}; // lim_{x->0} Cf(x) = 0
  //   return 2.0 * psi_v2(vx(p));
  // };

  auto constexpr Cf_v2 = [=](P p) {
    if (p < 1.E-6) return P{0}; // lim_{x->0} Cf(x) = 0
    P const vp = vx(p);
    P const dphi_dx  = 2.0 / std::sqrt(M_PI) * std::exp(-vp * vp);
    P const phi_dphi = 0.5 * (phi(vp) - vp * dphi_dx);
    return std::pow(gamma(p), 2) * phi_dphi;
  };

  // termC1 == 1/p^2 * d/dp * p^2 * Ca * df/dp
  {
    auto g_div = [=](std::vector<P> const &p, std::vector<P> &fp) {
      for (size_t i = 0; i < p.size(); i++) {
        fp[i] = -std::sqrt(Ca_v2(p[i]));
        // std::cout << p[i] << "   " << fp[i] << "\n";
      }
    };
    auto g_grad = [=](std::vector<P> const &p, std::vector<P> &fp) {
      for (size_t i = 0; i < p.size(); i++)
        fp[i] = std::sqrt(Ca_v2(p[i]));
    };

    asgard::term_1d<P> const dpp({
            asgard::term_div<P>{g_div, asgard::flux_type::upwind, asgard::boundary_type::left},
            asgard::term_grad<P>{g_grad, asgard::flux_type::upwind, asgard::boundary_type::right}
        });

    pde += asgard::term_md<P>({dpp, asgard::term_identity{}});
  }

  // termC2 == 1/p^2 * d/dp * p^2 * Cf * f
  {
    auto g_div = [=](std::vector<P> const &p, std::vector<P> &fp) {
      for (size_t i = 0; i < p.size(); i++) {
        fp[i] = - Cf_v2(p[i]);
      }
    };
    pde += asgard::term_md<P>({asgard::term_div<P>{g_div, asgard::flux_type::upwind, asgard::boundary_type::right},
                              asgard::term_identity{}});
  }

  // termC3 == Cb(p)/p^4 * d/dz( (1-z^2) * df/dz )
  {
    auto g_vol = [=](std::vector<P> const &p, std::vector<P> &fp) {
      for (size_t i = 0; i < p.size(); i++) {
        fp[i] = std::sqrt( Cb_v2(p[i]) );
        // std::cout << p[i] << "   " << fp[i] << "\n";
        fp[i] = 0;
      }
    };
    auto g_z3_pos = [=](std::vector<P> const &z, std::vector<P> &fz) {
      for (size_t i = 0; i < z.size(); i++)
        fz[i] = std::sqrt(1.0 - z[i] * z[i]);
    };
    auto g_z3_neg = [=](std::vector<P> const &z, std::vector<P> &fz) {
      for (size_t i = 0; i < z.size(); i++)
        fz[i] = -std::sqrt(1.0 - z[i] * z[i]);
    };

    asgard::term_1d<P> cmass = asgard::term_volume<P>{g_vol};

    // TODO: check if this should be downwind or upwind
    asgard::term_1d<P> div = asgard::term_div<P>{g_z3_neg, asgard::flux_type::upwind, asgard::boundary_type::bothsides};
    asgard::term_1d<P> grad = asgard::term_div<P>{g_z3_pos, asgard::flux_type::upwind};

    asgard::term_md<P> link1 = {cmass, div};
    asgard::term_md<P> link2 = {cmass, grad};

    link2.set_mass({asgard::term_volume<P>{vec_dp}, asgard::term_identity{}});

    pde += {link1, link2};
  }

  // termE1 == -E*z*f(z) * 1/p^2 (d/dp p^2 f(p))
  {
    auto Edp = [=](std::vector<P> const &p, std::vector<P> &fp)
        -> void
      {
        for (size_t i = 0; i < p.size(); i++)
          fp[i] = E * dp(p[i]);
      };
    auto positive = [](std::vector<P> const &z, std::vector<P> &fz)
        -> void
      {
        for (size_t i = 0; i < z.size(); i++)
          fz[i] = std::max(P{0}, z[i]);
      };

    // terms are split into positive and negative
    auto negative = [](std::vector<P> const &z, std::vector<P> &fz)
        -> void
      {
        for (size_t i = 0; i < z.size(); i++)
          fz[i] = std::min(P{0}, z[i]);
      };

      pde += {asgard::term_div<P>{Edp, asgard::flux_type::upwind, asgard::boundary_type::right},
              asgard::term_volume<P>{positive}};
      pde += {asgard::term_div<P>{Edp, asgard::flux_type::downwind, asgard::boundary_type::right},
              asgard::term_volume<P>{negative}};
  }

  // termE2 == -E*p*f(p) * d/dz (1-z^2) f(z)
  {
    auto vecp = [=](std::vector<P> const &p, std::vector<P> &fp)
        -> void
      {
        fp = p;
      };
    auto dz = [=](std::vector<P> const &z, std::vector<P> &fz)
        -> void
      {
        for (size_t i = 0; i < z.size(); i++)
          fz[i] = E * std::sqrt(P{1} - z[i] * z[i]);
      };

    pde += {asgard::term_volume<P>{vecp},
            asgard::term_div<P>{dz, asgard::flux_type::upwind}};
  }

  // defining the separable initial condition
  auto icp = [&](std::vector<P> const &p, P /* time */, std::vector<P> &fp) {
    for (size_t i = 0; i < p.size(); i++)
      fp[i] = dp(p[i]) * ((p[i] <= 5) ? P{3.0 / 250.0} : 0);
  };
  auto icz = [](std::vector<P> const &z, P /* time */, std::vector<P> &fz) {
    for (size_t i = 0; i < z.size(); i++)
      fz[i] = (z[i] <= 0) ? P{3.0 / 250.0} : 0;
  };

  pde.add_initial(asgard::separable_func<P>({icp, icz}));

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_fokkerplanck_el make]
#endif
}

/*!
 * \ingroup asgard_examples_continuity_md
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
 * \snippet continuity.cpp continuity_md get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_md get-err]
#endif

  // using the orthogonality of the basis and ignoring quadrature error
  // in the projection of the exact solution onto the current basis
  // the error has two components:
  // - difference between the current state and the projection
  // - the L^2 norm of the exact solution minus the projection

  int const num_dims = disc.num_dims();

  // using the fact that the initial condition is the exact solution
  // disc.get_pde2().ic_sep() returns the separable initial conditions
  // disc.project_function() projects a set of separable functions
  // onto the current sparse grid basis and returns the coefficients
  std::vector<P> const eref = disc.project_function(disc.get_pde2().ic_sep());

  double constexpr space1d = 2 * PI; // integral of sin(x)^2 over (-2 * PI, 2 * PI)
  double const time_val    = std::cos(disc.time_params().time());

  // this is the L^2 norm-squared of the exact solution
  // powi works the same as std::pow but the second input is an integer
  double const enorm = asgard::fm::powi(space1d, num_dims) * time_val * time_val;

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
//! [continuity_md get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_continuity_md
 * \brief main() for the Fokker-Planck example
 *
 * The main() processes the command line arguments and calls both
 * make_continuity_pde() and get_error_l2().
 *
 * \snippet fokkerpanck_electron.cpp asgard_fokkerplanck_el main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_fokkerplanck_el main]
#endif

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this file and the two additional options accepted for this problem
  if (options.show_help) {
    std::cout << "\n solves the continuity equation:\n";
    std::cout << "    f_t + div f = s(t, x)\n";
    std::cout << " with periodic boundary conditions \n"
                 " and source term that generates a known artificial solution\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-dims            -dm     int        accepts: 1 - 6\n";
    std::cout << "                                    the number of dimensions\n\n";
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
  asgard::discretization_manager<P> disc(make_fokkerplanck(options),
                                         asgard::verbosity_level::high);

  // disc.print_mats();

  // time-integration is performed using the advance_time() method
  // advance_time(disc, n); will integrate for n time-steps
  // skipping n (or using a negative) will integrate until the end

  // if (not disc.stop_verbosity())
  //   std::cout << " -- error in the initial conditions: " << get_error_l2(disc) << "\n";

  disc.advance_time(); // integrate until num-steps or stop-time

  disc.progress_report();

  // if (not disc.stop_verbosity())
  //   std::cout << " -- final error: " << get_error_l2(disc) << "\n";

  // for (auto x : disc.current_state())
  //   std::cout << x << "   ";
  // std::cout << "\n";

  disc.save_final_snapshot(); // only if output filename is provided

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_fokkerplanck_el main]
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

// template<typename P>
// void dotest(double tol, int num_dims, std::string const &opts) {
//   current_test<P> test_(opts, num_dims);
//
//   auto options = make_opts(opts);
//
//   discretization_manager<P> disc(make_continuity_pde<P>(num_dims, options),
//                                  verbosity_level::quiet);
//
//   while (disc.time_params().num_remain() > 0)
//   {
//     disc.advance_time(1);
//
//     double const err = get_error_l2(disc);
//
//     tcheckless(disc.time_params().step(), err, tol);
//   }
// }
//
// template<typename P>
// void dolongtest(double tol, int num_dims, std::string const &opts) {
//   current_test<P> test_(opts, num_dims);
//
//   auto options = make_opts(opts);
//
//   discretization_manager<P> disc(make_continuity_pde<P>(num_dims, options),
//                                  verbosity_level::quiet);
//
//   disc.advance_time();
//
//   double const err = get_error_l2(disc);
//
//   tcheckless(disc.time_params().step(), err, tol);
// }
//
// template<typename P>
// void dotest(double tol, int num_dims, std::string const &opts, int np) {
//   current_test<P> test_(opts, num_dims);
//
//   auto options = make_opts(opts);
//
//   discretization_manager<P> disc(make_continuity_pde<P>(num_dims, options),
//                                  verbosity_level::quiet);
//
//   // makes a dense grid over the domain using np points each direction
//   vector2d<double> const mesh = make_grid<double>(disc.get_pde2().domain(), np);
//
//   // the reconstruction is always done in double-precision even if the data
//   // coming from the discretization_manager is in floats
//   // thus, use the double-precision version of the exact solution
//   auto sin_1d = [](std::vector<double> const &x, double, std::vector<double> &fx) ->
//     void {
//       for (size_t i = 0; i < x.size(); i++)
//         fx[i] = std::sin(x[i]);
//     };
//
//   auto cos_t = [](double t) -> double { return std::cos(t); };
//
//   separable_func<double> exact(
//       std::vector<svector_func1d<double>>(num_dims, sin_1d), cos_t);
//
//   std::vector<double> ref(mesh.num_strips());
//   std::vector<double> com(mesh.num_strips());
//
//   while (disc.time_params().num_remain() > 0)
//   {
//     disc.advance_time(1);
//
//     double const time = disc.time_params().time();
// #pragma omp parallel for
//     for (int64_t i = 0; i < mesh.num_strips(); i++)
//       ref[i] = exact.eval(mesh[i], time);
//
//     auto shot = disc.get_snapshot();
//
//     shot.reconstruct(mesh[0], mesh.num_strips(), com.data());
//
//     double err = 0;
//     for (size_t i = 0; i < ref.size(); i++)
//       err = std::max(err, std::abs(com[i] - ref[i]));
//
//     tcheckless(disc.time_params().step(), err, tol);
//   }
// }

void self_test() {
  all_tests testing_("continuity equation:", " f_t + div f = sources");

  // continuity is a simple pde and tests are cheap
  // thus, we can use to indirectly test multiple aspects of ASGarD
  // we still want tests to run fast but also show the important dynamics

#ifdef ASGARD_ENABLE_DOUBLE
  // test convergence with respect to sparse grid level

#endif

#ifdef ASGARD_ENABLE_FLOAT

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
