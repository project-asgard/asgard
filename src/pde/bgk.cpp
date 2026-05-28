#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file bgk.cpp
 * \brief Bhatnagar-Gross-Krook
 * \author The ASGarD Team
 * \ingroup asgard_examples_bgk
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_bgk Example: Bhatnagar-Gross-Krook (BGK)
 *
 * \par Bhatnagar-Gross-Krook
 * The (simple) BGK collision operator used here is defined as
 * \f[ \mathcal{C}_{sBGK}[f](x, v, t) = \nu ( M(f) - f) \f]
 * with
 * \f[ M(f)(x, v) = \frac{n(x)}{\left( 2\pi \theta(x) \right)^{3/2}} \exp \left( - \frac{|v - u(x)|^2}{2 \theta(x)} \right) \f]
 * where
 * \f[ n(x) = \int_v f dv, \qquad u(x) = (u_1, u_2, u_3), \quad u_i(x) = \frac{1}{n(x)} \int_v v_i f dv \f]
 * and
 * \f[ \theta(x) = \frac{1}{3 n(x)} \int_v |v|^2 f dv - \frac{1}{3} \| u(x) \|^2  \f]
 * Implementing the moment requires moment dependence in the source term.
 *
 * This example uses three different sub-problems.
 *
 * \par Poisson problem
 * The same problem as the \ref asgard_examples_vplb "Vlasov-Poisson-Lenard-Bernstein" example
 * but with the BGK operator in place of Lenard-Bernstein:
 * \f[ \frac{\partial}{\partial t} f(x, v, t) + v \nabla_x f(x, v, t) + E(x, t) \cdot \nabla_v f(x, v, t) =
 *  \mathcal{C}_{sBGK}[f](x, v, t) \f]
 * The definition of the electric field E(x, t) and the initial conditions is the same.
 * This problem corresponds to pde_mode::poisson
 *
 * \par The 1D and 2D shock problems
 * Using a variation of the problem borrowed from
 * <a href="https://link.springer.com/book/10.1007/b79761">
 * E. F. Toro. "Riemann Solvers and Numerical Methods for Fluid Dynamics" </a>,
 * page 586, section 17.1.
 * The problem contains a large mass in the middle of the domain and at each position point
 * the initial condition is Maxwellian in the velocity dimension.
 * The solution spreads like an "explosion" and creates a staircase pattern for the mass distribution.
 * The example presented here is a simplified version of the one provided in the paper,
 * namely, the initial condition features a sharp but continuous jump, as opposed to being
 * a pure step function.
 *
 * \par
 * The equation also omits the terms for the electric field
 * \f[ \frac{\partial}{\partial t} f(x, v, t) + v \nabla_x f(x, v, t) = \nu ( M(f) - f) \f]
 *
 * \par
 * The focus of this example is to show the usage of the moment dependence in the sources and
 * the specialized solver asgard::solver_method::scaled_identity that is designed
 * for problems where the operator is a scaled identity and therefore trivial to invert.
 *
 * \par
 * <i>This is still work-in-progress, the documentation needs more work.</i>
 */

/*!
 * \ingroup asgard_examples_bgk
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_bgk
 * \brief Indicates which problem is being simulated
  */
enum class pde_mode
{
  //! Coupling between BGK and Poisson capabilities, identical to \ref asgard_examples_vplb "VPLB"
  poisson,
  //! shock in 1d
  shock1d,
  //! shock in 2d
  shock2d
};

/*!
 * \ingroup asgard_examples_bgk
 * \brief Make single BGK PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param pde_mode is type ofPDE to set
 * \param options is the set of options
 *
 * \returns the asgard::pde_scheme definition
 *
 * \snippet bgk.cpp asgard_examples_bgk make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_bgk(pde_mode mode, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk make]
#endif

  // number of position and velocity dimensions
  // the total number of dimensions is 2 * dims
  int const dims = (mode == pde_mode::shock2d) ? 2 : 1;

  options.title = "Bhatnagar-Gross-Krook "
                 + std::to_string(dims) + "x" + std::to_string(dims) + "v";

  // get the collision frequency
  P const nu = options.extra_cli_value_group<P>({"-nu", }).value_or(1.0);
  options.subtitle = "collision frequency " + std::to_string(nu);

  options.subtitle += (mode == pde_mode::poisson) ? " with poisson" : " shock example";

  std::vector<asgard::domain_range> ranges;
  ranges.reserve(2 * dims);
  if (mode == pde_mode::poisson) {
    ranges.emplace_back(-2 * PI, 2 * PI);
  } else {
    for (int x = 0; x < dims; x++)
      ranges.emplace_back(-1.0, 1.0);
  }
  for (int v = 0; v < dims; v++)
    ranges.emplace_back(-6.0, 6.0);

  // the domain has one position and multiple velocity dimensions
  asgard::pde_domain<P> domain(asgard::position_dims{dims}, asgard::velocity_dims{dims}, ranges);

  // setting some default options
  options.default_degree = 2;
  options.default_start_levels = {6, };

  // default: using implicit-explicit stepper
  // since both the implicit and explicit components are linear,
  // this PDE can use fully implicit time-stepping
  options.default_step_method = asgard::time_method::imex2;

  // cfl condition for the explicit component
  options.default_dt = 0.0128 * domain.min_cell_size(options.max_level());

  options.default_stop_time = 1.0;

  // select an appropriate solver
  if (mode == pde_mode::poisson)
  {
    // the poisson solver creates nonlinear coupling
    // using an imex stepper is the proper way to do this problem
    options.throw_if_not_imex_stepper();

    options.default_solver = asgard::solver_method::scaled_identity;
  }
  else
  {
    // if the user has selected a time-stepping method that is not imex
    if (options.step_method and not asgard::is_imex(options.step_method.value())) {
      // then we default to the GMRES solver
      options.default_solver = asgard::solver_method::gmres;

      options.default_isolver_tolerance  = 1.E-8;
      options.default_isolver_iterations = 400;
      options.default_isolver_inner_iterations = 50;
    } else {
      // if IMEX is selected or default, using specialized solver
      options.default_solver = asgard::solver_method::scaled_identity;
    }
  }

  // the BGK example requires adaptivity to avoid instabilities, especially in 4D
  // instabilities can lead to locally negative density and non-physical results
  // the following code ensures that adaptivity is enabled by default
  if (not options.restarting()) {
    // the adaptive tolerance is stored in the restart file
    // if no adaptivity is specified, that will be used by default

    // if not restating and no adaptivity is specified
    // note that this will always respect the adaptive thresholds specified at runtime
    if (not options.adapt_threshold and not options.adapt_relative)
    {
      if (mode == pde_mode::poisson) {
        // the perturbation is at a low scale
        // needs tight tolerance threshold
        options.adapt_threshold = 1.E-6;
      } else {
        // coarser tolerance is permissible here
        options.adapt_threshold = 1.E-5;
      }
    }
  }

  // create a pde from the given options and domain
  asgard::pde_scheme<P> pde(options, domain);

  // common building blocks
  // see the two-stream instability example for details
  auto func_pos = [](std::vector<P> const &x, std::vector<P> &y)
        -> void {
        #pragma omp parallel for
        for (size_t i = 0; i < x.size(); i++)
          y[i] = std::max(P{0}, x[i]);
      };

  auto func_neg = [](std::vector<P> const &x, std::vector<P> &y)
        -> void {
        #pragma omp parallel for
        for (size_t i = 0; i < x.size(); i++)
          y[i] = std::min(P{0}, x[i]);
      };

  asgard::term_1d<P> positive = asgard::term_volume<P>(func_pos);

  asgard::term_1d<P> negative = asgard::term_volume<P>(func_neg);

  asgard::term_1d<P> div_up = asgard::term_div<P>(1, asgard::flux_type::upwind,
                                                  asgard::boundary_type::periodic);

  asgard::term_1d<P> div_do = asgard::term_div<P>(1, asgard::flux_type::downwind,
                                                  asgard::boundary_type::periodic);

  // I is a very expressive way to indicate the identity
  // but it can cause severe conflicts, so it is not included in the main library
  asgard::term_1d<P> I = asgard::term_identity{};

  // adding the advection terms in the explicit group
  // the explicit_id will persist until new_term_group() is called again
  int const explicit_id = pde.new_term_group();

  if (mode == pde_mode::poisson)
  {
    pde += asgard::term_md<P>({div_up, positive});
    pde += asgard::term_md<P>({div_do, negative});

    pde += asgard::term_md<P>({
        asgard::volume_electric<P>(func_pos),
        asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides)
      });
    pde += asgard::term_md<P>({
        asgard::volume_electric<P>(func_neg),
        asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides)
      });
  }
  else
  {
    switch (dims) {
      case 1:
        pde += asgard::term_md{div_up, positive};
        pde += asgard::term_md{div_do, negative};
        break;
      case 2:
        pde += asgard::term_md{div_up, I, positive, I};
        pde += asgard::term_md{div_do, I, negative, I};
        pde += asgard::term_md{I, div_up, I, positive};
        pde += asgard::term_md{I, div_do, I, negative};
        break;
      case 3:
        pde += asgard::term_md{div_up, I, I, positive, I, I};
        pde += asgard::term_md{div_do, I, I, negative, I, I};
        pde += asgard::term_md{I, div_up, I, I, positive, I};
        pde += asgard::term_md{I, div_do, I, I, negative, I};
        pde += asgard::term_md{I, I, div_up, I, I, positive};
        pde += asgard::term_md{I, I, div_do, I, I, negative};
        break;
    }
  }

  // the right-hand-side of the equation is set for the implicit group
  int const implicit_id = pde.new_term_group();

  // setting the nu * f term
  std::vector<asgard::term_1d<P>> nuI(2 * dims, asgard::term_identity{});
  nuI[0] = asgard::term_volume<P>{nu};

  double const dt = options.dt.value_or(options.default_dt.value());

  if (dims == 1) {
    asgard::moment_id im0 = pde.register_moment(asgard::moment(0));
    asgard::moment_id im1 = pde.register_moment(asgard::moment(1));
    asgard::moment_id im2 = pde.register_moment(asgard::moment(2));

    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m1 = moments[im1];
      std::vector<P> const &m2 = moments[im2];

      int64_t const num_nodes = nodes.num_strips();
      assert(vals.size() == static_cast<size_t>(num_nodes));
      assert(m0.size() == static_cast<size_t>(num_nodes));
      assert(m1.size() == static_cast<size_t>(num_nodes));
      assert(m2.size() == static_cast<size_t>(num_nodes));
      #pragma omp parallel for
      for (int64_t i = 0; i < num_nodes; i++) {
        // P const x = nodes[i][0]; // no explicit spatial dependence
        P const v = nodes[i][1];

        P const n = m0[i];
        P const u = m1[i] / m0[i];
        P const t = m2[i] / m0[i] - u * u;

        vals[i] = nu * n / std::sqrt(2 * PI * t);
        P const d = v - u;
        vals[i] *= std::exp(- P{0.5} * d * d / t);
      }
    };

    #ifdef ASGARD_USE_GPU
    // If GPU capabilities are enabled in ASGarD, then it is preferable to use
    // the builtin BGK operator, which will invoke GPU kernels.
    // Using the CPU callable function fbgk is allowed, but it will result in
    // data back-forth between the CPU/GPU and will result in slower performance.
    pde += asgard::operators::simple_bgk_collisions{nu};
    // The adapt weight should reflect all interpolation terms of the pde_scheme
    // thus, the weight is not automatically set with the simple BGK operator.
    // If the scheme has multiple interpolatory terms then a different weight is needed,
    // but in this case, we can use the default builtin weight.
    pde.set_adapt_weight(asgard::operators::simple_bgk_collisions{nu});

    std::ignore = fbgk; // ignore the variable above, suppresses compiler warning
    #else
    // If GPU capabilities are not enabled, the builtin BGK operator is identical
    // to the one implemented in this example.
    pde += asgard::term_md<P>(nuI);
    pde += asgard::source<P>(fbgk, {im0, im1, im2});

    auto abgk = [=](P time, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> const &,
                    std::vector<P> &vals)
    {
      fbgk(time, nodes, moments, vals);
    };
    pde.set_adapt_weight(abgk, {im0, im1, im2});
    #endif

  } else if (dims == 2) {

    asgard::moment_id im0 = pde.register_moment(asgard::moment(0, 0));
    asgard::moment_id im10 = pde.register_moment(asgard::moment(1, 0));
    asgard::moment_id im01 = pde.register_moment(asgard::moment(0, 1));
    asgard::moment_id im20 = pde.register_moment(asgard::moment(2, 0));
    asgard::moment_id im02 = pde.register_moment(asgard::moment(0, 2));

    std::vector<asgard::moment_id> const mids = {im0, im10, im01, im20, im02};

    auto fbgk = [=](P /* time */, asgard::vector2d<P> const &nodes,
                    asgard::momentset<P> const &moments, std::vector<P> &vals)
    {
      std::vector<P> const &m0 = moments[im0];
      std::vector<P> const &m10 = moments[im10];
      std::vector<P> const &m01 = moments[im01];
      std::vector<P> const &m20 = moments[im20];
      std::vector<P> const &m02 = moments[im02];

      int64_t const num_nodes = nodes.num_strips();
      #pragma omp parallel for
      for (int64_t i = 0; i < num_nodes; i++) {
        P const n = m0[i];
        P const u0 = m10[i] / m0[i];
        P const u1 = m01[i] / m0[i];
        P const t = 0.5 * ((m20[i] + m02[i]) / m0[i] - u0 * u0 - u1 * u1);

        vals[i] = nu * n / (2 * PI * t);
        P const vu0 = nodes[i][2] - u0;
        P const vu1 = nodes[i][3] - u1;
        P const d = vu0 * vu0 + vu1 * vu1;
        vals[i] *= std::exp(- P{0.5} * d / t);
      }
    };

    P const adapt_scale = 10.0 * nu * dt;

    #ifdef ASGARD_USE_GPU
    // see the 1x1v case
    pde += asgard::operators::simple_bgk_collisions{nu};
    // using square initial condition and with weight divided by 10
    // and ./bgk -shock2d -nu 100 -m 8 -a 5.E-5 -n -> see pattern
    // add default adaptivity (??) think about it
    pde.set_adapt_weight(asgard::operators::simple_bgk_collisions{adapt_scale});
    std::ignore = fbgk;
    #else
    pde += asgard::term_md<P>(nuI);
    pde.set_source(fbgk, mids);

    pde.set_adapt_weight(asgard::operators::simple_bgk_collisions{adapt_scale});
    #endif

  } else /* if (dims == 3) */ {

    // the simple for of the BGK operator (shows above) is built into the ASGarD library
    // it can be used for any combination of position/velocity dimensions 1 - 3
    pde += asgard::operators::simple_bgk_collisions{nu};
    pde.set_adapt_weight(asgard::operators::simple_bgk_collisions{nu});
  }

  // set the implicit and explicit operator groups
  pde.set(asgard::imex_implicit_group{implicit_id},
          asgard::imex_explicit_group{explicit_id});

  // setting the initial conditions
  if (mode == pde_mode::poisson)
  {
    // separable initial conditions in x and v
    auto ic_x = [](std::vector<P> const &x, std::vector<P> &fx) ->
      void {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = 1.0 + 1.E-4 * std::cos(0.5 * x[i]);
      };

    auto ic_v = [](std::vector<P> const &v, std::vector<P> &fv) ->
      void {
        P const c = P{1} / std::sqrt(2 * PI);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * std::exp(-0.5 * v[i] * v[i]);
      };

    pde.add_initial(asgard::separable_func<P>({ic_x, ic_v}));
  }
  else if (mode == pde_mode::shock1d)
  {
    auto icmd = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &vals)
          -> void {

        // the initial condition always has a Maxwellian component in velocity
        // m * std::exp( - 0.5 * v * v / t)  / std::sqrt(2 * PI * t)
        // where m and t have spatial dependence in x
        // there are 3 regions, the inner and outer regions (m, t) is constant
        // at (1, 1) and (0.125, 0.8) respectively
        // in the transition region with size dr we have a smooth polynomial dependence

        P constexpr inner_m = 1;
        P constexpr outer_m = 0.125;

        P constexpr inner_t = 1;
        P constexpr outer_t = 0.8;

        P constexpr inner_bound = 0.3;
        P constexpr outer_bound = 0.4;
        P constexpr dr = outer_bound - inner_bound;

        P constexpr cm = 6 * (outer_m - inner_m) / (- dr * dr * dr);
        P constexpr ct = 6 * (outer_t - inner_t) / (- dr * dr * dr);

        // for r in inner_bound < r < outer_bound we have that m and s are given by
        // (r * r * r / 3 - 0.5 * dr * r * r) * cm/t + inner_m/t

        for (int64_t i = 0; i < nodes.num_strips(); i++) {
          P const x = nodes[i][0];
          P const v = nodes[i][1];

          P const ax = std::abs(x);

          if (ax <= inner_bound) {
            vals[i] = inner_m * exp(- P{0.5} * v * v / inner_t) / std::sqrt(2 * PI * inner_t);
          } else if (ax >= outer_bound) {
            vals[i] = outer_m * exp(- P{0.5} * v * v / outer_t) / std::sqrt(2 * PI * outer_t);
          } else {
            // transition region
            P const r = ax - inner_bound;
            P const m = (r * r * r / 3 - 0.5 * dr * r * r) * cm + inner_m;
            P const t = (r * r * r / 3 - 0.5 * dr * r * r) * ct + inner_t;
            vals[i] = m * exp(- P{0.5} * v * v / t) / std::sqrt(2 * PI * t);
          }
        }
      };

    pde.set_initial(icmd);
  }
  else
  {
    auto icmd = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &vals)
          -> void {

        // see the shock1d, using the same logic here except the three regions
        // form concentric circles

        P constexpr inner_m = 1;
        P constexpr outer_m = 0.125;

        P constexpr inner_t = 1;
        P constexpr outer_t = 0.8;

        P constexpr inner_bound = 0.38;
        P constexpr outer_bound = 0.42;
        P constexpr dr = outer_bound - inner_bound;

        P constexpr cm = 6 * (outer_m - inner_m) / (- dr * dr * dr);
        P constexpr ct = 6 * (outer_t - inner_t) / (- dr * dr * dr);

        for (int64_t i = 0; i < nodes.num_strips(); i++) {
          P const x0 = nodes[i][0];
          P const x1 = nodes[i][1];
          P const v0 = nodes[i][2];
          P const v1 = nodes[i][3];

          P const ax = std::abs(x0);
          P const ay = std::abs(x1);

          if (ax >= outer_bound or ay >= outer_bound) {
            vals[i] = outer_m * exp(- P{0.5} * v0 * v0 / outer_t)
                              * exp(- P{0.5} * v1 * v1 / outer_t) / (2 * PI * outer_t);
            continue;
          }

          if (ax <= inner_bound) {
            vals[i] = inner_m * std::exp(- P{0.5} * v0 * v0 / inner_t) / std::sqrt(2 * PI * inner_t);
          } else if (ax >= outer_bound) {
            vals[i] = std::sqrt(outer_m) * std::exp(- P{0.5} * v0 * v0 / outer_t) / std::sqrt(2 * PI * outer_t);
          } else {
            // transition region
            P const r = ax - inner_bound;
            P const m = (r * r * r / 3 - 0.5 * dr * r * r) * cm + inner_m;
            P const t = (r * r * r / 3 - 0.5 * dr * r * r) * ct + inner_t;
            vals[i] = std::sqrt(m) * std::exp(- P{0.5} * v0 * v0 / t) / std::sqrt(2 * PI * t);
          }

          if (ay <= inner_bound) {
            vals[i] *= inner_m * std::exp(- P{0.5} * v1 * v1 / inner_t) / std::sqrt(2 * PI * inner_t);
          } else if (ay >= outer_bound) {
            vals[i] *= std::sqrt(outer_m) * std::exp(- P{0.5} * v1 * v1 / outer_t) / std::sqrt(2 * PI * outer_t);
          } else {
            // transition region
            P const r = ay - inner_bound;
            P const m = (r * r * r / 3 - 0.5 * dr * r * r) * cm + inner_m;
            P const t = (r * r * r / 3 - 0.5 * dr * r * r) * ct + inner_t;

            vals[i] *= std::sqrt(m) * std::exp(- P{0.5} * v1 * v1 / t) / std::sqrt(2 * PI * t);
          }
        }
      };

    pde.set_initial(icmd);
  }

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk make]
#endif
}

/*!
 * \ingroup asgard_examples_bgk
 * \brief Computes the perturbation between the Maxwellian and the current state
 *
 * The initial condition is a small perturbation of a Maxwellian, which is not
 * visible on a regular plot of the state.
 * This method computes the difference between the current state of the
 * discretization and the final Maxwellian distribution.
 *
 * \tparam P is the precision to use, float or double
 *
 * \param disc is a discretization of a PDE created with make_vplb()
 *
 * \returns the difference between the current state and the Maxwellian
 *          projected on the current sparse grid
 *
 * \snippet bgk.cpp asgard_examples_bgk compute_perturbation
 */
template<typename P = asgard::default_precision>
std::vector<P> compute_perturbation(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk compute_perturbation]
#endif

  // The Maxwellian is the initial condition
  // but with constant value of 1.0 set in dimension 0 (the position dimension)
  asgard::separable_func<P> maxw = disc.initial_cond_sep().front();

  // set dimension 0 to be a constant function with value 1
  maxw.set(asgard::dimension_id{0}, P{1});

  // project the Maxwellian onto the current grid
  std::vector<P> proj_max = disc.project_function(maxw);

  // subtract the current state
  std::vector<P> const &state = disc.current_state();

  // the projected size will always match the size of the current state
  if (proj_max.size() != state.size())
    throw std::runtime_error("this will never happen");

  size_t n = state.size();
  for (size_t i = 0; i < n; i++)
    proj_max[i] -= state[i];

  return proj_max;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk compute_perturbation]
#endif
}

/*!
 * \ingroup asgard_examples_bgk
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls make_two_stream().
 *
 * \snippet bgk.cpp asgard_examples_bgk main
 *
 * This example also comes with a Python driver that show how to plot the moments of the stored
 * solution.
 *
 * \snippet bgk.py bgk_py python
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk main]
#endif

  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves the two stream Vlasov-Poisson in 1x-(1v-3v) dimensions\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << R"help(<< additional options for this file >>
-poisson                 -          sets a 1x1v problem similar to the vplb.cpp example
-shock1d                 -          sets a 1x1v problem with a shock in the initial cond.
-shock2d                 -          sets a 2x2v problem with a shock in the initial cond.
-nu                      double     accepts: a positive number
                                    collision frequency

-test                               perform self-testing
)help";
    return 0;
  }

  // this is an optional step, check if there are misspelled or incorrect cli entries
  // the first set/vector of entries are those that can appear by themselves
  // the second set/vector requires extra parameters
  options.throw_if_argv_not_in({"-test", "--test", "-poisson", "-shock1d", "-shock2d"}, {"-nu", });

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  pde_mode const mode = [&]() -> pde_mode {
        if (options.has_cli_entry("-shock2d"))
          return pde_mode::shock2d;
        else if (options.has_cli_entry("-shock1d"))
          return pde_mode::shock1d;
        else // if (options.has_cli_entry("-poisson"))
          return pde_mode::poisson;
      }();

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(make_bgk<P>(mode, options),
                                         asgard::verbosity_level::high);

  if (mode == pde_mode::poisson) {
    // save the perturbation as an auxiliary field, for plotting
    disc.add_aux_field({"initial perturbation", compute_perturbation(disc)});

    disc.advance_time(); // integrate until num-steps or stop-time

    // save the final perturbation
    disc.add_aux_field({"final perturbation", compute_perturbation(disc)});

  } else {

    disc.advance_time(); // integrate until num-steps or stop-time
  }

  disc.final_output();

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [asgard_examples_bgk main]
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

template<typename P>
void test_energy(pde_mode const mode, std::string const &opt_str) {
  int const dims = (mode == pde_mode::shock2d) ? 2 : 1;
  current_test<P> test_(opt_str, 2 * dims);
  // analytic solution is not available, hence we use energy conservation for
  // the test quantity in place of an L^2 error

  prog_opts const options = make_opts(opt_str);

  auto pde = make_bgk<P>(mode, options);
  moment_id const m0 = (dims == 1) ? pde.register_moment({0}) : pde.register_moment({0, 0});

  moment_id const m2 = (dims == 1) ? pde.register_moment({2}) : moment_id::unset();

  discretization_manager disc(std::move(pde), verbosity_level::quiet);

  int64_t const n = disc.remaining_steps();

  P constexpr tol = (is_double<P>) ? 5.E-7 : 5.E-3;

  // std::cout << std::scientific;
  // std::cout.precision(6);

  P const mass0   = disc.get_moment(m0)[0]; // initial total moments
  P const energy0 = disc.get_moment(m2)[0];

  for (int64_t i = 0; i < n; i++)
  {
    tassert( disc.advance_time(1) );

    P const mass = disc.get_moment(m0)[0];

    std::cout << std::abs(mass - mass0) << "  " << mass << "  " << mass0 << "\n";
    tassert(std::abs(mass - mass0) < tol);

    if (dims == 1) {
      P const energy = disc.get_moment(m2)[0];

      // std::cout << std::abs(energy - energy0) << "  " << energy << "  " << energy0 << "\n";
      tassert(std::abs(energy - energy0) < tol);
    }
  }
}

void self_test() {
  all_tests testing_("Bhatnagar-Gross-Krook");

#ifdef ASGARD_ENABLE_DOUBLE

  test_energy<double>(pde_mode::poisson, "-m 8 -n 100 -s imex1");
  test_energy<double>(pde_mode::poisson, "-m 8 -n 100 -s imex2");

#endif

#ifdef ASGARD_ENABLE_FLOAT

  test_energy<float>(pde_mode::poisson, "-m 8 -n 100 -s imex2");

#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
