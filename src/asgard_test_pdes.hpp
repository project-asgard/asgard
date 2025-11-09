#pragma once

#include "asgard.hpp"

/*!
 * \internal
 * \file asgard_testpde.cpp
 * \brief Simple PDEs used for testing, not included in the library
 * \author The ASGarD Team
 * \ingroup asgard_testing
 *
 * \endinternal
 */

namespace asgard
{

#ifndef __ASGARD_DOXYGEN_SKIP_INTERNAL

/*!
 * \internal
 * \ingroup asgard_testing
 * \brief Simpler version of the continuity-md example using cos-waves in +/- 1.5 PI
 *
 * \endinternal
 */
struct pde_contcos {};
/*!
 * \internal
 * \ingroup asgard_testing
 * \brief Most simple PDE using Poisson solver, but no analytic solution
 *
 * \endinternal
 */
struct pde_twostream {};
/*!
 * \internal
 * \ingroup asgard_testing
 * \brief Simple PDE using interpolation and imex solver
 *
 * \endinternal
 */
struct pde_burgers {};

#endif

/*!
 * \internal
 * \ingroup asgard_testing
 * \brief Creates a simple test pde for the given dimensions and options
 *
 * \endinternal
 */
template<typename pde_type, typename P>
pde_scheme<P> make_testpde(int num_dims, prog_opts options) {

  if constexpr (std::is_same_v<pde_type, pde_contcos>)
  {
    options.title = "Simplified Continuity " + std::to_string(num_dims) + "D";

    std::vector<domain_range> ranges(num_dims, {-1.5 * PI, 1.5 * PI});

    pde_domain<P> domain(ranges);

    int const max_level = options.max_level();

    P const dx = domain.min_cell_size(max_level);

    options.default_dt = 0.5 * 0.1 * dx;

    pde_scheme<P> pde(std::move(options), std::move(domain));

    term_1d<P> div = term_div<P>(1, flux_type::upwind, boundary_type::periodic);

    // the multi-dimensional divergence, initially set to identity in md
    std::vector<term_1d<P>> ops(num_dims);
    for (int d = 0; d < num_dims; d++)
    {
      ops[d] = div;
      pde += term_md<P>(ops);
      ops[d] = term_identity{};
    }

    // put the time-parameter inside one of the cos-functions
    // tests the non-separable in time capabilities
    auto cos_1t = [](std::vector<P> const &x, P t, std::vector<P> &fx) ->
        void {
        for (size_t i = 0; i < x.size(); i++)
            fx[i] = std::cos(t) * std::cos(x[i]);
        };

    std::vector<svector_func1d<P>> func_md(num_dims, builtin_t<P>::cos);

    func_md[0] = cos_1t;
    pde.add_initial(func_md);
    func_md[0] = builtin_t<P>::cos;

    pde.add_source({func_md, builtin_s<P>::dcos}); // derivative in time

    func_md[0] = builtin_t<P>::dcos;
    pde.add_source({func_md, builtin_s<P>::cos});
    func_md[0] = cos_1t;

    // compute the spacial derivatives
    for (int d = 1; d < num_dims; d++)
    {
      func_md[d] = builtin_t<P>::dcos;
      pde.add_source(func_md);
      func_md[d] = builtin_t<P>::cos;
    }

    return pde;
  }
  else if constexpr (std::is_same_v<pde_type, pde_twostream>)
  {
    options.title = "Test Two Stream Instability";

    // the domain has one position and one velocity dimension: 1x1v
    pde_domain<P> domain(position_dims{1}, velocity_dims{1},
                         {{-2 * PI, 2 * PI}, {-2 * PI, 2 * PI}});

    options.default_degree = 2;

    // the CFL is more complicated
    int const k = options.degree.value_or(options.default_degree.value());
    int const n = (1 << options.max_level());
    options.default_dt = 3.0 / (2 * (2 * k + 1) * n);

    options.default_step_method = time_method::rk2;

    pde_scheme<P> pde(options, domain);

    pde += term_md<P>(std::vector<term_1d<P>>{
        term_div<P>(1, flux_type::upwind, boundary_type::periodic),
        term_volume<P>(builtin_v<P>::positive)
      });

    pde += term_md<P>(std::vector<term_1d<P>>{
        term_div<P>(1, flux_type::downwind, boundary_type::periodic),
        term_volume<P>(builtin_v<P>::negative),
      });

    pde += term_md<P>(std::vector<term_1d<P>>{
        volume_electric<P>(builtin_v<P>::positive),
        term_div<P>(1, flux_type::upwind, boundary_type::bothsides)
      });

    pde += term_md<P>(std::vector<term_1d<P>>{
        volume_electric<P>(builtin_v<P>::negative),
        term_div<P>(1, flux_type::downwind, boundary_type::bothsides)
      });

    // initial conditions in x and v
    auto ic_x = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
      void {
        for (size_t i = 0; i < x.size(); i++)
          fx[i] = 1.0 - 0.5 * std::cos(0.5 * x[i]);
      };

    auto ic_v = [](std::vector<P> const &v, P /* time */, std::vector<P> &fv) ->
      void {
        P const c = P{1} / std::sqrt(PI);

        for (size_t i = 0; i < v.size(); i++)
          fv[i] = c * v[i] * v[i] * std::exp(-v[i] * v[i]);
      };

    pde.add_initial(asgard::separable_func<P>({ic_x, ic_v}));

    moment_id const m0 = pde.register_moment({0, moment::inactive});
    moment_id const m1 = pde.register_moment({1, moment::inactive});
    moment_id const m2 = pde.register_moment({2, moment::inactive});

    return pde;
  }
  else if constexpr (std::is_same_v<pde_type, pde_burgers>)
  {
    options.title = "Burgers PDE 2D";

    P const nu = 0.01;

    // the 1D case is set on (-8, 8), the higher dimensions use (-1, 1)^d
    asgard::pde_domain<P> domain =
        asgard::pde_domain<P>(std::vector<asgard::domain_range>(num_dims, {-1.0, 1.0}));

    options.default_degree = 2;
    options.default_start_levels = {6, };

    // the inviscit equation can be done with an explicit time stepper
    options.default_step_method = asgard::time_method::imex1;

    P const dx = domain.min_cell_size(options.max_level());
    options.default_dt = 0.05 * dx;
    options.default_stop_time = 0.25;

    options.default_solver = asgard::solver_method::bicgstab;

    // defaults for iterative solvers, not necessarily optimal
    options.default_isolver_tolerance  = 1.E-8;
    options.default_isolver_iterations = 1000;

    asgard::pde_scheme<P> pde(options, std::move(domain));

    auto f2p = [=](P, asgard::vector2d<P> const &,
                  std::vector<P> const &f, std::vector<P> &vals) ->
      void {
        for (size_t i = 0; i < f.size(); i++) {
          vals[i] = (f[i] > 0) ? f[i] * f[i] : 0;
        }
      };
    auto f2n = [=](P, asgard::vector2d<P> const &,
                  std::vector<P> const &f, std::vector<P> &vals) ->
      void {
        for (size_t i = 0; i < f.size(); i++) {
          vals[i] = (f[i] < 0) ? f[i] * f[i] : 0;
        }
      };

    // setting up multidimensional volume term that uses interpolated coefficient
    asgard::term_md<P> term_f2_pos = asgard::term_interp<P>{f2p};
    asgard::term_md<P> term_f2_neg = asgard::term_interp<P>{f2n};

    // initial conditions and derivatives in x and y, also the exact solution in time
    auto icx   = [](P x) -> P { return P{1} + P{0.75} * x - P{0.25} * x * x; };
    auto icdx  = [](P x) -> P { return P{0.75} - P{0.5} * x; };
    auto icdxx = [](P) -> P { return - P{0.5}; };

    auto icy   = [](P y) -> P { return (P{1} - y * y); };
    auto icdy  = [](P y) -> P { return -2 * y; };
    auto icdyy = [](P) -> P { return -2; };

    auto exact_t = [](P t) -> P { return std::exp(-t); };

    // boundary conditions in y are homogeneous and simple to impose to all terms
    // boundary conditions in x are imposed only on the second order term
    asgard::term_md<P> divx_pos = {asgard::term_div<P>{0.5, asgard::flux_type::upwind},
                                    asgard::term_identity{}};
    asgard::term_md<P> divx_neg = {asgard::term_div<P>{0.5, asgard::flux_type::downwind},
                                    asgard::term_identity{}};
    asgard::term_md<P> divy_pos = {asgard::term_identity{},
                                    asgard::term_div<P>{0.5, asgard::boundary_type::left,
                                                        asgard::flux_type::upwind}, };
    asgard::term_md<P> divy_neg = {asgard::term_identity{},
                                    asgard::term_div<P>{0.5, asgard::boundary_type::right,
                                                        asgard::flux_type::downwind}, };

    // the group ids are needed for IMEX scheme in the viscous way
    int const non_linear_group_id = pde.new_term_group();

    pde += asgard::term_md<P>{divx_pos, term_f2_pos};
    pde += asgard::term_md<P>{divx_neg, term_f2_neg};
    pde += asgard::term_md<P>{divy_pos, term_f2_pos};
    pde += asgard::term_md<P>{divy_neg, term_f2_neg};

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

  } else {
    rassert(false, "Incorrect pde type for make_testpde");
  }
}

/*!
 * \internal
 * \ingroup asgard_testing
 * \brief Returns an indicator of the "health" of the PDE
 *
 * In cases when the PDE has a known analytic solution, this will simply return
 * the L^2 error between the current state and the known solution.
 *
 * In other cases, the indicator can be different, e.g., measuring energy conservation.
 *
 * \endinternal
 */
template<typename pde_type, typename P>
double get_qoi_indicator(asgard::discretization_manager<P> const &disc) {

  if constexpr (std::is_same_v<pde_type, pde_twostream>)
  {
    // there is no analytic solution, using the sum of particle potential and kinetic
    // energy as the indicator, it is not zero but must be near constant

    int const level0   = disc.get_grid().current_level(0);
    int const num_cell = fm::ipow2(level0);
    P const dx         = disc.domain().length(0) / num_cell;

    auto efield = disc.get_electric();

    P Ep = 0;
    for (auto e : efield) Ep += e * e;
    Ep *= dx;

    std::vector<P> mom2 = disc.get_moment(moment_id{2}); // cheating here, why exactly 2

    P const Ek = mom2[0] * std::sqrt(disc.domain().length(0));

    return 0.5 * (Ep + Ek);
  }

  if constexpr (std::is_same_v<pde_type, pde_burgers>)
  {
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
  }

  int const num_dims = disc.num_dims();

  std::vector<P> const eref = disc.project_function(disc.initial_cond_sep());

  auto [space1d, timev] = [&]() -> std::array<double, 2> {
    if constexpr (std::is_same_v<pde_type, pde_contcos>) {
      return {1.5 * PI, std::cos(disc.time())};
    } else { // no analytic solution, code will be intercepted above
      return {0, 0};
    }
  }();

  double const enorm = asgard::fm::powi(space1d, num_dims) * timev * timev;

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

  if (enorm < 1)
    return std::sqrt(ndiff + enorm - nself);
  else
    return std::sqrt((ndiff + enorm - nself) / enorm);
}

/*!
 * \internal
 * \ingroup asgard_testing
 * \brief Using the given PDE type and opts, integrate step-by-step and return max-L^2 error
 *
 * \endinternal
 */
template<typename pde_type, typename P>
double get_time_error(int num_dims, std::string const &opts) {

  auto options = make_opts(opts);

  discretization_manager<P> disc(make_testpde<pde_type, P>(num_dims, options));

  double max_err = 0;

  while (disc.time_params().num_remain() > 0)
  {
    advance_time(disc, 1);

    max_err = std::max(max_err, get_qoi_indicator<pde_type>(disc));
  }

  return max_err;
}

} // namespace asgard
