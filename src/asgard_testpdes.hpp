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

#endif

/*!
 * \internal
 * \ingroup asgard_testing
 * \brief Creates a simple test pde for the given dimensions and options
 *
 * \endinternal
 */
template<typename pde_type, typename P>
PDEv2<P> make_testpde(int num_dims, prog_opts options) {

  if constexpr (std::is_same_v<pde_type, pde_contcos>)
  {
    options.title = "Simplified Continuity " + std::to_string(num_dims) + "D";

    std::vector<domain_range<P>> ranges(num_dims, {-P{3} * PI / P{2}, P{3} * PI / P{2}});

    pde_domain<P> domain(ranges);

    int const max_level = options.max_level();

    P const dx = domain.min_cell_size(max_level);

    options.default_dt = 0.5 * 0.1 * dx;

    PDEv2<P> pde(std::move(options), std::move(domain));

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

    PDEv2<P> pde(options, domain);

    pde += term_md<P>(std::vector<term_1d<P>>{
        term_div<P>(1, flux_type::upwind, boundary_type::periodic),
        term_mass<P>(builtin_v<P>::positive)
      });

    pde += term_md<P>(std::vector<term_1d<P>>{
        term_div<P>(1, flux_type::downwind, boundary_type::periodic),
        term_mass<P>(builtin_v<P>::negative),
      });

    pde += term_md<P>(std::vector<term_1d<P>>{
        mass_electric<P>(builtin_v<P>::positive),
        term_div<P>(1, flux_type::upwind, boundary_type::dirichlet)
      });

    pde += term_md<P>(std::vector<term_1d<P>>{
        mass_electric<P>(builtin_v<P>::negative),
        term_div<P>(1, flux_type::downwind, boundary_type::dirichlet)
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
    int const num_moms = 3;
    int const pdof     = disc.degree() + 1;
    moments1d moms(num_moms, pdof - 1, disc.get_pde2().max_level(),
                   disc.get_pde2().domain());
    std::vector<P> mom_vec;

    moms.project_moments(disc.get_sgrid(), disc.current_state(), mom_vec);

    disc.do_poisson_update(disc.current_state()); // update the electric field

    auto const &efield = disc.get_terms().cdata.electric_field;

    int const level0   = disc.get_sgrid().current_level(0);
    int const num_cell = fm::ipow2(level0);
    double const dx    = disc.get_pde2().domain().length(0) / num_cell;

    double Ep = 0;
    for (auto e : efield)
      Ep += e * e;
    Ep *= dx;

    span2d<P> moments(num_moms * pdof, num_cell, mom_vec.data());

    double Ek = 0;
    for (int j : iindexof(num_cell))
      Ek += moments[j][2 * pdof]; // integrating the third moment
    Ek *= std::sqrt(disc.get_pde2().domain().length(0));

    return Ep + Ek;
  }

  int const num_dims = disc.num_dims();

  std::vector<P> const eref = disc.project_function(disc.get_pde2().ic_sep());

  auto [space1d, timev] = [&]() -> std::array<double, 2> {
    if constexpr (std::is_same_v<pde_type, pde_contcos>) {
      return {1.5 * PI, std::cos(disc.time_params().time())};
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
