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
    auto cos_1t = [](std::vector<P> const &x, P t, std::vector<P> &fx) ->
        void {
        for (size_t i = 0; i < x.size(); i++)
            fx[i] = std::cos(t) * std::cos(x[i]);
        };
    auto cos_1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
        void {
        for (size_t i = 0; i < x.size(); i++)
            fx[i] = std::cos(x[i]);
        };

    // the derivatives, d/dx cos(x) = -sin(x) and d/dt cos(t) = -sin(t)
    auto sin_1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
        void {
        for (size_t i = 0; i < x.size(); i++)
            fx[i] = -std::sin(x[i]);
        };
    auto cos_t  = [](P t) -> P { return  std::cos(t); };
    auto nsin_t = [](P t) -> P { return -std::sin(t); };

    std::vector<svector_func1d<P>> func_md(num_dims, cos_1d);

    func_md[0] = cos_1t;
    pde.add_initial(func_md);
    func_md[0] = cos_1d;

    pde.add_source({func_md, nsin_t}); // derivative in time

    func_md[0] = sin_1d;
    pde.add_source({func_md, cos_t});
    func_md[0] = cos_1t;

    // compute the spacial derivatives
    for (int d = 1; d < num_dims; d++)
    {
        func_md[d] = sin_1d;
        pde.add_source(func_md);
        func_md[d] = cos_1d;
    }

    return pde;
  }
}

/*!
 * \internal
 * \ingroup asgard_testing
 * \brief Returns the L^2 error for the current simple PDE
 *
 * \endinternal
 */
template<typename pde_type, typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc) {

  int const num_dims = disc.num_dims();

  std::vector<P> const eref = disc.project_function(disc.get_pde2().ic_sep());

  auto [space1d, timev] = [&]() -> std::array<double, 2> {
    if constexpr (std::is_same_v<pde_type, pde_contcos>) {
      return {1.5 * PI, std::cos(disc.time_params().time())};
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

    max_err = std::max(max_err, get_error_l2<pde_type>(disc));
  }

  return max_err;
}

} // namespace asgard
