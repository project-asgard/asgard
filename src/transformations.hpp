#pragma once

#include "element_table.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

template<typename P>
std::array<fk::matrix<P>, 6> generate_multi_wavelets(int const degree);

extern template std::array<fk::matrix<double>, 6>
generate_multi_wavelets(int const degree);
extern template std::array<fk::matrix<float>, 6>
generate_multi_wavelets(int const degree);

template<typename P>
fk::vector<P> combine_dimensions(Options const, element_table const &,
                                 std::vector<fk::vector<P>> const &, P const);

extern template fk::vector<double>
combine_dimensions(Options const, element_table const &,
                   std::vector<fk::vector<double>> const &, double const);
extern template fk::vector<float>
combine_dimensions(Options const, element_table const &,
                   std::vector<fk::vector<float>> const &, float const);

template<typename P>
fk::matrix<P> operator_two_scale(Options const opts);

extern template fk::matrix<double> operator_two_scale(Options const opts);
extern template fk::matrix<float> operator_two_scale(Options const opts);

// FIXME this interface is temporary. lmin, lmax, degree, and level
// will all be encapsulated by a dimension argument (member of pde)
// I haven't refitted the pde class yet, though, so that class doesn't
// exist yet - TM
template<typename P, typename F>
fk::vector<P> forwardMWT(Options const opts, P const domain_min,
                         P const domain_max, F function)
{
  int const num_levels = opts.get_level();
  int const degree     = opts.get_degree();

  assert(num_levels > 0);
  assert(degree > 0);
  assert(domain_max > domain_min);
  static_assert(std::is_invocable_v<decltype(function), fk::vector<P>>);

  // TODO may remove this call if want to create the FMWT matrix once and store
  // it in the appropriate dimension object passed to this function

  fk::matrix<P> const forward_transform = operator_two_scale<P>(opts);

  // get the Legendre-Gauss nodes and weights on the domain
  // [-1,+1] for performing quadrature.
  int const quadrature_num    = 10;
  auto const [roots, weights] = legendre_weights(
      quadrature_num, static_cast<P>(-1.0), static_cast<P>(1.0));

  // get the Legendre basis function evaluated at the Legendre-Gauss nodes up
  // to order k.
  fk::matrix<P> const basis = legendre(roots, degree)[0];

  // get grid spacing.

  // hate this name TODO
  int const n                  = static_cast<int>(std::pow(2, num_levels));
  P normalize                  = (domain_max - domain_min) / n;
  int const degrees_freedom_1d = degree * n;

  // this will be our return vector
  fk::vector<P> f(degrees_freedom_1d);

  // initial condition for f
  // hate this name also TODO
  fk::vector<P> x(n);
  for (int i = 0; i < n; ++i)
  {
    // map quad_x from [-1,+1] to [domain_min,domain_max] physical domain.
    for (int j = 0; j < n; ++j)
    {
      x[j] = normalize * (roots[j] / 2.0 + 1.0 / 2.0 + i) + domain_min;
    }

    // get the f(v) initial condition at the quadrature points.
    fk::vector<P> f_here = function(x);
    assert(f_here.size() == weights.size());
    std::transform(f_here.begin(), f_here.end(), weights.begin(),
                   f_here.begin(), std::multiplies<P>());

    // generate the coefficients for DG basis
    fk::vector<P> coeffs = basis * f_here;
    f.set(i * degree, coeffs);
  }
  f = f * (normalize * std::sqrt(static_cast<P>(1.0) / normalize) /
           static_cast<P>(2.0));

  // transfer to multi-DG bases
  return forward_transform * f;
}
