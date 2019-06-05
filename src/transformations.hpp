#pragma once

#include "element_table.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

template<typename P>
fk::vector<P>
combine_dimensions(int const, element_table const &,
                   std::vector<fk::vector<P>> const &, P const = 1.0);

template<typename P, typename F>
fk::vector<P> forward_transform(dimension<P> const &dim, F function)
{
  int const num_levels = dim.get_level();
  int const degree     = dim.get_degree();
  P const domain_min   = dim.domain_min;
  P const domain_max   = dim.domain_max;

  assert(num_levels > 0);
  assert(degree > 0);
  assert(domain_max > domain_min);

  // check to make sure the F function arg is a function type
  // that will accept a vector argument. we have a check for its
  // return below
  static_assert(std::is_invocable_v<decltype(function), fk::vector<P>>);

  fk::matrix<P> const forward_trans(dim.get_to_basis_operator());

  // get the Legendre-Gauss nodes and weights on the domain
  // [-1,+1] for performing quadrature.
  // we do the two-step store because we cannot have 'static' bindings
  int const quadrature_num = 10;
  static const auto legendre_values =
      legendre_weights<P>(quadrature_num, -1, 1);
  auto const [roots, weights] = legendre_values;

  // get grid spacing.
  // hate this name TODO
  int const n                  = two_raised_to(num_levels);
  int const degrees_freedom_1d = degree * n;

  // get the Legendre basis function evaluated at the Legendre-Gauss nodes   //
  // up to order k
  P const normalize         = (domain_max - domain_min) / n;
  fk::matrix<P> const basis = [&roots = roots, degree, normalize] {
    fk::matrix<P> legendre_ = legendre<P>(roots, degree)[0];
    return legendre_.transpose() * (static_cast<P>(1.0) / std::sqrt(normalize));
  }();

  // this will be our return vector
  fk::vector<P> transformed(degrees_freedom_1d);

  // initial condition for f
  // hate this name also TODO

  for (int i = 0; i < n; ++i)
  {
    // map quad_x from [-1,+1] to [domain_min,domain_max] physical domain.
    fk::vector<P> const mapped_roots = [&roots = roots, normalize, domain_min,
                                        i]() {
      fk::vector<P> mapped_roots(roots.size());
      std::transform(mapped_roots.begin(), mapped_roots.end(), roots.begin(),
                     mapped_roots.begin(), [&](P &elem, P const &root) {
                       return elem + (normalize * (root / 2.0 + 1.0 / 2.0 + i) +
                                      domain_min);
                     });
      return mapped_roots;
    }();

    // get the f(v) initial condition at the quadrature points.
    fk::vector<P> f_here = function(mapped_roots);
    // ensuring function returns vector of appropriate size
    assert(f_here.size() == weights.size());
    std::transform(f_here.begin(), f_here.end(), weights.begin(),
                   f_here.begin(), std::multiplies<P>());

    // generate the coefficients for DG basis
    fk::vector<P> coeffs = basis * f_here;

    transformed.set_subvector(i * degree, coeffs);
  }
  transformed = transformed * (normalize / 2.0);

  // transfer to multi-DG bases
  transformed = forward_trans * transformed;

  // zero out near-zero values resulting from transform to wavelet space
  std::transform(transformed.begin(), transformed.end(), transformed.begin(),
                 [](P &elem) {
                   P const compare = [] {
                     if constexpr (std::is_same<P, double>::value)
                     {
                       return static_cast<P>(1e-12);
                     }
                     return static_cast<P>(1e-4);
                   }();
                   return std::abs(elem) < compare ? static_cast<P>(0.0) : elem;
                 });

  return transformed;
}

extern template fk::vector<double>
combine_dimensions(int const, element_table const &,
                   std::vector<fk::vector<double>> const &, double const);
extern template fk::vector<float>
combine_dimensions(int const, element_table const &,
                   std::vector<fk::vector<float>> const &, float const);
