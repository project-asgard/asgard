#pragma once

#include "element_table.hpp"
#include "program_options.hpp"
#include "quadrature.hpp"
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
fk::vector<P> forward_transform(Options const opts, P const domain_min,
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

  fk::matrix<P> const forward_trans = operator_two_scale<P>(opts);

  // get the Legendre-Gauss nodes and weights on the domain
  // [-1,+1] for performing quadrature.
  int const quadrature_num    = 10;
  auto const [roots, weights] = legendre_weights<P>(quadrature_num, -1, 1);

  // get grid spacing.
  // hate this name TODO
  int const n                  = static_cast<int>(std::pow(2, num_levels));
  int const degrees_freedom_1d = degree * n;

  // get the Legendre basis function evaluated at the Legendre-Gauss nodes   //
  // up to order k
  P const normalize         = (domain_max - domain_min) / n;
  fk::matrix<P> const basis = [&roots = roots, degree, normalize] {
    fk::matrix<P> legendre_ = legendre<P>(roots, degree)[0];
    return legendre_.transpose() * (static_cast<P>(1.0) / std::sqrt(normalize));
  }();

  // this will be our return vector
  fk::vector<P> f(degrees_freedom_1d);

  // initial condition for f
  // hate this name also TODO

  for (int i = 0; i < n; ++i)
  {
    fk::vector<P> const x = [&roots = roots, normalize, domain_min, i]() {
      fk::vector<P> x(roots.size());
      // map quad_x from [-1,+1] to [domain_min,domain_max] physical domain.
      std::transform(x.begin(), x.end(), roots.begin(), x.begin(),
                     [&](P &elem, P const &root) {
                       return elem + (normalize * (root / 2.0 + 1.0 / 2.0 + i) +
                                      domain_min);
                     });
      return x;
    }();

    // get the f(v) initial condition at the quadrature points.
    fk::vector<P> f_here = function(x);

    assert(f_here.size() == weights.size());
    std::transform(f_here.begin(), f_here.end(), weights.begin(),
                   f_here.begin(), std::multiplies<P>());

    basis.print("p_val");
    // generate the coefficients for DG basis
    fk::vector<P> coeffs = basis * f_here;
    f.set(i * degree, coeffs);
    f.print("f");
  }
  f = f * (normalize / static_cast<P>(2.0));

  // transfer to multi-DG bases
  f = forward_trans * f;

  // zero out near-zero values resulting from transform to wavelet space
  std::transform(f.begin(), f.end(), f.begin(), [](P &elem) {
    return std::abs(elem) < 1e-12 ? static_cast<P>(0.0) : elem;
  });

  return f;
}
