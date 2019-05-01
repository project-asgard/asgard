#include "quadrature.hpp"
#include <iostream>

#include "matlab_utilities.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
// Evaluate Legendre polynomials on an input domain, trimmed to [-1,1]
// Virtually a direct translation of Ed's dlegendre2.m code
//
// Legendre matrix returned in the std::array[0], its derivative returned in [1]
// FIXME (is this right?) Each column of the matrix is such a polynomial (or its
// derivative) for each degree

template<typename P>
std::enable_if_t<std::is_floating_point<P>::value, std::array<fk::matrix<P>, 2>>
legendre(fk::vector<P> const domain, int const degree)
{
  assert(degree >= 0);
  assert(domain.size() > 0);

  // allocate and zero the output Legendre polynomials, their derivatives
  fk::matrix<P> legendre(domain.size(), std::max(1, degree));
  fk::matrix<P> legendre_prime(domain.size(), std::max(1, degree));

  legendre.update_col(0, std::vector<P>(domain.size(), static_cast<P>(1.0)));

  if (degree >= 2)
  {
    legendre.update_col(1, domain);
    legendre_prime.update_col(
        1, std::vector<P>(domain.size(), static_cast<P>(1.0)));
  }

  // if we are working to update column "n", then "_order" is the previous
  // column (i.e. n-1), and "_(order + 1)" is the one before that
  if (degree >= 3)
  {
    // initial values for n-1, n-2
    fk::vector<P> legendre_order =
        legendre.extract_submatrix(0, 1, domain.size(), 1);
    fk::vector<P> legendre_prime_order =
        legendre_prime.extract_submatrix(0, 1, domain.size(), 1);
    fk::vector<P> legendre_n_1 =
        legendre.extract_submatrix(0, 0, domain.size(), 1);
    fk::vector<P> legendre_prime_n_1 =
        legendre_prime.extract_submatrix(0, 0, domain.size(), 1);

    // set remaining columns
    for (int i = 0; i < (degree - 2); ++i)
    {
      int const n            = i + 1;
      int const column_index = i + 2;

      // element-wise multiplication
      fk::vector<P> product(domain.size());
      std::transform(domain.begin(), domain.end(), legendre_order.begin(),
                     product.begin(), std::multiplies<P>());

      P const factor = 1.0 / (n + 1.0);

      fk::vector<P> legendre_col = (product * static_cast<P>(2.0 * n + 1.0)) -
                                   (legendre_n_1 * static_cast<P>(n));
      legendre_col = legendre_col * factor;
      legendre.update_col(column_index, legendre_col);

      std::transform(domain.begin(), domain.end(), legendre_prime_order.begin(),
                     product.begin(), std::multiplies<P>());

      fk::vector<P> legendre_prime_col =
          (product + legendre_order) * static_cast<P>(2.0 * n + 1.0) -
          legendre_prime_n_1 * static_cast<P>(n);
      legendre_prime_col = legendre_prime_col * factor;
      legendre_prime.update_col(column_index, legendre_prime_col);

      // update columns for next iteration
      legendre_n_1         = legendre_order;
      legendre_order       = legendre_col;
      legendre_prime_n_1   = legendre_prime_order;
      legendre_prime_order = legendre_prime_col;
    }
  }

  // "normalizing"
  for (int i = 0; i < degree; ++i)
  {
    P const norm_2 = static_cast<P>(2.0) / (2.0 * i + 1.0);
    P const dscale = static_cast<P>(1.0) / std::sqrt(norm_2);

    fk::vector<P> const legendre_sub =
        legendre.extract_submatrix(0, i, domain.size(), 1);
    legendre.update_col(i, legendre_sub * dscale);

    fk::vector<P> const legendre_prime_sub =
        legendre_prime.extract_submatrix(0, i, domain.size(), 1);
    legendre_prime.update_col(i, legendre_prime_sub * dscale);
  }

  // "zero out points out of range"
  fk::vector<int> const out_of_range = find(domain, [](P const &elem) {
    return elem < static_cast<P>(-1.0) || elem > static_cast<P>(1.0);
  });
  for (int i : out_of_range)
  {
    legendre.update_row(
        i, std::vector<P>(std::max(degree, 1), static_cast<P>(0.0)));
    legendre_prime.update_row(
        i, std::vector<P>(std::max(degree, 1), static_cast<P>(0.0)));
  }

  if (degree > 0)
  {
    // "scaling to use normalization
    legendre       = legendre * static_cast<P>(std::sqrt(2.0));
    legendre_prime = legendre_prime * static_cast<P>(std::sqrt(2.0));
  }
  return {legendre, legendre_prime};
}

// From the matlab:

//% lgwt.m
//% This script is for computing definite integrals using Legendre-Gauss
//% Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
//% [a,b] with truncation order N
//%
//% Suppose you have a continuous function f(x) which is defined on [a,b]
//% which you can evaluate at any x in [a,b]. Simply evaluate it at all of
//% the values contained in the x vector to obtain a vector f. Then compute
//% the definite integral using sum(f.*w);
//%
//% Written by Greg von Winckel - 02/25/2004

// return[0] are the roots, return[1] are the weights

// FIXME we need to rename "xu", "y", "y0", etc., with meaningful
// variable names. need to discuss w tim.
template<typename P>
std::array<fk::vector<P>, 2>
legendre_weights(const int order, const int interval_start,
                 const int interval_end)
{
  assert(order > 0);
  assert(interval_start < interval_end);

  // prepare out vectors
  fk::vector<P> roots(order);
  fk::vector<P> weights(order);

  // xu=linspace(-1,1,N1)';
  fk::vector<P> const xu =
      linspace(static_cast<P>(-1.0), static_cast<P>(1.0), order);

  //% Initial guess
  // y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2);
  fk::vector<P> y =
      linspace(static_cast<P>(0.0), static_cast<P>((order - 1)), order);
  std::transform(y.begin(), y.end(), y.begin(), [&](P &elem) {
    return std::cos((2 * elem + 1) * M_PI /
                    static_cast<P>((2 * (order - 1) + 2)));
  });

  fk::vector<P> y2(xu);
  std::transform(y2.begin(), y2.end(), y2.begin(), [&](P &elem) {
    return (static_cast<P>(0.27) / order) *
           std::sin(M_PI * elem * (order - 1) / (order + 1));
  });

  y = y + y2;

  //% Legendre-Gauss Vandermonde Matrix
  // L=zeros(N1,N2);
  fk::matrix<P> legendre_gauss(order, (order + 1));
  fk::vector<P> legendre_p(order);

  //% Compute the zeros of the N+1 Legendre Polynomial
  // y0=2
  fk::vector<P> y0(order);
  std::fill(y0.begin(), y0.end(), static_cast<P>(2.0));
  P const eps = std::numeric_limits<P>::epsilon();

  //% Iterate until new points are uniformly within epsilon of old points
  // while max(abs(y-y0))>eps
  fk::vector<P> diff(order);
  auto const abs_diff = [&](P const &y_elem, P const &y0_elem) {
    return std::fabs(y_elem - y0_elem);
  };
  std::transform(y.begin(), y.end(), y0.begin(), diff.begin(), abs_diff);

  while (*std::max_element(diff.begin(), diff.end()) > eps)
  {
    // L(:,1)=1;
    legendre_gauss.update_col(
        0, std::vector<P>(legendre_gauss.nrows(), static_cast<P>(1.0)));

    // L(:,2)=y;
    legendre_gauss.update_col(1, y);

    // for k=2:N1
    // we set the i+1th column of L at each iter
    for (int i = 1; i < order; ++i)
    {
      fk::vector<P> const prev =
          legendre_gauss.extract_submatrix(0, i - 1, legendre_gauss.nrows(), 1);
      fk::vector<P> const current =
          legendre_gauss.extract_submatrix(0, i, legendre_gauss.nrows(), 1);

      fk::vector<P> next(current.size());

      // this loop for setting the next column is a little obscure, but doing
      // step by step with transforms was prohibitively slow when we invoke this
      // function from multiwavelet gen.
      P scale = (static_cast<P>(2.0) * (i + 1) - 1);
      for (int j = 0; j < next.size(); ++j)
      {
        next(j) = ((y(j) * scale * current(j)) - (prev(j) * i)) /
                  static_cast<P>(i + 1);
      }
      legendre_gauss.update_col(i + 1, next);
    }

    // Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);
    fk::vector<P> const legendre_n1 = legendre_gauss.extract_submatrix(
        0, order - 1, legendre_gauss.nrows(), 1);
    fk::vector<P> legendre_n2 = legendre_gauss.extract_submatrix(
        0, (order + 1) - 1, legendre_gauss.nrows(), 1);
    fk::vector<P> legendre_n2_scaled(legendre_n2.size());
    std::transform(legendre_n2.begin(), legendre_n2.end(), y.begin(),
                   legendre_n2_scaled.begin(), std::multiplies<P>());
    fk::vector<P> const y_operand = [&] {
      fk::vector<P> copy_y(y.size());
      std::transform(
          y.begin(), y.end(), y.begin(), copy_y.begin(),
          [](P &y, P &y_same) { return static_cast<P>(1.0) - y * y_same; });
      return copy_y;
    }();
    legendre_p = (legendre_n1 - legendre_n2_scaled) * (order + 1);
    auto const element_division = [](P const &one, P const &two) {
      return one / two;
    };
    std::transform(legendre_p.begin(), legendre_p.end(), y_operand.begin(),
                   legendre_p.begin(), element_division);

    y0 = y;

    // y=y0-L(:,N2)./Lp;
    std::transform(legendre_n2.begin(), legendre_n2.end(), legendre_p.begin(),
                   legendre_n2.begin(), element_division);
    y = y0 - legendre_n2;

    // diff = abs(y-y0)
    std::transform(y.begin(), y.end(), y0.begin(), diff.begin(), abs_diff);
  }

  //% Linear map from[-1,1] to [a,b]
  // x=(a*(1-y)+b*(1+y))/2;
  std::transform(y.begin(), y.end(), roots.begin(), [&](P &elem) {
    return (interval_start * (1 - elem) + interval_end * (1 + elem)) / 2;
  });

  //% Compute the weights
  // w=(b-a)./((1-y.^2).*Lp.^2)*(N2/N1)^2;
  std::transform(y.begin(), y.end(), legendre_p.begin(), weights.begin(),
                 [&](P &y_elem, P &lp_elem) {
                   return (interval_end - interval_start) /
                          ((static_cast<P>(1.0) - y_elem * y_elem) * lp_elem *
                           lp_elem) *
                          std::pow(static_cast<P>((order + 1)) / order, 2);
                 });

  // x=x(end:-1:1);
  // w=w(end:-1:1);
  std::reverse(roots.begin(), roots.end());
  std::reverse(weights.begin(), weights.end());

  return std::array<fk::vector<P>, 2>{roots, weights};
}

// explicit instatiations
template std::array<fk::matrix<float>, 2>
legendre(fk::vector<float> const domain, int const degree);
template std::array<fk::matrix<double>, 2>
legendre(fk::vector<double> const domain, int const degree);

template std::array<fk::vector<float>, 2>
legendre_weights(const int n, const int a, const int b);
template std::array<fk::vector<double>, 2>
legendre_weights(const int n, const int a, const int b);
