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
//% This script is for computing the Legendre-Gauss nodes (roots on x) and weights  
//% on an interval
//% [interval_start,interval_end] for Legendre polynomial degree polynomial_degree
//% These are later used for computing definite integrals using Legendre-Gauss
//% Quadrature.
//%
//% Suppose you have a continuous function f(x) which is defined on 
//% [interval_start,interval_end]
//% which you can evaluate at any x in [interval_start, interval_end]. 
//% Simply evaluate it at all of
//% the values contained in the x vector to obtain a vector f. Then compute
//% the definite integral using sum(f.*w);
//%
//% Written by Greg von Winckel - 02/25/2004

// return[0] are the x_roots, return[1] are the weights

template<typename P>
std::array<fk::vector<P>, 2>
legendre_weights(const int polynomial_degree, const int interval_start,
                 const int interval_end)
{
  assert(polynomial_degree > 0);
  assert(interval_start < interval_end);

  // prepare out vectors
  // the number of roots of a Legendre polynomial is equal to its degree
  fk::vector<P> x_roots(polynomial_degree);
  fk::vector<P> weights(polynomial_degree);

  // x_linspace=linspace(-1,1,polynomial_degree)';
  // This is a linearly spaced vector used to compose the initial guess
  // of the roots next
  fk::vector<P> const x_linspace =
      linspace(static_cast<P>(-1.0), static_cast<P>(1.0), polynomial_degree);

  //% Initial guess at the roots for the Legendre polynomial of degree
  // polynomial_degree
  // y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*x_linspace*N/N2);
  // It is unkown where this guess comes from, but seems to work well
  // The above operation is split into two pieces, the cos term and
  // then the sin term then they are added together
  x_roots =
      linspace(static_cast<P>(0.0), static_cast<P>((polynomial_degree - 1)), polynomial_degree);
  std::transform(x_roots.begin(), x_roots.end(), x_roots.begin(), [&](P &elem) {
    return std::cos((2 * elem + 1) * M_PI /
                    static_cast<P>((2 * (polynomial_degree - 1) + 2)));
  });

  fk::vector<P> x_roots2(x_linspace);
  std::transform(x_roots2.begin(), x_roots2.end(), x_roots2.begin(), [&](P &elem) {
    return (static_cast<P>(0.27) / polynomial_degree) *
           std::sin(M_PI * elem * (polynomial_degree - 1) / (polynomial_degree + 1));
  });

  x_roots = x_roots + x_roots2;

  //% Legendre-Gauss Vandermonde Matrix
  //% Legendre polynomial values for poly degree 0 to polynomial_degree_plus_one
  //% at the (estimated) zeros of the polynomial of degree polynomial_degree
  // L=zeros(N1,N2);
  fk::matrix<P> legendre_y_values(polynomial_degree, (polynomial_degree + 1));
  // The y values of the derivative of the legendre polynomial
  // of degree equals polynomial_degree at each of the (estimated) root locations
  fk::vector<P> legendre_prime_y_values(polynomial_degree);

  // Set x_roots_initial to a value that will fail the 
  // closeness comparison for the Newton iteration
  // y0=2
  fk::vector<P> x_roots_initial(polynomial_degree);
  std::fill(x_roots_initial.begin(), x_roots_initial.end(), static_cast<P>(2.0));
  P const eps = std::numeric_limits<P>::epsilon();

  //% Compute the zeros of the N+1 Legendre Polynomial
  //% Iterate until new points are uniformly within epsilon of old points
  // while max(abs(y-y0))>eps
  fk::vector<P> diff(polynomial_degree);
  auto const abs_diff = [&](P const &y_elem, P const &y0_elem) {
    return std::fabs(y_elem - y0_elem);
  };
  std::transform(x_roots.begin(), x_roots.end(), x_roots_initial.begin(), diff.begin(), abs_diff);

  while (*std::max_element(diff.begin(), diff.end()) > eps)
  {
    // L(:,1)=1;
    legendre_y_values.update_col(
        0, std::vector<P>(legendre_y_values.nrows(), static_cast<P>(1.0)));

    // L(:,2)=y;
    legendre_y_values.update_col(1, x_roots);

    // for k=2:N1
    // we set the i+1th column of L at each iter
    for (int i = 1; i < polynomial_degree; ++i)
    {
      fk::vector<P> const prev =
          legendre_y_values.extract_submatrix(0, i - 1, legendre_y_values.nrows(), 1);
      fk::vector<P> const current =
          legendre_y_values.extract_submatrix(0, i, legendre_y_values.nrows(), 1);

      fk::vector<P> next(current.size());

      // this loop for setting the next column is a little obscure, but doing
      // step by step with transforms was prohibitively slow when we invoke this
      // function from multiwavelet gen.
      P scale = (static_cast<P>(2.0) * (i + 1) - 1);
      for (int j = 0; j < next.size(); ++j)
      {
        next(j) = ((x_roots(j) * scale * current(j)) - (prev(j) * i)) /
                  static_cast<P>(i + 1);
      }
      legendre_y_values.update_col(i + 1, next);
    }

    // Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);
    fk::vector<P> const legendre_n1 = legendre_y_values.extract_submatrix(
        0, polynomial_degree - 1, legendre_y_values.nrows(), 1);
    fk::vector<P> legendre_n2 = legendre_y_values.extract_submatrix(
        0, (polynomial_degree + 1) - 1, legendre_y_values.nrows(), 1);
    fk::vector<P> legendre_n2_scaled(legendre_n2.size());
    std::transform(legendre_n2.begin(), legendre_n2.end(), x_roots.begin(),
                   legendre_n2_scaled.begin(), std::multiplies<P>());
    fk::vector<P> const y_operand = [&] {
      fk::vector<P> copy_y(x_roots.size());
      std::transform(
          x_roots.begin(), x_roots.end(), x_roots.begin(), copy_y.begin(),
          [](P &y, P &y_same) { return static_cast<P>(1.0) - y * y_same; });
      return copy_y;
    }();
    legendre_prime_y_values = (legendre_n1 - legendre_n2_scaled) * (polynomial_degree + 1);
    auto const element_division = [](P const &one, P const &two) {
      return one / two;
    };
    std::transform(legendre_prime_y_values.begin(), legendre_prime_y_values.end(), y_operand.begin(),
                   legendre_prime_y_values.begin(), element_division);

    x_roots_initial = x_roots;

    // y=y0-L(:,N2)./Lp;
    std::transform(legendre_n2.begin(), legendre_n2.end(), legendre_prime_y_values.begin(),
                   legendre_n2.begin(), element_division);
    x_roots = x_roots_initial - legendre_n2;

    // diff = abs(y-y0)
    std::transform(x_roots.begin(), x_roots.end(), x_roots_initial.begin(), diff.begin(), abs_diff);
  }

  //% Linear map from[-1,1] to [a,b]
  // x=(a*(1-y)+b*(1+y))/2;
  std::transform(x_roots.begin(), x_roots.end(), x_roots.begin(), [&](P &elem) {
    return (interval_start * (1 - elem) + interval_end * (1 + elem)) / 2;
  });

  //% Compute the weights
  // w=(b-a)./((1-y.^2).*Lp.^2)*(N2/N1)^2;
  legendre_prime_y_values.print("legendre_prime");
  std::transform(x_roots.begin(), x_roots.end(), legendre_prime_y_values.begin(), weights.begin(),
                 [&](P &y_elem, P &lp_elem) {
                   return (interval_end - interval_start) /
                          ((static_cast<P>(1.0) - y_elem * y_elem) * lp_elem *
                           lp_elem) *
                          std::pow(static_cast<P>((polynomial_degree + 1)) / polynomial_degree, 2);
                 });

  // x=x(end:-1:1);
  // w=w(end:-1:1);
  std::reverse(x_roots.begin(), x_roots.end());
  std::reverse(weights.begin(), weights.end());

  return std::array<fk::vector<P>, 2>{x_roots, weights};
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
