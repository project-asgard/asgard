#include "quadrature.hpp"

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

  // if we are working to update column "n", then "_n_1" is the previous column
  // (i.e. n-1), and "_n_2" is the one before that
  if (degree >= 3)
  {
    // initial values for n-1, n-2
    fk::vector<P> legendre_n_1 =
        legendre.extract_submatrix(0, 1, domain.size(), 1);
    fk::vector<P> legendre_prime_n_1 =
        legendre_prime.extract_submatrix(0, 1, domain.size(), 1);
    fk::vector<P> legendre_n_2 =
        legendre.extract_submatrix(0, 0, domain.size(), 1);
    fk::vector<P> legendre_prime_n_2 =
        legendre_prime.extract_submatrix(0, 0, domain.size(), 1);

    // set remaining columns
    for (int i = 0; i < (degree - 2); ++i)
    {
      int const n            = i + 1;
      int const column_index = i + 2;

      // element-wise multiplication
      fk::vector<P> product(domain.size());
      std::transform(domain.begin(), domain.end(), legendre_n_2.begin(),
                     product.begin(), std::multiplies<P>());

      P const factor = 1.0 / (n + 1.0);

      fk::vector<P> legendre_col = (product * static_cast<P>(2.0 * n + 1.0)) -
                                   (legendre_n_2 * static_cast<P>(n));
      legendre_col = legendre_col * factor;
      legendre.update_col(column_index, legendre_col);

      std::transform(domain.begin(), domain.end(), legendre_prime_n_2.begin(),
                     product.begin(), std::multiplies<P>());

      fk::vector<P> legendre_prime_col =
          (product + legendre_n_1) * static_cast<P>(2.0 * n + 1.0) -
          legendre_prime_n_2 * static_cast<P>(n);
      legendre_prime_col = legendre_prime_col * factor;
      legendre_prime.update_col(column_index, legendre_prime_col);

      // update columns for next iteration
      legendre_n_2       = legendre_n_1;
      legendre_n_1       = legendre_col;
      legendre_prime_n_2 = legendre_prime_n_1;
      legendre_prime_n_1 = legendre_prime_col;
    }
  }

  // "normalizing"
  for (int i = 0; i < degree; ++i)
  {
    P const norm_2 = 2.0 / (2.0 * i + 1.0);
    P const dscale = 1.0 / std::sqrt(norm_2);

    fk::matrix<P> legendre_sub =
        legendre.extract_submatrix(0, i, domain.size(), 1);
    legendre.set_submatrix(0, i, legendre_sub * dscale);

    fk::matrix<P> legendre_prime_sub =
        legendre_prime.extract_submatrix(0, i, domain.size(), 1);
    legendre_prime.set_submatrix(0, i, legendre_sub * dscale);
  }

  // "zero out points out of range"
  auto iter = domain.begin();
  while ((iter = std::find_if(iter, domain.end(), [](P elem) {
            return elem < static_cast<P>(1.0) || elem > static_cast<P>(1.0);
          })) != domain.end())
  {
    int const index = std::distance(domain.begin(), iter++);
    legendre.update_row(
        index, std::vector<P>(std::max(degree, 1), static_cast<P>(0.0)));
    legendre_prime.update_row(
        index, std::vector<P>(std::max(degree, 1), static_cast<P>(0.0)));
  }

  // "scaling to use normalization"
  legendre       = legendre * static_cast<P>(std::sqrt(2.0));
  legendre_prime = legendre_prime * static_cast<P>(std::sqrt(2.0));

  return {legendre, legendre_prime};
}

// explicit instatiations
template std::array<fk::matrix<float>, 2>
legendre(fk::vector<float> const domain, int const degree);
template std::array<fk::matrix<double>, 2>
legendre(fk::vector<double> const domain, int const degree);
