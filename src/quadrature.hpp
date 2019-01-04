#pragma once

#include "tensors.hpp"

// Legendre polynomials on [-1,1] function
// Virtually a direct translation of Ed's dlegendre2.m code
// Legendre returned in [0], derivative returned in [1]
template<typename P>
std::array<fk::vector<P>, 2> legendre(fk::vector<P> const x, int const degree)
{
  int const num_x = x.size();
  assert(degree >= 0);
  assert(num_x > 0);
  assert(std::is_floating_point<P>::value);

  fk::matrix<P> legendre(num_x, std::max(1, degree));
  fk::matrix<P> legendre_prime(num_x, std::max(1, degree));

  std::fill(legendre.begin(), legendre.end(), static_cast<P>(0.0));
  std::fill(legendre_prime.begin(), legendre_prime.end(), static_cast<P>(0.0));

  legendre.update_col(0, std::vector<P>(num_x, static_cast<P>(1.0)));

  if (degree >= 2)
  {
    legendre.update_col(1, x);
    legendre_prime.update_col(1, std::vector<P>(num_x, static_cast<P>(1.0)));
  }

  if (degree >= 3)
  {
    fk::vector<P> legendre_n_1 = legendre.extract_submatrix(0, 0, num_x, 1);
    fk::vector<P> legendre_prime_n_1 =
        legendre_prime.extract_submatrix(0, 0, num_x, 1);
    fk::vector<P> legendre_n_2 = legendre.extract_submatrix(0, 1, num_x, 1);
    fk::vector<P> legendre_prime_n_2 =
        legendre_prime.extract_submatrix(0, 1, num_x, 1);

    // set remaining columns
    int const columns_left = degree - 2;
    for (int i = 0; i < columns_left; ++i)
    {
      int const n            = i + 1;
      int const column_index = i + 2;

      fk::vector<P> product(num_x);
      std::transform(x.begin(), x.end(), legendre_n_1.begin(), product.begin(),
                     std::multiplies<P>());

      P factor = 1.0 / (n + 1.0);

      fk::vector<P> legendre_col = (product * static_cast<P>(2.0 * n + 1.0)) -
                                   (legendre_n_1 * static_cast<P>(n));
      legendre_col = legendre_col * factor;
      legendre.update_col(column_index, legendre_col);

      std::transform(x.begin(), x.end(), legendre_prime_n_1.begin(),
                     product.begin(), std::multiplies<P>());

      fk::vector<P> legendre_prime_col =
          (product + legendre_n_2) * static_cast<P>(2.0 * n + 1.0) -
          legendre_prime_n_1 * static_cast<P>(n);
      legendre_prime_col = legendre_prime_col * factor;
      legendre_prime.update_col(column_index, legendre_prime_col);

      legendre_n_1       = legendre_n_2;
      legendre_n_2       = legendre_col;
      legendre_prime_n_1 = legendre_prime_n_2;
      legendre_prime_n_2 = legendre_prime_col;
    }
  }

  // "normalizing"
  for (int i = 0; i < degree; ++i)
  {
    P norm_2                   = 2.0 / (2.0 * i + 1.0);
    P dscale                   = 1.0 / std::sqrt(norm_2);
    fk::matrix<P> legendre_sub = legendre.extract_submatrix(0, i, num_x, 1);
    legendre.set_submatrix(0, i, legendre_sub * dscale);

    fk::matrix<P> legendre_prime_sub =
        legendre_prime.extract_submatrix(0, i, num_x, 1);
    legendre_prime.set_submatrix(0, i, legendre_sub * dscale);
  }

  // "zero out points out of range"
  auto iter = x.begin();
  while ((iter = std::find_if(iter, x.end(), [](P elem) {
            return elem < static_cast<P>(1.0) || elem > static_cast<P>(1.0);
          })) != x.end())
  {
    int index = std::distance(x.begin(), iter++);
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
