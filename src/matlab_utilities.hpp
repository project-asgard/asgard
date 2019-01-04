#pragma once

//-----------------------------------------------------------------------------
//
// a collection of utility functions to make working with matlab-like
// code a little easier
//
// - matlab functions ported to c++
//    - linspace (scalar inputs only)
//
// - matlab/octave file IO
//    - readVectorFromTxtFile (tested for octave)
//    - readVectorFromBinFile (tested for octave and matlab)
//
//-----------------------------------------------------------------------------

#include "tensors.hpp"
#include <string>
#include <vector>

// matlab's "linspace(start, end, N)" function
//-----------------------------------------------------------------------------
//
// c++ implementation of matlab (a subset of) linspace() function
// initial c++ implementation by Tyler McDaniel
//
// -- linspace (START, END)
// -- linspace (START, END, N)
//     Return a row vector with N linearly spaced elements between START
//     and END.
//
//     If the number of elements is greater than one, then the endpoints
//     START and END are always included in the range.  If START is
//     greater than END, the elements are stored in decreasing order.  If
//     the number of points is not specified, a value of 100 is used.
//
//     The 'linspace' function returns a row vector when both START and
//     END are scalars.
//
//  (unsupported)
//     If one, or both, inputs are vectors, then
//     'linspace' transforms them to column vectors and returns a matrix
//     where each row is an independent sequence between
//     'START(ROW_N), END(ROW_N)'.
//
//     For compatibility with MATLAB, return the second argument (END)
//     when only a single value (N = 1) is requested.
//
//-----------------------------------------------------------------------------
template<typename P>
std::enable_if_t<std::is_floating_point<P>::value, fk::vector<P>>
linspace(P const start, P const end, unsigned int const num_elems = 100)
{
  assert(num_elems > 1); // must have at least 2 elements

  // create output vector
  fk::vector<P> points(num_elems);

  // find interval size
  P const interval_size = (end - start) / (num_elems - 1);

  // insert first and last elements
  points(0)             = start;
  points(num_elems - 1) = end;

  // fill in the middle
  for (unsigned int i = 1; i < num_elems - 1; ++i)
  {
    points(i) = start + i * interval_size;
  }

  return points;
}
//-----------------------------------------------------------------------------
//
// c++ implementation of (a subset of) eye() function
// The following are not supported here:
// - providing a third "CLASS" argument
// - providing a vector argument for the dimensions
//
// -- eye (N)
// -- eye (M, N)
// -- eye ([M N])
//     Return an identity matrix.
//
//     If invoked with a single scalar argument N, return a square NxN
//     identity matrix.
//
//     If supplied two scalar arguments (M, N), 'eye' takes them to be the
//     number of rows and columns.  If given a vector with two elements,
//     'eye' uses the values of the elements as the number of rows and
//     columns, respectively.  For example:
//
//          eye (3)
//           =>  1  0  0
//               0  1  0
//               0  0  1
//
//     The following expressions all produce the same result:
//
//          eye (2)
//          ==
//          eye (2, 2)
//          ==
//          eye (size ([1, 2; 3, 4]))
//
//     Calling 'eye' with no arguments is equivalent to calling it with an
//     argument of 1.  Any negative dimensions are treated as zero.  These
//     odd definitions are for compatibility with MATLAB.
//
//-----------------------------------------------------------------------------
template<typename P>
fk::matrix<P> eye(int const M = 1)
{
  fk::matrix<P> id(M, M);
  for (auto i = 0; i < M; ++i)
    id(i, i) = 1.0;
  return id;
}
template<typename P>
fk::matrix<P> eye(int const M, int const N)
{
  fk::matrix<P> id(M, N);
  for (auto i = 0; i < (M < N ? M : N); ++i)
    id(i, i) = 1.0;
  return id;
}

//-----------------------------------------------------------------------------
// C++ implementation of subset of matlab polyval
// Function for evaluating a polynomial.
//
// Returns the value of a polynomial p evaluated for
// x / each element of x.
// p is a vector of length n+1 whose elements are
// the coefficients of the polynomial in descending powers.

// y = p(0)*x^n + p(1)*x^(n-1) + ... + p(n-1)*x + p(n)
//-----------------------------------------------------------------------------
template<typename P>
P polyval(fk::vector<P> const p, P const x)
{
  int const num_terms = p.size();
  assert(num_terms > 0);

  P y = static_cast<P>(0.0);
  for (int i = 0; i < num_terms - 1; ++i)
  {
    int const deg = num_terms - i - 1;
    y += p(i) * static_cast<P>(std::pow(x, deg));
  }
  y += p(num_terms - 1);

  return y;
}

template<typename P>
fk::vector<P> polyval(fk::vector<P> const p, fk::vector<P> const x)
{
  int const num_terms = p.size();
  int const num_sols  = x.size();
  assert(num_terms > 0);
  assert(num_sols > 0);

  fk::vector<P> solutions(num_sols);
  for (int i = 0; i < num_sols; ++i)
  {
    solutions(i) = polyval(p, x(i));
  }

  return solutions;
}

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

  // FIXME I don't know what these vector names mean
  if (degree >= 3)
  {
    fk::vector<P> legendre_ml = legendre.extract_submatrix(0, 0, num_x, 1);
    fk::vector<P> legendre_prime_ml =
        legendre_prime.extract_submatrix(0, 0, num_x, 1);
    fk::vector<P> legendre_n = legendre.extract_submatrix(0, 1, num_x, 1);
    fk::vector<P> legendre_prime_n =
        legendre_prime.extract_submatrix(0, 1, num_x, 1);

    // set remaining columns
    int const columns_left = degree - 2;
    for (int i = 0; i < columns_left; ++i)
    {
      int const n            = i + 1;
      int const column_index = i + 2;

      fk::vector<P> product(num_x);
      std::transform(x.begin(), x.end(), legendre_ml.begin(), product.begin(),
                     std::multiplies<P>());

      P factor = 1.0 / (n + 1.0);
      x*factor;
      // FIXME I don't know what these names mean, either
      fk::vector<P> legendre_col = (product * static_cast<P>(2.0 * n + 1.0))  - (legendre_ml * static_cast<P>(n));
      legendre_col = legendre_col * factor;
      legendre.update_col(column_index, legendre_col);

      std::transform(x.begin(), x.end(), legendre_prime_ml.begin(),
                     product.begin(), std::multiplies<P>());

      fk::vector<P> legendre_prime_col =
         (product + legendre_n) * static_cast<P>(2.0 * n + 1.0)  - legendre_prime_ml * static_cast<P>(n);
      legendre_prime_col = legendre_prime_col * factor;
      legendre_prime.update_col(column_index, legendre_prime_col);

      legendre_ml       = legendre_n;
      legendre_n        = legendre_col;
      legendre_prime_ml = legendre_prime_n;
      legendre_prime_n  = legendre_prime_col;
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
                   return elem < static_cast<P>(1.0) ||
                          elem > static_cast<P>(1.0);
                 })) != x.end())
  {
    int index = std::distance(x.begin(), iter++);
    legendre.update_row(index, std::vector<P>(std::max(degree,1), static_cast<P>(0.0)));
    legendre_prime.update_row(index,
                              std::vector<P>(std::max(degree, 1), static_cast<P>(0.0)));
  }

  // "scaling to use normalization"
  legendre = legendre * static_cast<P>(std::sqrt(2.0));
  legendre_prime = legendre_prime * static_cast<P>(std::sqrt(2.0));

  return {legendre, legendre_prime};
}

// read a matlab vector from binary file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
fk::vector<double> readVectorFromBinFile(std::string const &path);

// read an octave vector from text file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
fk::vector<double> readVectorFromTxtFile(std::string const &path);

// read an octave matrix from text file into a fk::matrix
fk::matrix<double> readMatrixFromTxtFile(std::string const &path);
