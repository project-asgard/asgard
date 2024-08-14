#pragma once

#include "asgard_indexset.hpp"

#include "asgard_quadrature.hpp"

namespace asgard::basis
{
// hardcoded linear basis functions
template<typename P>
struct linear
{
  static P constexpr s3 = 1.73205080756887729; // sqrt(3.0)
  // projection basis
  static P pleg0(P) { return 1; }
  static P pleg1(P x) { return 2 * s3 * x - s3; }
  static P pwav0L(P x) { return s3 * (1 - 4 * x); }
  static P pwav0R(P x) { return s3 * (-3 + 4 * x); }
  static P pwav1L(P x) { return -1 + 6 * x; }
  static P pwav1R(P x) { return -5 + 6 * x; }
  // interpolation basis
  static P ibas0(P x) { return -3 * x + 2; }
  static P ibas1(P x) { return 3 * x - 1; }
  static P iwav0L(P x) { return -6 * x + 2; }
  static P iwav0R(P) { return 0; }
  static P iwav1L(P) { return 0; }
  static P iwav1R(P x) { return 6 * x - 4; }
};
// hardcoded quadratic basis functions
template<typename P>
struct quadratic
{
  static P constexpr s3 = linear<P>::s3; // sqrt(3.0)
  static P constexpr s5 = 2.2360679774997897;  // sqrt(5.0)

  static constexpr auto pleg0 = linear<P>::pleg0;
  static constexpr auto pleg1 = linear<P>::pleg1;
  static P pleg2(P x) { return s5 * (6.0 * x * x - 6.0 * x + 1.0); }

  static P pwav0L(P x) { return -(1.0 - 12.0 * x + 24.0 * x * x) * s5; }
  static P pwav1L(P x) { return s3 * (30.0 * x * x - 14.0 * x + 1.0); }
  static P pwav2L(P x) { return 1.0 - 6.0 * x; }

  static P pwav0R(P x) { return (13.0 - 36.0 * x + 24.0 * x * x) * s5; }
  static P pwav1R(P x) { return s3 * (30.0 * x * x - 46.0 * x + 17.0); }
  static P pwav2R(P x) { return 5.0 - 6.0 * x; }
};

// algorithms for general basis
// this is a cleaner version of the generate_multi_wavelets method
// in addition to the reduction of calls to new/delete the goal is
// to have less code and easier to read compared to C++-2-MATLAB-isms
// the inner product is still computed with a quadrature as the quadrature
// appears to be numerically more stable then the analytic integration
// of the individual monomials
// the monomial approach simply leads to numbers close in absolute value
// but with opposite signs

// these methods do a lot of polynomial manipulations
// here, a polynomial is defined by the monomial coefficients
// p(x) = sum_i c_i x^i -> p(x) is represented as a vector (c_0, c_1 ... c_n)
// addition/subtraction/scalar-mult applies the action to the vector/s

//! Indicates whether to integrate left, right sub-interval or all of (-1, 1)
enum class integ_range
{
  //! integrate over (-1,  0)
  left,
  //! integrate over ( 0, +1)
  right,
  //! integrate over (-1, +1)
  full
};

/*!
 * \brief Constructs the Legendre orthogonal polynomials
 *
 * The Legendre polynomials are represented as a coefficients for the
 * monomial basis functions. The actual values of the polynomials
 * can be computed from:
 * \code
 *   auto legendre = basis::legendre_poly(n);
 *
 *   // i-th Legendre polynomial of degree n at point x
 *   // is computed as follow
 *   P Li = 0.0;
 *   P mono = 1.0;
 *   for (int j = 0; j <= n; j++) {
 *     Li += legendre[i][j] * mono;
 *     mono *= x;
 *   }
 * \endcode
 *
 * The polynomials are rescaled to have unit norm over the specified range,
 * e.g., (-1, 0), (0, 1) or (-1, 1).
 */
template<typename P, integ_range range = integ_range::full>
vector2d<P> legendre_poly(int const degree)
{
  // the level 0 wavelets, in this case, the Legendre polynomials
  // so we use the recurrence relation to express the Legendre polynomials
  // in terms of the monomial basis

  int const pdof = degree + 1; // number of polynomial dof for given degree

  vector2d<P> legendre(pdof, pdof);
  legendre[0][0] = 1.0; // constant polynomial
  if (degree >= 1)
  {
    if constexpr (range == integ_range::full)
      legendre[1][1] = 1.0; // linear term
    else if constexpr (range == integ_range::right)
    {
      legendre[1][0] = -1.0; // linear term
      legendre[1][1] = 2.0;
    }
    else
    {
      legendre[1][0] = 1.0; // linear term
      legendre[1][1] = 2.0;
    }
  }
  if (degree >= 2)
  {
    // L_n = (1 / n) * ( (2n - 1) x L_{n-1} - (n - 1) L_{n - 2} )
    // L_n = (1 / n) * ( alpha * x * L_{n-1} - beta * L_{n - 2} )
    // the multiplication by x means that going from L_{n-1} to L_n
    // we shift the monomial coefficients by one (higher power)

    P alpha = (range == integ_range::full) ? 1 : 2;
    P beta  = 0;
    P gamma = (range == integ_range::right) ? -1 : 1; // used for the half-range
    for (int n = 2; n < pdof; n++)
    {
      alpha += (range == integ_range::full) ? 2 : 4;
      beta  += 1;
      if constexpr (range != integ_range::full)
        gamma += (range == integ_range::right) ? -2 : 2;

      if constexpr (range == integ_range::full)
        legendre[n][0] = - beta * legendre[n - 2][0] / n;
      else
        legendre[n][0] = (gamma * legendre[n - 1][0] - beta * legendre[n - 2][0]) / n;

      for (int k = 1; k < n; k++)
        if constexpr (range == integ_range::full)
          legendre[n][k] = (alpha * legendre[n - 1][k - 1]
                            - beta * legendre[n - 2][k]) / n;
        else
          legendre[n][k] = (alpha * legendre[n - 1][k - 1]
                            + gamma * legendre[n - 1][k]
                             - beta * legendre[n - 2][k]) / n;
      legendre[n][n] = alpha * legendre[n - 1][n - 1] / n;
    }
  }

  // normalize the polynomials
  for (auto n : indexof<int>(pdof))
  {
    P const scale = [&]() -> P {
      if constexpr (range == integ_range::full)
        return std::sqrt((2 * n + P{1}) / P{2});
      else
        return std::sqrt((2 * n + P{1}));
    }();

    P *p = legendre[n];
    for (auto i : indexof<int>(pdof))
      p[i] *= scale;
  }

  return legendre;
}

/*!
 * \brief Holds the quadrature points and weights for integration
 *
 * Creates (and reuses) quadrature points defined on (-1, 0) and (0, 1),
 * i.e., the two sub-intervals of the canonical (-1, 1).
 * The class allows for integration over the left/right part of
 * the subdomain, specifically take inner product of polynomials defined
 * by monomial coefficients.
 */
class canonical_integrator
{
public:
  //! Create and store the quadratures for (-1, 0) and (0, 1)
  canonical_integrator(int const degree_in) : degree_(degree_in)
  {
    auto [fkxl, fkwl] = legendre_weights<double>(2 * degree_ + 1, -1, 0);
    auto [fkxr, fkwr] = legendre_weights<double>(2 * degree_ + 1, 0, 1);

    wl = fkwl.to_std();
    xl = fkxl.to_std();
    wr = fkwr.to_std();
    xr = fkxr.to_std();

    work1.resize(xl.size());
    work2.resize(xl.size());
  }
  //! compute inner product of f and g over (-1, 0)
  template<typename P>
  double integrate_left(P const f[], P const g[]) const
  {
    return poly_inner_product<integ_range::left>(f, g);
  }
  //! compute inner product of f and g over (0, 1)
  template<typename P>
  double integrate_right(P const f[], P const g[]) const
  {
    return poly_inner_product<integ_range::right>(f, g);
  }
  //! returns the degree used to create the quadrature
  int degree() const { return degree_; }

private:
  //! common implementation for the inner product logic
  template<integ_range range, typename P>
  double poly_inner_product(P const f[], P const g[]) const
  {
    static_assert(range == integ_range::left or range == integ_range::right);

    auto const &w = (range == integ_range::left) ? wl : wr;
    auto const &x = (range == integ_range::left) ? xl : xr;

    std::fill(work1.begin(), work1.end(), f[degree_]);
    std::fill(work2.begin(), work2.end(), g[degree_]);

    for (int i = degree_ - 1; i >= 0; --i)
    {
      for (auto j : indexof(work1))
        work1[j] = work1[j] * x[j] + f[i];
      for (auto j : indexof(work2))
        work2[j] = work2[j] * x[j] + g[i];
    }

    double sum = 0;
    for (auto i : indexof(work1))
      sum += w[i] * work1[i] * work2[i];
    return sum;
  }

  int degree_;
  std::vector<double> wl, xl, wr, xr;

  mutable std::vector<double> work1, work2;
};

/*!
 * \brief Constructs the wavelets
 *
 * The wavelets have two components defined over the left/right subintervals.
 * The definition is similar to legendre, the difference is that we have
 * twice as many coefficients to match left-right segments.
 *
 * \code
 *   auto wavelets = basis::wavelet_poly(n);
 *
 *   // i-th wavelet polynomial of degree n at point x
 *   // is computed as follow
 *   int const offset = (x < 0.0) ? 0 : n + 1;
 *   P Li = 0.0;
 *   P mono = 1.0;
 *   for (int j = 0; j <= n; j++) {
 *     Li += wavelets[i][j + offset] * mono;
 *     mono *= x;
 *   }
 *   // if the wavelets are rescaled for (0, 1), then offset must be adjusted
 *   // offset = (x < 0.5) ? 0 : n + 1
 * \endcode
 *
 * The polynomials are rescaled to have unit norm over the specified range,
 * e.g., (0, 1) or (-1, 1).
 *
 * \param legendre must be the output of legendre_poly with full range scaling,
 *                 the full range must be used even if we are using different
 *                 scaling here
 */
template<typename P>
vector2d<P> wavelet_poly(vector2d<P> const &legendre,
                         canonical_integrator const &quad)
{
  int const pdof   = legendre.stride();
  int const degree = pdof - 1;
  // the direct application of Gram-Schmidt yields results in reversed order
  // and all odd number of wavelets have reverse sign
  // hence do the process in reverse and flip signs during rescaling

  // each wavelet has 2 sets of coefficients, over (-1, 0) and (0, 1)
  // the left segment has coefficients at wavelet[n][0] ... [n][pdof - 1]
  // the right segment starts at wavelet[n][0] + pdof
  vector2d<P> wavelets(2 * pdof, pdof);
  for (auto n : indexof<int>(pdof))
  { // initialize the wavelets
    wavelets[degree - n][n]        = -1; // -x^n on (-1, 0)
    wavelets[degree - n][n + pdof] =  1; //  x^n on ( 0, 1)
  }

  // before Gram-Schmidt, orthogonalize the wavelets w.r.t. legendre
  for (auto n : indexof<int>(pdof))
  {
    P *left  = wavelets[n];
    P *right = wavelets[n] + pdof;
    for (auto l : indexof<int>(pdof))
    {
      P const *leg = legendre[l];
      P const prod = quad.integrate_left(left, leg)
                    + quad.integrate_right(right, leg);

      for (auto i : indexof<int>(pdof))
      {
        left[i]  -= prod * leg[i];
        right[i] -= prod * leg[i];
      }
    }
  }

  // now we apply Gram-Schmidt orthogonalization
  for (int nr : indexof<int>(pdof)) // reverse the order
  {
    int const n = degree - nr;
    P *left  = wavelets[n];
    P *right = wavelets[n] + pdof;
    for (int ir : indexof<int>(nr))
    {
      int const i = degree - ir;

      P const prod = quad.integrate_left(left, wavelets[i])
                    + quad.integrate_right(right, wavelets[i] + pdof);
      for (int j : indexof<int>(2 * pdof))
        wavelets[n][j] -= prod * wavelets[i][j];
    }

    P const nrm2 = quad.integrate_left(left, left) + quad.integrate_right(right, right);
    P const nrm  = P{1} / std::sqrt(nrm2);
    for (auto i : indexof<int>(2 * pdof))
      wavelets[n][i] *= nrm;
  }

  { // shift everything to (0, 1), i.e., scale by sqrt(2)
    P s2 = std::sqrt(P{2});
    for (int n : indexof<int>(pdof))
    {
      P *w = wavelets[n];
      for (int j : indexof<int>(2 * pdof))
        w[j] *= s2;
      s2 = -s2;
    }
  }

  return wavelets;
}

//! overload, skips the canonical_integrator construction if used only once
template<typename P>
vector2d<P> wavelet_poly(vector2d<P> const &legendre, int degree)
{
  return wavelet_poly(legendre, canonical_integrator(degree));
}

} // namespace asgard::basis
