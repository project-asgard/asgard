#include "asgard_basis.hpp"

namespace asgard
{
// generate_multi_wavelets routine creates wavelet basis (phi_co)
// then uses these to generate the two-scale coefficients which can be
// used (outside of this routine) to construct the forward multi-wavelet
// transform
template<typename P>
std::array<fk::matrix<P>, 4> generate_multi_wavelets(int const degree)
{
  expect(degree >= 0);

  int const pdof = degree + 1;

  // These are the function outputs
  // g0,g1,h0, and h1 are two-scale coefficients
  // The returned phi_co is the wavelet basis
  // scalet_coefficients are the scaling function basis
  //   -- the scalet coefficients form the legendre basis
  //      from a monomial basis

  // hard-cording degree 0, 1, 2 (mostly for less rounding)
  if (degree <= 2)
  {
    constexpr P s2 = 1.41421356237309505;

    switch (degree)
    {
    case 0: {
      P const is2 = 1 / s2;
      fk::matrix<P> h0 = {{is2,},};
      fk::matrix<P> h1 = {{is2,},};
      fk::matrix<P> g0 = {{-is2,},};
      fk::matrix<P> g1 = {{is2,},};
      return {h0, h1, g0, g1};
    }
    case 1: {
      P const is2  = 1 / s2;
      P const is22 = 1 / (2 * s2);
      P const is6  = std::sqrt(P{6}) / 4;
      fk::matrix<P> h0 = {{is2, 0}, {-is6, is22}};
      fk::matrix<P> h1 = {{is2, 0}, {is6, is22}};
      fk::matrix<P> g0 = {{0, -is2}, {is22, is6}};
      fk::matrix<P> g1 = {{0, is2}, {-is22, is6}};
      return {h0, h1, g0, g1};
    }
    case 2: {
      P const is2  = 1 / s2;
      P const is22 = 1 / (2 * s2);
      P const is24 = 1 / (4 * s2);
      P const is6  = std::sqrt(P{6}) / 4;
      P const is30 = 15 / (P{4} * std::sqrt(P{30}));
      fk::matrix<P> h0 = {{is2, 0, 0}, {-is6, is22, 0}, {0, -is30, is24}};
      fk::matrix<P> h1 = {{is2, 0, 0}, {is6, is22, 0}, {0, is30, is24}};
      fk::matrix<P> g0 = {{0, 0, -is2}, {0, is24, is30}, {-is22, -is6, 0}};
      fk::matrix<P> g1 = {{0, 0, is2}, {0, -is24, is30}, {is22, -is6, 0}};
      return {h0, h1, g0, g1};
    }
    default:
      break;
    };
  }

  fk::matrix<P> g0(pdof, pdof);
  fk::matrix<P> g1(pdof, pdof);
  fk::matrix<P> h0(pdof, pdof);
  fk::matrix<P> h1(pdof, pdof);

  basis::canonical_integrator quad(degree);

  // those are the transposes compared to the matrices used in the rest of the code
  auto leg = basis::legendre_poly<P>(degree);
  auto wav = basis::wavelet_poly(leg, quad);

  P const  s2 = std::sqrt(P{2});
  P const is2 = P{1} / s2;

  fk::matrix<P> scalets(pdof, pdof);
  for (auto i : indexof<int>(pdof))
    for (auto j : indexof<int>(pdof))
      scalets(i, j) = s2 * leg[i][degree - j];

  fk::matrix<P> phi_co(pdof * 2, pdof);
  for (auto i : indexof<int>(pdof))
    for (auto j : indexof<int>(pdof))
      phi_co(i, j) = wav[i][degree - j];
  for (auto i : indexof<int>(pdof))
    for (auto j : indexof<int>(pdof))
      phi_co(i + pdof, j) = wav[i][pdof + degree - j];

  // Calculate Two-Scale Coefficients

  // Sums to directly generate H0, H1, G0, G1
  //  H0 and H1 are the "coarsening coefficients"
  //  These describe how two adjacent locations of a higher (finer resolution)
  //  level sum to give a lower (more coarse resolution) level coefficients
  //  G0 and G1 are the "refining or detail coefficients"
  //  These describe how lower level (more coarse resolution)
  //  is split into two higher (finer resolution) level coefficients
  //  H0 is the inner product of the scaling functions of two successive
  //   levels - thus the difference in roots
  // elem_1 is the scalet functions on (-1,0)
  // elem_2 is the scalet function of a lower level and therefore spans (-1,1)
  //  H1 is also the inner product of the scaling functions of two successive
  //   levels - thus the difference in roots
  // elem_3 is the scalet functions on (0,1)
  //  G0 is the inner product of the wavelet functions of one level
  //   with the scalet functions of a lower level
  //   - thus the difference in roots
  // elem_4 is the wavelet functions on (-1,0)
  //  G1 is also the inner product of the wavelet functions of one level
  //   with the scalet functions of a lower level
  // elem_5 is the scalet functions on (0,1)

  // if you have a function represented on a finer grid (say 2 n cells) with
  // a set of legendre coefficients per cell
  // representing the function using the n cell wavelets means multiplying by
  // the adjacent cells by G0 and G1 and adding them together
  // the remainder is formed by multiplying by H0 an H1 and adding it up
  // no we have the wavelets at level n and the corresponding remainder going up
  // on level 0, there will be just a remainders

  auto leg2 = basis::legendre_poly<P, basis::integ_range::right>(degree);
  P s = 1.0;

  for (int row = 0; row < pdof; ++row)
  {
    for (int col = 0; col < row; ++col)
    {
      P const it = quad.integrate_right(leg2[col], leg[row]);

      h1(row, col) = it;
      h0(row, col) = ((row - col) % 2 == 0) ? it : -it;
    }

    h0(row, row) = is2 / s;
    h1(row, row) = h0(row, row);

    s *= 2;
  }

  for (int row = 0; row < pdof; ++row)
  {
    for (int col = degree - row; col < pdof; ++col)
    {
      P const it = is2 * quad.integrate_right(leg2[col], wav[row] + pdof);

      g1(row, col) = it;
      g0(row, col) = ((col - row + degree) % 2 == 0) ? -it : it;
    }
  }

  P constexpr tol = (std::is_same_v<P, double>) ? 1.e-12 : 1.e-4;

  auto const normalize = [&](fk::matrix<P> &matrix) -> void {
    for (auto &m : matrix)
      if (std::abs(m) < tol)
        m = 0;
  };
  normalize(h0);
  normalize(h1);
  normalize(g0);
  normalize(g1);

  return {h0, h1, g0, g1};
}

#ifdef ASGARD_ENABLE_DOUBLE
template std::array<fk::matrix<double>, 4>
generate_multi_wavelets(int const degree);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template std::array<fk::matrix<float>, 4>
generate_multi_wavelets(int const degree);
#endif

} // namespace asgard
