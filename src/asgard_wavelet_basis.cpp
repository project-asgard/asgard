#include "asgard_wavelet_basis.hpp"

#include "asgard_small_mats.hpp"

namespace asgard::legendre
{
// generate_multi_wavelets routine creates wavelet basis (phi_co)
// then uses these to generate the two-scale coefficients which can be
// used (outside of this routine) to construct the forward multi-wavelet
// transform
std::array<std::vector<double>, 4> generate_multi_wavelets(int const degree)
{
  assert(degree >= 2); // if you need degree 0 or 1, see the switch statement below

  // Consider the first two levels of the wavelet hierarchy and the corresponding
  // cell-by-cell (non-hierarchical) Legendre basis representation.
  // If l0 and l1 are the vectors of Legendre coefficients on the left and right cells,
  // then the hierarchical coefficients for level 0 are H0 * l0 + H1 * l1
  // and the hierarchical coefficients for level 1 are G0 * l0 + G1 * l1
  // In other words, H0/H1 are the projections of the level 0 Legendre basis
  // onto the two adjacent cells, while G0/G1 are the projections of the level 1
  // wavelet coefficients.
  // Thus, two adjacent non-hierarchical cells are merged into a wavelet cell
  // and a remainder on the upper level.

  int const pdof = degree + 1;

  // hard-cording degree 0, 1, 2 (mostly for less rounding)
  // but levels 0 and 1 are even more hard-coded directly into the transform algorithms
  // thus, keeping the degree == 2 special case and a general case for 3 and above
  // if (degree <= 2)
  // {
  //   constexpr double s2 = 1.41421356237309505;
  //
  //   switch (degree)
  //   {
  //   case 0: {
  //     double const is2 = 1 / s2;
  //     std::vector<double> h0 = {is2,};
  //     std::vector<double> h1 = {is2,};
  //     std::vector<double> g0 = {-is2,};
  //     std::vector<double> g1 = {is2,};
  //     return {h0, h1, g0, g1};
  //   }
  //   case 1: {
  //     double const is2  = 1 / s2;
  //     double const is22 = 1 / (2 * s2);
  //     double const is6  = std::sqrt(6.0) / 4;
  //     std::vector<double> h0 = {is2, -is6, 0, is22};
  //     std::vector<double> h1 = {is2,  is6, 0, is22};
  //     std::vector<double> g0 = {0,  is22, -is2, is6};
  //     std::vector<double> g1 = {0, -is22,  is2, is6};
  //     return {h0, h1, g0, g1};
  //   }
  //   case 2: {
  //     double const is2  = 1 / s2;
  //     double const is22 = 1 / (2 * s2);
  //     double const is24 = 1 / (4 * s2);
  //     double const is6  = std::sqrt(6.0) / 4;
  //     double const is30 = 15 / (4.0 * std::sqrt(30.0));
  //     std::vector<double> h0 = {is2, -is6, 0, 0, is22, -is30, 0, 0, is24};
  //     std::vector<double> h1 = {is2,  is6, 0, 0, is22,  is30, 0, 0, is24};
  //     std::vector<double> g0 = {0, 0, -is22, 0,  is24, -is6, -is2, is30, 0};
  //     std::vector<double> g1 = {0, 0,  is22, 0, -is24, -is6,  is2, is30, 0};
  //     return {h0, h1, g0, g1};
  //   }
  //   default:
  //     break;
  //   };
  // }

  if (degree == 2) {
    constexpr double s2 = 1.41421356237309505; // std::sqrt(2.0)
    double const is2  = 1.0 / s2;
    double const is22 = 1.0 / (2 * s2);
    double const is24 = 1.0 / (4 * s2);
    double const is6  = std::sqrt(6.0) / 4;
    double const is30 = 15 / (4.0 * std::sqrt(30.0));
    std::vector<double> h0 = {is2, -is6, 0, 0, is22, -is30, 0, 0, is24};
    std::vector<double> h1 = {is2,  is6, 0, 0, is22,  is30, 0, 0, is24};
    std::vector<double> g0 = {0, 0, -is22, 0,  is24, -is6, -is2, is30, 0};
    std::vector<double> g1 = {0, 0,  is22, 0, -is24, -is6,  is2, is30, 0};
    return {h0, h1, g0, g1};
  }

  std::vector<double> g0(pdof * pdof);
  std::vector<double> g1(pdof * pdof);
  std::vector<double> h0(pdof * pdof);
  std::vector<double> h1(pdof * pdof);

  legendre::canonical_integrator quad(degree);

  // those are the transposes compared to the matrices used in the rest of the code
  auto leg = legendre::poly<double>(degree);
  auto wav = legendre::wavelet_poly(leg, quad);

  double constexpr  s2 = 1.41421356237309505;
  double constexpr is2 = 1.0 / s2;

  auto leg2 = legendre::poly<double, legendre::integ_range::right>(degree);
  double s = 1.0;

  for (int row = 0; row < pdof; ++row)
  {
    for (int col = 0; col < row; ++col)
    {
      double const it = quad.integrate_right(leg2[col], leg[row]);

      h1[row + col * pdof] = it;
      h0[row + col * pdof] = ((row - col) % 2 == 0) ? it : -it;
    }

    h0[row + row * pdof] = is2 / s;
    h1[row + row * pdof] = h0[row + row * pdof];

    s *= 2;
  }

  for (int row = 0; row < pdof; ++row)
  {
    for (int col = degree - row; col < pdof; ++col)
    {
      double const it = is2 * quad.integrate_right(leg2[col], wav[row] + pdof);

      g1[row + col * pdof] = it;
      g0[row + col * pdof] = ((col - row + degree) % 2 == 0) ? -it : it;
    }
  }

  double constexpr tol = 1.e-12;

  auto const normalize = [&](std::vector<double> &mat) -> void {
    for (auto &m : mat)
      if (std::abs(m) < tol)
        m = 0;
  };
  normalize(h0);
  normalize(h1);
  normalize(g0);
  normalize(g1);

  return {h0, h1, g0, g1};
}

vector2d<double> poly2diff(int const degree)
{
  vector2d<double> leg = poly<double, integ_range::full>(degree);

  int const pdof = degree + 1;

  smmat::scal(pdof * pdof, std::sqrt(2), leg[0]);

  vector2d<double> diff(pdof, pdof);
  for (int i = 0; i < pdof; i++) {
    for (int k = 1; k < pdof; k++)
      diff[i][k - 1] = static_cast<double>(k) * leg[i][k];
    diff[i][degree] = 0;
  }

  smmat::transp_swap(pdof, leg[0]);

  return diff;
}

} // namespace asgard::legendre
