#pragma once

#include "asgard_indexset.hpp"

namespace asgard
{

enum class quadrature_mode
{
  use_degree,
  use_fixed
};

// Legendre polynomials and quadrature points/weights are always computed
// in double precision (since it is hard to use quad-precision).
// After the fact, they are truncated to single-precision, when that is needed

// values of Legendre polynomials and their derivatives
std::array<std::vector<double>, 2>
legendre_vals(std::vector<double> const &points, int const degree);

// quadrature points and weights
std::array<std::vector<double>, 2>
legendre_weights(int const degree, double const lower_bound, double const upper_bound,
                 quadrature_mode const quad_mode = quadrature_mode::use_fixed);

template<typename P>
vector2d<P> make_quadrature(int const degree, no_deduce<P> const min,
                            no_deduce<P> const max,
                            quadrature_mode const qmode = quadrature_mode::use_fixed)
{
  auto [lx, lw] = legendre_weights(degree, min, max, qmode);

  vector2d<P> quad(lx.size(), 2);
  std::copy(lx.begin(), lx.end(), quad[0]); // points
  std::copy(lw.begin(), lw.end(), quad[1]); // weights
  return quad;
}

} // namespace asgard
