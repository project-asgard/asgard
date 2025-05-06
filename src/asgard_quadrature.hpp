#pragma once

#include "asgard_indexset.hpp"

namespace asgard
{
enum class legendre_normalization
{
  unnormalized,
  lin,
  matlab
};

enum class quadrature_mode
{
  use_degree,
  use_fixed
};

// value sof Legendre polynomials and their derivatives
template<typename P>
std::array<std::vector<P>, 2>
legendre_vals(std::vector<P> const &points, int const degree,
              legendre_normalization const norm = legendre_normalization::lin);

// quadrature points and weights
template<typename P>
std::array<std::vector<P>, 2>
legendre_weights(int const degree, no_deduce<P> const lower_bound,
                 no_deduce<P> const upper_bound,
                 quadrature_mode const quad_mode = quadrature_mode::use_fixed);

template<typename P>
vector2d<P> make_quadrature(int const degree, no_deduce<P> const min,
                            no_deduce<P> const max,
                            quadrature_mode const qmode = quadrature_mode::use_fixed)
{
  auto [lx, lw] = legendre_weights<P>(degree, min, max, qmode);
  vector2d<P> quad(lx.size(), 2);
  std::copy(lx.begin(), lx.end(), quad[0]); // points
  std::copy(lw.begin(), lw.end(), quad[1]); // weights
  return quad;
}

} // namespace asgard
