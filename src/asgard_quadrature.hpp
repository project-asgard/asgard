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

// Legendre polynomials and quadrature points/weights are always computed
// in double precision (since it is hard to use quad-precision).
// After the fact, they are truncated to single-precision, when that is needed

// values of Legendre polynomials and their derivatives
std::array<std::vector<double>, 2>
legendre_vals(std::vector<double> const &points, int const degree,
              legendre_normalization const norm = legendre_normalization::lin);

// quadrature points and weights
std::array<std::vector<double>, 2>
legendre_weights(int const degree, double const lower_bound, double const upper_bound,
                 quadrature_mode const quad_mode = quadrature_mode::use_fixed);

inline std::array<std::vector<float>, 2>
legendre_weights_float(int const degree, float const lower_bound,
                       float const upper_bound,
                       quadrature_mode const quad_mode = quadrature_mode::use_fixed)
{
  auto [p, w] = legendre_weights(degree, double{lower_bound}, double{upper_bound},
                                 quad_mode);

  std::vector<float> points(p.size());
  std::vector<float> weights(w.size());

  std::copy(p.begin(), p.end(), points.begin());
  std::copy(w.begin(), w.end(), weights.begin());

  return {points, weights};
}

template<typename P>
vector2d<P> make_quadrature(int const degree, no_deduce<P> const min,
                            no_deduce<P> const max,
                            quadrature_mode const qmode = quadrature_mode::use_fixed)
{
  auto [lx, lw] = [&]() -> std::array<std::vector<P>, 2>
  {
    if constexpr (std::is_same_v<double, P>) {
      return legendre_weights(degree, min, max, qmode);
    } else {
      return legendre_weights_float(degree, min, max, qmode);
    }
  }();

  vector2d<P> quad(lx.size(), 2);
  std::copy(lx.begin(), lx.end(), quad[0]); // points
  std::copy(lw.begin(), lw.end(), quad[1]); // weights
  return quad;
}

} // namespace asgard
