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

// exposed to the API for testing purposes
template<typename P>
std::enable_if_t<std::is_floating_point_v<P>, fk::vector<P>>
linspace(P const start, P const end, unsigned int const num_elems = 100);

template<typename P>
std::enable_if_t<std::is_floating_point_v<P>, std::array<fk::matrix<P>, 2>>
legendre(fk::vector<P> const &domain, int const degree,
         legendre_normalization const norm = legendre_normalization::lin);

// return[0] are the roots, return[1] are the weights
// num_points: number of quadrature points
// lower and upper bound are the bounds of integration

// if use_degree_points is set, use degree quadrature points
// otherwise, use max(10, degree + 1)
template<typename P>
std::array<fk::vector<P>, 2>
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
