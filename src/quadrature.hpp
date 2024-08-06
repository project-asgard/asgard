#pragma once

#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"

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
} // namespace asgard
