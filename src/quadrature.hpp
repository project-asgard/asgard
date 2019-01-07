#pragma once

#include "tensors.hpp"
#include <array>

template<typename P>
std::enable_if_t<std::is_floating_point<P>::value, std::array<fk::matrix<P>, 2>>
legendre(fk::vector<P> const domain, int const degree);

// suppress implicit instatiation
extern template std::array<fk::matrix<float>, 2>
legendre(fk::vector<float> const domain, int const degree);
extern template std::array<fk::matrix<double>, 2>
legendre(fk::vector<double> const domain, int const degree);
