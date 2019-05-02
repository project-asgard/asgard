#pragma once
#include "pde.hpp"
#include "tensors.hpp"

template<typename P>
fk::matrix<double>
generate_coefficients(dimension<P> const &dim, term<P> const term_1D,
                      double const time = 0.0);

extern template fk::matrix<double>
generate_coefficients(dimension<float> const &dim, term<float> const term_1D,
                      double const time = 0.0);
extern template fk::matrix<double>
generate_coefficients(dimension<double> const &dim, term<double> const term_1D,
                      double const time = 0.0);
