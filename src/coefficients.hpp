#pragma once
#include "pde.hpp"
#include "tensors.hpp"

template<typename P>
fk::matrix<double>
generate_coefficients(dimension<P> const &dim, term<P> &term_1D,
                      double const time = 0.0, bool const rotate = true);

extern template fk::matrix<double>
generate_coefficients(dimension<float> const &dim, term<float> &term_1D,
                      double const time = 0.0, bool const rotate = true);

extern template fk::matrix<double>
generate_coefficients(dimension<double> const &dim, term<double> &term_1D,
                      double const time = 0.0, bool const rotate = true);
