#pragma once
#include "pde.hpp"
#include "tensors.hpp"

template<typename P>
void generate_all_coefficients(PDE<P> &pde, P const time = 0.0,
                               bool const rotate = true);

template<typename P>
fk::matrix<double>
generate_coefficients(dimension<P> const &dim, term<P> const &term_1D,
                      partial_term<P> const &pterm, P const time = 0.0,
                      bool const rotate = true);
