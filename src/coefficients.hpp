#pragma once
#include "pde.hpp"
#include "tensors.hpp"

template<typename P>
void generate_all_coefficients(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    P const time = 0.0, bool const rotate = true);

template<typename P>
fk::matrix<P> generate_coefficients(
    dimension<P> const &dim, term<P> const &term_1D,
    partial_term<P> const &pterm,
    basis::wavelet_transform<P, resource::host> const &transformer,
    P const time = 0.0, bool const rotate = true);
