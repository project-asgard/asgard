#pragma once
#include "asgard_transformations.hpp"

namespace asgard
{
template<typename P>
void generate_all_coefficients(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    P const time = 0.0, bool const rotate = true);

template<typename P>
void generate_all_coefficients_max_level(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    P const time = 0.0, bool const rotate = true);

template<typename P>
void generate_dimension_mass_mat(
    PDE<P> &pde,
    basis::wavelet_transform<P, resource::host> const &transformer);

template<typename P>
fk::matrix<P> generate_coefficients(
    dimension<P> const &dim, partial_term<P> const &pterm,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const level, P const time = 0.0, bool const rotate = true);
} // namespace asgard
