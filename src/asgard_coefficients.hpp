#pragma once
#include "asgard_transformations.hpp"

namespace asgard
{
template<typename P>
void generate_all_coefficients(
    PDE<P> &pde, coefficient_matrices<P> &mats, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, P const time);

// explicit construction of the Kronecker matrix, expensive and used
// only for the implicit time-stepping
template<typename P>
void build_system_matrix(
    PDE<P> const &pde, std::function<fk::matrix<P>(int, int)> get_coeffs,
    elements::table const &elem_table, fk::matrix<P> &A,
    element_subgrid const &grid, imex_flag const imex = imex_flag::unspecified);

} // namespace asgard
