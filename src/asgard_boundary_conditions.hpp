#pragma once
#include "asgard_transformations.hpp"

// FIXME refactor this component
namespace asgard::boundary_conditions
{
template<typename P>
using unscaled_bc_parts =
    std::vector<std::vector<std::vector<asgard::fk::vector<P>>>>;

// generate all boundary conditions
// add the new transformer here and the coefficient matrices
template<typename P>
std::array<unscaled_bc_parts<P>, 2> make_unscaled_bc_parts(
    PDE<P> const &pde, elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    hierarchy_manipulator<P> const &hier, coefficient_matrices<P> &cmats,
    connection_patterns const &conns,
    int const start_element, int const stop_element, P const t_init = 0);

// group together the conditions that will be scaled by separable-in-time term
template<typename P>
fk::vector<P> generate_scaled_bc(unscaled_bc_parts<P> const &left_bc_parts,
                                 unscaled_bc_parts<P> const &right_bc_parts,
                                 PDE<P> const &pde, int const start_element,
                                 int const stop_element, P const time);

template<typename P>
fk::vector<P>
compute_left_boundary_condition(g_func_type<P> g_func, g_func_type<P> dv_func,
                                P const time, dimension<P> const &dim,
                                vector_func<P> const bc_func);
template<typename P>
fk::vector<P>
compute_right_boundary_condition(g_func_type<P> g_func, g_func_type<P> dv_func,
                                 P const time, dimension<P> const &dim,
                                 vector_func<P> const bc_func);

} // namespace asgard::boundary_conditions
