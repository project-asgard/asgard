#pragma once
#include "asgard_transformations.hpp"

// FIXME refactor this component
namespace asgard::boundary_conditions
{
template<typename P>
using unscaled_bc_parts =
    std::vector<std::vector<std::vector<asgard::fk::vector<P>>>>;

template<typename P>
std::array<unscaled_bc_parts<P>, 2> make_unscaled_bc_parts(
    PDE<P> const &pde, elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const start_element, int const stop_element, P const t_init = 0);

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

template<typename P>
std::vector<fk::vector<P>> generate_partial_bcs(
    std::vector<dimension<P>> const &dimensions, int const d_index,
    std::vector<vector_func<P>> const &bc_funcs,
    basis::wavelet_transform<P, resource::host> const &transformer,
    P const time, std::vector<term<P>> const &terms,
    std::vector<partial_term<P>> const &partial_terms, int const p_index,
    fk::vector<P> &&trace_bc);

} // namespace asgard::boundary_conditions
