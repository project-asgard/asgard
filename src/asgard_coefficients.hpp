#pragma once
#include "asgard_transformations.hpp"

namespace asgard
{

/*
 * \internal
 * \brief Allows to selectively update coefficients and avoid repeated work
 *
 * \endinternal
 */
enum class coeff_update_mode {
    //! update all coefficients
    all,
    //! update the imex explicit term-coefficients
    imex_explicit,
    //! update the imex implicit term-coefficients
    imex_implicit,
    //! update only the coefficients that depends on the poisson data
    poisson,
    //! update only the independent coefficients, i.e., do not depend on moments
    independent,
};

template<typename P>
void generate_coefficients(
    PDE<P> &pde, coefficient_matrices<P> &mats, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, P const time,
    coeff_update_mode mode = coeff_update_mode::all);

// explicit construction of the Kronecker matrix, expensive and used
// only for the implicit time-stepping
template<typename P>
void build_system_matrix(
    PDE<P> const &pde, std::function<fk::matrix<P>(int, int)> get_coeffs,
    elements::table const &elem_table, fk::matrix<P> &A,
    element_subgrid const &grid, imex_flag const imex = imex_flag::unspecified);

} // namespace asgard
