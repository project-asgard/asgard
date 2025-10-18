#pragma once

#include "asgard_small_mats.hpp"
#include "asgard_transformations.hpp"

/*!
 * \file asgard_legendre_matrices.hpp
 * \brief Private header containing algorithm linking Legendre basis and small matrices
 * \author The ASGarD Team
 * \ingroup asgard_smallmat
 */

namespace asgard::legendre
{

//! \brief convert single cell Legendre coefficients to values are the quadrature points
template<typename P>
void coeffs2quadvals(legendre_basis<P> const &basis, P const coeffs[], P quadvals[]) {
  smmat::gemv(basis.num_quad, basis.pdof, basis.leg, coeffs, quadvals);
}
//! \brief same as coeffs2quadvals but increments quadvals with the new values
template<typename P>
void add_coeffs2quadvals(legendre_basis<P> const &basis, P const coeffs[], P quadvals[]) {
  smmat::gemv1(basis.num_quad, basis.pdof, basis.leg, coeffs, quadvals);
}

//! \brief scale Legendre values are the quadrature points by given values
template<typename P>
void scale_quadvals(legendre_basis<P> const &basis, P const scales[], P vals[]) {
  smmat::col_scal(basis.num_quad, basis.pdof, scales, basis.leg, vals);
}

//! \brief integrate values at quadrature points against Legendre basis, increments integ[]
template<int dir, typename P>
void integrate(legendre_basis<P> const &basis, P const vals[], P integ[]) {
  static_assert(dir == 1 or dir == -1, "increment direction has to be +1 or -1");
  smmat::gemm_tn<dir>(basis.pdof, basis.num_quad, basis.legw, vals, integ);
}

}
