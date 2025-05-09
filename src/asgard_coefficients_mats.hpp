#pragma once
#include "asgard_transformations.hpp"

#include "asgard_small_mats.hpp"

// private header, exposes some of the coefficient methods for easier testing
// also reduces the clutter in asgard_coefficients.cpp

namespace asgard
{

enum class rhs_type {
  is_func, is_const
};

template<typename P, operation_type optype, rhs_type rtype, data_mode dmode = data_mode::replace>
void gen_tri_cmat(legendre_basis<P> const &basis, P xleft, P xright, int level,
                  sfixed_func1d<P> const &rhs, P const rhs_const, flux_type flux,
                  boundary_type boundary, rhs_raw_data<P> &rhs_raw, block_tri_matrix<P> &coeff)
{
  static_assert(optype != operation_type::volume
                and optype != operation_type::identity
                and optype != operation_type::chain,
                "identity, mass and chain operations yield diagonal matrices, "
                "should not be used in the tri-diagonal case");
  static_assert(dmode == data_mode::replace or dmode == data_mode::increment,
                "matrices can either replace existing data or add to it, "
                "use data_mode::replace or data_mode::increment");
  static_assert(not (optype == operation_type::penalty and rtype == rhs_type::is_func),
                "cannot use spatially dependant penalty term");

  if constexpr (optype == operation_type::grad) {
    // the grad operation flips the fixed and free boundary conditions
    switch (boundary) {
      case boundary_type::bothsides:
        boundary = boundary_type::none;
        break;
      case boundary_type::none:
        boundary = boundary_type::bothsides;
        break;
      case boundary_type::right:
        boundary = boundary_type::left;
        break;
      case boundary_type::left:
        boundary = boundary_type::right;
        break;
      default: // periodic, do nothing since it is symmetric anyway
        break;
    }
  }

  int const num_cells = fm::ipow2(level);
  P const dx = (xright - xleft) / num_cells;

  int const nblock = basis.pdof * basis.pdof;
  if constexpr (dmode == data_mode::replace) {
    coeff.resize_and_zero(nblock, num_cells);
  } else {
    expect(coeff.nblock() == nblock);
    expect(coeff.nrows() == num_cells);
  }

  // if not using a constant rhs, get the values from the function
  static std::vector<P> rhs_pnts;

  span2d<P> rhs_vals;
  if constexpr (rtype == rhs_type::is_func) {
    // left point, interior pnts, right-point, left/right in adjacent cells will match
    int const stride = (basis.num_quad + 1);
    rhs_raw.pnts.resize(stride * num_cells + 1);
    rhs_raw.vals.resize(rhs_raw.pnts.size());
#pragma omp parallel for
    for (int i = 0; i < num_cells; i++) {
      P const l = xleft + i * dx;
      rhs_raw.pnts[i * stride] = l;
      for (int k = 0; k < basis.num_quad; k++)
        rhs_raw.pnts[i * stride + k + 1] = (0.5 * basis.qp[k] + 0.5) * dx + l;
    }
    // right most cell
    rhs_raw.pnts.back() = xright;
    rhs(rhs_raw.pnts, rhs_raw.vals);

    // for the i-th cell rhs_vals[i][0] is the left-most point
    // rhs_vals[i][1] ... rhs_vals[i][num_quad] are the interior quadrature points
    // right-most point of the i-th cell is at rhs_vals[i+1][0]

    rhs_vals = span2d<P>(stride, num_cells, rhs_raw.vals.data());
  }

  P const fscale = -static_cast<int>(flux); // scale +/- 1

  P const escale = P{1} / dx; // edge scale
  P const vscale = P{2} / dx; // volume scale

  std::vector<P> const_mat;
  if constexpr (rtype == rhs_type::is_const and optype != operation_type::penalty) {
    // if the coefficient is constant, we have identical copies of the same matrix
    // compute once and reuse as needed,
    // also note that the penalty operation skips the volume component
    const_mat.resize(nblock);
    smmat::gemm_tn<-1>(basis.pdof, basis.num_quad,
                       basis.der, basis.legw, const_mat.data());
    smmat::scal(nblock, vscale * rhs_const, const_mat.data());
  }

#pragma omp parallel
  {
    // each thread will allocate it's own tmp matrix
    std::vector<P> tmp;
    if constexpr (rtype == rhs_type::is_func and optype != operation_type::penalty)
      tmp.resize(basis.num_quad * basis.pdof); // if not using const coefficient

    // tmp will be captured inside the lambda closure
    // no allocations will occur per call to the lambda
    auto apply_volume = [&](int i) -> void {
      // the penalty term does not include a volume integral
      if constexpr (optype != operation_type::penalty)
      {
        if constexpr (rtype == rhs_type::is_const) {
          std::copy_n(const_mat.data(), nblock, coeff.diag(i));
        } else {
          smmat::col_scal(basis.num_quad, basis.pdof,
                          vscale, rhs_vals[i] + 1, basis.legw, tmp.data());
          smmat::gemm_tn<-1>(basis.pdof, basis.num_quad, basis.der, tmp.data(), coeff.diag(i));
        }
      }
    };

#pragma omp for
    for (int i = 1; i < num_cells - 1; i++)
    {
      apply_volume(i);

      if constexpr (optype == operation_type::penalty)
      {
        smmat::axpy(nblock, -escale * rhs_const, basis.from_left, coeff.lower(i));
        smmat::axpy(nblock,  escale * rhs_const, basis.to_left, coeff.diag(i));

        smmat::axpy(nblock,  escale * rhs_const, basis.to_right, coeff.diag(i));
        smmat::axpy(nblock, -escale * rhs_const, basis.from_right, coeff.upper(i));
      }
      else
      {
        P const left  = 0.5 * ((rtype == rhs_type::is_const) ? rhs_const : rhs_vals[i][0]);
        P const right = 0.5 * ((rtype == rhs_type::is_const) ? rhs_const : rhs_vals[i + 1][0]);

        P const left_abs  = fscale * std::abs(left);
        P const right_abs = fscale * std::abs(right);

        smmat::axpy(nblock, escale * (-left - left_abs), basis.from_left, coeff.lower(i));
        smmat::axpy(nblock, escale * (-left + left_abs), basis.to_left, coeff.diag(i));

        smmat::axpy(nblock, escale * (right + right_abs), basis.to_right, coeff.diag(i));
        smmat::axpy(nblock, escale * (right - right_abs), basis.from_right, coeff.upper(i));
      }
    }

    // interior cells are done in parallel, the boundary conditions are done once
    // the first thread that exits the for-loop above will do this work
#pragma omp single
    {
      // need to consider various types of boundary conditions on left/right
      // but we have a possible case of 1 cell, so left-most is also right-most

      int const rmost = num_cells - 1; // right-most cell

      apply_volume(0);   // left-most cell
      if (num_cells > 1) // if the right-most cell is not the left-most cell
        apply_volume(rmost);

      if constexpr (optype == operation_type::penalty)
      {
        switch (boundary) {
          case boundary_type::bothsides:
          case boundary_type::left: // dirichelt on the left
            smmat::axpy(nblock, escale * rhs_const, basis.to_left, coeff.diag(0));
            break;
          case boundary_type::periodic:
            smmat::axpy(nblock, -escale * rhs_const, basis.from_left, coeff.lower(0));
            smmat::axpy(nblock,  escale * rhs_const, basis.to_left, coeff.diag(0));
            break;
          default: // free flux, no penalty applied
            break;
        };

        if (num_cells > 1) { // left-right most cells are different, build mid-conditions
          smmat::axpy(nblock,  escale * rhs_const, basis.to_right, coeff.diag(0));
          smmat::axpy(nblock, -escale * rhs_const, basis.from_right, coeff.upper(0));

          smmat::axpy(nblock, -escale * rhs_const, basis.from_left, coeff.lower(rmost));
          smmat::axpy(nblock,  escale * rhs_const, basis.to_left, coeff.diag(rmost));
        }

        switch (boundary) {
          case boundary_type::bothsides:
          case boundary_type::right: // dirichelt on the right
            smmat::axpy(nblock, escale * rhs_const, basis.to_right, coeff.diag(rmost));
            break;
          case boundary_type::periodic:
            smmat::axpy(nblock,  escale * rhs_const, basis.to_right, coeff.diag(rmost));
            smmat::axpy(nblock, -escale * rhs_const, basis.from_right, coeff.upper(rmost));
            break;
          default: // free flux, no penalty applied
            break;
        };
      }
      else // div or grad operation
      {
        // look at the left-boundary
        switch (boundary) {
          case boundary_type::none:
          case boundary_type::right: // free on the left
            smmat::axpy(nblock, -escale * ((rtype == rhs_type::is_const) ? rhs_const : rhs_vals[0][0]), basis.to_left, coeff.diag(0));
            break;
          case boundary_type::periodic: {
            P const left     = 0.5 * ((rtype == rhs_type::is_const) ? rhs_const : rhs_vals[0][0]);
            P const left_abs = fscale * std::abs(left);
            smmat::axpy(nblock, escale * (-left - left_abs), basis.from_left, coeff.lower(0));
            smmat::axpy(nblock, escale * (-left + left_abs), basis.to_left, coeff.diag(0));
            }
            break;
          default: // dirichlet flux, nothing to set
            break;
        };

        if (num_cells > 1) {
          P c    = 0.5 * ((rtype == rhs_type::is_const) ? rhs_const : rhs_vals[1][0]);
          P cabs = fscale * std::abs(c);

          smmat::axpy(nblock, escale * (c + cabs), basis.to_right, coeff.diag(0));
          smmat::axpy(nblock, escale * (c - cabs), basis.from_right, coeff.upper(0));

          c    = 0.5 * ((rtype == rhs_type::is_const) ? rhs_const : rhs_vals[rmost][0]);
          cabs = fscale * std::abs(c);

          smmat::axpy(nblock, escale * (-c - cabs), basis.from_left, coeff.lower(rmost));
          smmat::axpy(nblock, escale * (-c + cabs), basis.to_left, coeff.diag(rmost));
        }

        // look at the right-boundary
        switch (boundary) {
          case boundary_type::none:
          case boundary_type::left: // free on the right
            smmat::axpy(nblock, escale * ((rtype == rhs_type::is_const) ? rhs_const : rhs_raw.vals.back()), basis.to_right, coeff.diag(rmost));
            break;
          case boundary_type::periodic: {
            P const right     = 0.5 * ((rtype == rhs_type::is_const) ? rhs_const : rhs_raw.vals.back());
            P const right_abs = fscale * std::abs(right);
            smmat::axpy(nblock, escale * (right + right_abs), basis.to_right, coeff.diag(rmost));
            smmat::axpy(nblock, escale * (right - right_abs), basis.from_right, coeff.upper(rmost));
            }
            break;
          default: // dirichlet flux, nothing to set
            break;
        };
      }
    } // #pragma omp single
  }

  if constexpr (optype == operation_type::grad)
  {
    // take the negative transpose of div
#pragma omp parallel for
    for (int64_t r = 0; r < coeff.nrows() - 1; r++)
    {
      smmat::neg_transp_swap(basis.pdof, coeff.lower(r + 1), coeff.upper(r));
      smmat::neg_transp(basis.pdof, coeff.diag(r));
    }
    smmat::neg_transp(basis.pdof, coeff.diag(coeff.nrows() - 1));
    smmat::neg_transp_swap(basis.pdof, coeff.lower(0), coeff.upper(coeff.nrows() - 1));
  }
}

template<typename P, operation_type optype>
void gen_diag_cmat(legendre_basis<P> const &basis, int level,
                   P const rhs_const, block_diag_matrix<P> &coeff)
{
  static_assert(optype == operation_type::volume,
                "only mass matrices should be used to create mass terms");

  int const num_cells = fm::ipow2(level);
  int const nblock = basis.pdof * basis.pdof;

  coeff.resize_and_zero(nblock, num_cells);

  // if the coefficient is constant, we have identical copies of the same matrix
  // compute once and reuse as needed,
  // also note that the penalty operation skips the volume component
  std::vector<P> const_mat;
  const_mat.resize(nblock);

  for (int i = 0; i < basis.pdof; i++)
  {
    for (int j = 0; j < i; j++)
      const_mat[i * basis.pdof + j] = 0;
    const_mat[i * basis.pdof + i] = rhs_const;
    for (int j = i + 1; j < basis.pdof; j++)
      const_mat[i * basis.pdof + j] = 0;
  }

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
    std::copy_n(const_mat.data(), nblock, coeff[i]);
}

template<typename P>
void gen_diag_cmat_pwc(legendre_basis<P> const &basis, int level,
                       std::vector<P> const &pwc, block_diag_matrix<P> &coeff)
{
  int const num_cells = fm::ipow2(level);
  int const nblock = basis.pdof * basis.pdof;

  coeff.resize_and_zero(nblock, num_cells);

  std::vector<P> const_mat(nblock, P{0});
  for (int i = 0; i < basis.pdof; i++)
    const_mat[i * basis.pdof + i] = 1;

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
    smmat::axpy(nblock, pwc[i], const_mat.data(), coeff[i]);
}

template<typename P, operation_type optype,
         term_dependence depends = term_dependence::none>
void gen_diag_cmat(legendre_basis<P> const &basis, P xleft, P xright, int level,
                   sfixed_func1d<P> const &rhs, rhs_raw_data<P> &rhs_raw,
                   block_diag_matrix<P> &coeff)
{
  static_assert(optype == operation_type::volume,
                "only volume matrices should be used to create volume terms");

  int const num_cells = fm::ipow2(level);
  P const dx = (xright - xleft) / num_cells;

  int const nblock = basis.pdof * basis.pdof;
  coeff.resize_and_zero(nblock, num_cells);

  span2d<P> rhs_vals;
  if constexpr (depends == term_dependence::none) {
    rhs_raw.pnts.resize(basis.num_quad * num_cells);
    rhs_raw.vals.resize(rhs_raw.pnts.size());
#pragma omp parallel for
    for (int i = 0; i < num_cells; i++) {
      P const l = xleft + i * dx; // left edge of cell i
      for (int k = 0; k < basis.num_quad; k++)
        rhs_raw.pnts[i * basis.num_quad + k] = (0.5 * basis.qp[k] + 0.5) * dx + l;
    }
    // right most cell
    rhs(rhs_raw.pnts, rhs_raw.vals);

    rhs_vals = span2d<P>(basis.num_quad, num_cells, rhs_raw.vals.data());
  }

#pragma omp parallel
  {
    // each thread will allocate it's own tmp matrix
    std::vector<P> tmp;
    tmp.resize(basis.num_quad * basis.pdof); // if not using const coefficient

#pragma omp for
    for (int i = 0; i < num_cells; i++)
    {
      smmat::col_scal(basis.num_quad, basis.pdof,
                      rhs_vals[i], basis.legw, tmp.data());
      smmat::gemm_tn<1>(basis.pdof, basis.num_quad, basis.leg, tmp.data(), coeff[i]);
    }
  }
}

//! moment over moment zero
template<typename P, int multsign, term_dependence dep>
void gen_diag_mom_cases(
    legendre_basis<P> const &basis, int level, int mindex,
    std::vector<P> const &moms, block_diag_matrix<P> &coefficients)
{
  static_assert(multsign == 1 or multsign == -1);

  // setup jacobi of variable x and define coeff_mat
  int const num_cells = fm::ipow2(level);

  int const pdof     = basis.pdof;
  int const num_quad = basis.num_quad;

  coefficients.resize_and_zero(pdof * pdof, num_cells);

  size_t const wsize = [&]() -> size_t {
    if constexpr (dep == term_dependence::moment_divided_by_density)
      return num_quad * pdof + 2 * num_quad;
    else if constexpr (dep == term_dependence::lenard_bernstein_coll_theta_1x1v
                       or dep == term_dependence::lenard_bernstein_coll_theta_1x2v
                       or dep == term_dependence::lenard_bernstein_coll_theta_1x3v)
      return num_quad * pdof + 3 * num_quad;
  }();

  span2d<P const> moment(moms.size() / num_cells, num_cells, moms.data());

#pragma omp parallel
  {
    // each thread will allocate it's own tmp matrix
    std::vector<P> workspace(wsize);
    P *tmp   = workspace.data();
    P *gv    = tmp + num_quad * pdof;
    P *gdiv  = gv + num_quad;
    P *gv2   = (dep == term_dependence::moment_divided_by_density) ? nullptr : gdiv + num_quad;

    // workspace will be captured inside the lambda closure
    // no allocations will occur per call
#pragma omp for
    for (int i = 0; i < num_cells; ++i)
    {
      if constexpr (dep == term_dependence::moment_divided_by_density)
      {
        // make gv to be the values of rhs at the quad-nodes
        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + mindex * pdof, gv);
        smmat::gemv(num_quad, pdof, basis.leg, moment[i], gdiv);

        for (int k : iindexof(num_quad))
          gv[k] /= gdiv[k];
      }
      else if constexpr (dep == term_dependence::lenard_bernstein_coll_theta_1x1v)
      {
        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + pdof, gv);
        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + 2 * pdof, gv2);
        smmat::gemv(num_quad, pdof, basis.leg, moment[i], gdiv);

        for (int k : iindexof(num_quad))
          gv[k] = (gv2[k] / gdiv[k]) - gv[k] * gv[k] / (gdiv[k] * gdiv[k]);
      }
      else if constexpr (dep == term_dependence::lenard_bernstein_coll_theta_1x2v)
      {
        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + pdof, gv2);
        for (int k : iindexof(num_quad))
          gv[k] = gv2[k] * gv2[k];
        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + 2 * pdof, gv2);
        for (int k : iindexof(num_quad))
          gv[k] += gv2[k] * gv2[k];

        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + 3 * pdof, gv2);
        smmat::gemv1(num_quad, pdof, basis.leg, moment[i] + 4 * pdof, gv2);

        smmat::gemv(num_quad, pdof, basis.leg, moment[i], gdiv);

        for (int k : iindexof(num_quad))
          gv[k] = 0.5 * ((gv2[k] / gdiv[k]) - gv[k] / (gdiv[k] * gdiv[k]));
      }
      else if constexpr (dep == term_dependence::lenard_bernstein_coll_theta_1x3v)
      {
        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + 1 * pdof, gv2);
        for (int k : iindexof(num_quad))
          gv[k] = gv2[k] * gv2[k];
        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + 2 * pdof, gv2);
        for (int k : iindexof(num_quad))
          gv[k] += gv2[k] * gv2[k];
        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + 3 * pdof, gv2);
        for (int k : iindexof(num_quad))
          gv[k] += gv2[k] * gv2[k];

        smmat::gemv(num_quad, pdof, basis.leg, moment[i] + 4 * pdof, gv2);
        smmat::gemv1(num_quad, pdof, basis.leg, moment[i] + 5 * pdof, gv2);
        smmat::gemv1(num_quad, pdof, basis.leg, moment[i] + 6 * pdof, gv2);

        smmat::gemv(num_quad, pdof, basis.leg, moment[i], gdiv);

        for (int k : iindexof(num_quad))
          gv[k] = (P{1} / P{3}) * ((gv2[k] / gdiv[k]) - gv[k] / (gdiv[k] * gdiv[k]));
      }

      // multiply the values of rhs by the values of the Leg. polynomials
      smmat::col_scal(num_quad, pdof, gv, basis.leg, tmp);

      // multiply results in integration
      smmat::gemm_tn<multsign>(pdof, num_quad, basis.legw, tmp, coefficients[i]);
    }
  } // #pragma omp parallel
}

} // namespace asgard
