#pragma once
#include "asgard_transformations.hpp"

#include "asgard_small_mats.hpp"

// private header, exposes some of the coefficient methods for easier testing
// also reduces the clutter in asgard_coefficients.cpp

namespace asgard
{

template<typename P>
void generate_partial_mass(int const idim, dimension<P> const &dim,
                           partial_term<P> const &pterm,
                           hierarchy_manipulator<P> const &hier, P const time,
                           level_mass_matrces<P> &mass)
{
  if (not dim.volume_jacobian_dV and not pterm.lhs_mass_func())
    return;

  function_1d<P> dv = [&](std::vector<P> const &x, std::vector<P> &dvx) -> void
  {
    if (dim.volume_jacobian_dV) {
      if (pterm.lhs_mass_func()) {
        for (auto i : indexof(x))
          dvx[i] = pterm.lhs_mass_func()(x[i], time) * dim.volume_jacobian_dV(x[i], time);
      } else {
        for (auto i : indexof(x))
          dvx[i] = dim.volume_jacobian_dV(x[i], time);
      }
    } else {
      for (auto i : indexof(x))
        dvx[i] = pterm.lhs_mass_func()(x[i], time);
    }
  };

  int const level  = dim.get_level();

  hier.make_mass(idim, level, dv, mass);
};

// construct 1D coefficient matrix - new conventions
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matrices
//
// the coeff_type must match pterm.coeff_type, it is a template parameter
// so that we can simplify the code and avoid runtime cost with if-constexpr
template<typename P, coefficient_type coeff_type>
void gen_tri_cmat(dimension<P> const &dim, partial_term<P> const &pterm,
                  int const level, P const time,
                  block_tri_matrix<P> &coefficients)
{
  expect(time >= 0.0);
  expect(coeff_type == pterm.coeff_type());
  static_assert(has_flux_v<coeff_type>, "building block-tri-diagonal matrix with no flux");

  auto g_dv_func = [g_func  = pterm.g_func(),
                    dv_func = pterm.dv_func()]() -> g_func_type<P> {
    if (g_func && dv_func)
    {
      return [g_func, dv_func](P const x, P const t) {
        return g_func(x, t) * dv_func(x, t);
      };
    }
    else if (g_func)
    {
      return [=](P const x, P const t) { return g_func(x, t); };
    }
    else if (dv_func)
    {
      return [dv_func](P const x, P const t) { return dv_func(x, t); };
    }
    else
    {
      return [](P const x, P const t) {
        ignore(x);
        ignore(t);
        return P{1.0};
      };
    }
  }();

  // setup jacobi of variable x and define coeff_mat
  auto const num_cells = fm::ipow2(level);

  auto const grid_spacing       = (dim.domain_max - dim.domain_min) / num_cells;
  int const nblock = (dim.get_degree() + 1) * (dim.get_degree() + 1);
  coefficients.resize_and_zero(nblock, num_cells);

  // get quadrature points and quadrature_weights.
  // we do the two-step store because we cannot have 'static' bindings
  static auto const legendre_values =
      legendre_weights<P>(dim.get_degree(), -1.0, 1.0);
  auto const &quadrature_points  = legendre_values[0];
  auto const &quadrature_weights = legendre_values[1];

  auto const legendre_poly_LR = [&]() {
    auto [lP_L, lPP_L] = legendre(fk::vector<P>{-1}, dim.get_degree());
    lP_L               = lP_L * (1 / std::sqrt(grid_spacing));
    auto [lP_R, lPP_R] = legendre(fk::vector<P>{+1}, dim.get_degree());
    lP_R               = lP_R * (1 / std::sqrt(grid_spacing));
    // this is to get around unused warnings (until c++20)
    ignore(lPP_L);
    ignore(lPP_R);
    return std::array<fk::matrix<P>, 2>{lP_L, lP_R};
  }();
  auto const &legendre_poly_L = legendre_poly_LR[0];
  auto const &legendre_poly_R = legendre_poly_LR[1];

  // get the basis functions and derivatives for all k
  // this auto is std::array<fk::matrix<P>, 2>
  auto const legendre_poly_prime = [&]() {
    auto [lP, lPP] = legendre(quadrature_points, dim.get_degree());

    lP  = lP * (1.0 / std::sqrt(grid_spacing));
    lPP = lPP * (1.0 / std::sqrt(grid_spacing) * 2.0 / grid_spacing);

    return std::array<fk::matrix<P>, 2>{lP, lPP};
  }();

  int const degree = dim.get_degree();

  int const block_size = (degree + 1) * (degree + 1);

  // adds a matrix mat (scaled by alpha) into a block of coefficients
  auto coeff_axpy = [&](P const alpha, fk::matrix<P> const &mat, P c[])
      -> void {
    P const *s = mat.data();
    for (int k : indexof<int>(block_size))
      c[k] += alpha * s[k];
  };

  auto const &legendre_poly  = legendre_poly_prime[0];
  auto const &legendre_prime = legendre_poly_prime[1];

  // get jacobian
  auto const jacobi = grid_spacing / 2;

  fk::matrix<P> matrix_LtR(legendre_poly_L.ncols(), legendre_poly_R.ncols());
  fm::gemm(legendre_poly_L, legendre_poly_R, matrix_LtR, true, false, P{1}, P{0});

  fk::matrix<P> matrix_LtL(legendre_poly_L.ncols(), legendre_poly_L.ncols());
  fm::gemm(legendre_poly_L, legendre_poly_L, matrix_LtL, true, false, P{1}, P{0});

  fk::matrix<P> matrix_RtR(legendre_poly_R.ncols(), legendre_poly_R.ncols());
  fm::gemm(legendre_poly_R, legendre_poly_R, matrix_RtR, true, false, P{1}, P{0});

  fk::matrix<P> matrix_RtL(legendre_poly_R.ncols(), legendre_poly_L.ncols());
  fm::gemm(legendre_poly_R, legendre_poly_L, matrix_RtL, true, false, P{1}, P{0});

  // General algorithm
  // For each cell, we need to compute the volume integral within the cell
  // and then the fluxes given the neighboring cells to the left and right.
  // Computing three blocks per cell with size (degree + 1) by (degree + 1)
  // The blocks are denoted by:
  //    (current, current - degree - 1)
  //    (current, current),
  //    (current, current + degree + 1)
  // where "current" for cell "i" is "(degree + 1) * i"
  //
  // The key here is to add the properly scaled matrix LtR, LtL etc.
  // to the correct block of the coefficients matrix.
  //
  // If using periodic boundary, the left-most and right-most cells wrap around
  // i.e., the left cell for the left-most cell is the right-most cell.
  // Even without periodicity, left/right-most cells need special consideration
  // as they must use the Dirichlet or Neumann boundary condition in the flux.
  // Thus, the main for-loop works only on the interior cells
  // and the left most cell (0) and right most cell (num_cells - 1)
  // are handled separately.

#pragma omp parallel
  {
    // each thread will allocate it's own tmp matrix
    fk::matrix<P> tmp(legendre_poly.nrows(), legendre_poly.ncols());

    // tmp will be captured inside the lambda closure
    // no allocations will occur per call
    auto apply_volume = [&](int i) -> void {
      // the penalty term does not include a volume integral
      if constexpr (coeff_type != coefficient_type::penalty)
      {
        for (int k = 0; k < tmp.nrows(); k++)
        {
          P c = g_dv_func(
              (0.5 * quadrature_points[k] + 0.5 + i) * grid_spacing + dim.domain_min, time);
          c *= quadrature_weights(k) * jacobi;

          for (int j = 0; j < tmp.ncols(); j++)
            tmp(k, j) = c * legendre_poly(k, j);
        }

        if constexpr (coeff_type == coefficient_type::mass)
          smmat::gemm_tn<1>(legendre_poly.ncols(), legendre_poly.nrows(),
                            legendre_poly.data(), tmp.data(), coefficients.diag(i));
        else // div or grad falls here
          smmat::gemm_tn<-1>(legendre_prime.ncols(), legendre_prime.nrows(),
                             legendre_prime.data(), tmp.data(), coefficients.diag(i));
      }
    };

#pragma omp for
    for (int i = 1; i < num_cells - 1; ++i)
    {
      // looping over the interior cells

      // get left and right locations for this element
      P const x_left  = dim.domain_min + i * grid_spacing;
      P const x_right = x_left + grid_spacing;

      // get index for current block
      //int const current = (dim.get_degree() + 1) * i;

      apply_volume(i);

      if constexpr (coeff_type == coefficient_type::grad or
                    coeff_type == coefficient_type::div or
                    coeff_type == coefficient_type::penalty)
      {
        // setup numerical flux choice/boundary conditions
        //
        // - <funcCoef*{q},p>
        //----------------------------------------------
        // Numerical Flux is defined as
        // Flux = {{f}} + C/2*[[u]]
        //      = ( f_L + f_R )/2 + FunCoef*( u_R - u_L )/2
        // [[v]] = v_R - v_L

        // FIXME G functions should accept G(x,p,t,dat), since we don't know how
        // the dat is going to be used in the G function (above it is used as
        // linear multuplication but this is not always true)

        P fluxL2 = 0.5 * g_dv_func(x_left, time);
        P fluxR2 = 0.5 * g_dv_func(x_right, time);

        P const fluxL2abs = pterm.get_flux_scale() * std::abs(fluxL2);
        P const fluxR2abs = pterm.get_flux_scale() * std::abs(fluxR2);

        if constexpr (coeff_type == coefficient_type::penalty)
        {
          fluxL2 = 0;
          fluxR2 = 0;
        }

        // get the "trace" values
        // (values at the left and right of each element for all k)
        // -------------------------------------------------------------------------
        // More detailed explanation
        // Each trace_value_ evaluates <FLUX_f,[[v]]>
        // where v is a DG functions with support on I_i. The
        // difference between the trace_values_ varies with the edge the flux
        // is evaluated on and the support of the DG function f.
        // The legendre_poly_X is the trace of f and legende_poly_X_t is for v
        // We will use f=p_X for the polynomials where X=L (left boundary of cell)
        // or X=R (right boundary of cell).  Similar for v but depends on the
        // support.  Note matrix multiply ordering goes by
        // v_mat^T * f_mat for <f,v>

        // trace_value_1 is the interaction on x_{i-1/2} --
        // the edge between cell I_{i-1} and I_i or the left boundary of I_i.
        // f is a DG function with support on I_{i-1}
        // In this case:  {{f}} = p_R/2, [[f]] = p_R, [[v]] = -p_L
        // (in the code below and in all other cases, the expressions has been
        //  simplified by applying the negative or positive -p_L)
        //coeff_axpy(current, current - degree - 1, -fluxL2 - fluxL2abs, matrix_LtR);
        coeff_axpy(-fluxL2 - fluxL2abs, matrix_LtR, coefficients.lower(i));

        // trace_value_2 is the interaction on x_{i-1/2} --
        // the edge between cell I_{i-1} and I_i or the left boundary of I_i.
        // f is a DG function with support on I_{i}
        // In this case:  {{f}} = p_L/2, [[f]] = -p_L, [[v]] = -p_L
        //coeff_axpy(current, current, -fluxL2 + fluxL2abs, matrix_LtL);
        coeff_axpy(-fluxL2 + fluxL2abs, matrix_LtL, coefficients.diag(i));

        // trace_value_3 is the interaction on x_{i+1/2} --
        // the edge between cell I_i and I_{i+1} or the right boundary of I_i.
        // f is a DG function with support on I_{i}
        // In this case:  {{f}} = p_R/2, [[f]] = p_R, [[v]] = p_R
        //coeff_axpy(current, current, fluxR2 + fluxR2abs, matrix_RtR);
        coeff_axpy(fluxR2 + fluxR2abs, matrix_RtR, coefficients.diag(i));

        // trace_value_4 is the interaction on x_{i+1/2} --
        // the edge between cell I_i and I_{i+1} or the right boundary of I_i.
        // f is a DG function with support on I_{i+1}
        // In this case:  {{f}} = p_L/2, [[f]] = -p_L, [[v]] = p_R
        //coeff_axpy(current, current + degree + 1, fluxR2 - fluxR2abs, matrix_RtL);
        coeff_axpy(fluxR2 - fluxR2abs, matrix_RtL, coefficients.upper(i));

        // If dirichelt
        // u^-_LEFT = g(LEFT)
        // u^+_RIGHT = g(RIGHT)

        // Dirichlet Boundary Conditions
        // For div and grad, the boundary is not part of the bilinear operator,
        // but instead tranferred to the source.  Similar to an inflow condition.
        // For penalty, the operator <|gfunc|/2*f,v> is applied for the case where
        // f and v share the same volume support

        // If statement checking coeff_type is because gfunc can evaluate to nan
        // in 1/0 case.  Ex: gfunc = x, domain = [0,4] (possible in spherical
        // coordinates)

        // Neumann boundary conditions
        // For div and grad, the interior trace is used to calculate the flux,
        // similar to an outflow boundary condition. For penalty, nothing is
        // added.
      }
    } // for i

#pragma omp single
    {
      // special case, handle the left and right boundary conditions
      // the first thread that exits the for-loop above will do this work

      // need to consider various types of boundary conditions on left/right
      // but we have a possible case of 1 cell, so left-most is also right-most

      apply_volume(0);   // left-most cell
      if (num_cells > 1) // if right-most is not left-most
        apply_volume(num_cells - 1);

      if constexpr (coeff_type == coefficient_type::grad or
                    coeff_type == coefficient_type::div or
                    coeff_type == coefficient_type::penalty)
      {
        // get index for the last element (first is zero)
        P fluxL2 = 0.5 * g_dv_func(dim.domain_min, time);
        P fluxR2 = 0.5 * g_dv_func(dim.domain_min + grid_spacing, time);

        P fluxL2abs = pterm.get_flux_scale() * std::abs(fluxL2);
        P fluxR2abs = pterm.get_flux_scale() * std::abs(fluxR2);

        if constexpr (coeff_type == coefficient_type::penalty)
        {
          fluxL2 = 0;
          fluxR2 = 0;
        }

        // handle the left-boundary
        switch (pterm.ileft())
        {
        case boundary_condition::dirichlet:
          // If penalty then we add <|g|/2[f],[v]>
          // Else we're wanting no flux as this is handed by the
          // boundary conditions.
          if constexpr (coeff_type == coefficient_type::penalty)
            coeff_axpy(fluxL2abs, matrix_LtL, coefficients.diag(0));
          break;

        case boundary_condition::neumann:
          // If penalty then we add nothing
          // Else we want to standard (outflow) flux
          // <gf,v> = <g{f}/2,v>
          if constexpr (coeff_type != coefficient_type::penalty)
            coeff_axpy(-2.0 * fluxL2, matrix_LtL, coefficients.diag(0));
          break;

        default: // case boundary_condition::periodic
          coeff_axpy(-fluxL2 - fluxL2abs, matrix_LtR, coefficients.lower(0));
          coeff_axpy(-fluxL2 + fluxL2abs, matrix_LtL, coefficients.diag(0));
          break;
        }

        if (num_cells > 1)
        {
          // right boundary of the left-most cell is in the interior
          coeff_axpy(fluxR2 + fluxR2abs, matrix_RtR, coefficients.diag(0));
          coeff_axpy(fluxR2 - fluxR2abs, matrix_RtL, coefficients.upper(0));

          // at this point, we are done with the left-most cell
          // switch the flux to the right-most cell

          fluxL2 = 0.5 * g_dv_func(dim.domain_max - grid_spacing, time);
          fluxR2 = 0.5 * g_dv_func(dim.domain_max, time);

          fluxL2abs = pterm.get_flux_scale() * std::abs(fluxL2);
          fluxR2abs = pterm.get_flux_scale() * std::abs(fluxR2);

          if constexpr (coeff_type == coefficient_type::penalty)
          {
            fluxL2 = 0;
            fluxR2 = 0;
          }

          coeff_axpy(-fluxL2 - fluxL2abs, matrix_LtR, coefficients.lower(num_cells - 1));
          coeff_axpy(-fluxL2 + fluxL2abs, matrix_LtL, coefficients.diag(num_cells - 1));
        }

        // handle the right boundary condition
        switch (pterm.iright())
        {
        case boundary_condition::dirichlet:
          if constexpr (coeff_type == coefficient_type::penalty)
            coeff_axpy(fluxR2abs, matrix_RtR, coefficients.diag(num_cells - 1));
          break;

        case boundary_condition::neumann:
          if constexpr (coeff_type != coefficient_type::penalty)
            coeff_axpy(2.0 * fluxR2, matrix_RtR, coefficients.diag(num_cells - 1));
          break;

        default: // case boundary_condition::periodic
          coeff_axpy(fluxR2 + fluxR2abs, matrix_RtR, coefficients.diag(num_cells - 1));
          coeff_axpy(fluxR2 - fluxR2abs, matrix_RtL, coefficients.upper(num_cells - 1));
          break;
        }
      }
    } // #pragma omp single

  } // #pragma omp parallel

  if constexpr (coeff_type == coefficient_type::grad)
  {
    // take the negative transpose of div
#pragma omp parallel for
    for (int64_t r = 0; r < coefficients.nrows() - 1; r++)
    {
      smmat::neg_transp_swap(degree + 1, coefficients.lower(r + 1), coefficients.upper(r));
      smmat::neg_transp(degree + 1, coefficients.diag(r));
    }
    smmat::neg_transp(degree + 1, coefficients.diag(coefficients.nrows() - 1));
    smmat::neg_transp_swap(degree + 1, coefficients.lower(0), coefficients.upper(coefficients.nrows() - 1));
  }
}

// using extended definition of g-function, now accepting a cell-index (i)
// which allows us to read from a vector, e.g., with pre-computed values of the e-filed
template<typename P, coefficient_type coeff_type, typename gfunctor_type>
void gen_diag_cmat(dimension<P> const &dim, int const level, P const time,
                   gfunctor_type gfunc, block_diag_matrix<P> &coefficients)
{
  expect(time >= 0.0);
  static_assert(not has_flux_v<coeff_type>, "building block-diag-diagonal matrix for flux pterm");

  // setup jacobi of variable x and define coeff_mat
  int const num_cells = fm::ipow2(level);

  P const dx     = (dim.domain_max - dim.domain_min) / num_cells;
  int const pdof = dim.get_degree() + 1;

  coefficients.resize_and_zero(pdof * pdof, num_cells);

  // get quadrature points and quadrature_weights.
  // we do the two-step store because we cannot have 'static' bindings
  static auto const legendre_values =
      legendre_weights<P>(dim.get_degree(), -1.0, 1.0);
  auto const &quad_p = legendre_values[0];
  auto const &quad_w = legendre_values[1];

  int const num_quad = quad_p.size();

  // the values of the normalized Legendre polynomials at the quad points
  // Lv.nrows() == num_quad and Lv.ncols() == pdof
  fk::matrix<P> const Lv = [&]() {
    auto [lP, lPP] = legendre(quad_p, dim.get_degree());

    return lP * (P{1} / std::sqrt(dx));
  }();

  fk::matrix<P> const Lw = [&]() {
    fk::matrix<P> R = Lv;
    smmat::col_scal(num_quad, pdof, P{0.5} * dx, quad_w.data(), R.data());
    return R;
  }();

#pragma omp parallel
  {
    // each thread will allocate it's own tmp matrix
    fk::matrix<P> tmp(num_quad, pdof);

    // tmp will be captured inside the lambda closure
    // no allocations will occur per call
    auto apply_volume = [&](int i) -> void {
      for (int k = 0; k < tmp.nrows(); k++)
      {
        P c = gfunc(i, (0.5 * quad_p[k] + 0.5 + i) * dx + dim.domain_min, time);

        for (int j = 0; j < tmp.ncols(); j++)
          tmp(k, j) = c * Lv(k, j);
      }

      smmat::gemm_tn<1>(pdof, num_quad, Lw.data(), tmp.data(), coefficients[i]);
    };

#pragma omp for
    for (int i = 0; i < num_cells; ++i)
      apply_volume(i);
  } // #pragma omp parallel
}

// special case, term that corresponds to the mass operator
// \int lhs \phi_i \phi_j dx = \int rhs f \phi_j dx
// lhs and rhs (left/right hand sides) are given as cell-by-cell Legenre expansions
template<typename P, coefficient_type coeff_type, int irhs, int multsign = 1>
void gen_diag_mom_cmat(dimension<P> const &dim, int const level, P const time,
                       int const num_moments, std::vector<P> const &moms,
                       block_diag_matrix<P> &coefficients)
{
  expect(time >= 0.0);
  expect(0 < irhs and irhs < num_moments);
  static_assert(not has_flux_v<coeff_type>, "building block-diag-diagonal matrix for flux pterm");

  // setup jacobi of variable x and define coeff_mat
  int const num_cells = fm::ipow2(level);

  P const dx     = (dim.domain_max - dim.domain_min) / num_cells;
  int const pdof = dim.get_degree() + 1;

  coefficients.resize_and_zero(pdof * pdof, num_cells);

  // get quadrature points and quadrature_weights.
  // we do the two-step store because we cannot have 'static' bindings
  static auto const legendre_values =
      legendre_weights<P>(dim.get_degree(), -1.0, 1.0);
  auto const &quad_p = legendre_values[0];
  auto const &quad_w = legendre_values[1];

  int const num_quad = quad_p.size();

  // the values of the normalized Legendre polynomials at the quad points
  // Lv.nrows() == num_quad and Lv.ncols() == pdof
  fk::matrix<P> const Lv = [&]() {
    auto [lP, lPP] = legendre(quad_p, dim.get_degree());

    return lP * (P{1} / std::sqrt(dx));
  }();

  fk::matrix<P> const Lw = [&]() {
    fk::matrix<P> R = Lv;
    smmat::col_scal(num_quad, pdof, P{0.5} * dx, quad_w.data(), R.data());
    return R;
  }();

  expect(moms.size() == static_cast<size_t>(num_moments * pdof * num_cells));
  span2d<P const> moment(num_moments * pdof, num_cells, moms.data());

#pragma omp parallel
  {
    // each thread will allocate it's own tmp matrix
    std::vector<P> workspace(num_quad * pdof + pdof * pdof + num_quad);
    P *tmp   = workspace.data();
    P *lmass = tmp + num_quad * pdof;
    P *gv    = lmass + pdof * pdof;

    // workspace will be captured inside the lambda closure
    // no allocations will occur per call
    auto apply_volume = [&](int i) -> void {
      // make gv to be the values of rhs at the quad-nodes
      smmat::gemv(num_quad, pdof, Lv.data(), moment[i] + irhs * pdof, gv);

      // multiply the values of rhs by the values of the Leg. polynomials
      smmat::col_scal(num_quad, pdof, gv, Lv.data(), tmp);

      // multiply results in integration
      smmat::gemm_tn<multsign>(pdof, num_quad, Lw.data(), tmp, coefficients[i]);

      // make gv to be the values of lhs at the quad nodes
      smmat::gemv(num_quad, pdof, Lv.data(), moment[i], gv);

      // multiply the values of lhs by the values of the Leg. polynomials
      smmat::col_scal(num_quad, pdof, gv, Lv.data(), tmp);

      // multiply results in integration
      std::fill_n(lmass, pdof * pdof, P{0});
      smmat::gemm_tn<multsign>(pdof, num_quad, Lw.data(), tmp, lmass);

      switch (pdof)
      {
        case 1:
          coefficients[i][0] /= lmass[0];
          break;
        case 2:
          smmat::inv2by2(lmass);
          smmat::gemm2by2(lmass, coefficients[i]);
          break;
        default:
          // factorize and solve
          smmat::potrf(pdof, lmass);
          smmat::posvm(pdof, lmass, coefficients[i]);
          break;
      };
    };

#pragma omp for
    for (int i = 0; i < num_cells; ++i)
      apply_volume(i);
  } // #pragma omp parallel
}

// same as gen_diag_mom_cmat but uses division as opposed to inversion of a mass matrix
template<typename P, coefficient_type coeff_type, int irhs, int multsign = 1>
void gen_diag_mom_cmat_div(dimension<P> const &dim, int const level, P const time,
                           int const num_moments, std::vector<P> const &moms,
                           block_diag_matrix<P> &coefficients)
{
  expect(time >= 0.0);
  expect(0 < irhs and irhs < num_moments);
  static_assert(not has_flux_v<coeff_type>, "building block-diag-diagonal matrix for flux pterm");

  // setup jacobi of variable x and define coeff_mat
  int const num_cells = fm::ipow2(level);

  P const dx     = (dim.domain_max - dim.domain_min) / num_cells;
  int const pdof = dim.get_degree() + 1;

  coefficients.resize_and_zero(pdof * pdof, num_cells);

  // get quadrature points and quadrature_weights.
  // we do the two-step store because we cannot have 'static' bindings
  static auto const legendre_values =
      legendre_weights<P>(dim.get_degree(), -1.0, 1.0);
  auto const &quad_p = legendre_values[0];
  auto const &quad_w = legendre_values[1];

  int const num_quad = quad_p.size();

  // the values of the normalized Legendre polynomials at the quad points
  // Lv.nrows() == num_quad and Lv.ncols() == pdof
  fk::matrix<P> const Lv = [&]() {
    auto [lP, lPP] = legendre(quad_p, dim.get_degree());

    return lP * (P{1} / std::sqrt(dx));
  }();

  fk::matrix<P> const Lw = [&]() {
    fk::matrix<P> R = Lv;
    smmat::col_scal(num_quad, pdof, P{0.5} * dx, quad_w.data(), R.data());
    return R;
  }();

  expect(moms.size() == static_cast<size_t>(num_moments * pdof * num_cells));
  span2d<P const> moment(num_moments * pdof, num_cells, moms.data());

#pragma omp parallel
  {
    // each thread will allocate it's own tmp matrix
    std::vector<P> workspace(num_quad * pdof + 2 * num_quad);
    P *tmp   = workspace.data();
    P *gv    = tmp + num_quad * pdof;
    P *gdiv  = gv + num_quad;

    // workspace will be captured inside the lambda closure
    // no allocations will occur per call
    auto apply_volume = [&](int i) -> void {
      // make gv to be the values of rhs at the quad-nodes
      smmat::gemv(num_quad, pdof, Lv.data(), moment[i] + irhs * pdof, gv);
      smmat::gemv(num_quad, pdof, Lv.data(), moment[i], gdiv);

      for (int k : indexof(num_quad))
        gv[k] /= gdiv[k];

      // multiply the values of rhs by the values of the Leg. polynomials
      smmat::col_scal(num_quad, pdof, gv, Lv.data(), tmp);

      // multiply results in integration
      smmat::gemm_tn<multsign>(pdof, num_quad, Lw.data(), tmp, coefficients[i]);
    };

#pragma omp for
    for (int i = 0; i < num_cells; ++i)
      apply_volume(i);
  } // #pragma omp parallel
}

} // namespace asgard
