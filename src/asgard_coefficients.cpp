#include "asgard_coefficients.hpp"

#include "asgard_small_mats.hpp"

namespace asgard
{
// construct 1D coefficient matrix - new conventions
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matrices
//
// the coeff_type must match pterm.coeff_type, it is a template parameter
// so that we can simplify the code and avoid runtime cost with if-constexpr
template<typename P, coefficient_type coeff_type>
void generate_coefficients(dimension<P> const &dim, partial_term<P> const &pterm,
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
  //auto const degrees_freedom_1d = (dim.get_degree() + 1) * num_cells;
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

template<typename P>
void generate_coefficients(
    dimension<P> const &dim, partial_term<P> const &pterm,
    int const level, P const time, block_tri_matrix<P> &coefficients)
{
  switch (pterm.coeff_type())
  {
  case coefficient_type::mass:
    throw std::runtime_error("trying to generate block_tri_matrix from pterm with no flux in the coefficient");
    break;
  case coefficient_type::grad:
    generate_coefficients<P, coefficient_type::grad>(dim, pterm, level, time, coefficients);
    break;
  case coefficient_type::div:
    generate_coefficients<P, coefficient_type::div>(dim, pterm, level, time, coefficients);
    break;
  default: // case coefficient_type::penalty:
    generate_coefficients<P, coefficient_type::penalty>(dim, pterm, level, time, coefficients);
    break;
  };
}

template<typename P, coefficient_type coeff_type>
void generate_coefficients(dimension<P> const &dim, partial_term<P> const &pterm,
                           int const level, P const time,
                           block_diag_matrix<P> &coefficients)
{
  expect(time >= 0.0);
  expect(coeff_type == pterm.coeff_type());
  static_assert(not has_flux_v<coeff_type>, "building block-diag-diagonal matrix for flux pterm");

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

  auto const grid_spacing = (dim.domain_max - dim.domain_min) / num_cells;
  int const nblock        = (dim.get_degree() + 1) * (dim.get_degree() + 1);
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
                            legendre_poly.data(), tmp.data(), coefficients[i]);
        else // div or grad falls here
          smmat::gemm_tn<-1>(legendre_prime.ncols(), legendre_prime.nrows(),
                             legendre_prime.data(), tmp.data(), coefficients[i]);

      }
    };

#pragma omp for
    for (int i = 0; i < num_cells; ++i)
    {
      // looping over the interior cells
      apply_volume(i);
    }
  } // #pragma omp parallel
}

template<typename P>
void generate_coefficients(
    dimension<P> const &dim, partial_term<P> const &pterm,
    int const level, P const time, block_diag_matrix<P> &coefficients)
{
  expect(not has_flux(pterm.coeff_type()));
  // add a case statement if there are other coefficient_type instances with no flux
  generate_coefficients<P, coefficient_type::mass>(dim, pterm, level, time, coefficients);
}

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

template<typename P>
void generate_all_coefficients(
    PDE<P> &pde, coefficient_matrices<P> &mats, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, P const time)
{
  tools::time_event time_generating_("gen_coefficients");
  expect(time >= 0.0);

  static block_tri_matrix<P> raw_tri;
  static block_diag_matrix<P> raw_diag;

  int const num_dims = pde.num_dims();
  int const pdof     = hier.degree() + 1;

  for (int d : indexof<int>(num_dims))
  {
    dimension<P> const &dim = pde.get_dimensions()[d];

    int const level = dim.get_level();

    for (int t : indexof<int>(pde.num_terms()))
    {
      auto const &term1d = pde.get_terms()[t][d];
      auto const &pterms = term1d.get_partial_terms();

      // TODO: skip regenerating coefficients that are constant in time
      // "maybe" what about change in the level? Must apply that too
      //if (not term1d.time_dependent() and time > 0.0)
      //  continue;

      expect(pterms.size() >= 1);
      if (pterms.size() == 1)
      {
        // the single term case uses less scratch space
        generate_partial_mass(d, dim, pterms[0], hier, time,
                              mats.pterm_mass[t * num_dims + d][0]);

        if (has_flux(pterms[0].coeff_type()))
        {
          generate_coefficients<P>(dim, pterms[0], level, time, raw_tri);

          if (mats.pterm_mass[t * num_dims + d][0].has_level(level))
            invert_mass(pdof, mats.pterm_mass[t * num_dims + d][0][level], raw_tri);

          mats.term_coeffs[t * num_dims + d] = hier.tri2hierarchical(raw_tri, level, conn);
        }
        else // no-flux, e.g., mass matrix
        {
          generate_coefficients<P>(dim, pterms[0], level, time, raw_diag);

          if (mats.pterm_mass[t * num_dims + d][0].has_level(level))
            invert_mass(pdof, mats.pterm_mass[t * num_dims + d][0][level], raw_diag);

          mats.term_coeffs[t * num_dims + d] = hier.diag2hierarchical(raw_diag, level, conn);
        }
      }
      else
      {
        // additional scratch space matrices
        static block_tri_matrix<P> raw_tri0, raw_tri1;
        static block_diag_matrix<P> raw_diag0, raw_diag1;

        // switch to non-owning pointers for easier and cheaper swapping
        // will hold the final result
        block_tri_matrix<P> *rtri = &raw_tri;
        block_diag_matrix<P> *rdiag = &raw_diag;

        // used for scratch space
        block_tri_matrix<P> *rtri0 = &raw_tri0;
        block_tri_matrix<P> *rtri1 = &raw_tri1;
        block_diag_matrix<P> *rdiag0 = &raw_diag0;
        block_diag_matrix<P> *rdiag1 = &raw_diag1;

        // check if using diagonal or tri-diagonal structure
        bool is_tri_diagonal = false;
        for (auto const &p : pterms)
        {
          if (has_flux(p.coeff_type()))
          {
            is_tri_diagonal = true;
            break;
          }
        }

        if (is_tri_diagonal)
        {
          for (auto fi : indexof(pterms.size()))
          {
            // looping over partial-terms in reverse order
            int64_t const ip = static_cast<int64_t>(pterms.size()) - fi - 1;
            auto const &pterm = pterms[ip];
            generate_partial_mass(d, dim, pterm, hier, time,
                                  mats.pterm_mass[t * num_dims + d][fi]);

            if (fi == 0)
            {
              if (has_flux(pterm.coeff_type()))
              {
                generate_coefficients<P>(dim, pterm, level, time, *rtri);

                if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rtri);

                mats.pterm_coeffs[t * num_dims + d][ip] = hier.tri2hierarchical(*rtri, level, conn);
              }
              else
              {
                generate_coefficients<P>(dim, pterm, level, time, *rdiag);

                if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rdiag);

                *rtri = *rdiag;
                mats.pterm_coeffs[t * num_dims + d][ip] = hier.diag2hierarchical(*rdiag, level, conn);
              }
            }
            else
            {
              if (has_flux(pterm.coeff_type()))
              {
                generate_coefficients<P>(dim, pterm, level, time, *rtri0);

                if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rtri0);

                mats.pterm_coeffs[t * num_dims + d][ip] = hier.tri2hierarchical(*rtri0, level, conn);

                if (rtri1->nblock() != rtri->nblock() or rtri1->nrows() != rtri->nrows())
                  rtri1->resize_and_zero(*rtri);

                std::swap(rtri, rtri1);
                if (pterm.flux() == flux_type::upwind)
                  gemm_block_tri_ul(pdof, *rtri0, *rtri1, *rtri);
                else
                  gemm_block_tri_lu(pdof, *rtri0, *rtri1, *rtri);
              }
              else
              {
                generate_coefficients<P>(dim, pterm, level, time, *rdiag0);

                if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rdiag0);

                mats.pterm_coeffs[t * num_dims + d][ip] = hier.diag2hierarchical(*rdiag0, level, conn);

                if (rtri1->nblock() != rtri->nblock() or rtri1->nrows() != rtri->nrows())
                  rtri1->resize_and_zero(*rtri);

                std::swap(rtri, rtri1);
                gemm_diag_tri(pdof, *rdiag0, *rtri1, *rtri);
              }
            }
          }
          mats.term_coeffs[t * num_dims + d] = hier.tri2hierarchical(*rtri, level, conn);
        }
        else
        {
          // using a series of diagonal matrices
          for (auto fi : indexof(pterms.size()))
          {
            // looping over partial-terms in reverse order
            int64_t const ip = static_cast<int64_t>(pterms.size()) - fi - 1;
            auto const &pterm = pterms[ip];

            generate_partial_mass(d, dim, pterm, hier, time,
                                  mats.pterm_mass[t * num_dims + d][fi]);

            if (fi == 0)
            {
              generate_coefficients<P>(dim, pterm, level, time, *rdiag);

              if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rdiag);

              mats.pterm_coeffs[t * num_dims + d][ip] = hier.diag2hierarchical(*rdiag, level, conn);
            }
            else
            {
              generate_coefficients<P>(dim, pterm, level, time, *rdiag0);

              if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rdiag0);

              mats.pterm_coeffs[t * num_dims + d][ip] = hier.diag2hierarchical(*rdiag0, level, conn);

              if (rdiag1->nblock() != rdiag0->nblock() or rdiag1->nrows() != rdiag0->nrows())
                rdiag1->resize_and_zero(*rdiag0);

              std::swap(rdiag, rdiag1);
              gemm_block_diag(pdof, *rdiag0, *rdiag1, *rdiag);
            }
          }
          mats.term_coeffs[t * num_dims + d] = hier.diag2hierarchical(*rdiag, level, conn);
        }
      } // if multiple partial terms
    } // for num-terms
  } // for num-dims
}

template<typename P>
inline fk::vector<int>
linear_coords_to_indices(PDE<P> const &pde, int const degree,
                         fk::vector<int> const &coords)
{
  fk::vector<int> indices(coords.size());
  for (int d = 0; d < pde.num_dims(); ++d)
  {
    indices(d) = coords(d) * (degree + 1);
  }
  return indices;
}
template<typename P>
void build_system_matrix(
    PDE<P> const &pde, std::function<fk::matrix<P>(int, int)> get_coeffs,
    elements::table const &elem_table, fk::matrix<P> &A,
    element_subgrid const &grid, imex_flag const imex)
{
  // assume uniform degree for now
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = fm::ipow(degree + 1, pde.num_dims());

  int const A_cols = elem_size * grid.ncols();
  int const A_rows = elem_size * grid.nrows();
  expect(A.ncols() == A_cols && A.nrows() == A_rows);

  using key_type = std::pair<int, int>;
  using val_type = fk::matrix<P, mem_type::owner, resource::host>;
  std::map<key_type, val_type> coef_cache;

  // copy coefficients to host for subsequent use
  for (int k = 0; k < pde.num_terms(); ++k)
  {
    for (int d = 0; d < pde.num_dims(); d++)
    {
      coef_cache.emplace(key_type(k, d), get_coeffs(k, d));
    }
  }

  auto terms = pde.get_terms();

  // loop over elements
  for (auto i = grid.row_start; i <= grid.row_stop; ++i)
  {
    // first, get linearized indices for this element
    //
    // calculate from the level/cell indices for each
    // dimension
    fk::vector<int> const coords = elem_table.get_coords(i);
    expect(coords.size() == pde.num_dims() * 2);
    fk::vector<int> const elem_indices = linearize(coords);

    int const global_row = i * elem_size;

    // calculate the row portion of the
    // operator position used for this
    // element's gemm calls
    fk::vector<int> const operator_row =
        linear_coords_to_indices(pde, degree, elem_indices);

    // loop over connected elements. for now, we assume
    // full connectivity
    for (int j = grid.col_start; j <= grid.col_stop; ++j)
    {
      // get linearized indices for this connected element
      fk::vector<int> const coords_nD = elem_table.get_coords(j);
      expect(coords_nD.size() == pde.num_dims() * 2);
      fk::vector<int> const connected_indices = linearize(coords_nD);

      // calculate the col portion of the
      // operator position used for this
      // element's gemm calls
      fk::vector<int> const operator_col =
          linear_coords_to_indices(pde, degree, connected_indices);

      for (int k = 0; k < pde.num_terms(); ++k)
      {
        std::vector<fk::matrix<P>> kron_vals;
        fk::matrix<P> kron0(1, 1);
        // if using imex, include only terms that match the flag
        if (imex == imex_flag::unspecified || terms[k][0].flag() == imex)
        {
          kron0(0, 0) = 1.0;
        }
        else
        {
          kron0(0, 0) = 0.0;
        }
        kron_vals.push_back(std::move(kron0));
        for (int d = 0; d < pde.num_dims(); d++)
        {
          fk::matrix<P, mem_type::view> op_view = fk::matrix<P, mem_type::view>(
              coef_cache[key_type(k, d)], operator_row(d),
              operator_row(d) + degree, operator_col(d),
              operator_col(d) + degree);
          fk::matrix<P> k_new = kron_vals[d].kron(op_view);
          kron_vals.push_back(std::move(k_new));
        }

        // calculate the position of this element in the
        // global system matrix
        int const global_col = j * elem_size;
        auto const &k_tmp    = kron_vals.back();
        fk::matrix<P, mem_type::view> A_view(
            A, global_row - grid.row_start * elem_size,
            global_row + k_tmp.nrows() - 1 - grid.row_start * elem_size,
            global_col - grid.col_start * elem_size,
            global_col + k_tmp.ncols() - 1 - grid.col_start * elem_size);

        A_view = A_view + k_tmp;
      }
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template void generate_all_coefficients<double>(
    PDE<double> &, coefficient_matrices<double> &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double);

template void build_system_matrix<double>(
    PDE<double> const &, std::function<fk::matrix<double>(int, int)>,
    elements::table const &, fk::matrix<double> &,
    element_subgrid const &, imex_flag const);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void generate_all_coefficients<float>(
    PDE<float> &, coefficient_matrices<float> &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float);

template void build_system_matrix<float>(
    PDE<float> const &, std::function<fk::matrix<float>(int, int)>,
    elements::table const &, fk::matrix<float> &,
    element_subgrid const &, imex_flag const);
#endif

} // namespace asgard
