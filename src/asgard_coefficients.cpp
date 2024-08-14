#include "asgard_coefficients.hpp"

namespace asgard
{
// generate coefficient matrices for each 1D term in each dimension and
// underlying partial term coefficients matrices
template<typename P>
void generate_all_coefficients(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    P const time, bool const rotate)
{
  tools::time_event time_generating_("gen_coefficients");
  expect(time >= 0.0);

  for (auto i = 0; i < pde.num_dims(); ++i)
  {
    auto const &dim = pde.get_dimensions()[i];
    std::vector<int> ipiv((dim.get_degree() + 1) *
                          fm::two_raised_to(dim.get_level()));
    for (auto j = 0; j < pde.num_terms(); ++j)
    {
      auto const &term_1D       = pde.get_terms()[j][i];
      auto const &partial_terms = term_1D.get_partial_terms();

      // skip regenerating coefficients that are constant in time
      if (!term_1D.time_dependent() && time > 0.0)
      {
        continue;
      }

      for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
      {
        // TODO: refactor these changes, this is slow!
        partial_term<P> const lhs_mass_pterm = partial_term<P>(
            coefficient_type::mass, partial_terms[k].lhs_mass_func(), nullptr,
            flux_type::central, boundary_condition::periodic,
            boundary_condition::periodic, homogeneity::homogeneous,
            homogeneity::homogeneous, {}, nullptr, {}, nullptr,
            dim.volume_jacobian_dV);

        auto mass_coeff = generate_coefficients<P>(
            dim, lhs_mass_pterm, transformer, dim.get_level(), time, rotate);

        // precompute inv(mass) * coeff for each level up to max level
        // std::vector<fk::matrix<P>> pterm_coeffs;

        auto result = generate_coefficients<P>(
            dim, partial_terms[k], transformer, dim.get_level(), time, rotate);

        // for (int level = 0; level <= dim.get_level(); ++level)
        //{
        auto const dof  = (dim.get_degree() + 1) * fm::two_raised_to(dim.get_level());
        auto result_tmp = result.extract_submatrix(0, 0, dof, dof);
        if (partial_terms[k].dv_func() || partial_terms[k].g_func())
        {
          auto mass_tmp = mass_coeff.extract_submatrix(0, 0, dof, dof);
          fm::gesv(mass_tmp, result_tmp, ipiv);
        }
        // pterm_coeffs.emplace_back(std::move(result_tmp));
        //}

        pde.set_lhs_mass(j, i, k, std::move(mass_coeff));
        // pde.set_partial_coefficients(j, i, k, std::move(pterm_coeffs));
        pde.set_partial_coefficients(j, i, k, std::move(result_tmp));
      }
    }
    pde.rechain_dimension(i);
  }
}

template<typename P>
void generate_all_coefficients_max_level(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    P const time, bool const rotate)
{
  tools::time_event time_generating_("gen_coefficients");
  expect(time >= 0.0);

  for (auto i = 0; i < pde.num_dims(); ++i)
  {
    auto const &dim = pde.get_dimensions()[i];
    std::vector<int> ipiv((dim.get_degree() + 1) * fm::two_raised_to(pde.max_level()));
    for (auto j = 0; j < pde.num_terms(); ++j)
    {
      auto const &term_1D       = pde.get_terms()[j][i];
      auto const &partial_terms = term_1D.get_partial_terms();

      for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
      {
        // TODO: refactor these changes, this is slow!
        partial_term<P> const lhs_mass_pterm = partial_term<P>(
            coefficient_type::mass, partial_terms[k].lhs_mass_func(), nullptr,
            flux_type::central, boundary_condition::periodic,
            boundary_condition::periodic, homogeneity::homogeneous,
            homogeneity::homogeneous, {}, nullptr, {}, nullptr,
            dim.volume_jacobian_dV);

        auto mass_coeff = generate_coefficients<P>(
            dim, lhs_mass_pterm, transformer, pde.max_level(), time, rotate);

        // precompute inv(mass) * coeff for each level up to max level
        std::vector<fk::matrix<P>> pterm_coeffs;

        for (int level = 0; level <= pde.max_level(); ++level)
        {
          auto result = generate_coefficients<P>(
              dim, partial_terms[k], transformer, level, time, rotate);
          if (partial_terms[k].dv_func() || partial_terms[k].g_func())
          {
            auto const dof = (dim.get_degree() + 1) * fm::two_raised_to(level);
            auto mass_tmp  = mass_coeff.extract_submatrix(0, 0, dof, dof);
            fm::gesv(mass_tmp, result, ipiv);
          }
          pterm_coeffs.emplace_back(std::move(result));
        }

        pde.set_lhs_mass(j, i, k, std::move(mass_coeff));
        pde.set_partial_coefficients(j, i, k, std::move(pterm_coeffs));
      }
    }
    pde.rechain_dimension(i);
  }
}

template<typename P>
void generate_dimension_mass_mat(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer)
{
  for (auto i = 0; i < pde.num_dims(); ++i)
  {
    auto &dim = pde.get_dimensions()[i];

    for (int level = 0; level <= pde.max_level(); ++level)
    {
      partial_term<P> const lhs_mass_pterm = partial_term<P>(
          coefficient_type::mass, nullptr, nullptr, flux_type::central,
          boundary_condition::periodic, boundary_condition::periodic,
          homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
          nullptr, dim.volume_jacobian_dV);
      auto mass_mat = generate_coefficients<P>(dim, lhs_mass_pterm, transformer,
                                               level, 0.0, true);

      pde.update_dimension_mass_mat(i, std::move(mass_mat), level);
    }
  }
}

// construct 1D coefficient matrix - new conventions
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matricies
//
// the coeff_type must match pterm.coeff_type, it is a template parameter
// so that we can simplify the code and avoid runtime cost with if-constexpr
template<typename P, coefficient_type coeff_type>
fk::matrix<P> generate_coefficients(
    dimension<P> const &dim, partial_term<P> const &pterm,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const level, P const time, bool const rotate)
{
  expect(time >= 0.0);
  expect(transformer.degree == dim.get_degree());
  expect(transformer.max_level >= dim.get_level());
  expect(level <= transformer.max_level);
  expect(coeff_type == pterm.coeff_type());

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
  auto const num_cells = fm::two_raised_to(level);

  auto const grid_spacing       = (dim.domain_max - dim.domain_min) / num_cells;
  auto const degrees_freedom_1d = (dim.get_degree() + 1) * num_cells;
  fk::matrix<P> coefficients(degrees_freedom_1d, degrees_freedom_1d);

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

  // adds a matrix mat (scaled by alpha) into a block of coefficients
  auto coeff_axpy = [&](int begin, int end, P alpha, fk::matrix<P> const &mat)
      -> void {
    fk::matrix<P, mem_type::view> blk(coefficients, begin, begin + degree,
                                      end, end + degree);

    for (int j = 0; j <= degree; j++)
      for (int i = 0; i <= degree; i++)
        blk(i, j) += alpha * mat(i, j);
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
        int const current = (dim.get_degree() + 1) * i;

        for (int k = 0; k < tmp.nrows(); k++)
        {
          P c = g_dv_func(
              (0.5 * quadrature_points[k] + 0.5 + i) * grid_spacing + dim.domain_min, time);
          c *= quadrature_weights(k) * jacobi;

          for (int j = 0; j < tmp.ncols(); j++)
            tmp(k, j) = c * legendre_poly(k, j);
        }

        fk::matrix<P, mem_type::view> blk(coefficients, current,
                                          current + degree, current,
                                          current + degree);
        if constexpr (coeff_type == coefficient_type::mass)
          // volume integral where phi is trial(tmp) and psi is test(legendre_poly)
          //  \int phi(x)\psi(x) dx
          fm::gemm(legendre_poly, tmp, blk, true, false, P{1}, P{1});
        else // div or grad falls here
          // -\int \phi(x)\psi'(x) dx
          fm::gemm(legendre_prime, tmp, blk, true, false, P{-1}, P{1});
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
      int const current = (dim.get_degree() + 1) * i;

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
        coeff_axpy(current, current - degree - 1, -fluxL2 - fluxL2abs, matrix_LtR);

        // trace_value_2 is the interaction on x_{i-1/2} --
        // the edge between cell I_{i-1} and I_i or the left boundary of I_i.
        // f is a DG function with support on I_{i}
        // In this case:  {{f}} = p_L/2, [[f]] = -p_L, [[v]] = -p_L
        coeff_axpy(current, current, -fluxL2 + fluxL2abs, matrix_LtL);

        // trace_value_3 is the interaction on x_{i+1/2} --
        // the edge between cell I_i and I_{i+1} or the right boundary of I_i.
        // f is a DG function with support on I_{i}
        // In this case:  {{f}} = p_R/2, [[f]] = p_R, [[v]] = p_R
        coeff_axpy(current, current, fluxR2 + fluxR2abs, matrix_RtR);

        // trace_value_4 is the interaction on x_{i+1/2} --
        // the edge between cell I_i and I_{i+1} or the right boundary of I_i.
        // f is a DG function with support on I_{i+1}
        // In this case:  {{f}} = p_L/2, [[f]] = -p_L, [[v]] = p_R
        coeff_axpy(current, current + degree + 1, fluxR2 - fluxR2abs, matrix_RtL);

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
        int const last = (dim.get_degree() + 1) * (num_cells - 1);

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
            coeff_axpy(0, 0, fluxL2abs, matrix_LtL);
          break;

        case boundary_condition::neumann:
          // If penalty then we add nothing
          // Else we want to standard (outflow) flux
          // <gf,v> = <g{f}/2,v>
          if constexpr (coeff_type != coefficient_type::penalty)
            coeff_axpy(0, 0, -2.0 * fluxL2, matrix_LtL);
          break;

        default: // case boundary_condition::periodic
          coeff_axpy(0, last, -fluxL2 - fluxL2abs, matrix_LtR);
          coeff_axpy(0, 0, -fluxL2 + fluxL2abs, matrix_LtL);
          break;
        }

        if (num_cells > 1)
        {
          // right boundary of the left-most cell is in the interior
          coeff_axpy(0, 0, fluxR2 + fluxR2abs, matrix_RtR);
          coeff_axpy(0, degree + 1, fluxR2 - fluxR2abs, matrix_RtL);

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

          // left boundary of the right-most cell is in the interior
          coeff_axpy(last, last - degree - 1, -fluxL2 - fluxL2abs, matrix_LtR);
          coeff_axpy(last, last, -fluxL2 + fluxL2abs, matrix_LtL);
        }

        // handle the right boundary condition
        switch (pterm.iright())
        {
        case boundary_condition::dirichlet:
          if constexpr (coeff_type == coefficient_type::penalty)
            coeff_axpy(last, last, fluxR2abs, matrix_RtR);
          break;

        case boundary_condition::neumann:
          if constexpr (coeff_type != coefficient_type::penalty)
            coeff_axpy(last, last, 2.0 * fluxR2, matrix_RtR);
          break;

        default: // case boundary_condition::periodic
          coeff_axpy(last, last, fluxR2 + fluxR2abs, matrix_RtR);
          coeff_axpy(last, 0, fluxR2 - fluxR2abs, matrix_RtL);
          break;
        }
      }
    } // #pragma omp single

  } // #pragma omp parallel

  if constexpr (coeff_type == coefficient_type::grad)
  {
    // take the negative transpose of div
    coefficients.transpose();
    std::transform(coefficients.begin(), coefficients.end(),
                   coefficients.begin(), std::negate<P>());
  }

  if (rotate)
  {
    // transform matrix to wavelet space

    // These routines do the following operation:
    // coefficients = forward_trans * coefficients * forward_trans_transpose;
    coefficients = transformer.apply(
        transformer.apply(coefficients, level, basis::side::right,
                          basis::transpose::trans),
        level, basis::side::left, basis::transpose::no_trans);
  }

  return coefficients;
}

template<typename P>
fk::matrix<P> generate_coefficients(
    dimension<P> const &dim, partial_term<P> const &pterm,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const level, P const time, bool const rotate)
{
  switch (pterm.coeff_type())
  {
  case coefficient_type::mass:
    return generate_coefficients<P, coefficient_type::mass>(dim, pterm, transformer, level, time, rotate);
  case coefficient_type::grad:
    return generate_coefficients<P, coefficient_type::grad>(dim, pterm, transformer, level, time, rotate);
  case coefficient_type::div:
    return generate_coefficients<P, coefficient_type::div>(dim, pterm, transformer, level, time, rotate);
  default: // case coefficient_type::penalty:
    return generate_coefficients<P, coefficient_type::penalty>(dim, pterm, transformer, level, time, rotate);
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template fk::matrix<double> generate_coefficients<double>(
    dimension<double> const &dim, partial_term<double> const &pterm,
    basis::wavelet_transform<double, resource::host> const &transformer,
    int const level, double const time, bool const rotate);

template void generate_all_coefficients<double>(
    PDE<double> &pde,
    basis::wavelet_transform<double, resource::host> const &transformer,
    double const time, bool const rotate);

template void generate_all_coefficients_max_level<double>(
    PDE<double> &pde,
    basis::wavelet_transform<double, resource::host> const &transformer,
    double const time, bool const rotate);

template void generate_dimension_mass_mat<double>(
    PDE<double> &pde,
    basis::wavelet_transform<double, resource::host> const &transformer);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template fk::matrix<float> generate_coefficients<float>(
    dimension<float> const &dim, partial_term<float> const &pterm,
    basis::wavelet_transform<float, resource::host> const &transformer,
    int const level, float const time, bool const rotate);

template void generate_all_coefficients<float>(
    PDE<float> &pde,
    basis::wavelet_transform<float, resource::host> const &transformer,
    float const time, bool const rotate);

template void generate_all_coefficients_max_level<float>(
    PDE<float> &pde,
    basis::wavelet_transform<float, resource::host> const &transformer,
    float const time, bool const rotate);

template void generate_dimension_mass_mat<float>(
    PDE<float> &pde,
    basis::wavelet_transform<float, resource::host> const &transformer);
#endif

} // namespace asgard
