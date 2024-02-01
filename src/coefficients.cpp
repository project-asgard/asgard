#include "coefficients.hpp"

#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "basis.hpp"
#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tools.hpp"
#include "transformations.hpp"
#include <numeric>

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

  for (auto i = 0; i < pde.num_dims; ++i)
  {
    auto const &dim = pde.get_dimensions()[i];
    std::vector<int> ipiv(dim.get_degree() *
                          fm::two_raised_to(dim.get_level()));
    for (auto j = 0; j < pde.num_terms; ++j)
    {
      auto const &term_1D       = pde.get_terms()[j][i];
      auto const &partial_terms = term_1D.get_partial_terms();

      // skip regenerating coefficients that are constant in time
      if (!term_1D.time_dependent && time > 0.0)
      {
        continue;
      }

      for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
      {
        // TODO: refactor these changes, this is slow!
        partial_term<P> const lhs_mass_pterm = partial_term<P>(
            coefficient_type::mass, partial_terms[k].lhs_mass_func, nullptr,
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
        auto const dof  = dim.get_degree() * fm::two_raised_to(dim.get_level());
        auto result_tmp = result.extract_submatrix(0, 0, dof, dof);
        if (partial_terms[k].dv_func || partial_terms[k].g_func)
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

  for (auto i = 0; i < pde.num_dims; ++i)
  {
    auto const &dim = pde.get_dimensions()[i];
    std::vector<int> ipiv(dim.get_degree() * fm::two_raised_to(pde.max_level));
    for (auto j = 0; j < pde.num_terms; ++j)
    {
      auto const &term_1D       = pde.get_terms()[j][i];
      auto const &partial_terms = term_1D.get_partial_terms();

      for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
      {
        // TODO: refactor these changes, this is slow!
        partial_term<P> const lhs_mass_pterm = partial_term<P>(
            coefficient_type::mass, partial_terms[k].lhs_mass_func, nullptr,
            flux_type::central, boundary_condition::periodic,
            boundary_condition::periodic, homogeneity::homogeneous,
            homogeneity::homogeneous, {}, nullptr, {}, nullptr,
            dim.volume_jacobian_dV);

        auto mass_coeff = generate_coefficients<P>(
            dim, lhs_mass_pterm, transformer, pde.max_level, time, rotate);

        // precompute inv(mass) * coeff for each level up to max level
        std::vector<fk::matrix<P>> pterm_coeffs;

        for (int level = 0; level <= pde.max_level; ++level)
        {
          auto result = generate_coefficients<P>(
              dim, partial_terms[k], transformer, level, time, rotate);
          if (partial_terms[k].dv_func || partial_terms[k].g_func)
          {
            auto const dof = dim.get_degree() * fm::two_raised_to(level);
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
  for (auto i = 0; i < pde.num_dims; ++i)
  {
    auto &dim = pde.get_dimensions()[i];

    for (int level = 0; level <= pde.max_level; ++level)
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
template<typename P>
fk::matrix<P> generate_coefficients(
    dimension<P> const &dim, partial_term<P> const &pterm,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const level, P const time, bool const rotate)
{
  expect(time >= 0.0);
  expect(transformer.degree == dim.get_degree());
  expect(transformer.max_level >= dim.get_level());
  expect(level <= transformer.max_level);

  auto g_dv_func = [g_func  = pterm.g_func,
                    dv_func = pterm.dv_func]() -> g_func_type<P> {
    if (g_func && dv_func)
    {
      return [g_func, dv_func](P const x, P const t) {
        return g_func(x, t) * dv_func(x, t);
      };
    }
    else if (g_func)
    {
      return [g_func](P const x, P const t) { return g_func(x, t); };
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
  auto const degrees_freedom_1d = dim.get_degree() * num_cells;
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

  auto const legendre_poly_L_t = fk::matrix<P>(legendre_poly_L).transpose();
  auto const legendre_poly_R_t = fk::matrix<P>(legendre_poly_R).transpose();

  int const nrows = dim.get_degree();

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

  auto const legendre_poly_t  = fk::matrix<P>(legendre_poly).transpose();
  auto const legendre_prime_t = fk::matrix<P>(legendre_prime).transpose();

  // get jacobian
  auto const jacobi = grid_spacing / 2;

  auto matrix_LtR = (legendre_poly_L_t * legendre_poly_R);
  auto matrix_LtL = (legendre_poly_L_t * legendre_poly_L);
  auto matrix_RtR = (legendre_poly_R_t * legendre_poly_R);
  auto matrix_RtL = (legendre_poly_R_t * legendre_poly_L);

  // get index for first and last element
  int const first   = 0;
  int const last    = dim.get_degree() * (num_cells - 1);

#pragma omp parallel
{
  fk::matrix<P> tmp(legendre_poly.nrows(), legendre_poly.ncols());
  fk::matrix<P> block(dim.get_degree(), dim.get_degree());

  fk::matrix<P> trace_value_1(nrows, nrows);
  fk::matrix<P> trace_value_2(nrows, nrows);
  fk::matrix<P> trace_value_3(nrows, nrows);
  fk::matrix<P> trace_value_4(nrows, nrows);

  auto apply_volume = [&](int i)->void {
    int const current = dim.get_degree() * i;
    int const current_stop = current + dim.get_degree() - 1;

    if (pterm.coeff_type != coefficient_type::penalty) {
      for (int k = 0; k < tmp.nrows(); k++)
      {
        P c = g_dv_func((0.5 * quadrature_points[k] + 0.5 + i) * grid_spacing
                        + dim.domain_min, time);
        c *= quadrature_weights(k) * jacobi;

        for (int j = 0; j < tmp.ncols(); j++)
          tmp(k, j) = c * legendre_poly(k, j);
      }

      if (pterm.coeff_type == coefficient_type::mass)
      {
        fm::gemm(P{1}, legendre_poly_t, tmp, P{0}, block);
      }
      else if (pterm.coeff_type == coefficient_type::grad ||
               pterm.coeff_type == coefficient_type::div)
      {
        fm::gemm(P{-1}, legendre_prime_t, tmp, P{0}, block);
      }

      fk::matrix<P, mem_type::view> curr_block(coefficients, current,
                                               current_stop, current,
                                               current_stop);
      fm::mat_axpy(block, curr_block);
    }
  };

#pragma omp for
  for (auto i = 1; i < num_cells - 1; ++i)
  {
    // get left and right locations for this element
    P const x_left  = dim.domain_min + i * grid_spacing;
    P const x_right = x_left + grid_spacing;

    // get index for current block
    int const current = dim.get_degree() * i;

    apply_volume(i);

    if (pterm.coeff_type == coefficient_type::grad ||
        pterm.coeff_type == coefficient_type::div ||
        pterm.coeff_type == coefficient_type::penalty)
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

      // Penalty term is just <|gfunc|/2[[f]],[[v]]> so we need to remove the
      // central flux <gfunc{{f}},[[v]]> from the operators
      P const central_coeff =
          pterm.coeff_type == coefficient_type::penalty ? 0.0 : 1.0;

      P const flux_left  = g_dv_func(x_left, time);
      P const flux_right = g_dv_func(x_right, time);

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
      // support

      // trace_value_1 is the interaction on x_{i-1/2} --
      // the edge between cell I_{i-1} and I_i or the left boundary of I_i.
      // f is a DG function with support on I_{i-1}
      // In this case:  {{f}} = p_R/2, [[f]] = p_R, [[v]] = -p_L
      P coeff = central_coeff * (-1 * flux_left / 2) + (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      fm::mat_copy(coeff, matrix_LtR, trace_value_1);

      // trace_value_2 is the interaction on x_{i-1/2} --
      // the edge between cell I_{i-1} and I_i or the left boundary of I_i.
      // f is a DG function with support on I_{i}
      // In this case:  {{f}} = p_L/2, [[f]] = -p_L, [[v]] = -p_L
      coeff = central_coeff * (-1 * flux_left / 2) + (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      fm::mat_copy(coeff, matrix_LtL, trace_value_2);

      // trace_value_3 is the interaction on x_{i+1/2} --
      // the edge between cell I_i and I_{i+1} or the right boundary of I_i.
      // f is a DG function with support on I_{i}
      // In this case:  {{f}} = p_R/2, [[f]] = p_R, [[v]] = p_R
      coeff = central_coeff * (+1 * flux_right / 2) + (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      fm::mat_copy(coeff, matrix_RtR, trace_value_3);

      // trace_value_4 is the interaction on x_{i+1/2} --
      // the edge between cell I_i and I_{i+1} or the right boundary of I_i.
      // f is a DG function with support on I_{i+1}
      // In this case:  {{f}} = p_L/2, [[f]] = -p_L, [[v]] = p_R

      coeff = central_coeff * (+1 * flux_right / 2) + (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      fm::mat_copy(coeff, matrix_RtL, trace_value_4);

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

      // If neumann
      // (gradient u)*num_cells = g
      // by splitting grad u = q by LDG methods, the B.C is changed to
      // q*num_cells = g (=> q = g for 1D variable)
      // only work for derivatives greater than 1

      // Neumann boundary conditions
      // For div and grad, the interior trace is used to calculate the flux,
      // similar to an outflow boundary condition. For penalty, nothing is
      // added.

      int row1 = current;
      int col1 = current - dim.get_degree();

      int row2 = current;
      int col2 = current;

      int row3 = current;
      int col3 = current;

      int row4 = current;
      int col4 = current + dim.get_degree();

      // Add trace part 1
      fk::matrix<P, mem_type::view> block1(coefficients, row1,
                                            row1 + dim.get_degree() - 1, col1,
                                            col1 + dim.get_degree() - 1);
      fm::mat_axpy(trace_value_1, block1);

      // Add trace part 2
      fk::matrix<P, mem_type::view> block2(coefficients, row2,
                                           row2 + dim.get_degree() - 1, col2,
                                           col2 + dim.get_degree() - 1);
      fm::mat_axpy(trace_value_2, block2);

      // Add trace part 3
      fk::matrix<P, mem_type::view> block3(coefficients, row3,
                                           row3 + dim.get_degree() - 1, col3,
                                           col3 + dim.get_degree() - 1);
      fm::mat_axpy(trace_value_3, block3);

      // Add trace part 4
      fk::matrix<P, mem_type::view> block4(coefficients, row4,
                                            row4 + dim.get_degree() - 1, col4,
                                            col4 + dim.get_degree() - 1);
      fm::mat_axpy(trace_value_4, block4);
    }
  } // for i

#pragma omp single
{
  int const i = 0;

  // get left and right locations for this element
  P const x_left  = dim.domain_min + i * grid_spacing;
  P const x_right = x_left + grid_spacing;

  // get index for current block
  int const current = dim.get_degree() * i;

  apply_volume(i);

  if (pterm.coeff_type == coefficient_type::grad ||
      pterm.coeff_type == coefficient_type::div ||
      pterm.coeff_type == coefficient_type::penalty)
  {
    P const central_coeff =
        pterm.coeff_type == coefficient_type::penalty ? 0.0 : 1.0;

    if (num_cells == 1) { // i == 0 and i == num_cells - 1
      P const flux_left  = g_dv_func(x_left, time);
      P const flux_right = g_dv_func(x_right, time);

      P coeff = central_coeff * (-1 * flux_left / 2) + (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      fm::mat_copy(coeff, matrix_LtR, trace_value_1);

      coeff = central_coeff * (-1 * flux_left / 2) + (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      fm::mat_copy(coeff, matrix_LtL, trace_value_2);

      coeff = central_coeff * (+1 * flux_right / 2) + (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      fm::mat_copy(coeff, matrix_RtR, trace_value_3);

      coeff = central_coeff * (+1 * flux_right / 2) + (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      fm::mat_copy(coeff, matrix_RtL, trace_value_4);

      // If dirichelt
      // u^-_LEFT = g(LEFT)
      // u^+_RIGHT = g(RIGHT)
      boundary_condition const left  = pterm.ileft;
      boundary_condition const right = pterm.iright;

      // Dirichlet Boundary Conditions
      // For div and grad, the boundary is not part of the bilinear operator,
      // but instead tranferred to the source.  Similar to an inflow condition.
      // For penalty, the operator <|gfunc|/2*f,v> is applied for the case where
      // f and v share the same volume support

      // If statement checking coeff_type is because gfunc can evaluate to nan
      // in 1/0 case.  Ex: gfunc = x, domain = [0,4] (possible in spherical
      // coordinates)

      if (left == boundary_condition::dirichlet) // left dirichlet
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_2 = (legendre_poly_L_t * legendre_poly_L) *
                            (-1.0 * pterm.get_flux_scale() *
                              std::abs(flux_left) / 2.0 * -1.0);
          }
          else
          {
            trace_value_2 =
                (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) *
                (-1);
          }
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (right == boundary_condition::dirichlet) // right dirichlet
      {
        if (i == num_cells - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_3 =
                (legendre_poly_R_t * legendre_poly_R) *
                (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          }
          else
          {
            trace_value_3 =
                (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) *
                (+1);
          }
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      if (left == boundary_condition::neumann) // left neumann
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_2 =
                (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) *
                (-1);
          }
          else
          {
            trace_value_2 =
                (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left);
          }
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (right == boundary_condition::neumann) // right neumann
      {
        if (i == num_cells - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_3 =
                (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) *
                (+1);
          }
          else
          {
            trace_value_3 =
                (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right);
          }
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      // Add trace values to matrix

      int row1 = current;
      int col1 = current - dim.get_degree();

      int row2 = current;
      int col2 = current;

      int row3 = current;
      int col3 = current;

      int row4 = current;
      int col4 = current + dim.get_degree();

      if (left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        if (i == 0)
        {
          row1 = current;
          col1 = last;
        }
        if (i == num_cells - 1)
        {
          row4 = current;
          col4 = first;
        }
      }

      if (i != 0 || left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        // Add trace part 1
        fk::matrix<P, mem_type::view> block1(coefficients, row1,
                                              row1 + dim.get_degree() - 1, col1,
                                              col1 + dim.get_degree() - 1);
        //block1 = block1 + trace_value_1;
        for(int j=0; j<trace_value_1.ncols(); j++)
          for(int k=0; k<trace_value_1.nrows(); k++)
            block1(k, j) += trace_value_1(k, j);
      }

      // Add trace part 2
      fk::matrix<P, mem_type::view> block2(coefficients, row2,
                                            row2 + dim.get_degree() - 1, col2,
                                            col2 + dim.get_degree() - 1);
      //block2 = block2 + trace_value_2;
      for(int j=0; j<trace_value_2.ncols(); j++)
        for(int k=0; k<trace_value_2.nrows(); k++)
          block2(k, j) += trace_value_2(k, j);

      // Add trace part 3
      fk::matrix<P, mem_type::view> block3(coefficients, row3,
                                            row3 + dim.get_degree() - 1, col3,
                                            col3 + dim.get_degree() - 1);
      //block3 = block3 + trace_value_3;
      for(int j=0; j<trace_value_3.ncols(); j++)
        for(int k=0; k<trace_value_3.nrows(); k++)
          block3(k, j) += trace_value_3(k, j);

      if (i != num_cells - 1 || left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        // Add trace part 4
        fk::matrix<P, mem_type::view> block4(coefficients, row4,
                                              row4 + dim.get_degree() - 1, col4,
                                              col4 + dim.get_degree() - 1);
        //block4 = block4 + trace_value_4;
        for(int j=0; j<trace_value_4.ncols(); j++)
          for(int k=0; k<trace_value_4.nrows(); k++)
            block4(k, j) += trace_value_4(k, j);
      }
    } else {
      P const flux_left  = g_dv_func(x_left, time);
      P const flux_right = g_dv_func(x_right, time);

      P coeff = central_coeff * (-1 * flux_left / 2) + (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      fm::mat_copy(coeff, matrix_LtR, trace_value_1);

      coeff = central_coeff * (-1 * flux_left / 2) + (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      fm::mat_copy(coeff, matrix_LtL, trace_value_2);

      coeff = central_coeff * (+1 * flux_right / 2) + (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      fm::mat_copy(coeff, matrix_RtR, trace_value_3);

      coeff = central_coeff * (+1 * flux_right / 2) + (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      fm::mat_copy(coeff, matrix_RtL, trace_value_4);

      // If dirichelt
      // u^-_LEFT = g(LEFT)
      // u^+_RIGHT = g(RIGHT)
      boundary_condition const left  = pterm.ileft;
      boundary_condition const right = pterm.iright;

      // Dirichlet Boundary Conditions
      // For div and grad, the boundary is not part of the bilinear operator,
      // but instead tranferred to the source.  Similar to an inflow condition.
      // For penalty, the operator <|gfunc|/2*f,v> is applied for the case where
      // f and v share the same volume support

      // If statement checking coeff_type is because gfunc can evaluate to nan
      // in 1/0 case.  Ex: gfunc = x, domain = [0,4] (possible in spherical
      // coordinates)

      if (left == boundary_condition::dirichlet) // left dirichlet
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_2 = (legendre_poly_L_t * legendre_poly_L) *
                            (-1.0 * pterm.get_flux_scale() *
                              std::abs(flux_left) / 2.0 * -1.0);
          }
          else
          {
            trace_value_2 =
                (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) *
                (-1);
          }
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (right == boundary_condition::dirichlet) // right dirichlet
      {
        if (i == num_cells - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_3 =
                (legendre_poly_R_t * legendre_poly_R) *
                (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          }
          else
          {
            trace_value_3 =
                (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) *
                (+1);
          }
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      if (left == boundary_condition::neumann) // left neumann
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_2 =
                (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) *
                (-1);
          }
          else
          {
            trace_value_2 =
                (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left);
          }
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (right == boundary_condition::neumann) // right neumann
      {
        if (i == num_cells - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_3 =
                (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) *
                (+1);
          }
          else
          {
            trace_value_3 =
                (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right);
          }
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      // Add trace values to matrix

      int row1 = current;
      int col1 = current - dim.get_degree();

      int row2 = current;
      int col2 = current;

      int row3 = current;
      int col3 = current;

      int row4 = current;
      int col4 = current + dim.get_degree();

      if (left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        if (i == 0)
        {
          row1 = current;
          col1 = last;
        }
        if (i == num_cells - 1)
        {
          row4 = current;
          col4 = first;
        }
      }

      if (i != 0 || left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        // Add trace part 1
        fk::matrix<P, mem_type::view> block1(coefficients, row1,
                                              row1 + dim.get_degree() - 1, col1,
                                              col1 + dim.get_degree() - 1);
        //block1 = block1 + trace_value_1;
        for(int j=0; j<trace_value_1.ncols(); j++)
          for(int k=0; k<trace_value_1.nrows(); k++)
            block1(k, j) += trace_value_1(k, j);
      }

      // Add trace part 2
      fk::matrix<P, mem_type::view> block2(coefficients, row2,
                                            row2 + dim.get_degree() - 1, col2,
                                            col2 + dim.get_degree() - 1);
      //block2 = block2 + trace_value_2;
      for(int j=0; j<trace_value_2.ncols(); j++)
        for(int k=0; k<trace_value_2.nrows(); k++)
          block2(k, j) += trace_value_2(k, j);

      // Add trace part 3
      fk::matrix<P, mem_type::view> block3(coefficients, row3,
                                            row3 + dim.get_degree() - 1, col3,
                                            col3 + dim.get_degree() - 1);
      //block3 = block3 + trace_value_3;
      for(int j=0; j<trace_value_3.ncols(); j++)
        for(int k=0; k<trace_value_3.nrows(); k++)
          block3(k, j) += trace_value_3(k, j);

      if (i != num_cells - 1 || left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        // Add trace part 4
        fk::matrix<P, mem_type::view> block4(coefficients, row4,
                                              row4 + dim.get_degree() - 1, col4,
                                              col4 + dim.get_degree() - 1);
        //block4 = block4 + trace_value_4;
        for(int j=0; j<trace_value_4.ncols(); j++)
          for(int k=0; k<trace_value_4.nrows(); k++)
            block4(k, j) += trace_value_4(k, j);
      }
    }
  }
} // #pragma omp single


#pragma omp single
{
  if (num_cells > 1) {
    int const i = num_cells - 1;

    // get left and right locations for this element
    P const x_left  = dim.domain_min + i * grid_spacing;
    P const x_right = x_left + grid_spacing;

    // get index for current block
    int const current = dim.get_degree() * i;

    apply_volume(i);

    if (pterm.coeff_type == coefficient_type::grad ||
        pterm.coeff_type == coefficient_type::div ||
        pterm.coeff_type == coefficient_type::penalty)
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

      // Penalty term is just <|gfunc|/2[[f]],[[v]]> so we need to remove the
      // central flux <gfunc{{f}},[[v]]> from the operators
      P const central_coeff =
          pterm.coeff_type == coefficient_type::penalty ? 0.0 : 1.0;

      P const flux_left  = g_dv_func(x_left, time);
      P const flux_right = g_dv_func(x_right, time);

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
      // support

      // trace_value_1 is the interaction on x_{i-1/2} --
      // the edge between cell I_{i-1} and I_i or the left boundary of I_i.
      // f is a DG function with support on I_{i-1}
      // In this case:  {{f}} = p_R/2, [[f]] = p_R, [[v]] = -p_L
  //       trace_value_1 =
  //           (legendre_poly_L_t * legendre_poly_R) * central_coeff *
  //               (-1 * flux_left / 2) +
  //           (legendre_poly_L_t * legendre_poly_R) *
  //               (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);

      P coeff = central_coeff * (-1 * flux_left / 2) + (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      //fm::mat_axpy(coeff, matrix_LtR, trace_value_1);
      for(int j=0; j<nrows; j++)
        for(int k=0; k<nrows; k++)
          trace_value_1(k, j) = coeff * matrix_LtR(k, j);

      // trace_value_2 is the interaction on x_{i-1/2} --
      // the edge between cell I_{i-1} and I_i or the left boundary of I_i.
      // f is a DG function with support on I_{i}
      // In this case:  {{f}} = p_L/2, [[f]] = -p_L, [[v]] = -p_L
  //       auto trace_value_2 =
  //           (legendre_poly_L_t * legendre_poly_L) * central_coeff *
  //               (-1 * flux_left / 2) +
  //           (legendre_poly_L_t * legendre_poly_L) *
  //               (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);

      coeff = central_coeff * (-1 * flux_left / 2) + (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      for(int j=0; j<nrows; j++)
        for(int k=0; k<nrows; k++)
          trace_value_2(k, j) = coeff * matrix_LtL(k, j);

      // trace_value_3 is the interaction on x_{i+1/2} --
      // the edge between cell I_i and I_{i+1} or the right boundary of I_i.
      // f is a DG function with support on I_{i}
      // In this case:  {{f}} = p_R/2, [[f]] = p_R, [[v]] = p_R
  //       auto trace_value_3 =
  //           (legendre_poly_R_t * legendre_poly_R) * central_coeff *
  //               (+1 * flux_right / 2) +
  //           (legendre_poly_R_t * legendre_poly_R) *
  //               (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);

      coeff = central_coeff * (+1 * flux_right / 2) + (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      for(int j=0; j<nrows; j++)
        for(int k=0; k<nrows; k++)
          trace_value_3(k, j) = coeff * matrix_RtR(k, j);

      // trace_value_4 is the interaction on x_{i+1/2} --
      // the edge between cell I_i and I_{i+1} or the right boundary of I_i.
      // f is a DG function with support on I_{i+1}
      // In this case:  {{f}} = p_L/2, [[f]] = -p_L, [[v]] = p_R
  //       auto trace_value_4 =
  //           (legendre_poly_R_t * legendre_poly_L) * central_coeff *
  //               (+1 * flux_right / 2) +
  //           (legendre_poly_R_t * legendre_poly_L) *
  //               (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);

      coeff = central_coeff * (+1 * flux_right / 2) + (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      for(int j=0; j<nrows; j++)
        for(int k=0; k<nrows; k++)
          trace_value_4(k, j) = coeff * matrix_RtL(k, j);

      // If dirichelt
      // u^-_LEFT = g(LEFT)
      // u^+_RIGHT = g(RIGHT)
      boundary_condition const left  = pterm.ileft;
      boundary_condition const right = pterm.iright;

      // Dirichlet Boundary Conditions
      // For div and grad, the boundary is not part of the bilinear operator,
      // but instead tranferred to the source.  Similar to an inflow condition.
      // For penalty, the operator <|gfunc|/2*f,v> is applied for the case where
      // f and v share the same volume support

      // If statement checking coeff_type is because gfunc can evaluate to nan
      // in 1/0 case.  Ex: gfunc = x, domain = [0,4] (possible in spherical
      // coordinates)

      if (left == boundary_condition::dirichlet) // left dirichlet
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_2 = (legendre_poly_L_t * legendre_poly_L) *
                            (-1.0 * pterm.get_flux_scale() *
                              std::abs(flux_left) / 2.0 * -1.0);
          }
          else
          {
            trace_value_2 =
                (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) *
                (-1);
          }
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (right == boundary_condition::dirichlet) // right dirichlet
      {
        if (i == num_cells - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_3 =
                (legendre_poly_R_t * legendre_poly_R) *
                (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          }
          else
          {
            trace_value_3 =
                (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) *
                (+1);
          }
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      // If neumann
      // (gradient u)*num_cells = g
      // by splitting grad u = q by LDG methods, the B.C is changed to
      // q*num_cells = g (=> q = g for 1D variable)
      // only work for derivatives greater than 1

      // Neumann boundary conditions
      // For div and grad, the interior trace is used to calculate the flux,
      // similar to an outflow boundary condition. For penalty, nothing is
      // added.

      if (left == boundary_condition::neumann) // left neumann
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_2 =
                (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) *
                (-1);
          }
          else
          {
            trace_value_2 =
                (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left);
          }
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) *
                  central_coeff +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (right == boundary_condition::neumann) // right neumann
      {
        if (i == num_cells - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) *
                  central_coeff +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          if (pterm.coeff_type == coefficient_type::penalty)
          {
            trace_value_3 =
                (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) *
                (+1);
          }
          else
          {
            trace_value_3 =
                (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right);
          }
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      // Add trace values to matrix

      int row1 = current;
      int col1 = current - dim.get_degree();

      int row2 = current;
      int col2 = current;

      int row3 = current;
      int col3 = current;

      int row4 = current;
      int col4 = current + dim.get_degree();

      if (left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        if (i == 0)
        {
          row1 = current;
          col1 = last;
        }
        if (i == num_cells - 1)
        {
          row4 = current;
          col4 = first;
        }
      }

      if (i != 0 || left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        // Add trace part 1
        fk::matrix<P, mem_type::view> block1(coefficients, row1,
                                              row1 + dim.get_degree() - 1, col1,
                                              col1 + dim.get_degree() - 1);
        //block1 = block1 + trace_value_1;
        for(int j=0; j<trace_value_1.ncols(); j++)
          for(int k=0; k<trace_value_1.nrows(); k++)
            block1(k, j) += trace_value_1(k, j);
      }

      // Add trace part 2
      fk::matrix<P, mem_type::view> block2(coefficients, row2,
                                            row2 + dim.get_degree() - 1, col2,
                                            col2 + dim.get_degree() - 1);
      //block2 = block2 + trace_value_2;
      for(int j=0; j<trace_value_2.ncols(); j++)
        for(int k=0; k<trace_value_2.nrows(); k++)
          block2(k, j) += trace_value_2(k, j);

      // Add trace part 3
      fk::matrix<P, mem_type::view> block3(coefficients, row3,
                                            row3 + dim.get_degree() - 1, col3,
                                            col3 + dim.get_degree() - 1);
      //block3 = block3 + trace_value_3;
      for(int j=0; j<trace_value_3.ncols(); j++)
        for(int k=0; k<trace_value_3.nrows(); k++)
          block3(k, j) += trace_value_3(k, j);

      if (i != num_cells - 1 || left == boundary_condition::periodic ||
          right == boundary_condition::periodic)
      {
        // Add trace part 4
        fk::matrix<P, mem_type::view> block4(coefficients, row4,
                                              row4 + dim.get_degree() - 1, col4,
                                              col4 + dim.get_degree() - 1);
        //block4 = block4 + trace_value_4;
        for(int j=0; j<trace_value_4.ncols(); j++)
          for(int k=0; k<trace_value_4.nrows(); k++)
            block4(k, j) += trace_value_4(k, j);
      }
    }
  }
} // #pragma omp single

} // #pragma omp parallel

  if (pterm.coeff_type == coefficient_type::grad)
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
