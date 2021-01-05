#include "coefficients.hpp"

#include "basis.hpp"
#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "tools.hpp"
#include "transformations.hpp"
#include <numeric>

// generate coefficient matrices for each 1D term in each dimension and
// underlying partial term coefficients matrices
template<typename P>
void generate_all_coefficients(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    P const time, bool const rotate)
{
  tools::expect(time >= 0.0);

  for (auto i = 0; i < pde.num_dims; ++i)
  {
    auto const &dim = pde.get_dimensions()[i];

    for (auto j = 0; j < pde.num_terms; ++j)
    {
      auto const &term_1D       = pde.get_terms()[j][i];
      auto const &partial_terms = term_1D.get_partial_terms();

      for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
      {
        auto const partial_term_coeff = generate_coefficients<P>(
            dim, term_1D, partial_terms[k], transformer, time, rotate);
        pde.set_partial_coefficients(j, i, k, partial_term_coeff);
      }
    }
    pde.rechain_dimension(i);
  }
}

// construct 1D coefficient matrix - new conventions
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matricies
template<typename P>
fk::matrix<P> generate_coefficients(
    dimension<P> const &dim, term<P> const &term_1D,
    partial_term<P> const &pterm,
    basis::wavelet_transform<P, resource::host> const &transformer,
    P const time, bool const rotate)
{
  tools::expect(time >= 0.0);
  tools::expect(transformer.degree == dim.get_degree());
  tools::expect(transformer.max_level >= dim.get_level());

  // setup jacobi of variable x and define coeff_mat
  auto const num_points = fm::two_raised_to(transformer.max_level);

  auto const grid_spacing = (dim.domain_max - dim.domain_min) / num_points;
  auto const degrees_freedom_1d = dim.get_degree() * num_points;
  fk::matrix<P> coefficients(degrees_freedom_1d, degrees_freedom_1d);

  // get quadrature points and quadrature_weights.
  // we do the two-step store because we cannot have 'static' bindings
  static auto const legendre_values =
      legendre_weights<P>(dim.get_degree(), -1.0, 1.0);
  auto const [quadrature_points, quadrature_weights] = legendre_values;
  auto const [legendre_poly_L, legendre_poly_R]      = [&]() {
    auto [lP_L, lPP_L] = legendre(fk::vector<P>{-1}, dim.get_degree());
    lP_L               = lP_L * (1 / std::sqrt(grid_spacing));
    auto [lP_R, lPP_R] = legendre(fk::vector<P>{+1}, dim.get_degree());
    lP_R               = lP_R * (1 / std::sqrt(grid_spacing));
    // this is to get around unused warnings (until c++20)
    auto const ignore = [](auto ignored) { (void)ignored; };
    ignore(lPP_L);
    ignore(lPP_R);
    return std::array<fk::matrix<P>, 2>{lP_L, lP_R};
  }();

  auto const legendre_poly_L_t = fk::matrix<P>(legendre_poly_L).transpose();
  auto const legendre_poly_R_t = fk::matrix<P>(legendre_poly_R).transpose();

  // get the basis functions and derivatives for all k
  // this auto is std::array<fk::matrix<P>, 2>
  auto const [legendre_poly,
              legendre_prime] = [&, quadrature_points = quadrature_points]() {
    auto [lP, lPP] = legendre(quadrature_points, dim.get_degree());

    lP  = lP * (1.0 / std::sqrt(grid_spacing));
    lPP = lPP * (1.0 / std::sqrt(grid_spacing) * 2.0 / grid_spacing);

    return std::array<fk::matrix<P>, 2>{lP, lPP};
  }();

  auto const legendre_poly_t  = fk::matrix<P>(legendre_poly).transpose();
  auto const legendre_prime_t = fk::matrix<P>(legendre_prime).transpose();

  // get jacobian
  auto const jacobi = grid_spacing / 2;

  // convert term input data from wavelet space to realspace
  // FIXME during PDE rework, fix term's RAII issues...
  auto const &term_data = term_1D.get_data();
  fk::vector<P, mem_type::const_view> const data(term_data, 0,
                                                 degrees_freedom_1d - 1);

  auto const data_real = transformer.apply(
      data, transformer.max_level, basis::side::left, basis::transpose::trans);

  for (auto i = 0; i < num_points; ++i)
  {
    // get left and right locations for this element
    auto const x_left  = dim.domain_min + i * grid_spacing;
    auto const x_right = x_left + grid_spacing;

    // get index for current, first and last element
    auto const current = dim.get_degree() * i;
    auto const first   = 0;
    auto const last    = dim.get_degree() * (num_points - 1);

    // map quadrature points from [-1,1] to physical domain of this i element
    fk::vector<P> const quadrature_points_i = [&, quadrature_points =
                                                      quadrature_points]() {
      fk::vector<P> quadrature_points_copy(quadrature_points);
      std::transform(
          quadrature_points_copy.begin(), quadrature_points_copy.end(),
          quadrature_points_copy.begin(), [&](P const elem) {
            return ((elem + 1) / 2 + i) * grid_spacing + dim.domain_min;
          });
      return quadrature_points_copy;
    }();

    fk::vector<P> const g_func = [&, legendre_poly = legendre_poly]() {
      // get realspace data at quadrature points
      // NOTE : this is unused pending updating G functions to accept "dat"
      fk::vector<P> data_real_quad =
          legendre_poly *
          fk::vector<P, mem_type::const_view>(data_real, current,
                                              current + dim.get_degree() - 1);
      // get g(x,t,dat)
      // FIXME : add dat as a argument to the G functions
      fk::vector<P> g(quadrature_points_i.size());
      for (auto i = 0; i < quadrature_points_i.size(); ++i)
      {
        g(i) = pterm.g_func(quadrature_points_i(i), time);
      }
      return g;
    }();

    auto const block = [&, legendre_poly = legendre_poly,
                        quadrature_weights = quadrature_weights]() {
      fk::matrix<P> tmp(legendre_poly.nrows(), legendre_poly.ncols());

      for (int i = 0; i <= tmp.nrows() - 1; i++)
      {
        for (int j = 0; j <= tmp.ncols() - 1; j++)
        {
          tmp(i, j) =
              g_func(i) * legendre_poly(i, j) * quadrature_weights(i) * jacobi;
        }
      }
      fk::matrix<P> block(dim.get_degree(), dim.get_degree());

      if (pterm.coeff_type == coefficient_type::mass)
      {
        block = legendre_poly_t * tmp;
      }
      else if (pterm.coeff_type == coefficient_type::grad)
      {
        block = legendre_prime_t * tmp * (-1);
      }
      return block;
    }();

    // set the block at the correct position
    fk::matrix<P> const curr_block =
        fk::matrix<P, mem_type::view>(coefficients, current,
                                      current + dim.get_degree() - 1, current,
                                      current + dim.get_degree() - 1) +
        block;
    coefficients.set_submatrix(current, current, curr_block);

    if (pterm.coeff_type == coefficient_type::grad)
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

      auto const flux_left  = pterm.g_func(x_left, time);
      auto const flux_right = pterm.g_func(x_right, time);

      // get the "trace" values
      // (values at the left and right of each element for all k)
      auto trace_value_1 =
          (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) +
          (legendre_poly_L_t * legendre_poly_R) *
              (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      auto trace_value_2 =
          (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) +
          (legendre_poly_L_t * legendre_poly_L) *
              (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      auto trace_value_3 =
          (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) +
          (legendre_poly_R_t * legendre_poly_R) *
              (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      auto trace_value_4 =
          (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) +
          (legendre_poly_R_t * legendre_poly_L) *
              (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);

      // If dirichelt
      // u^-_LEFT = g(LEFT)
      // u^+_RIGHT = g(RIGHT)

      if (pterm.left == boundary_condition::dirichlet) // left dirichlet
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          trace_value_2 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (pterm.right == boundary_condition::dirichlet) // right dirichlet
      {
        if (i == num_points - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_3 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      // If neumann
      // (gradient u)*num_points = g
      // by splitting grad u = q by LDG methods, the B.C is changed to
      // q*num_points = g (=> q = g for 1D variable)
      // only work for derivatives greater than 1

      if (pterm.left == boundary_condition::neumann) // left neumann
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left);
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (pterm.right == boundary_condition::neumann) // right neumann
      {
        if (i == num_points - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right);
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      if (pterm.coeff_type == coefficient_type::grad)
      {
        // Add trace values to matrix

        int row1 = current;
        int col1 = current - dim.get_degree();

        int row2 = current;
        int col2 = current;

        int row3 = current;
        int col3 = current;

        int row4 = current;
        int col4 = current + dim.get_degree();

        if (pterm.left == boundary_condition::periodic ||
            pterm.right == boundary_condition::periodic)
        {
          if (i == 0)
          {
            row1 = current;
            col1 = last;
          }
          if (i == num_points - 1)
          {
            row4 = current;
            col4 = first;
          }
        }

        if (i != 0 || pterm.left == boundary_condition::periodic ||
            pterm.right == boundary_condition::periodic)
        {
          // Add trace part 1
          fk::matrix<P, mem_type::view> block1(
              coefficients, row1, row1 + dim.get_degree() - 1, col1,
              col1 + dim.get_degree() - 1);
          block1 = block1 + trace_value_1;
        }

        // Add trace part 2
        fk::matrix<P, mem_type::view> block2(coefficients, row2,
                                             row2 + dim.get_degree() - 1, col2,
                                             col2 + dim.get_degree() - 1);
        block2 = block2 + trace_value_2;

        // Add trace part 3
        fk::matrix<P, mem_type::view> block3(coefficients, row3,
                                             row3 + dim.get_degree() - 1, col3,
                                             col3 + dim.get_degree() - 1);
        block3 = block3 + trace_value_3;
        if (i != num_points - 1 || pterm.left == boundary_condition::periodic ||
            pterm.right == boundary_condition::periodic)
        {
          // Add trace part 4
          fk::matrix<P, mem_type::view> block4(
              coefficients, row4, row4 + dim.get_degree() - 1, col4,
              col4 + dim.get_degree() - 1);
          block4 = block4 + trace_value_4;
        }
      }
    }
  }

  if (rotate)
  {
    // transform matrix to wavelet space

    // These routines do the following operation:
    // coefficients = forward_trans * coefficients * forward_trans_transpose;
    coefficients = transformer.apply(
        transformer.apply(coefficients, transformer.max_level,
                          basis::side::right, basis::transpose::trans),
        transformer.max_level, basis::side::left, basis::transpose::no_trans);
  }

  return coefficients;
}

template fk::matrix<float> generate_coefficients<float>(
    dimension<float> const &dim, term<float> const &term_1D,
    partial_term<float> const &pterm,
    basis::wavelet_transform<float, resource::host> const &transformer,
    float const time, bool const rotate);

template fk::matrix<double> generate_coefficients<double>(
    dimension<double> const &dim, term<double> const &term_1D,
    partial_term<double> const &pterm,
    basis::wavelet_transform<double, resource::host> const &transformer,
    double const time, bool const rotate);

template void generate_all_coefficients<float>(
    PDE<float> &pde,
    basis::wavelet_transform<float, resource::host> const &transformer,
    float const time, bool const rotate);

template void generate_all_coefficients<double>(
    PDE<double> &pde,
    basis::wavelet_transform<double, resource::host> const &transformer,
    double const time, bool const rotate);
