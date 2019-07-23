#include "coefficients.hpp"

#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "transformations.hpp"
#include <numeric>

template<typename P>
void print(std::vector<P> const &input)
{
  for (auto it = input.cbegin(); it != input.cend(); it++)
  {
    std::cout << *it << ' ';
  }
}

// wrap generate_coefficients to allow for construction of the diffusion type
// operators
template<typename P>
fk::matrix<double>
generate_coefficients(dimension<P> const &dim, term<P> const term_1D,
                      double const time, bool const rotate)
{
  if (term_1D.coeff_type == coefficient_type::diff)
  {
    // LDG method for diff operator
    // Breaks second order operator to two first order operators with alternting
    // directions on the upwinding

    // Equation 1 of LDG

    dimension<P> dim_1(term_1D.BCL_1, term_1D.BCR_1, dim.domain_min,
                       dim.domain_max, dim.get_level(), dim.get_degree(),
                       dim.initial_condition, dim.name);

    term<P> term_1D_1(coefficient_type::grad, term_1D.g_func_1,
                      term_1D.time_dependent, term_1D.flux_1,
                      term_1D.get_data(), term_1D.name, dim_1);

    auto coefficients_1 =
        generate_mass_or_grad_coefficients(dim_1, term_1D_1, time, rotate);

    // Equation 2 of LDG

    dimension<P> dim_2(term_1D.BCL_2, term_1D.BCR_2, dim.domain_min,
                       dim.domain_max, dim.get_level(), dim.get_degree(),
                       dim.initial_condition, dim.name);

    term<P> term_1D_2(coefficient_type::grad, term_1D.g_func_2,
                      term_1D.time_dependent, term_1D.flux_2,
                      term_1D.get_data(), term_1D.name, dim_2);

    auto coefficients_2 =
        generate_mass_or_grad_coefficients(dim_2, term_1D_2, time, rotate);

    return coefficients_1 * coefficients_2;
  }
  else
  {
    auto coefficients =
        generate_mass_or_grad_coefficients(dim, term_1D, time, rotate);

    return coefficients;
  }
}

// construct 1D coefficient matrix - new conventions
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matricies
template<typename P>
fk::matrix<double>
generate_mass_or_grad_coefficients(dimension<P> const &dim,
                                   term<P> const term_1D, double const time,
                                   bool const rotate)
{
  assert(time >= 0.0);
  // setup jacobi of variable x and define coeff_mat
  int const num_points = fm::two_raised_to(dim.get_level());
  // note that grid_spacing is the symbol typically reserved for grid spacing
  double const grid_spacing    = (dim.domain_max - dim.domain_min) / num_points;
  int const degrees_freedom_1d = dim.get_degree() * num_points;
  fk::matrix<double> coefficients(degrees_freedom_1d, degrees_freedom_1d);

  // set number of quatrature points
  // FIXME should this be order dependent?
  // FIXME is this a global quantity??
  int const quad_num = 10;

  // get quadrature points and quadrature_weights.
  // we do the two-step store because we cannot have 'static' bindings
  static const auto legendre_values =
      legendre_weights<double>(quad_num, -1.0, 1.0);
  auto const [quadrature_points, quadrature_weights] = legendre_values;

  auto const [legendre_poly_L, legendre_poly_R] = [&]() {
    auto [lP_L, lPP_L] = legendre(fk::vector<double>{-1}, dim.get_degree());
    lP_L               = lP_L * (1 / std::sqrt(grid_spacing));
    auto [lP_R, lPP_R] = legendre(fk::vector<double>{+1}, dim.get_degree());
    lP_R               = lP_R * (1 / std::sqrt(grid_spacing));
    // this is to get around unused warnings (until c++20)
    auto const ignore = [](auto ignored) { (void)ignored; };
    ignore(lPP_L);
    ignore(lPP_R);
    return std::array<fk::matrix<double>, 2>{lP_L, lP_R};
  }();

  auto const legendre_poly_L_t =
      fk::matrix<double>(legendre_poly_L).transpose();
  auto const legendre_poly_R_t =
      fk::matrix<double>(legendre_poly_R).transpose();

  // get the basis functions and derivatives for all k
  // this auto is std::array<fk::matrix<P>, 2>
  auto const [legendre_poly,
              legendre_prime] = [&, quadrature_points = quadrature_points]() {
    auto [lP, lPP] = legendre(quadrature_points, dim.get_degree());

    lP  = lP * (1.0 / std::sqrt(grid_spacing));
    lPP = lPP * (1.0 / std::sqrt(grid_spacing) * 2.0 / grid_spacing);

    return std::array<fk::matrix<double>, 2>{lP, lPP};
  }();

  auto const legendre_poly_t  = fk::matrix<double>(legendre_poly).transpose();
  auto const legendre_prime_t = fk::matrix<double>(legendre_prime).transpose();

  // get jacobian
  auto jacobi = grid_spacing / 2;

  // convert term input data from wavelet space to realspace
  fk::matrix<double> const forward_trans_transpose =
      dim.get_from_basis_operator();
  fk::vector<double> const data = fk::vector<double>(term_1D.get_data());
  fk::vector<double> const data_real =
      forward_trans_transpose * fk::vector<double>(term_1D.get_data());

  for (int i = 0; i < num_points; ++i)
  {
    // get left and right locations for this element
    auto const x_left  = dim.domain_min + i * grid_spacing;
    auto const x_right = x_left + grid_spacing;

    // get index for current, firs and last element
    int const current = dim.get_degree() * i;
    int const first   = 0;
    int const last    = dim.get_degree() * (num_points - 1);

    // map quadrature points from [-1,1] to physical domain of this i element
    fk::vector<double> const quadrature_points_i =
        [&, quadrature_points = quadrature_points]() {
          fk::vector<double> quadrature_points_copy = quadrature_points;
          std::transform(
              quadrature_points_copy.begin(), quadrature_points_copy.end(),
              quadrature_points_copy.begin(), [&](double const elem) {
                return ((elem + 1) / 2 + i) * grid_spacing + dim.domain_min;
              });
          return quadrature_points_copy;
        }();

    fk::vector<double> const g_func = [&, legendre_poly = legendre_poly]() {
      // get realspace data at quadrature points
      // NOTE : this is unused pending updating G functions to accept "dat"
      fk::vector<double> data_real_quad =
          legendre_poly *
          fk::vector<double, mem_type::view>(data_real, current,
                                             current + dim.get_degree() - 1);
      // get g(x,t,dat)
      // FIXME : add dat as a argument to the G functions
      fk::vector<double> g(quadrature_points_i.size());
      for (int i = 0; i < quadrature_points_i.size(); ++i)
      {
        g(i) = term_1D.g_func(quadrature_points_i(i), time);
      }
      return g;
    }();

    auto const block = [&, legendre_poly = legendre_poly,
                        quadrature_weights = quadrature_weights]() {
      fk::matrix<double> tmp(legendre_poly.nrows(), legendre_poly.ncols());

      for (int i = 0; i <= tmp.nrows() - 1; i++)
      {
        for (int j = 0; j <= tmp.ncols() - 1; j++)
        {
          tmp(i, j) =
              g_func(i) * legendre_poly(i, j) * quadrature_weights(i) * jacobi;
        }
      }
      fk::matrix<double> block(dim.get_degree(), dim.get_degree());

      if (term_1D.coeff_type == coefficient_type::mass)
      {
        block = legendre_poly_t * tmp;
      }
      else if (term_1D.coeff_type == coefficient_type::grad)
      {
        block = legendre_prime_t * tmp * (-1);
      }
    }();

    // set the block at the correct position
    fk::matrix<double> const curr_block =
        fk::matrix<double, mem_type::view>(
            coefficients, current, current + dim.get_degree() - 1, current,
            current + dim.get_degree() - 1) +
        block;
    coefficients.set_submatrix(current, current, curr_block);

    // coefficients.print("Grad - pre BC");

    // setup numerical flux choice/boundary conditions
    //
    // - <funcCoef*{q},p>
    //----------------------------------------------
    // Numerical Flux is defined as
    // Flux = {{f}} + C/2*[[u]]
    //      = ( f_L + f_R )/2 + FunCoef*( u_R - u_L )/2
    // [[v]] = v_R - v_L

    // FIXME G functions should accept G(x,p,t,dat), since we don't know how the
    // dat is going to be used in the G function (above it is used as linear
    // multuplication but this is not always true)

    auto const FCL = term_1D.g_func(x_left, time);
    auto const FCR = term_1D.g_func(x_right, time);

    // get the "trace" values
    // (values at the left and right of each element for all k)
    auto trace_value_1 =
        (legendre_poly_L_t * legendre_poly_R) * (-1 * FCL / 2) +
        (legendre_poly_L_t * legendre_poly_R) *
            (+1 * term_1D.get_flux_scale() * std::abs(FCL) / 2 * -1);
    auto trace_value_2 =
        (legendre_poly_L_t * legendre_poly_L) * (-1 * FCL / 2) +
        (legendre_poly_L_t * legendre_poly_L) *
            (-1 * term_1D.get_flux_scale() * std::abs(FCL) / 2 * -1);
    auto trace_value_3 =
        (legendre_poly_R_t * legendre_poly_R) * (+1 * FCR / 2) +
        (legendre_poly_R_t * legendre_poly_R) *
            (+1 * term_1D.get_flux_scale() * std::abs(FCR) / 2 * +1);
    auto trace_value_4 =
        (legendre_poly_R_t * legendre_poly_L) * (+1 * FCR / 2) +
        (legendre_poly_R_t * legendre_poly_L) *
            (-1 * term_1D.get_flux_scale() * std::abs(FCR) / 2 * +1);

    // std::cout << "FCL: " << FCL << std::endl;
    // std::cout << "FCR: " << FCR << std::endl;
    // std::cout << "FluxVal: " << term_1D.get_flux_scale() << std::endl;

    // trace_value_1.print();
    // trace_value_2.print();
    // trace_value_3.print();
    // trace_value_4.print();

    // If dirichelt
    // u^-_LEFT = g(LEFT)
    // u^+_RIGHT = g(RIGHT)

    if (dim.left == boundary_condition::dirichlet) // left dirichlet
    {
      if (i == 0)
      {
        trace_value_1 =
            (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
        trace_value_2 =
            (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
        trace_value_3 =
            (legendre_poly_R_t * legendre_poly_R) * (+1 * FCR / 2) +
            (legendre_poly_R_t * legendre_poly_R) *
                (+1 * term_1D.get_flux_scale() * std::abs(FCR) / 2 * +1);
        trace_value_4 =
            (legendre_poly_R_t * legendre_poly_L) * (+1 * FCR / 2) +
            (legendre_poly_R_t * legendre_poly_L) *
                (-1 * term_1D.get_flux_scale() * std::abs(FCR) / 2 * +1);
      }
    }

    if (dim.right == boundary_condition::dirichlet) // right dirichlet
    {
      if (i == num_points - 1)
      {
        trace_value_1 =
            (legendre_poly_L_t * legendre_poly_R) * (-1 * FCL / 2) +
            (legendre_poly_L_t * legendre_poly_R) *
                (+1 * term_1D.get_flux_scale() * std::abs(FCL) / 2 * -1);
        trace_value_2 =
            (legendre_poly_L_t * legendre_poly_L) * (-1 * FCL / 2) +
            (legendre_poly_L_t * legendre_poly_L) *
                (-1 * term_1D.get_flux_scale() * std::abs(FCL) / 2 * -1);
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

    if (dim.left == boundary_condition::neumann) // left neumann
    {
      if (i == 0)
      {
        trace_value_1 =
            (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
        trace_value_2 = (legendre_poly_L_t * legendre_poly_L) * (-1 * FCL);
        trace_value_3 =
            (legendre_poly_R_t * legendre_poly_R) * (+1 * FCR / 2) +
            (legendre_poly_R_t * legendre_poly_R) *
                (+1 * term_1D.get_flux_scale() * std::abs(FCR) / 2 * +1);
        trace_value_4 =
            (legendre_poly_R_t * legendre_poly_L) * (+1 * FCR / 2) +
            (legendre_poly_R_t * legendre_poly_L) *
                (-1 * term_1D.get_flux_scale() * std::abs(FCR) / 2 * +1);
      }
    }

    if (dim.right == boundary_condition::neumann) // right neumann
    {
      if (i == num_points - 1)
      {
        trace_value_1 =
            (legendre_poly_L_t * legendre_poly_R) * (-1 * FCL / 2) +
            (legendre_poly_L_t * legendre_poly_R) *
                (+1 * term_1D.get_flux_scale() * std::abs(FCL) / 2 * -1);
        trace_value_2 =
            (legendre_poly_L_t * legendre_poly_L) * (-1 * FCL / 2) +
            (legendre_poly_L_t * legendre_poly_L) *
                (-1 * term_1D.get_flux_scale() * std::abs(FCL) / 2 * -1);
        trace_value_3 = (legendre_poly_R_t * legendre_poly_R) * (+1 * FCR);
        trace_value_4 =
            (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
      }
    }

    if (term_1D.coeff_type == coefficient_type::grad)
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

      if (dim.left == boundary_condition::periodic ||
          dim.right == boundary_condition::periodic)
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

      if (i != 0 || dim.left == boundary_condition::periodic ||
          dim.right == boundary_condition::periodic)
      {
        // Add trace part 1
        fk::matrix<double, mem_type::view> block1(
            coefficients, row1, row1 + dim.get_degree() - 1, col1,
            col1 + dim.get_degree() - 1);
        block1 = block1 + trace_value_1;
      }
      // Add trace part 2
      fk::matrix<double, mem_type::view> block2(
          coefficients, row2, row2 + dim.get_degree() - 1, col2,
          col2 + dim.get_degree() - 1);
      block2 = block2 + trace_value_2;

      // Add trace part 3
      fk::matrix<double, mem_type::view> block3(
          coefficients, row3, row3 + dim.get_degree() - 1, col3,
          col3 + dim.get_degree() - 1);
      block3 = block3 + trace_value_3;

      if (i != num_points - 1 || dim.left == boundary_condition::periodic ||
          dim.right == boundary_condition::periodic)
      {
        // Add trace part 4
        fk::matrix<double, mem_type::view> block4(
            coefficients, row4, row4 + dim.get_degree() - 1, col4,
            col4 + dim.get_degree() - 1);
        block4 = block4 + trace_value_4;
      }
    }

    // coefficients.print("Grad - post BC");
  }

  if (rotate)
  {
    // transform matrix to wavelet space
    fk::matrix<double> const forward_trans = dim.get_to_basis_operator();

    // These apply_*_fmwt() routines do the following operation:
    // coefficients = forward_trans * coefficients * forward_trans_transpose;
    coefficients = apply_right_fmwt_transposed(
        forward_trans,
        apply_left_fmwt(forward_trans, coefficients, dim.get_degree(),
                        dim.get_level()),
        dim.get_degree(), dim.get_level());
  }
  return coefficients;
}

template fk::matrix<double>
generate_coefficients(dimension<float> const &dim, term<float> const term_1D,
                      double const time, bool const rotate);

template fk::matrix<double>
generate_coefficients(dimension<double> const &dim, term<double> const term_1D,
                      double const time, bool const rotate);
