#include "coefficients.hpp"

#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "transformations.hpp"
#include <numeric>

// static helper functions

// perform volume integral to get degree x degree block
// FIXME this name, and description, are temporary -
// we need Tim or someone to clear this up a bit.
// issue open for this.
template<typename P>
static fk::matrix<double>
volume_integral(dimension<P> const &dim, term<P> const &term_1D,
                fk::matrix<double> const &basis,
                fk::matrix<double> const &basis_prime,
                fk::vector<double> const &quadrature_weights,
                fk::vector<double> const &data, double const h)
{
  fk::matrix<double> const basis_transpose =
      fk::matrix<double>(basis).transpose();
  fk::matrix<double> const basis_prime_transpose =
      fk::matrix<double>(basis_prime).transpose();

  // little helper tool
  // form a matrix that is ncols copies of the source vector appended
  // horizontally
  auto const expand = [](fk::vector<double> const source,
                         int const ncols) -> fk::matrix<double> {
    fk::matrix<double> expanded(source.size(), ncols);
    for (int i = 0; i < ncols; ++i)
    {
      expanded.update_col(i, fk::vector<double>(source));
    }
    return expanded;
  };

  //  expand to perform elementwise mult with basis
  fk::matrix<double> const data_expand = expand(data, dim.get_degree());
  fk::matrix<double> const quadrature_weights_expand =
      expand(quadrature_weights, dim.get_degree());
  // select factors based on coefficient type
  fk::matrix<double> const factor = term_1D.coeff == coefficient_type::mass
                                        ? basis_transpose
                                        : basis_prime_transpose;
  fk::matrix<double> middle_factor =
      term_1D.coeff == coefficient_type::stiffness ? basis_prime : basis;
  // form block
  for (int i = 0; i < middle_factor.nrows(); ++i)
  {
    for (int j = 0; j < middle_factor.ncols(); ++j)
    {
      middle_factor(i, j) = data_expand(i, j) * middle_factor(i, j) *
                            quadrature_weights_expand(i, j);
    }
  }
  fk::matrix<double> const block = (factor * middle_factor) * (h / 2.0);

  return block;
}

// get indices where flux should be applied
// FIXME Can tim or someone help us understand inputs/outputs here?
template<typename P>
static std::array<fk::matrix<int>, 2>
flux_or_boundary_indices(dimension<P> const &dim, int const index)
{
  int const two_to_lev             = fm::two_raised_to(dim.get_level());
  int const previous_index         = (index - 1) * dim.get_degree();
  int const current_index          = index * dim.get_degree();
  int const next_index             = (index + 1) * dim.get_degree();
  fk::matrix<int> const prev_mesh  = meshgrid(previous_index, dim.get_degree());
  fk::matrix<int> const curr_mesh  = meshgrid(current_index, dim.get_degree());
  fk::matrix<int> const curr_trans = fk::matrix<int>(curr_mesh).transpose();
  fk::matrix<int> const next_mesh  = meshgrid(next_index, dim.get_degree());

  // interior elements - setup for flux
  if (index < two_to_lev - 1 && index > 0)
  {
    fk::matrix<int> const col_indices =
        horz_matrix_concat<int>({prev_mesh, curr_mesh, curr_mesh, next_mesh});
    fk::matrix<int> const row_indices = horz_matrix_concat<int>(
        {curr_trans, curr_trans, curr_trans, curr_trans});
    return std::array<fk::matrix<int>, 2>{row_indices, col_indices};
  }

  // boundary elements - use boundary conditions
  //
  if (dim.left == boundary_condition::periodic ||
      dim.right == boundary_condition::periodic)
  {
    fk::matrix<int> const row_indices = horz_matrix_concat<int>(
        {curr_trans, curr_trans, curr_trans, curr_trans});
    // left boundary
    if (index == 0)
    {
      fk::matrix<int> const end_mesh =
          meshgrid(dim.get_degree() * (two_to_lev - 1), dim.get_degree());
      fk::matrix<int> const col_indices =
          horz_matrix_concat<int>({end_mesh, curr_mesh, curr_mesh, next_mesh});
      return std::array<fk::matrix<int>, 2>{row_indices, col_indices};
      // right boundary
    }
    else

    {
      fk::matrix<int> const start_mesh  = meshgrid(0, dim.get_degree());
      fk::matrix<int> const col_indices = horz_matrix_concat<int>(
          {prev_mesh, curr_mesh, curr_mesh, start_mesh});
      return std::array<fk::matrix<int>, 2>{row_indices, col_indices};
    }
  }

  // other boundary conditions use same indexing
  fk::matrix<int> const row_indices =
      horz_matrix_concat<int>({curr_trans, curr_trans, curr_trans});
  // left boundary
  if (index == 0)
  {
    fk::matrix<int> const col_indices =
        horz_matrix_concat<int>({curr_mesh, curr_mesh, next_mesh});
    return std::array<fk::matrix<int>, 2>{row_indices, col_indices};
    // right boundary
  }
  else
  {
    fk::matrix<int> const col_indices =
        horz_matrix_concat<int>({prev_mesh, curr_mesh, curr_mesh});
    return std::array<fk::matrix<int>, 2>{row_indices, col_indices};
  }
}

// FIXME issue opened to clarify this function's purpose/inputs & outputs
template<typename P>
static fk::matrix<double>
get_flux_operator(dimension<P> const &dim, term<P> const term_1D,
                  double const normalize, int const index)
{
  int const two_to_lev = fm::two_raised_to(dim.get_level());
  // compute the trace values (values at the left and right of each element for
  // all k) trace_left is 1 by degree trace_right is 1 by degree
  fk::matrix<double> const trace_left =
      legendre<double>(fk::vector<double>({-1.0}), dim.get_degree())[0];
  fk::matrix<double> const trace_right =
      legendre<double>(fk::vector<double>({1.0}), dim.get_degree())[0];

  fk::matrix<double> const trace_left_t =
      fk::matrix<double>(trace_left).transpose();
  fk::matrix<double> const trace_right_t =
      fk::matrix<double>(trace_right).transpose();

  // build default average and jump operators
  fk::matrix<double> avg_op =
      horz_matrix_concat<double>({(trace_left_t * -1.0) * trace_right,
                                  (trace_left_t * -1.0) * trace_left,
                                  trace_right_t * trace_right,
                                  trace_right_t * trace_left}) *
      (0.5 * 1.0 / normalize);

  fk::matrix<double> jmp_op =
      horz_matrix_concat<double>(
          {trace_left_t * trace_right, (trace_left_t * -1.0) * trace_left,
           (trace_right_t * -1.0) * trace_right, trace_right_t * trace_left}) *
      (0.5 * 1.0 / normalize);

  // cover boundary conditions, overwriting avg and jmp if necessary
  if (index == 0 && (dim.left == boundary_condition::dirichlet ||
                     dim.left == boundary_condition::neumann))
  {
    avg_op = horz_matrix_concat<double>({(trace_left_t * -1.0) * trace_left,
                                         trace_right_t * trace_right,
                                         trace_right_t * trace_left}) *
             (0.5 * 1.0 / normalize);

    jmp_op = horz_matrix_concat<double>({(trace_left_t * -1.0) * trace_left,
                                         (trace_right_t * -1.0) * trace_right,
                                         trace_right_t * trace_left}) *
             (0.5 * 1.0 / normalize);
  }

  if ((index == (two_to_lev - 1)) &&
      (dim.right == boundary_condition::dirichlet ||
       dim.right == boundary_condition::neumann))
  {
    avg_op = horz_matrix_concat<double>({(trace_left_t * -1.0) * trace_right,
                                         (trace_left_t * -1.0) * trace_left,
                                         trace_right_t * trace_right}) *
             (0.5 * 1.0 / normalize);

    jmp_op =
        horz_matrix_concat<double>({trace_left_t * trace_right,
                                    (trace_left_t * -1.0) * trace_left,
                                    (trace_right_t * -1.0) * trace_right}) *
        (0.5 * 1.0 / normalize);
  }

  fk::matrix<double> flux_op =
      avg_op +
      ((jmp_op * (1.0 / 2.0)) * static_cast<double>(term_1D.get_flux_scale()));

  return flux_op;
}

// apply flux operator to coeff at indices specified by
// row indices and col indices FIXME elaborate?
static fk::matrix<double> apply_flux_operator(fk::matrix<int> const row_indices,
                                              fk::matrix<int> const col_indices,
                                              fk::matrix<double> const flux,
                                              fk::matrix<double> coeff)
{
  assert(row_indices.nrows() == col_indices.nrows());
  assert(row_indices.nrows() == flux.nrows());
  assert(row_indices.ncols() == col_indices.ncols());
  assert(row_indices.ncols() == flux.ncols());

  for (int i = 0; i < flux.nrows(); ++i)
  {
    for (int j = 0; j < flux.ncols(); ++j)
    {
      int const row   = row_indices(i, j);
      int const col   = col_indices(i, j);
      coeff(row, col) = coeff(row, col) - flux(i, j);
    }
  }
  return coeff;
}

// construct 1D coefficient matrix - new conventions
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matricies
template<typename P>
fk::matrix<double>
generate_coefficients(dimension<P> const &dim, term<P> const term_1D,
                      double const time)
{
  assert(time >= 0.0);
  // setup jacobi of variable x and define coeff_mat
  int const N = fm::two_raised_to(dim.get_level());
  // note that h is the symbol typically reserved for grid spacing
  double const h               = (dim.domain_max - dim.domain_min) / N;
  int const degrees_freedom_1d = dim.get_degree() * N;
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

  // get the "trace" values
  // (values at the left and right of each element for all k)

  auto [legendre_poly_L, legendre_prime_L] =
      legendre(fk::vector<double>{-1}, dim.get_degree());
  auto [legendre_poly_R, legendre_prime_R] =
      legendre(fk::vector<double>{+1}, dim.get_degree());

  legendre_poly_L = legendre_poly_L * (1 / std::sqrt(h));
  legendre_poly_R = legendre_poly_R * (1 / std::sqrt(h));

  auto legendre_poly_L_t = fk::matrix<double>(legendre_poly_L).transpose();
  auto legendre_poly_R_t = fk::matrix<double>(legendre_poly_R).transpose();

  // get the basis functions and derivatives for all k
  // this auto is std::array<fk::matrix<P>, 2>
  auto [legendre_poly, legendre_prime] =
      legendre(quadrature_points, dim.get_degree());

  legendre_poly  = legendre_poly * (1.0 / std::sqrt(h));
  legendre_prime = legendre_prime * (1.0 / std::sqrt(h) * 2.0 / h);

  auto const legendre_poly_t  = fk::matrix<double>(legendre_poly).transpose();
  auto const legendre_prime_t = fk::matrix<double>(legendre_prime).transpose();

  // get jacobian
  auto jacobi = h / 2;

  // convert term input data from wavelet space to realspace
  fk::matrix<double> const forward_trans_transpose =
      dim.get_from_basis_operator();
  fk::vector<double> const data = fk::vector<double>(term_1D.get_data());
  fk::vector<double> const data_real =
      forward_trans_transpose * fk::vector<double>(term_1D.get_data());

  for (int i = 0; i < N; ++i)
  {
    // get left and right locations for this element
    auto xL = dim.domain_min + i * h;
    auto xR = xL + h;

    // get index for current, firs and last element
    int const current = dim.get_degree() * i;
    int const first   = 0;
    int const last    = dim.get_degree() * (N - 1);

    // map quadrature points from [-1,1] to physical domain of this i element
    fk::vector<double> const quadrature_points_i =
        [&, quadrature_points = quadrature_points]() {
          fk::vector<double> quadrature_points_copy = quadrature_points;
          std::transform(
              quadrature_points_copy.begin(), quadrature_points_copy.end(),
              quadrature_points_copy.begin(), [&](double const elem) {
                return ((elem + 1) / 2 + i) * h + dim.domain_min;
              });
          return quadrature_points_copy;
        }();

    fk::vector<double> const data_real_quad = [&, legendre_poly =
                                                      legendre_poly]() {
      // get realspace data at quadrature points
      fk::vector<double> data_real_quad =
          legendre_poly *
          fk::vector<double, mem_type::view>(data_real, current,
                                             current + dim.get_degree() - 1);

      // apply g_func
      std::transform(data_real_quad.begin(), data_real_quad.end(),
                     data_real_quad.begin(),
                     std::bind(term_1D.g_func, std::placeholders::_1, time));
      return data_real_quad;
    }();

    // perform volume integral to get a degree x degree block
    //// FIXME is this description correct?
    // fk::matrix<double> const block =
    //    volume_integral(dim, term_1D, basis, basis_prime, quadrature_weights,
    //                    data_real_quad, h);

    //    std::vector<double> const tmp(data_real_quad.size());
    //    std::transform(data_real_quad.begin(), data_real_quad.end(),
    //                   quadrature_weights.begin(), tmp.begin(),
    //                   std::multiplies<double>());

    fk::matrix<double> tmp(legendre_poly.nrows(), legendre_poly.ncols());

    for (int i = 0; i <= tmp.nrows() - 1; i++)
    {
      for (int j = 0; j <= tmp.ncols() - 1; j++)
      {
        tmp(i, j) = data_real_quad(i) * legendre_poly(i, j) *
                    quadrature_weights(i) * jacobi;
      }
    }

    fk::matrix<double> block(dim.get_degree(), dim.get_degree());

    if (term_1D.coeff == coefficient_type::mass)
    {
      block = legendre_poly_t * tmp;
    }
    else if (term_1D.coeff == coefficient_type::grad)
    {
      block = legendre_prime_t * tmp * (-1);
    }

    // std::copy(quadrature_weights.begin(), quadrature_weights.end(),
    //          std::ostream_iterator<P>(std::cout, " "));
    // std::cout << std::endl;
    // std::copy(data_real_quad.begin(), data_real_quad.end(),
    //          std::ostream_iterator<P>(std::cout, " "));
    // std::cout << std::endl;
    // legendre_poly.print();
    // std::cout << jacobi << std::endl;
    // std::cout << std::endl;

    // tmp.print();
    // block.print();

    // set the block at the correct position
    fk::matrix<double> curr_block =
        fk::matrix<double, mem_type::view>(
            coefficients, current, current + dim.get_degree() - 1, current,
            current + dim.get_degree() - 1) +
        block;
    coefficients.set_submatrix(current, current, curr_block);

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

    auto FCL = term_1D.g_func(xL, time);
    auto FCR = term_1D.g_func(xR, time);

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
      if (i == N - 1)
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
    // (gradient u)*n = g
    // by splitting grad u = q by LDG methods, the B.C is changed to
    // q*n = g (=> q = g for 1D variable)
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
      if (i == N - 1)
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
      if (i == N - 1)
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
    fk::matrix<double, mem_type::view> block2(coefficients, row2,
                                              row2 + dim.get_degree() - 1, col2,
                                              col2 + dim.get_degree() - 1);
    block2 = block2 + trace_value_2;

    // Add trace part 3
    fk::matrix<double, mem_type::view> block3(coefficients, row3,
                                              row3 + dim.get_degree() - 1, col3,
                                              col3 + dim.get_degree() - 1);
    block3 = block3 + trace_value_3;

    if (i != N - 1 || dim.left == boundary_condition::periodic ||
        dim.right == boundary_condition::periodic)
    {
      // Add trace part 4
      fk::matrix<double, mem_type::view> block4(
          coefficients, row4, row4 + dim.get_degree() - 1, col4,
          col4 + dim.get_degree() - 1);
      block4 = block4 + trace_value_4;
    }

    // if (term_1D.coeff == coefficient_type::grad)
    //{
    //  auto const [row_indices, col_indices] = flux_or_boundary_indices(dim,
    //  i);

    //  fk::matrix<double> const flux_op = get_flux_operator(dim, term_1D, h,
    //  i); coefficients =
    //      apply_flux_operator(row_indices, col_indices, flux_op,
    //      coefficients);
    //}

    // std::cout << term_1D.get_flux_scale() << std::endl;
    // std::cout << i << std::endl;
    // coefficients.print();
  }

  // transform matrix to wavelet space
  // FIXME does stiffness not need this transform?
  fk::matrix<double> const forward_trans = dim.get_to_basis_operator();

  // These apply_*_fmwt() routines do the following operation:
  // coefficients = forward_trans * coefficients * forward_trans_transpose;
  coefficients = apply_right_fmwt_transposed(
      forward_trans,
      apply_left_fmwt(forward_trans, coefficients, dim.get_degree(),
                      dim.get_level()),
      dim.get_degree(), dim.get_level());

  //// zero out near-zero values after conversion to wavelet space
  // double const threshold = 1e-10;
  // auto const normalize   = [threshold](fk::matrix<double> &matrix) {
  //  std::transform(matrix.begin(), matrix.end(), matrix.begin(),
  //                 [threshold](double &elem) {
  //                   return std::abs(elem) < threshold ? 0.0 : elem;
  //                 });
  //};
  // normalize(coefficients);
  coefficients.print();
  return coefficients;
}
// construct 1D coefficient matrix
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matricies
template<typename P>
fk::matrix<double>
generate_coefficients_old(dimension<P> const &dim, term<P> const term_1D,
                          double const time)
{
  assert(time >= 0.0);
  // note that N is the number of grid points at this level
  int const N = fm::two_raised_to(dim.get_level());
  // note that h is the symbol typically reserved for grid spacing
  double const h               = (dim.domain_max - dim.domain_min) / N;
  int const degrees_freedom_1d = dim.get_degree() * N;
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

  // get the basis functions and derivatives for all k
  auto const [legendre_poly, legendre_prime] =
      legendre(quadrature_points, dim.get_degree());

  // these matrices are quad_num by degree
  fk::matrix<double> const basis = legendre_poly * (1.0 / std::sqrt(h));
  fk::matrix<double> const basis_prime =
      legendre_prime * (1.0 / std::sqrt(h) * 2.0 / h);

  // convert term input data from wavelet space to realspace
  fk::matrix<double> const forward_trans_transpose =
      dim.get_from_basis_operator();
  fk::vector<double> const data = fk::vector<double>(term_1D.get_data());
  fk::vector<double> const data_real =
      forward_trans_transpose * fk::vector<double>(term_1D.get_data());

  for (int i = 0; i < N; ++i)
  {
    // get index for current element
    int const current = dim.get_degree() * i;

    // map quadrature points from [-1,1] to physical domain of this i element
    fk::vector<double> const quadrature_points_i =
        [&, quadrature_points = quadrature_points]() {
          fk::vector<double> quadrature_points_copy = quadrature_points;
          std::transform(
              quadrature_points_copy.begin(), quadrature_points_copy.end(),
              quadrature_points_copy.begin(), [&](double const elem) {
                return ((elem + 1) / 2 + i) * h + dim.domain_min;
              });
          return quadrature_points_copy;
        }();

    fk::vector<double> const data_real_quad = [&]() {
      // get realspace data at quadrature points
      fk::vector<double> data_real_quad =
          basis * fk::vector<double, mem_type::view>(
                      data_real, current, current + dim.get_degree() - 1);

      // apply g_func
      std::transform(data_real_quad.begin(), data_real_quad.end(),
                     data_real_quad.begin(),
                     std::bind(term_1D.g_func, std::placeholders::_1, time));
      return data_real_quad;
    }();

    // perform volume integral to get a degree x degree block
    fk::matrix<double> const block =
        volume_integral(dim, term_1D, basis, basis_prime, quadrature_weights,
                        data_real_quad, h);
    // set the block at the correct position
    fk::matrix<double> curr_block =
        fk::matrix<double, mem_type::view>(
            coefficients, current, current + dim.get_degree() - 1, current,
            current + dim.get_degree() - 1) +
        block;
    coefficients.set_submatrix(current, current, curr_block);

    // setup numerical flux choice/boundary conditions
    // FIXME is this grad only? not sure yet
    if (term_1D.coeff == coefficient_type::grad)
    {
      auto const [row_indices, col_indices] = flux_or_boundary_indices(dim, i);

      fk::matrix<double> const flux_op = get_flux_operator(dim, term_1D, h, i);
      coefficients =
          apply_flux_operator(row_indices, col_indices, flux_op, coefficients);
    }
  }

  // transform matrix to wavelet space
  // FIXME does stiffness not need this transform?
  fk::matrix<double> const forward_trans = dim.get_to_basis_operator();

  // These apply_*_fmwt() routines do the following operation:
  // coefficients = forward_trans * coefficients * forward_trans_transpose;
  coefficients = apply_right_fmwt_transposed(
      forward_trans,
      apply_left_fmwt(forward_trans, coefficients, dim.get_degree(),
                      dim.get_level()),
      dim.get_degree(), dim.get_level());

  // zero out near-zero values after conversion to wavelet space
  double const threshold = 1e-10;
  auto const normalize   = [threshold](fk::matrix<double> &matrix) {
    std::transform(matrix.begin(), matrix.end(), matrix.begin(),
                   [threshold](double &elem) {
                     return std::abs(elem) < threshold ? 0.0 : elem;
                   });
  };
  normalize(coefficients);
  return coefficients;
}

template fk::matrix<double>
generate_coefficients_old(dimension<float> const &dim,
                          term<float> const term_1D, double const time);

template fk::matrix<double>
generate_coefficients_old(dimension<double> const &dim,
                          term<double> const term_1D, double const time);

template fk::matrix<double> generate_coefficients(dimension<float> const &dim,
                                                  term<float> const term_1D,
                                                  double const time);

template fk::matrix<double> generate_coefficients(dimension<double> const &dim,
                                                  term<double> const term_1D,
                                                  double const time);
