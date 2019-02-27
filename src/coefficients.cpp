#include "coefficients.hpp"

#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "transformations.hpp"
#include <numeric>

// helper functions

// stitch matrices together side by side (all must have same # rows)
template<typename P>
static fk::matrix<P>
horz_matrix_concat(std::vector<fk::matrix<P>> const matrices)
{
  assert(matrices.size() > 0);
  auto const [nrows, ncols] = [&]() {
    int col_accum   = 0;
    int const nrows = matrices[0].nrows();
    for (auto const &mat : matrices)
    {
      col_accum += mat.ncols();
      assert(mat.nrows() == nrows);
    }
    return std::array<int, 2>{nrows, col_accum};
  }();
  fk::matrix<P> concat(nrows, ncols);
  int col_index = 0;
  for (auto const &mat : matrices)
  {
    concat.set_submatrix(0, col_index, mat);
    col_index += mat.ncols();
  }
  return concat;
}

// limited subset of matlab meshgrid
template<typename P>
static fk::matrix<P> meshgrid(int const start, int const length)
{
  fk::matrix<P> mesh(length, length);
  fk::vector<P> const row = [=]() {
    fk::vector<P> row(length);
    std::iota(row.begin(), row.end(), start);
    return row;
  }();
  for (int i = 0; i < mesh.nrows(); ++i)
  {
    mesh.update_row(i, row);
  }
  return mesh;
}

// perform volume integral to get degree x degree block
template<typename P>
static fk::matrix<double>
volume_integral(dimension<P> const dim, term<P> const term_1D,
                fk::matrix<double> const basis,
                fk::matrix<double> const basis_prime,
                fk::vector<double> const weights, fk::vector<double> const data,
                double const normalized_domain)
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
    fk::matrix<P> expanded(source.size(), ncols);
    for (int i = 0; i < ncols; ++i)
    {
      expanded.update_col(i, source);
    }
    return expanded;
  };

  fk::matrix<double> const block = [&, &weights = weights]() {
    fk::matrix<double> block(dim.get_degree(), dim.get_degree());
    //  expand to perform elementwise mult with basis
    fk::matrix<double> const data_expand    = expand(data, dim.get_degree());
    fk::matrix<double> const weights_expand = expand(weights, dim.get_degree());
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
        middle_factor(i, j) =
            data_expand(i, j) * middle_factor(i, j) * weights_expand(i, j);
      }
    }
    return (factor * middle_factor) * (normalized_domain / 2.0);
  }();
  return block;
}

// get indices where flux should be applied
template<typename P>
static std::array<fk::matrix<int>, 2>
flux_or_boundary_indices(dimension<P> const dim, int const index)
{
  int const two_to_lev = static_cast<int>(std::pow(2, dim.get_level()));
  int const prev       = (index - 1) * dim.get_degree();
  int const curr       = index * dim.get_degree();
  int const next       = (index + 1) * dim.get_degree();
  fk::matrix<int> const prev_mesh  = meshgrid<int>(prev, dim.get_degree());
  fk::matrix<int> const curr_mesh  = meshgrid<int>(curr, dim.get_degree());
  fk::matrix<int> const curr_trans = fk::matrix<int>(curr_mesh).transpose();
  fk::matrix<int> const next_mesh  = meshgrid<int>(next, dim.get_degree());

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
          meshgrid<int>(dim.get_degree() * (two_to_lev - 1), dim.get_degree());
      fk::matrix<int> const col_indices =
          horz_matrix_concat<int>({end_mesh, curr_mesh, curr_mesh, next_mesh});
      return std::array<fk::matrix<int>, 2>{row_indices, col_indices};
      // right boundary
    }
    else

    {
      fk::matrix<int> const start_mesh  = meshgrid<int>(0, dim.get_degree());
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

template<typename P>
static fk::matrix<double>
get_flux_operator(dimension<P> const dim, term<P> const term_1D,
                  double const normalize, int const index)
{
  int const two_to_lev = static_cast<int>(std::pow(2, dim.get_level()));
  // compute the trace values (values at the left and right of each element for
  // all k) trace_left is 1 by degree trace_right is 1 by degree
  fk::matrix<double> const trace_left =
      legendre<double>(fk::vector<double>({-1.0}), dim.get_degree())[0];
  fk::matrix<double> const trace_right =
      legendre<double>(fk::vector<double>({1.0}), dim.get_degree())[0];

  fk::matrix<double> const trace_left_t = [&] {
    fk::matrix<double> trace_left_transpose = trace_left;
    trace_left_transpose.transpose();
    return trace_left_transpose;
  }();

  fk::matrix<double> const trace_right_t = [&] {
    fk::matrix<double> trace_right_transpose = trace_right;
    trace_right_transpose.transpose();
    return trace_right_transpose;
  }();

  // build default average and jump operators
  fk::matrix<double> const avg_op =
      horz_matrix_concat<double>({(trace_left_t * -1.0) * trace_right,
                                  (trace_left_t * -1.0) * trace_left,
                                  trace_right_t * trace_right,
                                  trace_right_t * trace_left}) *
      (0.5 * 1.0 / normalize);

  fk::matrix<double> const jmp_op =
      horz_matrix_concat<double>(
          {trace_left_t * trace_right, (trace_left_t * -1.0) * trace_left,
           (trace_right_t * -1.0) * trace_right, trace_right_t * trace_left}) *
      (0.5 * 1.0 / normalize);

  // cover boundary conditions

  if (index == 0 && (dim.left == boundary_condition::dirichlet ||
                     dim.left == boundary_condition::neumann))
  {
    fk::matrix<double> const avg_op =
        horz_matrix_concat<double>({(trace_left_t * -1.0) * trace_left,
                                    trace_right_t * trace_right,
                                    trace_right_t * trace_left}) *
        (0.5 * 1.0 / normalize);

    fk::matrix<double> const jmp_op =
        horz_matrix_concat<double>({(trace_left_t * -1.0) * trace_left,
                                    (trace_right_t * -1.0) * trace_right,
                                    trace_right_t * trace_left}) *
        (0.5 * 1.0 / normalize);
  }

  if ((index == (two_to_lev - 1)) &&
      (dim.right == boundary_condition::dirichlet ||
       dim.right == boundary_condition::neumann))
  {
    fk::matrix<double> const avg_op =
        horz_matrix_concat<double>({(trace_left_t * -1.0) * trace_right,
                                    (trace_left_t * -1.0) * trace_left,
                                    trace_right_t * trace_right}) *
        (0.5 * 1.0 / normalize);

    fk::matrix<double> const jmp_op =
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

// apply flux operator at given indices
static fk::matrix<double> &
apply_flux_operator(fk::matrix<int> const row_indices,
                    fk::matrix<int> const col_indices,
                    fk::matrix<double> const flux, fk::matrix<double> &coeff)
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

// construct 1D coefficient matrix
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matricies.

template<typename P>
fk::matrix<double>
generate_coefficients(dimension<P> const dim, term<P> const term_1D,
                      double const time)
{
  assert(time >= 0.0);
  // setup jacobi of variable x and define coeff_mat
  int const two_to_level = static_cast<int>(std::pow(2, dim.get_level()));
  double const normalized_domain =
      (dim.domain_max - dim.domain_min) / two_to_level;
  int const degrees_freedom_1d = dim.get_degree() * two_to_level;
  fk::matrix<double> coefficients(degrees_freedom_1d, degrees_freedom_1d);

  // set number of quatrature points (should this be order dependent?)
  // FIXME is this a global quantity??
  int const quad_num = 10;

  // get quadrature points and weights.
  auto const [roots, weights] = legendre_weights<double>(quad_num, -1.0, 1.0);

  // get the basis functions and derivatives for all k
  auto const [legendre_poly, legendre_prime] =
      legendre(roots, dim.get_degree());

  // these matrices are quad_num by degree
  fk::matrix<double> const basis =
      legendre_poly * (1.0 / std::sqrt(normalized_domain));
  fk::matrix<double> const basis_prime =
      legendre_prime *
      (1.0 / std::sqrt(normalized_domain) * 2.0 / normalized_domain);

  // convert term input data from wavelet space to realspace

  fk::matrix<double> const forward_trans = operator_two_scale<P, double>(dim);
  fk::matrix<double> const forward_trans_transpose =
      fk::matrix<double>(forward_trans).transpose();
  fk::vector<double> const data = term_1D.get_data();
  fk::vector<double> const data_real =
      forward_trans_transpose * term_1D.get_data();

  for (int i = 0; i < two_to_level; ++i)
  {
    // get index for current element
    int const current = dim.get_degree() * i;

    // map quadrature points from [-1,1] to physical domain of this i element
    fk::vector<double> const roots_i = [&, roots = roots]() {
      fk::vector<double> roots_copy = roots;
      std::transform(roots_copy.begin(), roots_copy.end(), roots_copy.begin(),
                     [&](double const elem) {
                       return ((elem + 1) / 2 + i) * normalized_domain +
                              dim.domain_min;
                     });
      return roots_copy;
    }();

    fk::vector<double> const data_real_quad = [&]() {
      // get realspace data at quadrature points
      fk::vector<double> data_real_quad =
          basis * data_real.extract(current, current + dim.get_degree() - 1);
      // apply g_func
      std::transform(data_real_quad.begin(), data_real_quad.end(),
                     data_real_quad.begin(),
                     std::bind2nd(term_1D.g_func, time));
      return data_real_quad;
    }();

    // perform volume integral to get a degree x degree block //FIXME is this
    // description correct?
    fk::matrix<double> const block =
        volume_integral(dim, term_1D, basis, basis_prime, weights,
                        data_real_quad, normalized_domain);
    // set the block at the correct position
    fk::matrix<double> const curr_block =
        coefficients.extract_submatrix(current, current, dim.get_degree(),
                                       dim.get_degree()) +
        block;
    coefficients.set_submatrix(current, current, curr_block);
    // setup numerical flux choice/boundary conditions
    // FIXME is this grad only? not sure yet
    if (term_1D.coeff == coefficient_type::grad)
    {
      auto const [row_indices, col_indices] = flux_or_boundary_indices(dim, i);

      fk::matrix<double> const flux_op =
          get_flux_operator(dim, term_1D, normalized_domain, i);
      coefficients =
          apply_flux_operator(row_indices, col_indices, flux_op, coefficients);
    }
  }

  // transform matrix to wavelet space
  // FIXME does stiffness not need this transform?
  coefficients = forward_trans * coefficients * forward_trans_transpose;

  // zero out near-zero values after conversion to wavelet space
  double const compare = 1e-6;
  auto const normalize = [compare](fk::matrix<double> &matrix) {
    std::transform(matrix.begin(), matrix.end(), matrix.begin(),
                   [compare](double &elem) {
                     return std::abs(elem) < compare ? 0.0 : elem;
                   });
  };
  normalize(coefficients);
  return coefficients;
}

template fk::matrix<double> generate_coefficients(dimension<float> const dim,
                                                  term<float> const term_1D,
                                                  double const time);

template fk::matrix<double> generate_coefficients(dimension<double> const dim,
                                                  term<double> const term_1D,
                                                  double const time);
