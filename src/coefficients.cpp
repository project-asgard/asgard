#include "coefficients.hpp"

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
                fk::vector<double> const &weights,
                fk::vector<double> const &data, double const normalized_domain)
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
  fk::matrix<double> const block =
      (factor * middle_factor) * (normalized_domain / 2.0);

  return block;
}

// get indices where flux should be applied
// FIXME Can tim or someone help us understand inputs/outputs here?
template<typename P>
static std::array<fk::matrix<int>, 2>
flux_or_boundary_indices(dimension<P> const &dim, int const index)
{
  int const two_to_lev             = two_raised_to(dim.get_level());
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
  int const two_to_lev = two_raised_to(dim.get_level());
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

// construct 1D coefficient matrix
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
  int const two_to_level = two_raised_to(dim.get_level());
  double const normalized_domain =
      (dim.domain_max - dim.domain_min) / two_to_level;
  int const degrees_freedom_1d = dim.get_degree() * two_to_level;
  fk::matrix<double> coefficients(degrees_freedom_1d, degrees_freedom_1d);

  // set number of quatrature points
  // FIXME should this be order dependent?
  // FIXME is this a global quantity??
  int const quad_num = 10;

  // get quadrature points and weights.
  // we do the two-step store because we cannot have 'static' bindings
  static const auto legendre_values =
      legendre_weights<double>(quad_num, -1.0, 1.0);
  auto const [roots, weights] = legendre_values;

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
  fk::matrix<double> const forward_trans_transpose =
      dim.get_from_basis_operator();
  fk::vector<double> const data = fk::vector<double>(term_1D.get_data());
  fk::vector<double> const data_real =
      forward_trans_transpose * fk::vector<double>(term_1D.get_data());

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
          basis * fk::vector<double, mem_type::view>(
                      data_real, current, current + dim.get_degree() - 1);

      // apply g_func
      std::transform(data_real_quad.begin(), data_real_quad.end(),
                     data_real_quad.begin(),
                     std::bind(term_1D.g_func, std::placeholders::_1, time));
      return data_real_quad;
    }();

    // perform volume integral to get a degree x degree block
    // FIXME is this description correct?
    fk::matrix<double> const block =
        volume_integral(dim, term_1D, basis, basis_prime, weights,
                        data_real_quad, normalized_domain);
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

      fk::matrix<double> const flux_op =
          get_flux_operator(dim, term_1D, normalized_domain, i);
      coefficients =
          apply_flux_operator(row_indices, col_indices, flux_op, coefficients);
    }
  }

  // transform matrix to wavelet space
  // FIXME does stiffness not need this transform?
  fk::matrix<double> const forward_trans = dim.get_to_basis_operator();
  coefficients =
      apply_fmwt(forward_trans,
                 apply_fmwt(forward_trans, coefficients, dim.get_degree(),
                            dim.get_level(), true, false),
                 dim.get_degree(), dim.get_level(), false, true);

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

template fk::matrix<double> generate_coefficients(dimension<float> const &dim,
                                                  term<float> const term_1D,
                                                  double const time);

template fk::matrix<double> generate_coefficients(dimension<double> const &dim,
                                                  term<double> const term_1D,
                                                  double const time);
