#include "coefficients.hpp"

#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "transformations.hpp"
#include <numeric>

// construct 1D coefficient matrix
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matricies.

template<typename P>
fk::matrix<P> generate_coefficients(dimension<P> const dim,
                                    term<P> const term_1D, P const time)
{
  assert(time >= 0.0);
  // setup jacobi of variable x and define coeff_mat
  int const two_to_level    = static_cast<int>(std::pow(2, dim.level));
  P const normalized_domain = (dim.domain_max - dim.domain_min) / two_to_level;
  int const degrees_freedom_1d = dim.degree * two_to_level;
  fk::matrix<P> coefficients(degrees_freedom_1d, degrees_freedom_1d);

  // set number of quatrature points (should this be order dependent?)
  // FIXME is this a global quantity??
  int const quad_num = 10;

  // get quadrature points and weights.
  auto const [roots, weights] = legendre_weights<P>(quad_num, -1.0, 1.0);

  // compute the trace values (values at the left and right of each element for
  // all k) trace_left is 1 by degree trace_right is 1 by degree
  // FIXME should these be vectors?
  fk::matrix<P> const trace_left =
      legendre<P>(fk::vector<P>({-1.0}), dim.degree)[0];
  fk::matrix<P> const trace_right =
      legendre<P>(fk::vector<P>({1.0}), dim.degree)[0];

  // get the basis functions and derivatives for all k
  auto const [legendre_poly, legendre_prime] = legendre(roots, dim.degree);

  // these matrices are quad_num by degree
  fk::matrix<P> const basis =
      legendre_poly * (1.0 / std::sqrt(normalized_domain));
  fk::matrix<P> const basis_prime =
      legendre_prime *
      (1.0 / std::sqrt(normalized_domain) * 2.0 / normalized_domain);
  fk::matrix<P> const basis_transpose = fk::matrix<P>(basis).transpose();
  fk::matrix<P> const basis_prime_transpose =
      fk::matrix<P>(basis_prime).transpose();

  // convert term input data from wavelet space to realspace

  fk::matrix<P> const forward_trans = operator_two_scale<P>(dim);
  fk::matrix<P> const forward_trans_transpose =
      fk::matrix<P>(forward_trans).transpose();
  fk::vector<P> const data_real = forward_trans_transpose * term_1D.get_data();

  // define three small tools I'll need
  // limited subset of matlab meshgrid functionality
  auto const meshgrid = [](int const start, int const length) -> fk::matrix<P> {
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
  };

  // horizontally concatenate set of matrices w/ same number of rows
  auto const horz_matrix_concat =
      [](std::vector<fk::matrix<P>> const matrices) -> fk::matrix<P> {
    assert(matrices.size() > 0);
    auto const [nrows, ncols] = [&]() {
      int row_accum   = 0;
      int const ncols = matrices[0].ncols();
      for (auto const &mat : matrices)
      {
        row_accum += mat.nrows();
        assert(mat.ncols() == ncols);
      }
      return std::array<int, 2>{row_accum, ncols};
    }();
    fk::matrix<P> concat(nrows, ncols);
    int col_index = 0;
    for (auto const &mat : matrices)
    {
      concat.set_submatrix(0, col_index += ncols, mat);
    }
    return concat;
  };
  // form a matrix that is ncols copies of the source vector appended
  // horizontally
  auto const expand = [](fk::vector<P> const source,
                         int const ncols) -> fk::matrix<P> {
    fk::matrix<P> expanded(source.size(), ncols);
    for (int i = 0; i < ncols; ++i)
    {
      expanded.update_col(i, source);
    }
    return expanded;
  };

  for (int i = 0; i < two_to_level; ++i)
  {
    // get index for current, next, prev.
    int const current = dim.degree * i;
    int const prev    = dim.degree * (i - 1);
    int const next    = dim.degree * (i + 1);

    // map quadrature points from [-1,1] to physical domain of this i element
    fk::vector<P> const roots_i = [&, roots = roots]() {
      std::transform(
          roots.begin(), roots.end(), roots.begin(), [&](P const elem) {
            return ((elem + 1) / 2 + i) * normalized_domain + dim.domain_min;
          });
      return roots;
    }();

    // get realspace data at quadrature points, w/ g_func applied
    fk::vector<P> const data_real_quad = [&]() {
      fk::vector<P> data_real_quad =
          basis * data_real.extract(current, current + dim.degree);
      std::transform(data_real.begin(), data_real.end(), data_real.begin(),
                     std::bind2nd(term_1D.g_func, time));
      return data_real_quad;
    }();

    // perform volume integral to get a degree x degree block //FIXME is this
    // corrrect?
    fk::matrix<P> const block = [&, &weights = weights]() {
      fk::matrix<P> block(dim.degree, dim.degree);
      //  expand to perform elementwise mult with basis
      fk::matrix<P> const data_expand    = expand(data_real_quad, dim.degree);
      fk::matrix<P> const weights_expand = expand(weights, dim.degree);
      // select factors based on coefficient type
      fk::matrix<P> const factor = term_1D.coeff == coefficient_type::mass
                                       ? basis_transpose
                                       : basis_prime_transpose;
      fk::matrix<P> middle_factor =
          term_1D.coeff == coefficient_type::stiffness ? basis_prime : basis;
      // form blck
      for (int i = 0; i < middle_factor.nrows(); ++i)
      {
        for (int j = 0; j < middle_factor.ncols(); ++j)
        {
          middle_factor(i, j) =
              data_expand(i, j) * middle_factor(i, j) * weights_expand(i, j);
        }
      }
      return (factor * block) * (normalized_domain / 2.0);
    }();
  }

  return fk::matrix<P>();
}

template fk::matrix<float> generate_coefficients(dimension<float> const dim,
                                                 term<float> const term_1D,
                                                 float const time);

template fk::matrix<double> generate_coefficients(dimension<double> const dim,
                                                  term<double> const term_1D,
                                                  double const time);
