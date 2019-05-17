#include "basis.hpp"

#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

// generate_multi_wavelets routine creates wavelet basis (phi_co)
// then uses these to generate the two-scale coefficients which can be
// used (outside of this routine) to construct the forward multi-wavelet
// transform
template<typename P>
std::array<fk::matrix<P>, 6> generate_multi_wavelets(int const degree)
{
  assert(degree > 0);

  // These are the function outputs
  // g0,g1,h0, and h1 are two-scale coefficients
  // The returned phi_co is the wavelet basis
  // scalet_coefficients are the scaling function basis
  fk::matrix<P> g0(degree, degree);
  fk::matrix<P> g1(degree, degree);
  fk::matrix<P> h0(degree, degree);
  fk::matrix<P> h1(degree, degree);
  fk::matrix<P> phi_co(degree * 2, degree);
  fk::matrix<P> scalet_coefficients(degree, degree);

  // Set up parameters for quadrature - used for inner product evaluation
  // stepping sets accuracy
  // In this routine there are three intervals for the quadrature
  // Negative one to zero --zero to one -- and negative one to one
  // since this is done on the "reference element" -1 to 1
  // the intervals are fixed to be (-1,0), (0,1), and (-1,1)
  int const stepping = two_raised_to(6);
  P const neg1       = -1.0;
  P const zero       = 0.0;
  P const one        = 1.0;

  // Step 1 of 2 in creating scalets
  // generate Legendre polynomial coefficients
  // size_legendre degree+2 is used so that this algorithm will
  // work with degree 0
  // the recurrence relation used here can be found at
  // http://mathworld.wolfram.com/LegendrePolynomial.html equation (43)
  int const size_legendre = degree + 2;
  fk::matrix<P> legendre(size_legendre, size_legendre);
  legendre(0, size_legendre - 1) = 1.0;
  legendre(1, size_legendre - 2) = 1.0;
  for (int row_pos = 0; row_pos < degree - 2; ++row_pos)
  {
    // The Legendre polynomial of order row_pos+2 is constructed
    // from Legendre polynomial order row_pos and row_pos+1
    // single row matrix; using matrix here for set submatrix
    fk::matrix<P> const P_row_pos =
        legendre.extract_submatrix(row_pos + 1, 0, 1, size_legendre);
    fk::matrix<P> const P_row_pos_minus1 =
        legendre.extract_submatrix(row_pos, 0, 1, size_legendre);

    // Shifting in matrix form is equivalent to multiplying by x
    fk::matrix<P> row_shift(1, size_legendre);
    row_shift.set_submatrix(
        0, size_legendre - row_pos - 3,
        P_row_pos.extract_submatrix(0, size_legendre - row_pos - 2, 1,
                                    row_pos + 2));
    legendre.update_row(
        row_pos + 2,
        fk::vector<P>(
            row_shift *
                ((2.0 * (row_pos + 1.0) + 1.0) / ((row_pos + 1.0) + 1.0)) -
            P_row_pos_minus1 * ((row_pos + 1.0) / ((row_pos + 1.0) + 1.0))));
  }

  // Step 2 of 2 in creating the scalets
  // scale the matrix phi_j = sqrt(2*j+1)*P_j(2*x-1)
  // where P_j is Legendre polynomial of degree j
  for (int row_pos = 0; row_pos < size_legendre; ++row_pos)
  {
    fk::vector<P> const row =
        legendre.extract_submatrix(row_pos, 0, 1, size_legendre);
    legendre.update_row(row_pos, row * std::sqrt(2.0 * (row_pos) + 1));
  }

  // Roots and weights of Legendre polynomials on (-1,0), (0,1), and (-1,1)
  // we do the two-step store because we cannot have 'static' bindings
  static auto const legendre_neg1_zero = legendre_weights<P>(
      stepping, static_cast<int>(neg1), static_cast<int>(zero));
  auto const [roots_neg1to0, weights_neg1to0] = legendre_neg1_zero;

  static auto const legendre_zero_one = legendre_weights<P>(
      stepping, static_cast<int>(zero), static_cast<int>(one));
  auto const [roots_0to1, weights_0to1] = legendre_zero_one;

  static auto const legendre_neg1_one = legendre_weights<P>(
      stepping, static_cast<int>(neg1), static_cast<int>(one));
  auto const [roots_neg1to1, weights_neg1to1] = legendre_neg1_one;

  // this is to get around unused warnings
  // because can't unpack only some args w structured binding (until c++20)
  auto const ignore = [](auto ignored) { (void)ignored; };
  ignore(weights_neg1to1);

  auto const get_row = [](fk::matrix<P> const mat,
                          int const row_pos) -> fk::vector<P> {
    return mat.extract_submatrix(row_pos, 0, 1, mat.ncols());
  };

  // Formulation of phi_co before orthogonalization begins
  // phi_co is the coefficients of the polynomials
  //        f_j = |  x^(j-1) on (0,1)
  //              | -x^(j-1) on (-1,0)
  fk::matrix<P> const norm_co = [&] {
    fk::matrix<P> flip_identity = eye<P>(degree);
    for (int i = 0; i < degree; ++i)
    {
      fk::vector<int> row = fk::vector<int>(get_row(flip_identity, i));
      std::reverse(row.begin(), row.end());
      flip_identity.update_row(i, fk::vector<P>(row));
    }
    return flip_identity;
  }();

  scalet_coefficients.set_submatrix(
      0, 0,
      legendre.extract_submatrix(0, size_legendre - degree, degree, degree));

  phi_co.set_submatrix(0, 0, norm_co * -1);
  phi_co.set_submatrix(degree, 0, norm_co);

  auto const weighted_sum_products = [&](fk::vector<P> const vect_1,
                                         fk::vector<P> const vect_2,
                                         fk::vector<P> const weights) -> P {
    P sum = 0.0;
    for (int i = 0; i < weights.size(); ++i)
    {
      sum += vect_1(i) * vect_2(i) * weights(i);
    }
    return sum * 0.5;
  };

  // Gram-Schmidt orthogonalization of phi_co
  // with respect to scalets
  // The inner products are taken of:
  // elem_1 - phi_co from -1 to 0
  // elem_2 - scalet_coefficients (Legendre Poly.) from -1 to 0
  // elem_3 - phi_co from 0 to 1
  // elem_4 - scalet_coefficients (Legendre Poly.) from 0 to 1
  for (int row_pos = 0; row_pos < degree; ++row_pos)
  {
    fk::vector<P> proj(degree);
    for (int row_2 = 0; row_2 < degree; ++row_2)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(phi_co, row_pos), roots_neg1to0);
      fk::vector<P> const elem_2 =
          polyval(get_row(scalet_coefficients, row_2), roots_neg1to0);
      fk::vector<P> const elem_3 =
          polyval(get_row(phi_co, row_pos + degree), roots_0to1);
      fk::vector<P> const elem_4 =
          polyval(get_row(scalet_coefficients, row_2), roots_0to1);

      P const sum = weighted_sum_products(elem_1, elem_2, weights_neg1to0) +
                    weighted_sum_products(elem_3, elem_4, weights_0to1);
      proj = proj + (get_row(scalet_coefficients, row_2) * sum);
    }
    phi_co.update_row(row_pos, (get_row(phi_co, row_pos) - proj));
    phi_co.update_row(row_pos + degree,
                      (get_row(phi_co, row_pos + degree) - proj));
  }

  // Gram-Schmidt orthogonalization of phi_co
  // with respect to each degree less than itself
  // e.g. row k is orthogonalized with respect to rows k-1, k-2 to zero
  // row k+degree is orthogonalized with respect to rows
  // k+degree-1, k+degree-2, to degree
  // The inner products are taken of:
  // elem_1 - phi_co from -1 to 0 of order k
  // elem_2 - phi_co from -1 to 0 of order k-1
  // elem_3 - phi_co from 0 to 1 - of order k
  // elem_4 - phi_co from 0 to 1 - of order k-1
  // sum_1 is the inner product of polynomial degree k
  //    with polynomial degree k-1
  // sum_2 is the norm of polynomial degree k-1
  for (int row_pos = 1; row_pos < degree; ++row_pos)
  {
    fk::vector<P> proj_1(degree);
    fk::vector<P> proj_2(degree);
    for (int row_2 = 0; row_2 < row_pos; ++row_2)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(phi_co, row_pos), roots_neg1to0);
      fk::vector<P> const elem_2 =
          polyval(get_row(phi_co, row_2), roots_neg1to0);
      fk::vector<P> const elem_3 =
          polyval(get_row(phi_co, row_pos + degree), roots_0to1);
      fk::vector<P> const elem_4 =
          polyval(get_row(phi_co, row_2 + degree), roots_0to1);

      P const sum_1 = weighted_sum_products(elem_1, elem_2, weights_neg1to0) +
                      weighted_sum_products(elem_3, elem_4, weights_0to1);

      P const sum_2 = weighted_sum_products(elem_2, elem_2, weights_neg1to0) +
                      weighted_sum_products(elem_4, elem_4, weights_0to1);

      fk::vector<P> proj_1_add = get_row(phi_co, row_2) * sum_1;
      fk::vector<P> proj_2_add = get_row(phi_co, row_2 + degree) * sum_1;
      for (int i = 0; i < proj_1_add.size(); ++i)
      {
        proj_1_add(i) /= sum_2;
        proj_2_add(i) /= sum_2;
      }
      proj_1 = proj_1 + proj_1_add;
      proj_2 = proj_2 + proj_2_add;
    }

    phi_co.update_row(row_pos, (get_row(phi_co, row_pos) - proj_1));
    phi_co.update_row(row_pos + degree,
                      (get_row(phi_co, row_pos + degree) - proj_2));
  }

  // Normalization of phi_co basis vectors
  for (int row_pos = 0; row_pos < degree; ++row_pos)
  {
    fk::vector<P> const elem_1 =
        polyval(get_row(phi_co, row_pos), roots_neg1to0);
    fk::vector<P> const elem_2 =
        polyval(get_row(phi_co, row_pos + degree), roots_0to1);

    P const sum = weighted_sum_products(elem_1, elem_1, weights_neg1to0) +
                  weighted_sum_products(elem_2, elem_2, weights_0to1);

    fk::vector<P> row   = get_row(phi_co, row_pos);
    fk::vector<P> row_2 = get_row(phi_co, row_pos + degree);

    for (int i = 0; i < row.size(); ++i)
    {
      row(i) /= std::sqrt(sum);
      row_2(i) /= std::sqrt(sum);
    }

    phi_co.update_row(row_pos, row);
    phi_co.update_row(row_pos + degree, row_2);
  }

  // build a degree by degree matrix with alternating rows
  // of -1.0 and 1.0, then stack it vertically
  // This restores sign of polynomials after normalization
  fk::matrix<P> rep_mat(2 * degree, degree);
  std::vector<P> const pos_one(degree, 1.0);
  std::vector<P> const neg_one(degree, -1.0);
  for (int i = 0; i < degree; ++i)
  {
    if ((i % 2) == 0)
    {
      rep_mat.update_row(i, pos_one);
      rep_mat.update_row(i + degree, pos_one);
    }
    else
    {
      rep_mat.update_row(i, neg_one);
      rep_mat.update_row(i + degree, neg_one);
    }
  }

  // reverse the rows of degree by degree
  // vertical slices of phi_co
  for (int i = 0; i < 2; ++i)
  {
    fk::matrix<P> phi_co_part =
        phi_co.extract_submatrix(i * degree, 0, degree, degree);
    fk::matrix<P> const phi_co_part_copy(phi_co_part);
    for (int j = 0; j < phi_co_part.nrows(); ++j)
    {
      fk::vector<P> const new_row = phi_co_part_copy.extract_submatrix(
          phi_co_part.nrows() - 1 - j, 0, 1, phi_co_part.ncols());
      phi_co_part.update_row(j, new_row);
    }
    phi_co.set_submatrix(i * degree, 0, phi_co_part);
  }

  // finally, multiply the two elementwise
  std::transform(phi_co.begin(), phi_co.end(), rep_mat.begin(), phi_co.begin(),
                 [](P &elem_1, P &elem_2) { return elem_1 * elem_2; });

  // Calculate Two-Scale Coefficients

  // Sums to directly generate H0, H1, G0, G1
  //  H0 and H1 are the "coarsening coefficients"
  //  These describe how two adjacent locations of a higher (finer resolution)
  //  level sum to give a lower (more coarse resolution) level coefficients
  //  G0 and G1 are the "refining or detail coefficients"
  //  These describe how lower level (more coarse resolution)
  //  is split into two higher (finer resolution) level coefficients
  //  H0 is the inner product of the scaling functions of two successive
  //   levels - thus the difference in roots
  // elem_1 is the scalet functions on (-1,0)
  // elem_2 is the scalet function of a lower level and therefore spans (-1,1)
  //  H1 is also the inner product of the scaling functions of two successive
  //   levels - thus the difference in roots
  // elem_3 is the scalet functions on (0,1)
  //  G0 is the inner product of the wavelet functions of one level
  //   with the scalet functions of a lower level
  //   - thus the difference in roots
  // elem_4 is the wavelet functions on (-1,0)
  //  G1 is also the inner product of the wavelet functions of one level
  //   with the scalet functions of a lower level
  // elem_5 is the scalet functions on (0,1)

  for (int row = 0; row < degree; ++row)
  {
    for (int col = 0; col < degree; ++col)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(scalet_coefficients, row), roots_neg1to0);
      fk::vector<P> const elem_2 =
          polyval(get_row(scalet_coefficients, col), roots_neg1to1);
      h0(row, col) = 2.0 *
                     weighted_sum_products(elem_1, elem_2, weights_neg1to0) /
                     std::sqrt(2.0);

      fk::vector<P> const elem_3 =
          polyval(get_row(scalet_coefficients, row), roots_0to1);
      h1(row, col) = 2.0 * weighted_sum_products(elem_3, elem_2, weights_0to1) /
                     std::sqrt(2.0);

      fk::vector<P> const elem_4 = polyval(get_row(phi_co, row), roots_neg1to0);
      g0(row, col)               = 2.0 *
                     weighted_sum_products(elem_4, elem_2, weights_neg1to0) /
                     std::sqrt(2.0);

      fk::vector<P> const elem_5 =
          polyval(get_row(phi_co, row + degree), roots_0to1);
      g1(row, col) = 2.0 * weighted_sum_products(elem_5, elem_2, weights_0to1) /
                     std::sqrt(2.0);
    }
  }

  P const compare = [] {
    if constexpr (std::is_same<P, double>::value)
    {
      return static_cast<P>(1e-12);
    }
    return static_cast<P>(1e-4);
  }();
  auto const normalize = [compare](fk::matrix<P> &matrix) {
    std::transform(
        matrix.begin(), matrix.end(), matrix.begin(),
        [compare](P &elem) { return std::abs(elem) < compare ? 0.0 : elem; });
  };
  normalize(h0);
  normalize(h1);
  normalize(g0);
  normalize(g1);
  normalize(phi_co);
  normalize(scalet_coefficients);

  return std::array<fk::matrix<P>, 6>{h0, h1,     g0,
                                      g1, phi_co, scalet_coefficients};
}

template<typename R>
fk::matrix<R> operator_two_scale(int const degree, int const num_levels)
{
  assert(degree > 0);
  assert(num_levels > 0);

  int const max_level = two_raised_to(num_levels);

  // this is to get around unused warnings
  // because can't unpack only some args w structured binding (until c++20)
  auto const ignore = [](auto ignored) { (void)ignored; };
  auto const [h0, h1, g0, g1, phi_co, scale_co] =
      generate_multi_wavelets<R>(degree);
  ignore(phi_co);
  ignore(scale_co);

  fk::matrix<R> fmwt(degree * max_level, degree * max_level);

  fk::matrix<R> const h_block = fk::matrix<R>(h0.nrows(), h0.ncols() * 2)
                                    .set_submatrix(0, 0, h0)
                                    .set_submatrix(0, h0.ncols(), h1);
  fk::matrix<R> const g_block = fk::matrix<R>(g0.nrows(), g0.ncols() * 2)
                                    .set_submatrix(0, 0, g0)
                                    .set_submatrix(0, g0.ncols(), g1);

  // set the top vertical half of fmwt along the block diagonal with h_block
  // the lower half is set in the same manner, but with g_block
  for (int i = 0; i < max_level / 2; ++i)
  {
    fmwt.set_submatrix(degree * i, 2 * degree * i, h_block);
    fmwt.set_submatrix(degree * (i + max_level / 2), 2 * degree * i, g_block);
  }

  fk::matrix<R> fmwt_comp = eye<R>(degree * max_level, degree * max_level);

  int const n = std::floor(std::log2(max_level));
  for (int j = 1; j <= n; j++)
  {
    fk::matrix<R> cfmwt(degree * max_level, degree * max_level);
    if (j == 1)
    {
      cfmwt = fmwt;
    }
    else
    {
      int const cn = two_raised_to(n - j + 1.0) * degree;

      std::fill(cfmwt.begin(), cfmwt.end(), 0.0);
      cfmwt.set_submatrix(cn, cn, eye<R>(degree * max_level - cn));
      cfmwt.set_submatrix(
          0, 0, fk::matrix<R, mem_type::view>(fmwt, 0, cn / 2 - 1, 0, cn - 1));
      cfmwt.set_submatrix(cn / 2, 0,
                          fk::matrix<R, mem_type::view>(
                              fmwt, degree * max_level / 2,
                              degree * max_level / 2 + cn / 2 - 1, 0, cn - 1));
    }
    fmwt_comp = cfmwt * fmwt_comp;
  }
  std::transform(fmwt_comp.begin(), fmwt_comp.end(), fmwt_comp.begin(),
                 [](R &elem) { return std::abs(elem) < 1e-12 ? 0.0 : elem; });
  return fmwt_comp;
}

template std::array<fk::matrix<double>, 6>
generate_multi_wavelets(int const degree);
template std::array<fk::matrix<float>, 6>
generate_multi_wavelets(int const degree);

template fk::matrix<double>
operator_two_scale(int const degree, int const num_levels);
template fk::matrix<float>
operator_two_scale(int const degree, int const num_levels);
