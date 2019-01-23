#include "transformations.hpp"

#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

// temp
#include <iostream>
#include <numeric>
template<typename ForwardIterator, typename P>
static void strided_iota(ForwardIterator first, ForwardIterator last, P value,
                         P const stride)
{
  while (first != last)
  {
    *first++ = value;
    value += stride;
  }
}

template<typename P>
multi_wavelets<P>::multi_wavelets(int const degree)
    : phi_co(degree * 2, degree), scalet_coefficients(degree, degree)
{
  assert(degree > 0);
  // get the quadrature stuff...for evaluating some integral? don't remember...
  int const stepping           = static_cast<int>(std::pow(2, 10));
  P const step                 = static_cast<P>(2.0) / stepping;
  P const start                = -1.0;
  P const end                  = 1.0;
  int const num_steps          = static_cast<int>((end - start) / step);
  fk::vector<P> const x_coords = [=] {
    fk::vector<P> x(num_steps);
    strided_iota(x.begin(), x.end(), start, step);
    return x;
  }();

  // isn't this the coefficient matrix? or no...
  int const size_legendre = 3 * degree + 2;
  fk::matrix<P> legendre(size_legendre, size_legendre);
  legendre(0, size_legendre - 1) = static_cast<P>(1.0);
  legendre(1, size_legendre - 2) = static_cast<P>(1.0);
  for (int row_pos = 0; row_pos < 3 * degree; ++row_pos)
  {
    // single row matrix; using matrix here for set submatrix

    fk::matrix<P> const row_shift =
        legendre.extract_submatrix(row_pos + 1, 1, 1, size_legendre - 1);
    fk::matrix<P> row(1, size_legendre);
    row.set_submatrix(0, 0, row_shift);
    fk::vector<P> const row_last = row;
    fk::vector<P> const row_last_scaled =
        row_last * static_cast<P>(2 * (row_pos + 1) + 1) *
        (static_cast<P>(1.0) / (row_pos + 2));

    fk::vector<P> const row_prev =
        legendre.extract_submatrix(row_pos, 0, 1, size_legendre);
    fk::vector<P> const row_prev_scaled = row_prev *
                                          static_cast<P>((row_pos + 1)) *
                                          (static_cast<P>(1.0) / (row_pos + 2));

    legendre.update_row(row_pos + 2, (row_last_scaled - row_prev_scaled));
  }

  // scale the matrix
  for (int row_pos = 0; row_pos < size_legendre; ++row_pos)
  {
    fk::vector<P> const row =
        legendre.extract_submatrix(row_pos, 0, 1, size_legendre);
    legendre.update_row(row_pos,
                        row * std::sqrt(static_cast<P>(2.0) * (row_pos) + 1));
  }

  auto const [roots, weights] = legendre_weights<P>(
      stepping, static_cast<int>(start), static_cast<int>(end));
  fk::vector<P> const weights_scaled = weights * static_cast<P>(0.5);

  auto const get_row = [](fk::matrix<P> const mat,
                          int const row_pos) -> fk::vector<P> {
    return mat.extract_submatrix(row_pos, 0, 1, mat.ncols());
  };

  fk::matrix<P> const norm_co = [&] {
    fk::matrix<P> flip_identity = eye<P>(degree);
    for (int i = 0; i < degree; ++i)
    {
      fk::vector<int> row = get_row(flip_identity, i);
      std::reverse(row.begin(), row.end());
      flip_identity.update_row(i, row);
    }
    return flip_identity;
  }();

  scalet_coefficients.set_submatrix(
      0, 0,
      legendre.extract_submatrix(0, size_legendre - degree, degree, degree));

  phi_co.set_submatrix(0, 0, norm_co * -1);
  phi_co.set_submatrix(degree, 0, norm_co);

  auto const [roots_minus_scaled, roots_plus_scaled] = [&, roots = roots] {
    fk::vector<P> roots_min_1  = roots;
    fk::vector<P> roots_plus_1 = roots;
    for (int i = 0; i < roots.size(); ++i)
    {
      roots_min_1(i)  = (roots(i) - static_cast<P>(1.0)) / static_cast<P>(2.0);
      roots_plus_1(i) = (roots(i) + static_cast<P>(1.0)) / static_cast<P>(2.0);
    }
    return std::array<fk::vector<P>, 2>{roots_min_1, roots_plus_1};
  }();

  auto const weighted_sum_products = [&](fk::vector<P> const vect_1,
                                         fk::vector<P> const vect_2) -> P {
    P sum = 0.0;
    for (int i = 0; i < weights_scaled.size(); ++i)
    {
      sum += vect_1(i) * vect_2(i) * weights_scaled(i);
    }
    return sum * 0.5;
  };

  for (int row_pos = 0; row_pos < degree; ++row_pos)
  {
    fk::vector<P> proj(degree);
    for (int row_2 = 0; row_2 < degree; ++row_2)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(phi_co, row_pos), roots_minus_scaled);
      fk::vector<P> const elem_2 =
          polyval(get_row(scalet_coefficients, row_2), roots_minus_scaled);
      fk::vector<P> const elem_3 =
          polyval(get_row(phi_co, row_pos + degree), roots_plus_scaled);
      fk::vector<P> const elem_4 =
          polyval(get_row(scalet_coefficients, row_2), roots_plus_scaled);

      P const sum = weighted_sum_products(elem_1, elem_2) +
                    weighted_sum_products(elem_3, elem_4);
      proj = proj + (get_row(scalet_coefficients, row_2) * sum);
    }
    phi_co.update_row(row_pos, (get_row(phi_co, row_pos) - proj));
    phi_co.update_row(row_pos + degree,
                      (get_row(phi_co, row_pos + degree) - proj));
  }

  // "boost normalization to higher polynomials"
  for (int row_pos = 1; row_pos < degree; ++row_pos)
  {
    fk::vector<P> proj_1(degree);
    fk::vector<P> proj_2(degree);
    for (int row_2 = 0; row_2 < row_pos; ++row_2)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(phi_co, row_pos), roots_minus_scaled);
      fk::vector<P> const elem_2 =
          polyval(get_row(phi_co, row_2), roots_minus_scaled);
      fk::vector<P> const elem_3 =
          polyval(get_row(phi_co, row_pos + degree), roots_plus_scaled);
      fk::vector<P> const elem_4 =
          polyval(get_row(phi_co, row_2 + degree), roots_plus_scaled);

      P const sum_1 = weighted_sum_products(elem_1, elem_2) +
                      weighted_sum_products(elem_3, elem_4);

      fk::vector<P> const elem_5 =
          polyval(get_row(phi_co, row_2), roots_minus_scaled);
      fk::vector<P> const elem_6 = elem_5;
      fk::vector<P> const elem_7 =
          polyval(get_row(phi_co, row_2 + degree), roots_plus_scaled);
      fk::vector<P> const elem_8 = elem_7;

      P const sum_2 = weighted_sum_products(elem_5, elem_6) +
                      weighted_sum_products(elem_7, elem_8);

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

  for (int row_pos = 0; row_pos < degree; ++row_pos)
  {
    fk::vector<P> const elem_1 =
        polyval(get_row(phi_co, row_pos), roots_minus_scaled);
    fk::vector<P> const elem_2 = elem_1;
    fk::vector<P> const elem_3 =
        polyval(get_row(phi_co, row_pos + degree), roots_plus_scaled);
    fk::vector<P> const elem_4 = elem_3;

    P const sum = weighted_sum_products(elem_1, elem_2) +
                  weighted_sum_products(elem_3, elem_4);

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
  fk::matrix<P> rep_mat(2*degree, degree);
  std::vector<P> const pos_one(degree, static_cast<P>(1.0));
  std::vector<P> const neg_one(degree, static_cast<P>(-1.0));
  for(int i = 0; i < degree; ++i) {
    if((i % 2) == 0) {
        rep_mat.update_row(i, pos_one);
	rep_mat.update_row(i + degree, pos_one);
    } else {
	rep_mat.update_row(i, neg_one);
	rep_mat.update_row(i + degree, neg_one);
    }
  } 

  // reverse the rows of degree by degree 
  // vertical slices of phi_co
  for(int i = 0; i < 2; ++i) {
    fk::matrix<P> phi_co_part = phi_co.extract_submatrix(i*degree, 0, degree, degree);
    std::reverse(phi_co_part.begin(), phi_co_part.end());
    for(int j = 0; j < degree; ++j) {
	fk::vector<P> phi_co_row = phi_co_part.extract_submatrix(j, 0, 1, degree);
	std::reverse(phi_co_row.begin(), phi_co_row.end());
	phi_co_part.update_row(j, phi_co_row);
    }
    phi_co.set_submatrix(i*degree, 0, phi_co_part);
  }

  phi_co = rep_mat * phi_co;

  // "determine the Two Scale Coeffecients"

  fk::matrix<P> g0(degree, degree);
  fk::matrix<P> h0(degree, degree);
  fk::matrix<P> g1(degree, degree);
  fk::matrix<P> h1(degree, degree);

  fk::vector<P> const norm_legendre_roots = roots_plus_scaled;
  fk::vector<P> const norm_legendre_minus = [&] {
	fk::vector<P> norm_legendre = norm_legendre_roots;
	std::transform(norm_legendre.begin(), norm_legendre.end(), norm_legendre.begin(),[](P& elem) {
			return elem-static_cast<P>(1.0);});
	return norm_legendre;
  }();
  fk::vector<P> const norm_legendre_two = [&] {
	fk::vector<P> norm_legendre = norm_legendre_roots;
	std::transform(norm_legendre.begin(), norm_legendre.end(), norm_legendre.begin(),[](P& elem) {
			return (elem*static_cast<P>(2.0))-static_cast<P>(1.0);});
	return norm_legendre;
  }();


  for (int row = 0; row < degree; ++row)
  {
    for (int col = 0; col < degree; ++col)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(scalet_coefficients, row), norm_legendre_minus);
      fk::vector<P> const elem_2 =
          polyval(get_row(scalet_coefficients, col), norm_legendre_two);
      h0(row, col) = weighted_sum_products(elem_1, elem_2) / std::sqrt(static_cast<P>(2.0));

      fk::vector<P> const elem_3 =
          polyval(get_row(scalet_coefficients, row), norm_legendre_roots);
      h1(row, col) = weighted_sum_products(elem_3, elem_2) / std::sqrt(static_cast<P>(2.0));


      fk::vector<P> const elem_4 =
          polyval(get_row(phi_co, row), norm_legendre_minus);
      g0(row, col) = weighted_sum_products(elem_4, elem_2) / std::sqrt(static_cast<P>(2.0));

      fk::vector<P> const elem_5 =
	      polyval(get_row(phi_co, row+degree), norm_legendre_roots);
      g1(row, col) = weighted_sum_products(elem_5, elem_2) / std::sqrt(static_cast<P>(2.0));
    }

  }

  return std::array<fk::matrix<P>, 

}

template<typename P>
fk::matrix<P> multi_wavelets<P>::get_g0()
{
  return g0;
}

template<typename P>
fk::matrix<P> multi_wavelets<P>::get_h0()
{
  return h0;
}

template<typename P>
fk::matrix<P> multi_wavelets<P>::get_phi_co()
{
  return phi_co;
}

template<typename P>
fk::matrix<P> multi_wavelets<P>::get_scalet_coefficients()
{
  return scalet_coefficients;
}

template class multi_wavelets<double>;
template class multi_wavelets<float>;
