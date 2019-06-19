#pragma once

#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

template<typename P>
std::array<fk::matrix<P>, 6> generate_multi_wavelets(int const degree);

template<typename P>
fk::matrix<P> operator_two_scale(int const degree, int const num_levels);

template<typename P>
fk::matrix<P>
apply_fmwt(fk::matrix<P> const fmwt, fk::matrix<P> const coefficient_matrix,
           int const kdegree, int const num_levels, bool const fmwt_left,
           bool const fmwt_trans);

extern template std::array<fk::matrix<double>, 6>
generate_multi_wavelets(int const degree);
extern template std::array<fk::matrix<float>, 6>
generate_multi_wavelets(int const degree);

extern template fk::matrix<double> operator_two_scale(int const, int const);
extern template fk::matrix<float> operator_two_scale(int const, int const);

extern template fk::matrix<double>
apply_fmwt(fk::matrix<double> const fmwt,
           fk::matrix<double> const coefficient_matrix, int const kdegree,
           int const num_levels, bool const fmwt_left, bool const fmwt_trans);
extern template fk::matrix<float>
apply_fmwt(fk::matrix<float> const fmwt,
           fk::matrix<float> const coefficient_matrix, int const kdegree,
           int const num_levels, bool const fmwt_left, bool const fmwt_trans);
