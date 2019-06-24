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
fk::matrix<P> apply_left_fmwt(fk::matrix<P> const &fmwt,
                              fk::matrix<P> const &coefficient_matrix,
                              int const kdeg, int const num_levels);
template<typename P>
fk::matrix<P>
apply_left_fmwt_transposed(fk::matrix<P> const &fmwt,
                           fk::matrix<P> const &coefficient_matrix,
                           int const kdeg, int const num_levels);
template<typename P>
fk::matrix<P> apply_right_fmwt(fk::matrix<P> const &fmwt,
                               fk::matrix<P> const &coefficient_matrix,
                               int const kdeg, int const num_levels);
template<typename P>
fk::matrix<P>
apply_right_fmwt_transposed(fk::matrix<P> const &fmwt,
                            fk::matrix<P> const &coefficient_matrix,
                            int const kdeg, int const num_levels);

extern template std::array<fk::matrix<double>, 6>
generate_multi_wavelets(int const degree);
extern template std::array<fk::matrix<float>, 6>
generate_multi_wavelets(int const degree);

extern template fk::matrix<double> operator_two_scale(int const, int const);
extern template fk::matrix<float> operator_two_scale(int const, int const);

extern template fk::matrix<double> apply_left_fmwt(fk::matrix<double> const &,
                                                   fk::matrix<double> const &,
                                                   int const, int const);
extern template fk::matrix<float> apply_left_fmwt(fk::matrix<float> const &,
                                                  fk::matrix<float> const &,
                                                  int const, int const);

extern template fk::matrix<double>
apply_left_fmwt_transposed(fk::matrix<double> const &,
                           fk::matrix<double> const &, int const, int const);
extern template fk::matrix<float>
apply_left_fmwt_transposed(fk::matrix<float> const &, fk::matrix<float> const &,
                           int const, int const);

extern template fk::matrix<double> apply_right_fmwt(fk::matrix<double> const &,
                                                    fk::matrix<double> const &,
                                                    int const, int const);
extern template fk::matrix<float> apply_right_fmwt(fk::matrix<float> const &,
                                                   fk::matrix<float> const &,
                                                   int const, int const);

extern template fk::matrix<double>
apply_right_fmwt_transposed(fk::matrix<double> const &,
                            fk::matrix<double> const &, int const, int const);
extern template fk::matrix<float>
apply_right_fmwt_transposed(fk::matrix<float> const &,
                            fk::matrix<float> const &, int const, int const);
