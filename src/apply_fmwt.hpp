#pragma once

#include "element_table.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

template<typename P>
fk::matrix<P> reshape(fk::matrix<P> mat, int const dim1, int const dim2);

extern template fk::matrix<double>
reshape(fk::matrix<double> mat, int const dim1, int const dim2);
extern template fk::matrix<float>
reshape(fk::matrix<float> mat, int const dim1, int const dim2);

template<typename P>
fk::matrix<P> apply_fmwt(fk::matrix<P> matrix1, fk::matrix<P> matrix2,
                         int const kdeg, int const lev, int const isLeft,
                         int const isTrans, int const method);

extern template fk::matrix<double>
apply_fmwt(fk::matrix<double> matrix1, fk::matrix<double> matrix2,
           int const kdeg, int const lev, int const isLeft, int const isTrans,
           int const method);
extern template fk::matrix<float>
apply_fmwt(fk::matrix<float> matrix1, fk::matrix<float> matrix2, int const kdeg,
           int const lev, int const isLeft, int const isTrans,
           int const method);
