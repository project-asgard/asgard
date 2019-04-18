#include "apply_fmwt.hpp"

#include "connectivity.hpp"
#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

template<typename P>
fk::matrix<P> apply_fmwt(fk::matrix<P> matrix1,fk::matrix<P> matrix2)
{

  fk::matrix<P> product = matrix1*matrix2;

  return product;
}

template fk::matrix<double>
apply_fmwt(fk::matrix<double> matrix1,fk::matrix<double> matrix2);
template fk::matrix<float>
apply_fmwt(fk::matrix<float> matrix1,fk::matrix<float> matrix2);

