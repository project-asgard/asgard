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
std::array<fk::matrix<P>, 1> apply_fmwt(fk::matrix<P> matrix1,fk::matrix<P> matrix2)
{

  fk::matrix<P> product = matrix1*matrix2;

  return std::array<fk::matrix<P>, 1>{product};
}

