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
std::array<fk::matrix<P>, 1> apply_fmwt(fk::matrix<P> matrix1,fk::matrix<P> matrix2);

