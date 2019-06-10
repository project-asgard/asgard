#pragma once

#include "blas_wrapped.hpp"
#include "tensors.hpp"

// axpy - add the argument vector scaled by alpha
template<typename P, mem_type mem, mem_type omem>
fk::vector<P, mem> &
axpy(P const alpha, fk::vector<P, omem> const &x, fk::vector<P, mem> &y);

// copy(x,y) - copy vector x into y
template<typename P, mem_type mem, mem_type omem>
fk::vector<P, mem> &copy(fk::vector<P, omem> const &x, fk::vector<P, mem> &y);

// scal - scale a vector
template<typename P, mem_type mem>
fk::vector<P, mem> &scal(P const alpha, fk::vector<P, mem> &x);
