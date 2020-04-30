#pragma once
#include "tensors.hpp"

namespace solver
{
// simple, node-local test version of gmres
template<typename P>
P simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
               fk::matrix<P> const &M, int const restart, int const max_iter,
               P const tolerance);
} // namespace solver
