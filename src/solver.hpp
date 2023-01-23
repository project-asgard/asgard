#pragma once

#include "batch.hpp"
#include "pde.hpp"
#include "tensors.hpp"

namespace asgard::solver
{
// simple, node-local test version of gmres
template<typename P>
P simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
               fk::matrix<P> const &M, int const restart, int const max_iter,
               P const tolerance);

template<typename P>
P simple_gmres(PDE<P> const &pde, elements::table const &elem_table,
               options const &program_options,
               element_subgrid const &my_subgrid, fk::vector<P> &x,
               fk::vector<P> const &b, fk::matrix<P> const &M,
               int const restart, int const max_iter, P const tolerance);

} // namespace asgard::solver
