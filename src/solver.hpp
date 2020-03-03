#pragma once

#include "batch.hpp"
#include "chunk.hpp"
#include "distribution.hpp"
#include "pde.hpp"
#include "tensors.hpp"

namespace solver
{
// perform GMRES using apply_A for tensor encoded matrix*vector
// returns the residual
// approximate solution stored in host_space.x
template<typename P>
P gmres(PDE<P> const &pde, element_table const &elem_table,
        distribution_plan const &plan, std::vector<element_chunk> const &chunks,
        host_workspace<P> &host_space, rank_workspace<P> &rank_space,
        std::vector<batch_operands_set<P>> &batches, P const dt,
        P const threshold, int const restart);

// simple, node-local test version
template<typename P>
P simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
               fk::matrix<P> const &M, int const restart, int const max_iter,
               P const tolerance);
} // namespace solver
