#pragma once
#ifdef ASGARD_USE_CUDA
#define USE_GPU
#endif
#include "distribution.hpp"
#include "element_table.hpp"
#include "pde.hpp"
#include "tensors.hpp"

// this component is designed to interface with E. D'Azevedo's
// kronmult library (in contrib, built as dependency).

namespace kronmult
{
// execute one subgrid by breaking into smaller subgrids to
// fit workspace limit MB
template<typename P>
fk::vector<P, mem_type::owner, resource::host>
execute(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid, int const workspace_size_MB,
        fk::vector<P, mem_type::owner, resource::host> const &x);

} // namespace kronmult
