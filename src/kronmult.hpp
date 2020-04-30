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
template<typename P>
fk::vector<P, mem_type::owner, resource::host>
execute(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid,
        fk::vector<P, mem_type::owner, resource::host> const &x);
}
