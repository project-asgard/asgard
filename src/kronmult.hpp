#pragma once
#ifdef ASGARD_USE_CUDA
#define USE_GPU
#endif
#include "distribution.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "tensors.hpp"

// this component is designed to interface with E. D'Azevedo's
// kronmult library (in contrib, built as dependency).

namespace asgard::kronmult
{
// execute one subgrid by breaking into smaller subgrids to
// fit workspace limit MB
template<typename P>
fk::vector<P, mem_type::owner, resource::host>
execute(PDE<P> const &pde, elements::table const &elem_table,
        options const &program_options, element_subgrid const &my_subgrid,
        fk::vector<P, mem_type::owner, resource::host> const &x,
        imex_flag const imex = imex_flag::unspecified);

} // namespace asgard::kronmult
