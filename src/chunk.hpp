#pragma once

#include "distribution.hpp"
#include "element_table.hpp"
#include "tensors.hpp"

using element_chunk = std::map<int, grid_limits>;

template<typename P>
double get_MB(uint64_t const num_elems)
{
  assert(num_elems > 0);
  double const bytes = num_elems * sizeof(P);
  double const MB    = bytes * 1e-6;
  return MB;
}

// convenience functions when working with element chunks
int num_elements_in_chunk(element_chunk const &g);
int max_connected_in_chunk(element_chunk const &g);

grid_limits columns_in_chunk(element_chunk const &g);
grid_limits rows_in_chunk(element_chunk const &g);

// functions to assign chunks
template<typename P>
int get_num_chunks(element_subgrid const &grid, PDE<P> const &pde,
                   int const rank_size_MB = 1000);

std::vector<element_chunk>
assign_elements(element_subgrid const &grid, int const num_chunks);

// reduce an element chunk's results after batched gemm
template<typename P, resource resrc>
fk::vector<P, mem_type::owner, resrc> const &
reduce_chunk(PDE<P> const &pde,
             fk::vector<P, mem_type::owner, resrc> const &reduction_space,
             fk::vector<P, mem_type::owner, resrc> &output,
             fk::vector<P, mem_type::owner, resrc> const &unit_vector,
             element_subgrid const &subgrid, element_chunk const &chunk);
