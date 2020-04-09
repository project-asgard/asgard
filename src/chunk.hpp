#pragma once

#include "distribution.hpp"
#include "element_table.hpp"
#include "pde.hpp"
#include "tensors.hpp"

using element_chunk = std::map<int, grid_limits>;

// convenience functions when working with element chunks
int num_elements_in_chunk(element_chunk const &g);
int max_connected_in_chunk(element_chunk const &g);

grid_limits columns_in_chunk(element_chunk const &g);
grid_limits rows_in_chunk(element_chunk const &g);

// FIXME we should eventually put this in the pde class?
auto const element_segment_size = [](auto const &pde) {
  int const degree = pde.get_dimensions()[0].get_degree();
  return static_cast<int>(std::pow(degree, pde.num_dims));
};

template<typename P>
double get_MB(uint64_t const num_elems)
{
  assert(num_elems > 0);
  double const bytes = num_elems * sizeof(P);
  double const MB    = bytes * 1e-6;
  return MB;
}

// FIXME going away
// host-side memory space holding the assigned portion of input/output
// vectors.
template<typename P>
class host_workspace
{
public:
  host_workspace(PDE<P> const &pde, element_subgrid const &grid,
                 int const memory_limit_MB);
  // working vectors for time advance (e.g. intermediate RK result vects,
  // source vector space)
  fk::vector<P> scaled_source;
  fk::vector<P> x_orig;
  fk::vector<P> x;
  fk::vector<P> fx;
  fk::vector<P> reduced_fx;
  fk::vector<P> result_1;
  fk::vector<P> result_2;
  fk::vector<P> result_3;

  double size_MB() const
  {
    int64_t num_elems = scaled_source.size() + x_orig.size() + fx.size() +
                        reduced_fx.size() + x.size() + result_1.size() +
                        result_2.size() + result_3.size();
    double const bytes     = static_cast<double>(num_elems) * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };
};

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
