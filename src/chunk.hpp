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

// workspace for the primary computation in time advance. along with
// the coefficient matrices, we need this space resident on whatever
// accelerator we are using
template<typename P>
class rank_workspace
{
public:
  rank_workspace(PDE<P> const &pde, std::vector<element_chunk> const &chunks);
  fk::vector<P, mem_type::owner, resource::device> const &
  get_unit_vector() const;

  // input, output, workspace for batched gemm/reduction
  fk::vector<P, mem_type::owner, resource::device> batch_input;
  fk::vector<P, mem_type::owner, resource::device> reduction_space;
  fk::vector<P, mem_type::owner, resource::device> batch_intermediate;
  fk::vector<P, mem_type::owner, resource::device> batch_output;
  double size_MB() const
  {
    int64_t num_elems = batch_input.size() + reduction_space.size() +
                        batch_intermediate.size() + batch_output.size() +
                        unit_vector_.size();

    double const bytes     = static_cast<double>(num_elems) * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

private:
  fk::vector<P, mem_type::owner, resource::device> unit_vector_;
};

// larger, host-side memory space holding the assigned portion of input/output
// vectors.
template<typename P>
class host_workspace
{
public:
  host_workspace(PDE<P> const &pde, element_subgrid const &grid);
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

// data management functions
// FIXME move...to distro? idk.
template<typename P>
void copy_grid_inputs(PDE<P> const &pde, element_subgrid const &grid,
                      rank_workspace<P> &rank_space,
                      host_workspace<P> const &host_space);

template<typename P>
void copy_chunk_outputs(PDE<P> const &pde, element_subgrid const &grid,
                        rank_workspace<P> const &rank_space,
                        host_workspace<P> &host_space,
                        element_chunk const &chunk);

// reduce an element chunk's results after batched gemm
template<typename P>
void reduce_chunk(PDE<P> const &pde, rank_workspace<P> &rank_space,
                  element_chunk const &chunk);

extern template int get_num_chunks(element_subgrid const &grid,
                                   PDE<float> const &pde,
                                   int const rank_size_MB);
extern template int get_num_chunks(element_subgrid const &grid,
                                   PDE<double> const &pde,
                                   int const rank_size_MB);

extern template void copy_grid_inputs(PDE<float> const &pde,
                                      element_subgrid const &grid,
                                      rank_workspace<float> &rank_space,
                                      host_workspace<float> const &host_space);

extern template void copy_grid_inputs(PDE<double> const &pde,
                                      element_subgrid const &grid,
                                      rank_workspace<double> &rank_space,
                                      host_workspace<double> const &host_space);

extern template void copy_chunk_outputs(PDE<float> const &pde,
                                        element_subgrid const &grid,
                                        rank_workspace<float> const &rank_space,
                                        host_workspace<float> &host_space,
                                        element_chunk const &chunk);

extern template void
copy_chunk_outputs(PDE<double> const &pde, element_subgrid const &grid,
                   rank_workspace<double> const &rank_space,
                   host_workspace<double> &host_space,
                   element_chunk const &chunk);

extern template void reduce_chunk(PDE<float> const &pde,
                                  rank_workspace<float> &rank_space,
                                  element_chunk const &chunk);

extern template void reduce_chunk(PDE<double> const &pde,
                                  rank_workspace<double> &rank_space,
                                  element_chunk const &chunk);
