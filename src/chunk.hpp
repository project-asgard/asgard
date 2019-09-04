#pragma once
#include "element_table.hpp"
#include "pde.hpp"
#include "tensors.hpp"

struct limits
{
  limits(int const start, int const stop) : start(start), stop(stop){};
  limits(limits const &l) : start(l.start), stop(l.stop){};
  limits(limits const &&l) : start(l.start), stop(l.stop){};
  bool operator==(const limits &rhs) const
  {
    return start == rhs.start && stop == rhs.stop;
  }
  int const start;
  int const stop;
};

using element_chunk = std::map<int, limits>;

// convenience functions when working with element chunks
int num_elements_in_chunk(element_chunk const &g);
int max_connected_in_chunk(element_chunk const &g);

limits columns_in_chunk(element_chunk const &g);
limits rows_in_chunk(element_chunk const &g);

// FIXME we should eventually put this in the pde class?
auto const element_segment_size = [](auto const &pde) {
  int const degree = pde.get_dimensions()[0].get_degree();
  return static_cast<int>(std::pow(degree, pde.num_dims));
};

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

// larger, host-side memory space holding the entire input/output vectors.
// FIXME when we split the problem with MPI, we can restrict this to
// only the portions of x and y needed for a given rank's
// assigned element chunks.
template<typename P, typename T>
class host_workspace
{
public:
  host_workspace(PDE<P> const &pde, element_table<T> const &table);
  // working vectors for time advance (e.g. intermediate RK result vects,
  // source vector space)
  fk::vector<P> scaled_source;
  fk::vector<P> x_orig;
  fk::vector<P> x;
  fk::vector<P> fx;
  fk::vector<P> result_1;
  fk::vector<P> result_2;
  fk::vector<P> result_3;

  double size_MB() const
  {
    int64_t num_elems = scaled_source.size() + x_orig.size() + fx.size() +
                        x.size() + result_1.size() + result_2.size() +
                        result_3.size();
    double const bytes     = static_cast<double>(num_elems) * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };
};

// functions to assign chunks
template<typename P, typename T>
int get_num_chunks(element_table<T> const &table, PDE<P> const &pde,
                   int const num_ranks = 1, int const rank_size_MB = 1000);

template<typename T>
std::vector<element_chunk>
assign_elements(element_table<T> const &table, int const num_chunks);

// data management functions
template<typename P, typename T>
void copy_chunk_inputs(PDE<P> const &pde, rank_workspace<P> &rank_space,
                       host_workspace<P, T> const &host_space,
                       element_chunk const &chunk);

template<typename P, typename T>
void copy_chunk_outputs(PDE<P> const &pde, rank_workspace<P> &rank_space,
                        host_workspace<P, T> const &host_space,
                        element_chunk const &chunk);

// reduce an element chunk's results after batched gemm
template<typename P>
void reduce_chunk(PDE<P> const &pde, rank_workspace<P> &rank_space,
                  element_chunk const &chunk);

extern template int get_num_chunks(element_table<int> const &table,
                                   PDE<float> const &pde, int const num_ranks,
                                   int const rank_size_MB);
extern template int get_num_chunks(element_table<long int> const &table,
                                   PDE<float> const &pde, int const num_ranks,
                                   int const rank_size_MB);
extern template int get_num_chunks(element_table<int> const &table,
                                   PDE<double> const &pde, int const num_ranks,
                                   int const rank_size_MB);
extern template int get_num_chunks(element_table<long int> const &table,
                                   PDE<double> const &pde, int const num_ranks,
                                   int const rank_size_MB);

extern template void
copy_chunk_inputs(PDE<float> const &pde, rank_workspace<float> &rank_space,
                  host_workspace<float, int> const &host_space,
                  element_chunk const &chunk);
extern template void
copy_chunk_inputs(PDE<float> const &pde, rank_workspace<float> &rank_space,
                  host_workspace<float, long int> const &host_space,
                  element_chunk const &chunk);

extern template void
copy_chunk_inputs(PDE<double> const &pde, rank_workspace<double> &rank_space,
                  host_workspace<double, int> const &host_space,
                  element_chunk const &chunk);
extern template void
copy_chunk_inputs(PDE<double> const &pde, rank_workspace<double> &rank_space,
                  host_workspace<double, long int> const &host_space,
                  element_chunk const &chunk);

extern template void
copy_chunk_outputs(PDE<float> const &pde, rank_workspace<float> &rank_space,
                   host_workspace<float, int> const &host_space,
                   element_chunk const &chunk);
extern template void
copy_chunk_outputs(PDE<float> const &pde, rank_workspace<float> &rank_space,
                   host_workspace<float, long int> const &host_space,
                   element_chunk const &chunk);

extern template void
copy_chunk_outputs(PDE<double> const &pde, rank_workspace<double> &rank_space,
                   host_workspace<double, int> const &host_space,
                   element_chunk const &chunk);
extern template void
copy_chunk_outputs(PDE<double> const &pde, rank_workspace<double> &rank_space,
                   host_workspace<double, long int> const &host_space,
                   element_chunk const &chunk);

extern template void reduce_chunk(PDE<float> const &pde,
                                  rank_workspace<float> &rank_space,
                                  element_chunk const &chunk);

extern template void reduce_chunk(PDE<double> const &pde,
                                  rank_workspace<double> &rank_space,
                                  element_chunk const &chunk);

extern template std::vector<element_chunk>
assign_elements(element_table<int> const &table, int const num_chunks);
extern template std::vector<element_chunk>
assign_elements(element_table<long int> const &table, int const num_chunks);
