#pragma once

#include "chunk.hpp"
#include "element_table.hpp"
#include "pde/pde_base.hpp"
#include "tensors.hpp"
#include <array>

// wrapper around an array of pointers to matrices or
// vectors for a call to batch gemm/gemv; i.e., the class
// represents the information for a batch operand
template<typename P,
         resource resrc =
             resource::device> // default to device - batch building functions
                               // only support this type for now
class batch
{
public:
  batch(int const num_entries, int const nrows, int const ncols,
        int const stride, bool const do_trans);
  batch(batch<P, resrc> const &other);
  batch &operator=(batch<P, resrc> const &other);
  batch(batch<P, resrc> &&other);
  batch &operator=(batch<P, resrc> &&other);
  ~batch();

  bool operator==(batch<P, resrc>) const;
  P *operator()(int const) const;

  void assign_entry(fk::matrix<P, mem_type::view, resrc> const a,
                    int const position);
  bool clear_entry(int const position);

  P **get_list() const;

  bool is_filled() const;
  batch &clear_all();

  int num_entries() const { return num_entries_; }
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  int get_stride() const { return stride_; }
  bool get_trans() const { return do_trans_; }

  // using P* const * because P*const *const because the
  // last const on non-class return would be ignored
  using const_iterator = P *const *;
  const_iterator begin() const { return batch_; }
  const_iterator end() const { return batch_ + num_entries(); }

private:
  int const num_entries_; // number of matrices/vectors in the batch
  int const nrows_;  // number of rows in matrices/size of vectors in this batch
  int const ncols_;  // number of cols in matrices (1 for vectors) in this batch
  int const stride_; // leading dimension passed into BLAS call for matrices;
                     // stride of vectors
  bool const do_trans_; // transpose passed into BLAS call for matrices

  P **batch_; // array of pointers to pass into blas call

  // want these for convenience in the class
  // don't want to expose them publicly...
  using iterator = P **;
  iterator begin() { return batch_; }
  iterator end() { return batch_ + num_entries(); }
};

// execute a batched gemm given a, b, c batch lists
template<typename P, resource resrc>
void batched_gemm(batch<P, resrc> const &a, batch<P, resrc> const &b,
                  batch<P, resrc> const &c, P const alpha, P const beta);

// execute a batched gemv given a, b, c batch lists
template<typename P, resource resrc>
void batched_gemv(batch<P, resrc> const &a, batch<P, resrc> const &b,
                  batch<P, resrc> const &c, P const alpha, P const beta);

// this could be named better
struct matrix_size_set
{
  int const rows_a;
  int const cols_a;
  int const rows_b;
  int const cols_b;
  matrix_size_set(int const rows_a, int const cols_a, int const rows_b,
                  int const cols_b)
      : rows_a(rows_a), cols_a(cols_a), rows_b(rows_b), cols_b(cols_b){};
};

// alias for a set of batch operands
// e.g., a, b, and c where a*b will be
// stored into c
template<typename P>
using batch_operands_set = std::vector<batch<P>>;

// create empty batches w/ correct dims and settings
// for batching
// num_elems is the total number of connected elements
// in the simulation; typically, size of element table squared
template<typename P>
std::vector<batch_operands_set<P>>
allocate_batches(PDE<P> const &pde, int const num_elems);

// given num_dims many square matrices of size degree by degree,
// and a vector x of size degree^num_dims, and an output
// vector y of the same size,
// enqueue the parameters for the batched gemm operations
// to perform the multiplication A*x=y, where
// A is the tensor product of the input matrices.
//
// i.e., enqueue small gemms to perform A*x=y,
// where A is tensor encoded and not explicitly formed.
//
// work array is the workspace for intermediate products for the gemms.
// each element should be degree^num_dims in size.
// the array must contain num_dims-1 such elements.
//
// the result of this function is that each a,b,c in each batch operand set,
// for each dimension, are assigned values for the small gemms that will
// do the arithmetic for a single connected element.
template<typename P>
void kronmult_to_batch_sets(
    std::vector<fk::matrix<P, mem_type::view, resource::device>> const A,
    fk::vector<P, mem_type::view, resource::device> x,
    fk::vector<P, mem_type::view, resource::device> y,
    std::vector<fk::vector<P, mem_type::view, resource::device>> const work,
    std::vector<batch_operands_set<P>> &batches, int const batch_offset,
    PDE<P> const &pde);

template<typename P>
std::vector<batch_operands_set<P>>
build_batches(PDE<P> const &pde, element_table const &elem_table,
              rank_workspace<P> const &workspace, element_chunk const &chunk);

template<typename P>
void build_implicit_system(PDE<P> const &pde, element_table const &elem_table,
                           element_chunk const &chunk, fk::matrix<P> &A);

template<typename P>
fk::matrix<P>
build_implicit_system(PDE<P> const &pde, element_table const &elem_table,
                      int const connected_start    = 0,
                      int const elements_per_batch = -1);

extern template class batch<float>;
extern template class batch<double>;
extern template class batch<float, resource::host>;
extern template class batch<double, resource::host>;

extern template void batched_gemm(batch<float> const &a, batch<float> const &b,
                                  batch<float> const &c, float const alpha,
                                  float const beta);
extern template void
batched_gemm(batch<double> const &a, batch<double> const &b,
             batch<double> const &c, double const alpha, double const beta);

extern template void batched_gemm(batch<float, resource::host> const &a,
                                  batch<float, resource::host> const &b,
                                  batch<float, resource::host> const &c,
                                  float const alpha, float const beta);
extern template void batched_gemm(batch<double, resource::host> const &a,
                                  batch<double, resource::host> const &b,
                                  batch<double, resource::host> const &c,
                                  double const alpha, double const beta);

extern template void batched_gemv(batch<float> const &a, batch<float> const &b,
                                  batch<float> const &c, float const alpha,
                                  float const beta);
extern template void
batched_gemv(batch<double> const &a, batch<double> const &b,
             batch<double> const &c, double const alpha, double const beta);

extern template void batched_gemv(batch<float, resource::host> const &a,
                                  batch<float, resource::host> const &b,
                                  batch<float, resource::host> const &c,
                                  float const alpha, float const beta);
extern template void batched_gemv(batch<double, resource::host> const &a,
                                  batch<double, resource::host> const &b,
                                  batch<double, resource::host> const &c,
                                  double const alpha, double const beta);

extern template std::vector<batch_operands_set<float>>
allocate_batches(PDE<float> const &pde, int const num_elems);
extern template std::vector<batch_operands_set<double>>
allocate_batches(PDE<double> const &pde, int const num_elems);

extern template void kronmult_to_batch_sets(
    std::vector<fk::matrix<float, mem_type::view, resource::device>> const A,
    fk::vector<float, mem_type::view, resource::device> x,
    fk::vector<float, mem_type::view, resource::device> y,
    std::vector<fk::vector<float, mem_type::view, resource::device>> const work,
    std::vector<batch_operands_set<float>> &batches, int const batch_offset,
    PDE<float> const &pde);

extern template void kronmult_to_batch_sets(
    std::vector<fk::matrix<double, mem_type::view, resource::device>> const A,
    fk::vector<double, mem_type::view, resource::device> x,
    fk::vector<double, mem_type::view, resource::device> y,
    std::vector<fk::vector<double, mem_type::view, resource::device>> const
        work,
    std::vector<batch_operands_set<double>> &batches, int const batch_offset,
    PDE<double> const &pde);

extern template std::vector<batch_operands_set<float>>
build_batches(PDE<float> const &pde, element_table const &elem_table,
              rank_workspace<float> const &workspace,
              element_chunk const &chunk);
extern template std::vector<batch_operands_set<double>>
build_batches(PDE<double> const &pde, element_table const &elem_table,
              rank_workspace<double> const &workspace,
              element_chunk const &chunk);

extern template fk::matrix<double>
build_implicit_system(PDE<double> const &pde, element_table const &elem_table,
                      int const connected_start    = 0,
                      int const elements_per_batch = -1);

extern template fk::matrix<float>
build_implicit_system(PDE<float> const &pde, element_table const &elem_table,
                      int const connected_start    = 0,
                      int const elements_per_batch = -1);

extern template void
build_implicit_system(PDE<double> const &pde, element_table const &elem_table,
                      element_chunk const &chunk, fk::matrix<double> &A);
extern template void
build_implicit_system(PDE<float> const &pde, element_table const &elem_table,
                      element_chunk const &chunk, fk::matrix<float> &A);
