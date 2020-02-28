#pragma once

#include "chunk.hpp"
#include "element_table.hpp"
#include "pde/pde_base.hpp"
#include "tensors.hpp"
#include <array>
#include <numeric>

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
  batch(int const capacity, int const nrows, int const ncols, int const stride,
        bool const do_trans);
  batch(batch<P, resrc> const &other);
  batch &operator=(batch<P, resrc> const &other);
  batch(batch<P, resrc> &&other);
  batch &operator=(batch<P, resrc> &&other);
  ~batch();

  bool operator==(batch<P, resrc> const &) const;
  P *operator()(int const) const;

  template<mem_type mem>
  void assign_entry(fk::matrix<P, mem, resrc> const &a, int const position);
  void assign_raw(P *const a, int const position);
  bool clear_entry(int const position);

  P **get_list() const;

  bool is_filled() const;
  batch &clear_all();

  int get_capacity() const { return capacity_; }

  int num_entries() const { return num_entries_; }
  batch &set_num_entries(int const new_num_entries)
  {
    assert(new_num_entries > 0);
    assert(new_num_entries <= get_capacity());
    num_entries_ = new_num_entries;
    return *this;
  }
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
  int const capacity_; // number of matrices/vectors this batch can hold
  int num_entries_;    // number of matrices/vectors for this chunk
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

/*

Problem relevant to class batch_chain:

given a vector "x" of length "x_size" and list of matrices of arbitrary
dimension in "matrix": { m0, m1, ... , m_last }, calculate ( m0 kron m1 kron ...
kron m_end ) * x

*/

template<typename P, resource resrc>
class batch_chain
{
public:
  /* allocates batches and assigns data */
  batch_chain(
      std::vector<fk::matrix<P, mem_type::const_view, resrc>> const &matrices,
      fk::vector<P, mem_type::const_view, resrc> const &x,
      std::array<fk::vector<P, mem_type::view, resrc>, 2> &workspace,
      fk::vector<P, mem_type::view, resrc> &final_output);

  void execute_batch_chain();

private:
  std::vector<batch<P, resrc>> left;

  std::vector<batch<P, resrc>> right;

  std::vector<batch<P, resrc>> product;
};

/* Calculates necessary workspace length for the Kron algorithm. See .cpp file
 * for more details */
template<typename P, resource resrc>
int calculate_workspace_length(
    std::vector<fk::matrix<P, mem_type::const_view, resrc>> const &matrices,
    int const x_size);

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
    std::vector<fk::matrix<P, mem_type::const_view, resource::device>> const &A,
    fk::vector<P, mem_type::const_view, resource::device> const &x,
    fk::vector<P, mem_type::const_view, resource::device> const &y,
    std::vector<fk::vector<P, mem_type::const_view, resource::device>> const
        &work,
    std::vector<batch_operands_set<P>> &batches, int const batch_offset,
    PDE<P> const &pde);

// unsafe version of kronmult to batch sets function
// conceptually performs the same operations, but works
// with raw pointers rather than views.

// we chose to implement this because the reference counting
// in the view class incurs a prohibitive runtime cost when used
// to batch millions (or billions) of GEMMs, e.g. for a 6d problem
template<typename P>
void unsafe_kronmult_to_batch_sets(std::vector<P *> const &A, P *const x,
                                   P *const y, std::vector<P *> const &work,
                                   std::vector<batch_operands_set<P>> &batches,
                                   int const batch_offset, PDE<P> const &pde);
template<typename P>
void build_batches(PDE<P> const &pde, element_table const &elem_table,
                   rank_workspace<P> const &workspace,
                   element_subgrid const &subgrid, element_chunk const &chunk,
                   std::vector<batch_operands_set<P>> &batches);

template<typename P>
void build_system_matrix(PDE<P> const &pde, element_table const &elem_table,
                         element_chunk const &chunk, fk::matrix<P> &A);

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
    std::vector<fk::matrix<float, mem_type::const_view, resource::device>> const
        &A,
    fk::vector<float, mem_type::const_view, resource::device> const &x,
    fk::vector<float, mem_type::const_view, resource::device> const &y,
    std::vector<fk::vector<float, mem_type::const_view, resource::device>> const
        &work,
    std::vector<batch_operands_set<float>> &batches, int const batch_offset,
    PDE<float> const &pde);

extern template void kronmult_to_batch_sets(
    std::vector<
        fk::matrix<double, mem_type::const_view, resource::device>> const &A,
    fk::vector<double, mem_type::const_view, resource::device> const &x,
    fk::vector<double, mem_type::const_view, resource::device> const &y,
    std::vector<
        fk::vector<double, mem_type::const_view, resource::device>> const &work,
    std::vector<batch_operands_set<double>> &batches, int const batch_offset,
    PDE<double> const &pde);

extern template void
unsafe_kronmult_to_batch_sets(std::vector<float *> const &A, float *const x,
                              float *const y, std::vector<float *> const &work,
                              std::vector<batch_operands_set<float>> &batches,
                              int const batch_offset, PDE<float> const &pde);

extern template void
unsafe_kronmult_to_batch_sets(std::vector<double *> const &A, double *const x,
                              double *const y,
                              std::vector<double *> const &work,
                              std::vector<batch_operands_set<double>> &batches,
                              int const batch_offset, PDE<double> const &pde);

extern template void
build_batches(PDE<float> const &pde, element_table const &elem_table,
              rank_workspace<float> const &workspace,
              element_subgrid const &subgrid, element_chunk const &chunk,
              std::vector<batch_operands_set<float>> &);
extern template void
build_batches(PDE<double> const &pde, element_table const &elem_table,
              rank_workspace<double> const &workspace,
              element_subgrid const &subgrid, element_chunk const &chunk,
              std::vector<batch_operands_set<double>> &);

extern template void
build_system_matrix(PDE<double> const &pde, element_table const &elem_table,
                    element_chunk const &chunk, fk::matrix<double> &A);
extern template void
build_system_matrix(PDE<float> const &pde, element_table const &elem_table,
                    element_chunk const &chunk, fk::matrix<float> &A);

extern template class batch_chain<double, resource::device>;
extern template class batch_chain<double, resource::host>;
extern template class batch_chain<float, resource::device>;
extern template class batch_chain<float, resource::host>;
/*
extern template int calculate_workspace_length(
    std::vector<fk::matrix<double, mem_type::view, resource::device>> const
        &matrices,
    int const x_size);

extern template int calculate_workspace_length(
    std::vector<fk::matrix<double, mem_type::view, resource::host>> const
        &matrices,
    int const x_size);

extern template int calculate_workspace_length(
    std::vector<fk::matrix<float, mem_type::view, resource::device>> const
        &matrices,
    int const x_size);

extern template int calculate_workspace_length(
    std::vector<fk::matrix<float, mem_type::view, resource::host>> const
        &matrices,
    int const x_size);
*/
