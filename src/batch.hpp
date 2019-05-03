#pragma once

#include "element_table.hpp"
#include "pde/pde_base.hpp"
#include "tensors.hpp"
#include <array>

// wrapper around an array of pointers to matrices for
// a call to batch gemm; i.e., the class represents the
// information for a batch gemm operand
template<typename P>
class batch_list
{
public:
  batch_list(int const num_batch, int const nrows, int const ncols,
             int const stride, bool const do_trans);
  batch_list(batch_list<P> const &other);
  batch_list &operator=(batch_list<P> const &other);
  batch_list(batch_list<P> &&other);
  batch_list &operator=(batch_list<P> &&other);
  ~batch_list();

  bool operator==(batch_list<P>) const;
  P *operator()(int const) const;

  void insert(fk::matrix<P, mem_type::view> const a, int const position);
  bool clear(int const position);

  P **get_list() const;

  bool is_filled() const;
  batch_list &clear_all();

  int const num_batch; // number of matrices in the batch
  int const nrows;     // number of rows in matrices in this batch
  int const ncols;     // number of cols in matrices in this batch
  int const stride;    // leading dimension passed into BLAS call
  bool const do_trans; // transpose passed into BLAS call

  using const_iterator = P *const *;
  const_iterator begin() const { return batch_list_; }
  const_iterator end() const { return batch_list_ + num_batch; }

private:
  P **batch_list_; // array of pointers to pass into blas call

  // want these for convenience in the class
  // don't want to expose them publicly...
  using iterator = P **;
  iterator begin() { return batch_list_; }
  iterator end() { return batch_list_ + num_batch; }
};

// execute a batched gemm given a, b, c batch lists
template<typename P>
void batched_gemm(batch_list<P> const a, batch_list<P> const b,
                  batch_list<P> const c, P const alpha, P const beta);

struct gemm_dims
{
  int const rows_a;
  int const cols_a;
  int const rows_b;
  int const cols_b;
  gemm_dims(int const rows_a, int const cols_a, int const rows_b,
            int const cols_b)
      : rows_a(rows_a), cols_a(cols_a), rows_b(rows_b), cols_b(cols_b){};
};

template<typename P>
using batch_set = std::vector<batch_list<P>>;

// create empty batches w/ correct dims and settings
// for batching
template<typename P>
std::vector<batch_set<P>>
allocate_batches(PDE<P> const &pde, int const num_elems);

// given num_dims many square matrices of size degree,
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

template<typename P>
void batch_for_kronmult(std::vector<fk::matrix<P, mem_type::view>> const A,
                        fk::vector<P, mem_type::view> x,
                        fk::vector<P, mem_type::view> y,
                        std::vector<fk::vector<P, mem_type::view>> const work,
                        std::vector<batch_set<P>> &batch_lists,
                        int const batch_offsets, PDE<P> const &pde);

// use info from pde and element table to create and populate the batch lists
template<typename P>
std::vector<batch_set<P>>
build_batches(PDE<P> const &pde, element_table const &elem_table,
              fk::vector<P> const &x, fk::vector<P> const &y,
              fk::vector<P> const &work);

extern template class batch_list<float>;
extern template class batch_list<double>;

extern template void
batched_gemm(batch_list<float> const a, batch_list<float> const b,
             batch_list<float> const c, float const alpha, float const beta);
extern template void
batched_gemm(batch_list<double> const a, batch_list<double> const b,
             batch_list<double> const c, double const alpha, double const beta);

extern template std::vector<batch_set<float>>
allocate_batches(PDE<float> const &pde, int const num_elems);
extern template std::vector<batch_set<double>>
allocate_batches(PDE<double> const &pde, int const num_elems);

extern template void
batch_for_kronmult(std::vector<fk::matrix<float, mem_type::view>> const A,
                   fk::vector<float, mem_type::view> x,
                   fk::vector<float, mem_type::view> y,
                   std::vector<fk::vector<float, mem_type::view>> const work,
                   std::vector<batch_set<float>> &batch_lists,
                   int const batch_offset, PDE<float> const &pde);

extern template void
batch_for_kronmult(std::vector<fk::matrix<double, mem_type::view>> const A,
                   fk::vector<double, mem_type::view> x,
                   fk::vector<double, mem_type::view> y,
                   std::vector<fk::vector<double, mem_type::view>> const work,
                   std::vector<batch_set<double>> &batch_lists,
                   int const batch_offset, PDE<double> const &pde);

extern template std::vector<batch_set<float>>
build_batches(PDE<float> const &pde, element_table const &elem_table,
              fk::vector<float> const &x, fk::vector<float> const &y,
              fk::vector<float> const &work);
extern template std::vector<batch_set<double>>
build_batches(PDE<double> const &pde, element_table const &elem_table,
              fk::vector<double> const &x, fk::vector<double> const &y,
              fk::vector<double> const &work);
