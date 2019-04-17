#pragma once
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
             int const stride);
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
// and other blas information
template<typename P>
void batchedGemm(batch_list<P> const a, batch_list<P> const b,
                 batch_list<P> const c, P alpha, P beta, bool trans_a,
                 bool trans_b);

extern template class batch_list<float>;
extern template class batch_list<double>;

extern template void batchedGemm(batch_list<float> const a,
                                 batch_list<float> const b,
                                 batch_list<float> const c, float alpha,
                                 float beta, bool trans_a, bool trans_b);
extern template void batchedGemm(batch_list<double> const a,
                                 batch_list<double> const b,
                                 batch_list<double> const c, double alpha,
                                 double beta, bool trans_a, bool trans_b);
