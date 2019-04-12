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
  batch_list(batch_list<P> const &&other);
  batch_list &operator=(batch_list<P> const &&other);

  ~batch_list();

  void insert(fk::matrix<P, mem_type::view> const a, int position);

  int const num_batch; // number of matrices in the batch
  int const nrows;     // number of rows in matrices in this batch
  int const ncols;     // number of cols in matrices in this batch
  int const stride;    // leading dimension passed into BLAS call

private:
  P **batch_list_; // array of pointers to pass into blas call
};

template<typename P>
batch_list<P>::batch_list(int const num_batch, int const nrows, int const ncols,
                          int const stride)
    : num_batch(num_batch), nrows(nrows),
      ncols(ncols), batch_list_{new P *[num_batch]()}
{}

template<typename P>
batch_list<P>::batch_list(batch_list<P> const &other)
    : num_batch(other.num_batch), nrows(other.nrows),
      ncols(other.ncols), batch_list_{new P *[other.num_batch()]}
{
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
}

template<typename P>
batch_list<P> &batch_list<P>::operator=(batch_list<P> const &other)
{
  assert(num_batch == other.num_batch);
  assert(nrows == other.nrows);
  assert(ncols == other.ncols);
  assert(stride == other.stride);
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
}

template<typename P>
batch_list<P>::batch_list(batch_list<P> const &&other)
    : num_batch(other.num_batch), nrows(other.nrows),
      ncols(other.ncols), batch_list_{other.batch_list_}
{
  other.batch_list_ = nullptr;
}

template<typename P>
batch_list<P> &batch_list<P>::operator=(batch_list<P> const &&other)
{
  assert(num_batch == other.num_batch);
  assert(nrows == other.nrows);
  assert(ncols == other.ncols);
  assert(stride == other.stride);

  batch_list_       = other.batch_list_;
  other.batch_list_ = nullptr;
}

template<typename P>
batch_list<P>::~batch_list()
{
  delete[] batch_list_;
}

template<typename P>
void batch_list<P>::insert(fk::matrix<P, mem_type::view> const a, int position)
{
  // make sure this matrix is the
  // same dimensions as others in batch
  assert(a.nrows() == nrows);
  assert(a.ncols() == ncols);
  assert(a.stride() == stride);

  // ensure position is valid
  assert(position >= 0);
  assert(position < num_batch);

  batch_list_[position] = a.data();
}
