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
             int const stride, bool const do_trans, P const scale);
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
  P const scale;       // alpha/beta for BLAS call

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
                  batch_list<P> const c);

// compute how many gemms a single call to batch_for_kronmult
// adds for every dimension
template<int num_dims>
std::array<int, num_dims> compute_batch_sizes(int const degree)
{
  assert(num_dims > 0);
  std::array<int, num_dims> sizes;

  // lowest level; 1 gemm
  sizes[0] = 1;
  if (num_dims == 1)
  {
    return sizes;
  }

  // intermediate levels
  for (int i = 1; i < num_dims - 1; ++i)
  {
    sizes[i] = std::pow(degree, num_dims) / std::pow(degree, i);
  }

  // highest level; 1 gemm
  sizes[num_dims - 1] = 1;
}

template<typename P>
using batch_set = std::array<batch_list<P>, 3>;

// helper for lowest level of kronmult
template<typename P, int num_dims>
static void
kron_base(fk::matrix<P, mem_type::view> const A,
          fk::vector<P, mem_type::view> x, fk::matrix<P, mem_type::view> y,
          batch_set<P> batch_lists, int const batch_offset, int const degree)
{
  batch_lists[0].insert(A, batch_offset);
  int const nrows = degree;
  int const ncols = std::pow(degree, num_dims - 1);
  fk::matrix<P, mem_type::view> x_view(x, 0, nrows, ncols);
  batch_lists[1].insert(x_view, batch_offset);
  fk::matrix<P, mem_type::view> y_view(y, 0, nrows, ncols);
  batch_lists[2].insert(y_view, batch_offset);
}

// given num_dims many square matrices of size degree,
// and a vector x of size degree^num_dims, and an output
// vector y of the same size,
// enqueue the parameters for the batched gemm operations
// that will perform the multiplication A*x=y, where
// A is the tensor product of the input matrices.
//
// i.e., enqueue small gemms to perform A*x=y, where A is tensor encoded
// and not explicitly formed.
//
// work array is the workspace for intermediate products for the gemms.
// each element should be degree^num_dims in size.

template<typename P, int num_dims>
void batch_for_kronmult(
    std::array<fk::matrix<P, mem_type::view>, num_dims> const A,
    fk::vector<P, mem_type::view> x, fk::vector<P, mem_type::view> y,
    std::array<fk::vector<P, mem_type::view>, std::max(num_dims - 1, 0)> const
        work,
    std::array<batch_set<P>, num_dims> batch_lists,
    std::array<int, num_dims> const batch_offsets, int const degree)
{
  // check for valid inputs
  assert(num_dims > 0);
  assert(degree > 0);

  // check vector sizes
  int const result_size = std::pow(degree, num_dims);
  assert(x.size() == result_size);
  assert(y.size() == result_size);

  // check workspace sizes
  for (int i = 0; i < work.size(); ++i)
  {
    assert(work[i].size() == result_size);
  }

  // check matrix sizes
  for (int i = 0; i < A.size(); ++i)
  {
    assert(A[i].nrows() == degree);
    assert(A[i].ncols() == degree);
  }
}

extern template class batch_list<float>;
extern template class batch_list<double>;

extern template void batched_gemm(batch_list<float> const a,
                                  batch_list<float> const b,
                                  batch_list<float> const c);
extern template void batched_gemm(batch_list<double> const a,
                                  batch_list<double> const b,
                                  batch_list<double> const c);
