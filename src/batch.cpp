#include "batch.hpp"
#include "tensors.hpp" // for views/blas

template<typename P>
batch_list<P>::batch_list(int const num_batch, int const nrows, int const ncols,
                          int const stride, bool const do_trans, P const scale)
    : num_batch(num_batch), nrows(nrows), ncols(ncols), stride(stride),
      do_trans(do_trans), scale(scale), batch_list_{new P *[num_batch]()}
{
  assert(num_batch > 0);
  assert(nrows > 0);
  assert(ncols > 0);
  assert(stride >= nrows);

  for (P *&ptr : (*this))
  {
    ptr = nullptr;
  }
}

template<typename P>
batch_list<P>::batch_list(batch_list<P> const &other)
    : num_batch(other.num_batch), nrows(other.nrows), ncols(other.ncols),
      stride(other.stride), do_trans(other.do_trans),
      scale(other.scale), batch_list_{new P *[other.num_batch]()}
{
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
}

template<typename P>
batch_list<P> &batch_list<P>::operator=(batch_list<P> const &other)
{
  if (&other == this)
  {
    return *this;
  }
  assert(num_batch == other.num_batch);
  assert(nrows == other.nrows);
  assert(ncols == other.ncols);
  assert(stride == other.stride);
  assert(do_trans == other.do_trans);
  assert(std::abs(scale - other.scale) <= TOL);
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
  return *this;
}

template<typename P>
batch_list<P>::batch_list(batch_list<P> &&other)
    : num_batch(other.num_batch), nrows(other.nrows), ncols(other.ncols),
      stride(other.stride), do_trans(other.do_trans),
      scale(other.scale), batch_list_{other.batch_list_}
{
  other.batch_list_ = nullptr;
}

template<typename P>
batch_list<P> &batch_list<P>::operator=(batch_list<P> &&other)
{
  if (&other == this)
  {
    return *this;
  }
  assert(num_batch == other.num_batch);
  assert(nrows == other.nrows);
  assert(ncols == other.ncols);
  assert(stride == other.stride);

  assert(std::abs(scale - other.scale) <= TOL);
  assert(do_trans == other.do_trans);
  batch_list_       = other.batch_list_;
  other.batch_list_ = nullptr;
  return *this;
}

template<typename P>
batch_list<P>::~batch_list()
{
  delete[] batch_list_;
}

template<typename P>
bool batch_list<P>::operator==(batch_list<P> other) const
{
  if (nrows != other.nrows)
  {
    return false;
  }
  if (ncols != other.ncols)
  {
    return false;
  }
  if (stride != other.stride)
  {
    return false;
  }
  if (num_batch != other.num_batch)
  {
    return false;
  }
  if (do_trans != other.do_trans)
  {
    return false;
  }
  if (std::abs(scale - other.scale) > TOL)
  {
    return false;
  }

  for (int i = 0; i < num_batch; ++i)
  {
    if (batch_list_[i] != other.batch_list_[i])
    {
      return false;
    }
  }

  return true;
}

template<typename P>
P *batch_list<P>::operator()(int const position) const
{
  assert(position >= 0);
  assert(position < num_batch);
  return batch_list_[position];
}

// insert the provided view's data pointer
// at the index indicated by position argument
// cannot overwrite previous assignment
template<typename P>
void batch_list<P>::insert(fk::matrix<P, mem_type::view> const a,
                           int const position)
{
  // make sure this matrix is the
  // same dimensions as others in batch
  assert(a.nrows() == nrows);
  assert(a.ncols() == ncols);
  assert(a.stride() == stride);

  // ensure position is valid
  assert(position >= 0);
  assert(position < num_batch);

  // ensure nothing already assigned
  assert(!batch_list_[position]);

  batch_list_[position] = a.data();
}

// clear one assignment
// returns true if there was a previous assignment,
// false if nothing was assigned
template<typename P>
bool batch_list<P>::clear(int const position)
{
  P *temp               = batch_list_[position];
  batch_list_[position] = nullptr;
  return temp;
}

// get a pointer to the batch_list's
// pointers for batched blas call
// for performance, may have to
// provide a direct access to P**
// from batch_list_, but avoid for now
template<typename P>
P **batch_list<P>::get_list() const
{
  P **const list_copy = new P *[num_batch]();
  std::memcpy(list_copy, batch_list_, num_batch * sizeof(P *));
  return list_copy;
}

// verify that every allocated pointer
// has been assigned to
template<typename P>
bool batch_list<P>::is_filled() const
{
  for (P *const ptr : (*this))
  {
    if (!ptr)
    {
      return false;
    }
  }
  return true;
}

// clear assignments
template<typename P>
batch_list<P> &batch_list<P>::clear_all()
{
  for (P *&ptr : (*this))
  {
    ptr = nullptr;
  }
  return *this;
}

// execute a batched gemm given a, b, c batch lists
// and other blas information
// if we store info in the batch about where it is
// resident, this could be an abstraction point
// for calling cpu/gpu blas etc.
template<typename P>
void batched_gemm(batch_list<P> const a, batch_list<P> const b,
                  batch_list<P> const c)
{
  // check data validity
  assert(a.is_filled() && b.is_filled() && c.is_filled());

  // check cardinality of sets
  assert(a.num_batch == b.num_batch);
  assert(b.num_batch == c.num_batch);
  int const num_batch = a.num_batch;

  assert(!c.do_trans);

  P const alpha = a.scale;
  P const beta  = c.scale;
  assert(b.scale == 1.0);

  // check dimensions for gemm
  int const rows_a = a.do_trans ? a.ncols : a.nrows;
  int const cols_a = a.do_trans ? a.nrows : a.ncols;
  int const rows_b = b.do_trans ? b.ncols : b.nrows;
  int const cols_b = b.do_trans ? b.nrows : b.ncols;
  assert(cols_a == rows_b);
  assert(c.nrows == rows_a);
  assert(c.ncols == cols_b);

  // setup blas args
  int m                  = rows_a;
  int n                  = cols_b;
  int k                  = cols_a;
  int lda                = a.stride;
  int ldb                = b.stride;
  int ldc                = c.stride;
  char const transpose_a = a.do_trans ? 't' : 'n';
  char const transpose_b = b.do_trans ? 't' : 'n';
  P alpha_               = alpha;
  P beta_                = beta;

  if constexpr (std::is_same<P, double>::value)
  {
    for (int i = 0; i < num_batch; ++i)
    {
      fk::dgemm_(&transpose_a, &transpose_b, &m, &n, &k, &alpha_, a(i), &lda,
                 b(i), &ldb, &beta_, c(i), &ldc);
    }
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    for (int i = 0; i < num_batch; ++i)
    {
      fk::sgemm_(&transpose_a, &transpose_b, &m, &n, &k, &alpha_, a(i), &lda,
                 b(i), &ldb, &beta_, c(i), &ldc);
    }
  }
}

template class batch_list<float>;
template class batch_list<double>;

template void batched_gemm(batch_list<float> const a, batch_list<float> const b,
                           batch_list<float> const c);

template void batched_gemm(batch_list<double> const a,
                           batch_list<double> const b,
                           batch_list<double> const c);
