#include "batch.hpp"

template<typename P>
batch_list<P>::batch_list(int const num_batch, int const nrows, int const ncols,
                          int const stride)
    : num_batch(num_batch), nrows(nrows), ncols(ncols),
      stride(stride), batch_list_{new P *[num_batch]()}
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
      stride(other.stride), batch_list_{new P *[other.num_batch]()}
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
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
  return *this;
}

template<typename P>
batch_list<P>::batch_list(batch_list<P> &&other)
    : num_batch(other.num_batch), nrows(other.nrows), ncols(other.ncols),
      stride(other.stride), batch_list_{other.batch_list_}
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

  for (int i = 0; i < num_batch; ++i)
  {
    if (batch_list_[i] != other.batch_list_[i])
    {
      return false;
    }
  }

  return true;
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

template class batch_list<float>;
template class batch_list<double>;
