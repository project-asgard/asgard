#pragma once

#include "lib_dispatch.hpp"
#include <memory>
#include <string>
#include <vector>

/* tolerance for answer comparisons */
#define TOL std::numeric_limits<P>::epsilon() * 2

// allows a private member function to declare via its parameter list who from
// outside the class is allowed to call it. you must hold an "access badge".
template<typename badge_holder>
class access_badge
{
  friend badge_holder;
  access_badge(){};
};

enum class mem_type
{
  owner,
  view
};

template<mem_type mem>
using enable_for_owner = std::enable_if_t<mem == mem_type::owner>;

template<mem_type mem>
using enable_for_view = std::enable_if_t<mem == mem_type::view>;

namespace fk
{
// forward declarations
template<typename P, mem_type mem = mem_type::owner> // default to be an owner
class vector;
template<typename P, mem_type mem = mem_type::owner>
class matrix;

template<typename P, mem_type mem>
class vector
{
  // all types of vectors are mutual friends
  template<typename, mem_type>
  friend class vector;

public:
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector();
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  explicit vector(int const size);
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector(std::initializer_list<P> list);
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector(std::vector<P> const &);
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector(fk::matrix<P> const &);

  // create view from owner.
  template<mem_type m_ = mem, typename = enable_for_view<m_>>
  explicit vector(fk::vector<P, mem_type::owner> const &owner,
                  int const start_index, int const stop_index);
  // overload for default case - whole vector
  template<mem_type m_ = mem, typename = enable_for_view<m_>>
  explicit vector(fk::vector<P, mem_type::owner> const &owner);

  ~vector();

  // constructor/assignment (required to be same to same T==T)
  vector(vector<P, mem> const &);
  vector<P, mem> &operator=(vector<P, mem> const &);

  // move precision constructor/assignment (required to be same to same)
  vector(vector<P, mem> &&);
  vector<P, mem> &operator=(vector<P, mem> &&);

  // converting constructor/assignment overloads
  template<typename PP, mem_type omem, mem_type m_ = mem,
           typename = enable_for_owner<m_>>
  explicit vector(vector<PP, omem> const &);
  template<typename PP, mem_type omem>
  vector<P, mem> &operator=(vector<PP, omem> const &);

  //
  // copy out of std::vector
  //
  vector<P, mem> &operator=(std::vector<P> const &);

  //
  // copy into std::vector
  //
  std::vector<P> to_std() const;

  //
  // subscripting operators
  //
  P &operator()(int const);
  P operator()(int const) const;
  //
  // comparison operators
  //
  template<mem_type omem>
  bool operator==(vector<P, omem> const &) const;
  template<mem_type omem>
  bool operator!=(vector<P, omem> const &) const;
  template<mem_type omem>
  bool operator<(vector<P, omem> const &) const;
  //
  // math operators
  //

  template<mem_type omem>
  vector<P> operator+(vector<P, omem> const &right) const;
  template<mem_type omem>
  vector<P> operator-(vector<P, omem> const &right) const;
  template<mem_type omem>
  P operator*(vector<P, omem> const &)const;

  template<mem_type omem>
  vector<P> operator*(matrix<P, omem> const &)const;

  vector<P> operator*(P const) const;

  template<mem_type omem>
  vector<P> single_column_kron(vector<P, omem> const &) const;

  vector<P, mem> &scale(P const x);
  //
  // basic queries to private data
  //
  int size() const { return size_; }
  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  // this can be offsetted for views
  P *data(int const elem = 0) const { return &data_[elem]; }
  // this is to allow specific other types to access the private ref counter of
  // owners - specifically, we want to allow a matrix<view> to be made from a
  // vector<owner>
  std::shared_ptr<int>
      get_ref_count(access_badge<matrix<P, mem_type::view>>) const
  {
    return ref_count_;
  }

  //
  // utility functions
  //
  void print(std::string const label = "") const;
  void dump_to_octave(char const *) const;

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  fk::vector<P, mem> &resize(int const size = 0);

  template<mem_type omem>
  vector<P, mem> &set_subvector(int const, vector<P, omem> const);

  vector<P> extract(int const, int const) const;

  template<mem_type omem, mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector<P, mem> &concat(vector<P, omem> const &right);

  typedef P *iterator;
  typedef const P *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }
  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + size(); }

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  int get_num_views() const;

private:
  P *data_;  //< pointer to elements
  int size_; //< dimension
  std::shared_ptr<int> ref_count_ = nullptr;
};

template<typename P, mem_type mem>
class matrix
{
  template<typename, mem_type>
  friend class matrix; // so that views can access owner sharedptr/rows

  // template on pointer/ref type to get iterator and const iterator
  template<typename T, typename R>
  class matrix_iterator; // forward declaration for custom iterator; defined
                         // out of line

public:
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix();
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix(int rows, int cols);
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix(std::initializer_list<std::initializer_list<P>> list);

  // create view from owner.
  template<mem_type m_ = mem, typename = enable_for_view<m_>>
  explicit matrix(fk::matrix<P, mem_type::owner> const &owner,
                  int const start_row, int const stop_row, int const start_col,
                  int const stop_col);
  // overload for default case - whole matrix
  template<mem_type m_ = mem, typename = enable_for_view<m_>>
  explicit matrix(fk::matrix<P, mem_type::owner> const &owner);

  // create matrix view from vector
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem>
  explicit matrix(fk::vector<P, omem> const &source, int const num_rows,
                  int const num_cols, int const start_index = 0);

  ~matrix();

  matrix(matrix<P, mem> const &);
  matrix<P, mem> &operator=(matrix<P, mem> const &);

  template<typename PP, mem_type omem, mem_type m_ = mem,
           typename = enable_for_owner<m_>>
  explicit matrix(matrix<PP, omem> const &);
  template<typename PP, mem_type omem>
  matrix<P, mem> &operator=(matrix<PP, omem> const &);

  matrix(matrix<P, mem> &&);
  matrix<P, mem> &operator=(matrix<P, mem> &&);

  //
  // copy out of fk::vector
  //
  template<mem_type omem>
  matrix<P, mem> &operator=(fk::vector<P, omem> const &);
  //
  // subscripting operators
  //
  P &operator()(int const, int const);
  P operator()(int const, int const) const;
  //
  // comparison operators
  //
  template<mem_type omem>
  bool operator==(matrix<P, omem> const &) const;
  template<mem_type omem>
  bool operator!=(matrix<P, omem> const &) const;
  template<mem_type omem>
  bool operator<(matrix<P, omem> const &) const;
  //
  // math operators
  //
  matrix<P> operator*(P const) const;
  template<mem_type omem>
  vector<P> operator*(vector<P, omem> const &)const;
  template<mem_type omem>
  matrix<P> operator*(matrix<P, omem> const &)const;
  template<mem_type omem>
  matrix<P> operator+(matrix<P, omem> const &) const;
  template<mem_type omem>
  matrix<P> operator-(matrix<P, omem> const &) const;

  matrix<P, mem> &transpose();

  template<mem_type omem>
  matrix<P> kron(matrix<P, omem> const &) const;

  // clang-format off
  template<typename U = P>
  std::enable_if_t<
    std::is_floating_point<U>::value && std::is_same<P, U>::value,
  matrix<P, mem> &> invert();


  template<typename U = P>
  std::enable_if_t<
      std::is_floating_point<U>::value && std::is_same<P, U>::value,
  P> determinant() const;
  // clang-format on

  //
  // basic queries to private data
  //
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  // for owners: stride == nrows
  // for views:  stride == owner's nrows
  int stride() const { return stride_; }
  int size() const { return nrows() * ncols(); }
  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  P *data(int const i = 0, int const j = 0) const
  {
    // return &data_[i * stride() + j]; // row-major
    return &data_[j * stride() + i]; // column-major
  }
  //
  // utility functions
  //

  template<mem_type omem>
  matrix<P, mem> &update_col(int const, fk::vector<P, omem> const &);
  matrix<P, mem> &update_col(int const, std::vector<P> const &);
  template<mem_type omem>
  matrix<P, mem> &update_row(int const, fk::vector<P, omem> const &);
  matrix<P, mem> &update_row(int const, std::vector<P> const &);

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix<P> &clear_and_resize(int const, int const);

  template<mem_type omem>
  matrix<P, mem> &set_submatrix(int const row_idx, int const col_idx,
                                fk::matrix<P, omem> const &submatrix);
  matrix<P> extract_submatrix(int const row_idx, int const col_idx,
                              int const num_rows, int const num_cols) const;
  void print(std::string const label = "") const;
  void dump_to_octave(char const *name) const;

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  int get_num_views() const;

  using iterator       = matrix_iterator<P *, P &>;
  using const_iterator = matrix_iterator<P const *, P const &>;

  iterator begin() { return iterator(data(), stride(), nrows()); }

  iterator end()
  {
    return iterator(data() + stride() * ncols(), stride(), nrows());
  }

  const_iterator begin() const
  {
    return const_iterator(data(), stride(), nrows());
  }

  const_iterator end() const
  {
    return const_iterator(data() + stride() * ncols(), stride(), nrows());
  }

private:
  P *data_;    //< pointer to elements
  int nrows_;  //< row dimension
  int ncols_;  //< column dimension
  int stride_; //< leading dimension;
               // number of elements in memory between successive matrix
               // elements in a row
  std::shared_ptr<int> ref_count_ = nullptr;
};
} // namespace fk

//
// This would otherwise be the start of the tensors.cpp, if we were still doing
// the explicit instantiations
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

//-----------------------------------------------------------------------------
//
// fk::vector class implementation starts here
//
//-----------------------------------------------------------------------------
template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem>::vector()
    : data_{nullptr}, size_{0}, ref_count_{std::make_shared<int>(0)}
{}
// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem>::vector(int const size)
    : data_{new P[size]()}, size_{size}, ref_count_{std::make_shared<int>(0)}
{}

// can also do this with variadic template constructor for constness
// https://stackoverflow.com/a/5549918
// but possibly this is "too clever" for our needs right now

template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem>::vector(std::initializer_list<P> list)
    : data_{new P[list.size()]}, size_{static_cast<int>(list.size())},
      ref_count_{std::make_shared<int>(0)}
{
  std::copy(list.begin(), list.end(), data_);
}

template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem>::vector(std::vector<P> const &v)
    : data_{new P[v.size()]}, size_{static_cast<int>(v.size())},
      ref_count_{std::make_shared<int>(0)}
{
  std::copy(v.begin(), v.end(), data_);
}

//
// matrix conversion constructor linearizes the matrix, i.e. stacks the columns
// of the matrix into a single vector
//
template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem>::vector(fk::matrix<P> const &mat)
    : data_{new P[mat.size()]}, ref_count_{std::make_shared<int>(0)}
{
  size_ = mat.size();
  if ((*this).size() == 0)
  {
    delete[] data_;
    data_ = nullptr;
  }
  else
  {
    int i = 0;
    for (auto const &elem : mat)
    {
      (*this)(i++) = elem;
    }
  }
}

// vector view constructor given a start and total length
template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem>::vector(fk::vector<P> const &vec, int const start_index,
                           int const stop_index)
    : ref_count_{vec.ref_count_}
{
  data_ = nullptr;
  size_ = 0;

  if (vec.size() > 0)
  {
    assert(start_index >= 0);
    assert(stop_index < vec.size());
    assert(stop_index >= start_index);

    data_ = vec.data_ + start_index;
    size_ = stop_index - start_index + 1;
  }
}

// delegating constructor to extract view from owner. overload for default case
// of viewing the entire owner
template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem>::vector(fk::vector<P> const &a)
    : vector(a, 0, std::max(0, a.size() - 1))
{}

template<typename P, mem_type mem>
fk::vector<P, mem>::~vector()
{
  if constexpr (mem == mem_type::owner)
  {
    assert(ref_count_.use_count() == 1);
    delete[] data_;
  }
}

//
// vector copy constructor for like types (like types only)
//
template<typename P, mem_type mem>
fk::vector<P, mem>::vector(vector<P, mem> const &a) : size_{a.size_}
{
  if constexpr (mem == mem_type::owner)
  {
    data_      = new P[a.size()];
    ref_count_ = std::make_shared<int>(0);
    std::memcpy(data_, a.data(), a.size() * sizeof(P));
  }
  else
  {
    data_      = a.data();
    ref_count_ = a.ref_count_;
  }
}

//
// vector copy assignment
// this can probably be optimized better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem>
fk::vector<P, mem> &fk::vector<P, mem>::operator=(vector<P, mem> const &a)
{
  if (&a == this)
    return *this;

  assert(size() == a.size());

  std::memcpy(data_, a.data(), a.size() * sizeof(P));

  return *this;
}

//
// vector move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
template<typename P, mem_type mem>
fk::vector<P, mem>::vector(vector<P, mem> &&a) : data_{a.data_}, size_{a.size_}
{
  if constexpr (mem == mem_type::owner)
  {
    assert(a.ref_count_.use_count() == 1);
  }
  ref_count_ = std::make_shared<int>(0);
  ref_count_.swap(a.ref_count_);
  a.data_ = nullptr; // b/c a's destructor will be called
  a.size_ = 0;
}

//
// vector move assignment
//
template<typename P, mem_type mem>
fk::vector<P, mem> &fk::vector<P, mem>::operator=(vector<P, mem> &&a)
{
  if (&a == this)
    return *this;

  if constexpr (mem == mem_type::owner)
  {
    assert(ref_count_.use_count() == 1);
    assert(a.ref_count_.use_count() == 1);
  }

  assert(size() == a.size());
  size_      = a.size_;
  ref_count_ = std::make_shared<int>(0);
  ref_count_.swap(a.ref_count_);
  P *temp{data_};
  data_   = a.data_;
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// converting vector constructor
//
template<typename P, mem_type mem>
template<typename PP, mem_type omem, mem_type, typename>
fk::vector<P, mem>::vector(vector<PP, omem> const &a)
    : data_{new P[a.size()]}, size_{a.size()}, ref_count_{
                                                   std::make_shared<int>(0)}
{
  for (auto i = 0; i < a.size(); ++i)
  {
    (*this)(i) = static_cast<P>(a(i));
  }
}

//
// converting vector assignment overload
// this can probably be optimized better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem>
template<typename PP, mem_type omem>
fk::vector<P, mem> &fk::vector<P, mem>::operator=(vector<PP, omem> const &a)
{
  assert(size() == a.size());

  size_ = a.size();
  for (auto i = 0; i < a.size(); ++i)
  {
    (*this)(i) = static_cast<P>(a(i));
  }

  return *this;
}

//
// copy out of std::vector
//
template<typename P, mem_type mem>
fk::vector<P, mem> &fk::vector<P, mem>::operator=(std::vector<P> const &v)
{
  assert(size() == static_cast<int>(v.size()));
  std::memcpy(data_, v.data(), v.size() * sizeof(P));
  return *this;
}

//
// copy into std::vector
//
template<typename P, mem_type mem>
std::vector<P> fk::vector<P, mem>::to_std() const
{
  return std::vector<P>(data(), data() + size());
}

// vector subscript operator
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename P, mem_type mem>
P &fk::vector<P, mem>::operator()(int i)
{
  assert(i < size_);
  return data_[i];
}

template<typename P, mem_type mem>
P fk::vector<P, mem>::operator()(int i) const
{
  assert(i < size_);
  return data_[i];
}

// vector comparison operators - set default tolerance above
// see https://stackoverflow.com/a/253874/6595797
// FIXME do we need to be more careful with these fp comparisons?
template<typename P, mem_type mem>
template<mem_type omem>
bool fk::vector<P, mem>::operator==(vector<P, omem> const &other) const
{
  if constexpr (omem == mem)
    if (&other == this)
      return true;
  if (size() != other.size())
    return false;
  for (auto i = 0; i < size(); ++i)
    if constexpr (std::is_floating_point<P>::value)
    {
      if (std::abs((*this)(i)-other(i)) > TOL)
      {
        return false;
      }
    }
    else
    {
      if ((*this)(i) != other(i))
      {
        return false;
      }
    }
  return true;
}
template<typename P, mem_type mem>
template<mem_type omem>
bool fk::vector<P, mem>::operator!=(vector<P, omem> const &other) const
{
  return !(*this == other);
}

template<typename P, mem_type mem>
template<mem_type omem>
bool fk::vector<P, mem>::operator<(vector<P, omem> const &other) const
{
  return std::lexicographical_compare(begin(), end(), other.begin(),
                                      other.end());
}

//
// vector addition operator
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::vector<P> fk::vector<P, mem>::operator+(vector<P, omem> const &right) const
{
  assert(size() == right.size());
  vector<P> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i) + right(i);
  return ans;
}

//
// vector subtraction operator
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::vector<P> fk::vector<P, mem>::operator-(vector<P, omem> const &right) const
{
  assert(size() == right.size());
  vector<P> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i)-right(i);
  return ans;
}

//
// vector*vector multiplication operator
//
template<typename P, mem_type mem>
template<mem_type omem>
P fk::vector<P, mem>::operator*(vector<P, omem> const &right) const
{
  assert(size() == right.size());
  int n           = size();
  int one         = 1;
  vector const &X = (*this);

  return lib_dispatch::dot(&n, X.data(), &one, right.data(), &one);
}

//
// vector*matrix multiplication operator
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::vector<P> fk::vector<P, mem>::operator*(fk::matrix<P, omem> const &A) const
{
  // check dimension compatibility
  assert(size() == A.nrows());

  vector const &X = (*this);
  vector<P> Y(A.ncols());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = A.stride();
  int one_i = 1;

  P zero = 0.0;
  P one  = 1.0;
  lib_dispatch::gemv("t", &m, &n, &one, A.data(), &lda, X.data(), &one_i, &zero,
                     Y.data(), &one_i);
  return Y;
}

//
// vector*scalar multiplication operator
//
template<typename P, mem_type mem>
fk::vector<P> fk::vector<P, mem>::operator*(P const x) const
{
  vector<P> a(*this);
  int one_i = 1;
  int n     = a.size();
  P alpha   = x;

  lib_dispatch::scal(&n, &alpha, a.data(), &one_i);

  return a;
}

//
// perform the matrix kronecker product by
// interpreting vector operands/return vector
// as single column matrices.
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::vector<P>
fk::vector<P, mem>::single_column_kron(vector<P, omem> const &right) const
{
  fk::vector<P> product((*this).size() * right.size());
  for (int i = 0; i < (*this).size(); ++i)
  {
    for (int j = 0; j < right.size(); ++j)
    {
      product(i * right.size() + j) = (*this)(i)*right(j);
    }
  }
  return product;
}

template<typename P, mem_type mem>
fk::vector<P, mem> &fk::vector<P, mem>::scale(P const x)
{
  int one_i = 1;
  int n     = this->size();
  P alpha   = x;

  lib_dispatch::scal(&n, &alpha, this->data(), &one_i);

  return *this;
}

//
// utility functions
//

//
// Prints out the values of a vector
//
// @param[in]   label   a string label printed with the output
// @param[in]   b       the vector from the batch to print out
// @return      Nothing
//
template<typename P, mem_type mem>
void fk::vector<P, mem>::print(std::string const label) const
{
  if constexpr (mem == mem_type::owner)
    std::cout << label << "(owner, ref_count = " << ref_count_.use_count()
              << ")" << '\n';
  else
    std::cout << label << "(view)" << '\n';

  if constexpr (std::is_floating_point<P>::value)
  {
    for (auto i = 0; i < size(); ++i)
      std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                << std::right << (*this)(i);
  }
  else
  {
    for (auto i = 0; i < size(); ++i)
      std::cout << std::right << (*this)(i) << " ";
  }
  std::cout << '\n';
}

//
// Dumps to file a vector that can be read straight into octave
// Same as the matrix:: version
//
// @param[in]   label   a string label printed with the output
// @param[in]   b       the vector from the batch to print out
// @return      Nothing
//
template<typename P, mem_type mem>
void fk::vector<P, mem>::dump_to_octave(char const *filename) const
{
  std::ofstream ofile(filename);
  auto coutbuf = std::cout.rdbuf(ofile.rdbuf());
  for (auto i = 0; i < size(); ++i)
    std::cout << std::setprecision(12) << (*this)(i) << " ";

  std::cout.rdbuf(coutbuf);
}

//
// resize the vector
// (currently supports a subset of the std::vector.resize() interface)
//
template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem> &fk::vector<P, mem>::resize(int const new_size)
{
  if (new_size == this->size())
    return *this;
  P *old_data{data_};
  data_ = new P[new_size]();
  if (size() > 0 && new_size > 0)
  {
    if (size() < new_size)
      std::memcpy(data_, old_data, size() * sizeof(P));
    else
      std::memcpy(data_, old_data, new_size * sizeof(P));
  }

  size_ = new_size;
  delete[] old_data;
  return *this;
}

template<typename P, mem_type mem>
template<mem_type omem, mem_type, typename>
fk::vector<P, mem> &fk::vector<P, mem>::concat(vector<P, omem> const &right)
{
  int const old_size = this->size();
  int const new_size = this->size() + right.size();
  P *old_data{data_};
  data_ = new P[new_size]();
  std::memcpy(data_, old_data, old_size * sizeof(P));
  std::memcpy(data(old_size), right.data(), right.size() * sizeof(P));
  size_ = new_size;
  delete[] old_data;
  return *this;
}

// set a subvector beginning at provided index
template<typename P, mem_type mem>
template<mem_type omem>
fk::vector<P, mem> &
fk::vector<P, mem>::set_subvector(int const index,
                                  fk::vector<P, omem> const sub_vector)
{
  assert(index >= 0);
  assert((index + sub_vector.size()) <= this->size());
  std::memcpy(&(*this)(index), sub_vector.data(),
              sub_vector.size() * sizeof(P));
  return *this;
}

// extract subvector, indices inclusive
template<typename P, mem_type mem>
fk::vector<P> fk::vector<P, mem>::extract(int const start, int const stop) const
{
  assert(start >= 0);
  assert(stop < this->size());
  assert(stop > start);

  int const sub_size = stop - start + 1;
  fk::vector<P> sub_vector(sub_size);
  for (int i = 0; i < sub_size; ++i)
  {
    sub_vector(i) = (*this)(i + start);
  }
  return sub_vector;
}

// get number of outstanding views for an owner
template<typename P, mem_type mem>
template<mem_type, typename>
int fk::vector<P, mem>::get_num_views() const
{
  return ref_count_.use_count() - 1;
}
//-----------------------------------------------------------------------------
//
// fk::matrix class implementation starts here
//
//-----------------------------------------------------------------------------

template<typename P, mem_type mem>
template<mem_type, typename>
fk::matrix<P, mem>::matrix()
    : data_{nullptr}, nrows_{0}, ncols_{0}, stride_{nrows_},
      ref_count_{std::make_shared<int>(0)}

{}

// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region

template<typename P, mem_type mem>
template<mem_type, typename>
fk::matrix<P, mem>::matrix(int M, int N)
    : data_{new P[M * N]()}, nrows_{M}, ncols_{N}, stride_{nrows_},
      ref_count_{std::make_shared<int>(0)}

{}

template<typename P, mem_type mem>
template<mem_type, typename>
fk::matrix<P, mem>::matrix(
    std::initializer_list<std::initializer_list<P>> llist)
    : data_{new P[llist.size() * llist.begin()->size()]()},
      nrows_{static_cast<int>(llist.size())}, ncols_{static_cast<int>(
                                                  llist.begin()->size())},
      stride_{nrows_}, ref_count_{std::make_shared<int>(0)}
{
  int row_idx = 0;
  for (auto const &row_list : llist)
  {
    // much simpler for row-major storage
    // std::copy(row_list.begin(), row_list.end(), data(row_idx));
    int col_idx = 0;
    for (auto const &col_elem : row_list)
    {
      (*this)(row_idx, col_idx) = col_elem;
      ++col_idx;
    }
    ++row_idx;
  }
}

// create view from owner.
template<typename P, mem_type mem>
template<mem_type, typename>
fk::matrix<P, mem>::matrix(fk::matrix<P, mem_type::owner> const &owner,
                           int const start_row, int const stop_row,
                           int const start_col, int const stop_col)
    : ref_count_(owner.ref_count_)
{
  data_   = nullptr;
  nrows_  = 0;
  ncols_  = 0;
  stride_ = 0;

  int const view_rows = stop_row - start_row + 1;
  int const view_cols = stop_col - start_col + 1;
  if (owner.size() > 0)
  {
    assert(start_row >= 0);
    assert(start_col >= 0);
    assert(stop_col < owner.ncols());
    assert(stop_row < owner.nrows());
    assert(stop_row >= start_row);

    data_   = owner.data(start_row, start_col);
    nrows_  = view_rows;
    ncols_  = view_cols;
    stride_ = owner.nrows();
  }
}

// overload for default case - whole matrix
template<typename P, mem_type mem>
template<mem_type, typename>
fk::matrix<P, mem>::matrix(fk::matrix<P, mem_type::owner> const &owner)
    : matrix(owner, 0, std::max(0, owner.nrows() - 1), 0,
             std::max(0, owner.ncols() - 1))
{}

// create matrix view of an existing vector
template<typename P, mem_type mem>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem>::matrix(fk::vector<P, omem> const &source,
                           int const num_rows, int const num_cols,
                           int const start_index)
    : ref_count_(source.get_ref_count({}))
{
  assert(start_index >= 0);
  assert(num_rows > 0);
  assert(num_cols > 0);

  int const size = num_rows * num_cols;
  assert(start_index + size <= source.size());

  data_   = nullptr;
  nrows_  = 0;
  ncols_  = 0;
  stride_ = 0;

  if (size > 0)
  {
    data_   = source.data(start_index);
    nrows_  = num_rows;
    ncols_  = num_cols;
    stride_ = num_rows;
  }
}

template<typename P, mem_type mem>
fk::matrix<P, mem>::~matrix()
{
  if constexpr (mem == mem_type::owner)
  {
    assert(ref_count_.use_count() == 1);
    delete[] data_;
  }
}

//
// matrix copy constructor
//
template<typename P, mem_type mem>
fk::matrix<P, mem>::matrix(matrix<P, mem> const &a)
    : nrows_{a.nrows()}, ncols_{a.ncols()}, stride_{a.stride()}

{
  if constexpr (mem == mem_type::owner)
  {
    data_      = new P[a.size()]();
    ref_count_ = std::make_shared<int>(0);

    // for optimization - if the matrices are contiguous, use memcpy
    // for performance
    if (stride() == nrows() && a.stride() == a.nrows())
    {
      std::memcpy(data_, a.data(), a.size() * sizeof(P));

      // else copy using loops. noticably slower in testing
    }
    else
    {
      for (auto j = 0; j < a.ncols(); ++j)
        for (auto i = 0; i < a.nrows(); ++i)
        {
          (*this)(i, j) = a(i, j);
        }
    }
  }
  else
  {
    data_      = a.data();
    ref_count_ = a.ref_count_;
  }
}

//
// matrix copy assignment
// this can probably be done better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem>
fk::matrix<P, mem> &fk::matrix<P, mem>::operator=(matrix<P, mem> const &a)
{
  if (&a == this)
    return *this;

  assert((nrows() == a.nrows()) && (ncols() == a.ncols()));

  // for optimization - if the matrices are contiguous, use memcpy
  // for performance
  if (stride() == nrows() && a.stride() == a.nrows())
  {
    std::memcpy(data_, a.data(), a.size() * sizeof(P));

    // else copy using loops. noticably slower in testing
  }
  else
  {
    for (auto j = 0; j < a.ncols(); ++j)
      for (auto i = 0; i < a.nrows(); ++i)
      {
        (*this)(i, j) = a(i, j);
      }
  }

  return *this;
}

//
// converting matrix copy constructor
//
template<typename P, mem_type mem>
template<typename PP, mem_type omem, mem_type, typename>
fk::matrix<P, mem>::matrix(matrix<PP, omem> const &a)
    : data_{new P[a.size()]()}, nrows_{a.nrows()}, ncols_{a.ncols()},
      stride_{a.nrows()}, ref_count_{std::make_shared<int>(0)}

{
  for (auto j = 0; j < a.ncols(); ++j)
    for (auto i = 0; i < a.nrows(); ++i)
    {
      (*this)(i, j) = static_cast<P>(a(i, j));
    }
}

//
// converting matrix copy assignment
// this can probably be done better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//

template<typename P, mem_type mem>
template<typename PP, mem_type omem>
fk::matrix<P, mem> &fk::matrix<P, mem>::operator=(matrix<PP, omem> const &a)
{
  assert((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  for (auto j = 0; j < a.ncols(); ++j)
    for (auto i = 0; i < a.nrows(); ++i)
    {
      (*this)(i, j) = static_cast<P>(a(i, j));
    }
  return *this;
}

//
// matrix move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//

template<typename P, mem_type mem>
fk::matrix<P, mem>::matrix(matrix<P, mem> &&a)
    : data_{a.data()}, nrows_{a.nrows()}, ncols_{a.ncols()}, stride_{a.stride()}
{
  if constexpr (mem == mem_type::owner)
  {
    assert(a.ref_count_.use_count() == 1);
  }

  ref_count_ = std::make_shared<int>(0);
  ref_count_.swap(a.ref_count_);

  a.data_  = nullptr; // b/c a's destructor will be called
  a.nrows_ = 0;
  a.ncols_ = 0;
}

//
// matrix move assignment
//
template<typename P, mem_type mem>
fk::matrix<P, mem> &fk::matrix<P, mem>::operator=(matrix<P, mem> &&a)
{
  if (&a == this)
    return *this;

  assert((nrows() == a.nrows()) &&
         (ncols() == a.ncols() && stride() == a.stride()));

  // check for destination orphaning; see below
  if constexpr (mem == mem_type::owner)
  {
    assert(ref_count_.use_count() == 1 && a.ref_count_.use_count() == 1);
  }
  ref_count_ = std::make_shared<int>(0);
  ref_count_.swap(a.ref_count_);

  P *temp{data_};
  // this would orphan views on the destination
  data_   = a.data();
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// copy out of fk::vector - assumes the vector is column-major
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::matrix<P, mem> &fk::matrix<P, mem>::operator=(fk::vector<P, omem> const &v)
{
  assert(nrows() * ncols() == v.size());

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      (*this)(i, j) = v(j + i * ncols());

  return *this;
}

//
// matrix subscript operator - row-major ordering
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename P, mem_type mem>
P &fk::matrix<P, mem>::operator()(int const i, int const j)
{
  assert(i < nrows() && j < ncols());
  return *(data(i, j));
}

template<typename P, mem_type mem>
P fk::matrix<P, mem>::operator()(int const i, int const j) const
{
  assert(i < nrows() && j < ncols());
  return *(data(i, j));
}

// matrix comparison operators - set default tolerance above
// see https://stackoverflow.com/a/253874/6595797
// FIXME we may need to be more careful with these comparisons
template<typename P, mem_type mem>
template<mem_type omem>
bool fk::matrix<P, mem>::operator==(matrix<P, omem> const &other) const
{
  if constexpr (mem == omem)
  {
    if (&other == this)
      return true;
  }

  if (nrows() != other.nrows() || ncols() != other.ncols())
    return false;
  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      if constexpr (std::is_floating_point<P>::value)
      {
        if (std::abs((*this)(i, j) - other(i, j)) > TOL)
        {
          return false;
        }
      }
      else
      {
        if ((*this)(i, j) != other(i, j))
        {
          return false;
        }
      }
  return true;
}

template<typename P, mem_type mem>
template<mem_type omem>
bool fk::matrix<P, mem>::operator!=(matrix<P, omem> const &other) const
{
  return !(*this == other);
}

template<typename P, mem_type mem>
template<mem_type omem>
bool fk::matrix<P, mem>::operator<(matrix<P, omem> const &other) const
{
  return std::lexicographical_compare(this->begin(), this->end(), other.begin(),
                                      other.end());
}

//
// matrix addition operator
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::matrix<P> fk::matrix<P, mem>::operator+(matrix<P, omem> const &right) const
{
  assert(nrows() == right.nrows() && ncols() == right.ncols());

  matrix<P> ans(nrows(), ncols());
  ans.nrows_ = nrows();
  ans.ncols_ = ncols();

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      ans(i, j) = (*this)(i, j) + right(i, j);

  return ans;
}

//
// matrix subtraction operator
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::matrix<P> fk::matrix<P, mem>::operator-(matrix<P, omem> const &right) const
{
  assert(nrows() == right.nrows() && ncols() == right.ncols());

  matrix<P> ans(nrows(), ncols());
  ans.nrows_ = nrows();
  ans.ncols_ = ncols();

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      ans(i, j) = (*this)(i, j) - right(i, j);

  return ans;
}

//
// matrix*scalar multiplication operator
//
template<typename P, mem_type mem>
fk::matrix<P> fk::matrix<P, mem>::operator*(P const right) const
{
  matrix<P> ans(nrows(), ncols());
  ans.nrows_ = nrows();
  ans.ncols_ = ncols();

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      ans(i, j) = (*this)(i, j) * right;

  return ans;
}

//
// matrix*vector multiplication operator
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::vector<P> fk::matrix<P, mem>::
operator*(fk::vector<P, omem> const &right) const
{
  // check dimension compatibility
  assert(ncols() == right.size());

  matrix<P, mem> const &A = (*this);
  vector<P> Y(A.nrows());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = A.stride();
  int one_i = 1;

  P one  = 1.0;
  P zero = 0.0;
  lib_dispatch::gemv("n", &m, &n, &one, A.data(), &lda, right.data(), &one_i,
                     &zero, Y.data(), &one_i);

  return Y;
}

//
// matrix*matrix multiplication operator C[m,n] = A[m,k] * B[k,n]
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::matrix<P> fk::matrix<P, mem>::operator*(matrix<P, omem> const &B) const
{
  assert(ncols() == B.nrows()); // k == k

  // just aliases for easier reading
  matrix const &A = (*this);
  int m           = A.nrows();
  int n           = B.ncols();
  int k           = B.nrows();

  matrix<P> C(m, n);

  int lda = A.stride();
  int ldb = B.stride();
  int ldc = C.stride();

  P one  = 1.0;
  P zero = 0.0;
  lib_dispatch::gemm("n", "n", &m, &n, &k, &one, A.data(), &lda, B.data(), &ldb,
                     &zero, C.data(), &ldc);

  return C;
}

//
// Transpose a matrix (overwrites original)
// @return  the transposed matrix
//
// FIXME could be worthwhile to optimize the matrix transpose
template<typename P, mem_type mem>
fk::matrix<P, mem> &fk::matrix<P, mem>::transpose()
{
  matrix<P> temp(ncols(), nrows());

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      temp(j, i) = (*this)(i, j);

  // inelegant manual "move assignment"
  // unlike actual move assignment, need to delete the old pointer
  nrows_  = temp.nrows();
  ncols_  = temp.ncols();
  stride_ = nrows();
  delete[] data_;
  data_      = temp.data();
  temp.data_ = nullptr;

  return *this;
}

// Simple quad-loop kron prod
// @return the product
//
// FIXME this is NOT optimized.
// we will use the batch gemm method
// for performance-critical (large)
// krons
template<typename P, mem_type mem>
template<mem_type omem>
fk::matrix<P> fk::matrix<P, mem>::kron(matrix<P, omem> const &B) const
{
  fk::matrix<P> C(nrows() * B.nrows(), ncols() * B.ncols());
  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
    {
      for (auto k = 0; k < B.nrows(); ++k)
      {
        for (auto l = 0; l < B.ncols(); ++l)
        {
          C((i * B.nrows() + k), (j * B.ncols() + l)) +=
              (*this)(i, j) * B(k, l);
        }
      }
    }
  }
  return C;
}

//
// Invert a square matrix (overwrites original)
// disabled for non-fp types; haven't written a routine to do it
// @return  the inverted matrix
//
template<typename P, mem_type mem>
template<typename U>
std::enable_if_t<std::is_floating_point<U>::value && std::is_same<P, U>::value,
                 fk::matrix<P, mem> &>
fk::matrix<P, mem>::invert()
{
  assert(nrows() == ncols());

  int *ipiv{new int[ncols()]};
  int lwork{nrows() * ncols()};
  int lda = stride();
  P *work{new P[nrows() * ncols()]};
  int info;

  lib_dispatch::getrf(&ncols_, &ncols_, data(0, 0), &lda, ipiv, &info);
  lib_dispatch::getri(&ncols_, data(0, 0), &lda, ipiv, work, &lwork, &info);

  delete[] ipiv;
  delete[] work;
  return *this;
}

//
// Get the determinant of the matrix  (non destructive)
// (based on src/Numerics/DeterminantOperators.h)
// (note possible problems with over/underflow
// - see Ed's emails 12/5/16, 10/14/16, 10/10/16.
// how is this handled / is it necessary in production?
// possibly okay for small KxK matrices - can build in a check/warning)
//
//
// disabled for non-float types; haven't written a routine to do it
//
// @param[in]   mat   integer matrix (walker) to get determinant from
// @return  the determinant (type double)
//
template<typename P, mem_type mem>
template<typename U>
std::enable_if_t<std::is_floating_point<U>::value && std::is_same<P, U>::value,
                 P>
fk::matrix<P, mem>::determinant() const
{
  assert(nrows() == ncols());

  matrix temp{*this}; // get temp copy to do LU
  int *ipiv{new int[ncols()]};
  int info;
  int n   = ncols();
  int lda = stride();

  lib_dispatch::getrf(&n, &n, temp.data(0, 0), &lda, ipiv, &info);

  P det    = 1.0;
  int sign = 1;
  for (auto i = 0; i < nrows(); ++i)
  {
    if (ipiv[i] != i + 1)
      sign *= -1;
    det *= temp(i, i);
  }
  det *= static_cast<P>(sign);
  delete[] ipiv;
  return det;
}

//
// Update a specific col of a matrix, given a fk::vector<P> (overwrites
// original)
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::matrix<P, mem> &
fk::matrix<P, mem>::update_col(int const col_idx, fk::vector<P, omem> const &v)
{
  assert(nrows() == static_cast<int>(v.size()));
  assert(col_idx < ncols());

  int n{v.size()};
  int one{1};
  int stride = 1;

  lib_dispatch::copy(&n, v.data(), &one, data(0, col_idx), &stride);

  return *this;
}

//
// Update a specific col of a matrix, given a std::vector (overwrites original)
//
template<typename P, mem_type mem>
fk::matrix<P, mem> &
fk::matrix<P, mem>::update_col(int const col_idx, std::vector<P> const &v)
{
  assert(nrows() == static_cast<int>(v.size()));
  assert(col_idx < ncols());

  int n{static_cast<int>(v.size())};
  int one{1};
  int stride = 1;

  lib_dispatch::copy(&n, const_cast<P *>(v.data()), &one, data(0, col_idx),
                     &stride);

  return *this;
}

//
// Update a specific row of a matrix, given a fk::vector<P> (overwrites
// original)
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::matrix<P, mem> &
fk::matrix<P, mem>::update_row(int const row_idx, fk::vector<P, omem> const &v)
{
  assert(ncols() == v.size());
  assert(row_idx < nrows());

  int n{v.size()};
  int one{1};
  int lda = stride();

  lib_dispatch::copy(&n, v.data(), &one, data(row_idx, 0), &lda);

  return *this;
}

//
// Update a specific row of a matrix, given a std::vector (overwrites original)
//
template<typename P, mem_type mem>
fk::matrix<P, mem> &
fk::matrix<P, mem>::update_row(int const row_idx, std::vector<P> const &v)
{
  assert(ncols() == static_cast<int>(v.size()));
  assert(row_idx < nrows());

  int n{static_cast<int>(v.size())};
  int one{1};
  int lda = stride();

  lib_dispatch::copy(&n, const_cast<P *>(v.data()), &one, data(row_idx, 0),
                     &lda);

  return *this;
}

//
// Resize, clearing all data
//
template<typename P, mem_type mem>
template<mem_type, typename>
fk::matrix<P> &
fk::matrix<P, mem>::clear_and_resize(int const rows, int const cols)
{
  assert(ref_count_.use_count() == 1);

  assert(rows >= 0);
  assert(cols >= 0);
  if (rows == 0 || cols == 0)
    assert(cols == rows);
  delete[] data_;
  data_   = new P[rows * cols]();
  nrows_  = rows;
  ncols_  = cols;
  stride_ = nrows_;
  return *this;
}

//
// Set a submatrix within the matrix, given another (smaller) matrix
//
template<typename P, mem_type mem>
template<mem_type omem>
fk::matrix<P, mem> &
fk::matrix<P, mem>::set_submatrix(int const row_idx, int const col_idx,
                                  matrix<P, omem> const &submatrix)
{
  assert(row_idx >= 0);
  assert(col_idx >= 0);
  assert(row_idx + submatrix.nrows() <= nrows());
  assert(col_idx + submatrix.ncols() <= ncols());

  matrix &matrix = *this;
  for (auto i = 0; i < submatrix.nrows(); ++i)
  {
    for (auto j = 0; j < submatrix.ncols(); ++j)
    {
      matrix(i + row_idx, j + col_idx) = submatrix(i, j);
    }
  }
  return matrix;
}

//
// Extract a rectangular submatrix from within the matrix
//
template<typename P, mem_type mem>
fk::matrix<P>
fk::matrix<P, mem>::extract_submatrix(int const row_idx, int const col_idx,
                                      int const num_rows,
                                      int const num_cols) const
{
  assert(row_idx >= 0);
  assert(col_idx >= 0);
  assert(row_idx + num_rows <= nrows());
  assert(col_idx + num_cols <= ncols());

  matrix<P> submatrix(num_rows, num_cols);
  auto matrix = *this;
  for (auto i = 0; i < num_rows; ++i)
  {
    for (auto j = 0; j < num_cols; ++j)
    {
      submatrix(i, j) = matrix(i + row_idx, j + col_idx);
    }
  }

  return submatrix;
}

// Prints out the values of a matrix
// @return  Nothing
//
template<typename P, mem_type mem>
void fk::matrix<P, mem>::print(std::string label) const
{
  if constexpr (mem == mem_type::owner)
    std::cout << label << "(owner, "
              << "outstanding views == " << std::to_string(get_num_views())
              << ")" << '\n';

  else
    std::cout << label << "(view, "
              << "stride == " << std::to_string(stride()) << ")" << '\n';

  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
    {
      if constexpr (std::is_floating_point<P>::value)
      {
        std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                  << std::right << (*this)(i, j);
      }
      else
      {
        std::cout << (*this)(i, j) << " ";
      }
    }
    std::cout << '\n';
  }
}

//
// Dumps to file a matrix that can be read data straight into octave
// e.g.
//
//      dump_to_matrix ("A.dat");
//      ...
//      octave> load A.dat
//
// @return  Nothing
//
template<typename P, mem_type mem>
void fk::matrix<P, mem>::dump_to_octave(char const *filename) const
{
  std::ofstream ofile(filename);
  auto coutbuf = std::cout.rdbuf(ofile.rdbuf());
  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
      std::cout << std::setprecision(12) << (*this)(i, j) << " ";

    std::cout << std::setprecision(4) << '\n';
  }
  std::cout.rdbuf(coutbuf);
}

// get number of outstanding views for an owner
template<typename P, mem_type mem>
template<mem_type, typename>
int fk::matrix<P, mem>::get_num_views() const
{
  return ref_count_.use_count() - 1;
}

template<typename P, mem_type mem>
template<typename T, typename R>
class fk::matrix<P, mem>::matrix_iterator
{
public:
  using self_type         = matrix_iterator;
  using value_type        = P;
  using reference         = R;
  using pointer           = T;
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = int;
  matrix_iterator(pointer ptr, int const stride, int const rows)
      : ptr_(ptr), start_(ptr), stride_(stride), rows_(rows)
  {}

  difference_type increment()
  {
    difference_type const next_pos = ptr_ - start_ + 1;

    if (next_pos % rows_ != 0)
    {
      return 1;
    }
    else
    {
      start_ += stride_;
      return stride_ - rows_ + 1;
    }
  }

  self_type operator++(int)
  {
    self_type i = *this;
    ptr_ += increment();
    return i;
  }
  self_type operator++()
  {
    ptr_ += increment();
    return *this;
  }

  reference operator*() const { return *ptr_; }
  pointer operator->() const { return ptr_; }
  bool operator==(const self_type &rhs) const { return ptr_ == rhs.ptr_; }
  bool operator!=(const self_type &rhs) const { return ptr_ != rhs.ptr_; }

private:
  pointer ptr_;
  pointer start_;
  int stride_;
  int rows_;
};
