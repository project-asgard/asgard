#pragma once
#include "asgard_lib_dispatch.hpp"

namespace asgard
{
// resource arguments allow developers to select host (CPU only) or device
// (accelerator) allocation for tensors. device
// tensors have a restricted API - most member functions are disabled. the fast
// math component is designed to allow BLAS on host and device tensors.

// mem_type arguments allow for the selection of owner or view (read/write
// window into underlying owner memory) semantics.

// device owners can be constructed with no-arg, size, or
// initializer list constructors.
//
// device owners are allocated in accelerator DRAM when
// the appropriate build option is set, with allocation
// falling back to CPU RAM otherwise.
//
// additionally, device owners can be transfer constructed from a
// host owner or copy/move constructed from another device owner.
//
// host owners can be created with any of the below constructors.
//
// device views can only be created from a device owner, and host
// views can only be constructor from host owners

/*!
 * \defgroup tensors
 *
 * One- and two-dimensional tensors managing memory allocation and destruction.
 */
namespace fk
{
template<typename P, mem_type mem, resource resrc>
class vector;

template<typename P, mem_type mem = mem_type::owner,
         resource resrc = resource::host>
class matrix
{
  template<typename, mem_type, resource>
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

  // create const view
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit matrix(fk::matrix<P, omem, resrc> const &owner, int const start_row,
                  int const stop_row, int const start_col, int const stop_col);
  // create modifiable view
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit matrix(fk::matrix<P, omem, resrc> &owner, int const start_row,
                  int const stop_row, int const start_col, int const stop_col);

  // overloads for default case - whole matrix
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit matrix(fk::matrix<P, omem, resrc> const &owner);

  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit matrix(fk::matrix<P, omem, resrc> &owner);

  // create matrix view from vector
  // const view version
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit matrix(fk::vector<P, omem, resrc> const &source, int const num_rows,
                  int const num_cols, int const start_index = 0);
  // modifiable view version
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit matrix(fk::vector<P, omem, resrc> &source, int const num_rows,
                  int const num_cols, int const start_index = 0);

  ~matrix();

  // copy constructor/assign
  matrix(matrix<P, mem, resrc> const &);
  matrix<P, mem, resrc> &operator=(matrix<P, mem, resrc> const &);

  // copy construct owner from view values
  template<mem_type omem, mem_type m_ = mem, typename = enable_for_owner<m_>,
           mem_type m__ = omem, typename = enable_for_all_views<m__>>
  explicit matrix(matrix<P, omem, resrc> const &);

  // assignment owner <-> view
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>>
  matrix<P, mem, resrc> &operator=(matrix<P, omem, resrc> const &);

  // converting construction/assign
  template<typename PP, mem_type omem, mem_type m_ = mem,
           typename = enable_for_owner<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  explicit matrix(matrix<PP, omem> const &);

  template<typename PP, mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &operator=(matrix<PP, omem> const &);

  // host to device, new matrix
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  fk::matrix<P, mem_type::owner, resource::device> clone_onto_device() const;
  // host to device copy
  template<mem_type omem, resource r_ = resrc, typename = enable_for_device<r_>>
  matrix<P, mem, resrc> &transfer_from(matrix<P, omem, resource::host> const &);
  // device to host, new matrix
  template<resource r_ = resrc, typename = enable_for_device<r_>>
  fk::matrix<P, mem_type::owner, resource::host> clone_onto_host() const;
  // device to host copy
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem, resrc> &
  transfer_from(matrix<P, omem, resource::device> const &);

  // move constructor/assign
  matrix(matrix<P, mem, resrc> &&);
  matrix<P, mem, resrc> &operator=(matrix<P, mem, resrc> &&);

  //
  // copy out of fk::vector
  //
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &operator=(fk::vector<P, omem, resrc> const &);

  //
  // subscripting operators
  //
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  P &operator()(int const, int const);

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  P operator()(int const, int const) const;

  //
  // comparison operators
  //
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator==(matrix<P, omem> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator!=(matrix<P, omem> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator<(matrix<P, omem> const &) const;

  //
  // math operators
  //
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> operator*(P const) const;
  template<mem_type omem>
  vector<P, mem_type::owner, resrc>
  operator*(vector<P, omem, resrc> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> operator*(matrix<P, omem, resrc> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> operator+(matrix<P, omem, resrc> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> operator-(matrix<P, omem, resrc> const &) const;

  template<mem_type m_ = mem, typename = enable_for_owner<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem, resrc> &transpose();

  template<mem_type m_ = mem, typename = enable_for_owner<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem, resrc> &ip_transpose();

  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> kron(matrix<P, omem> const &) const;

  template<typename U  = P,
           typename    = std::enable_if_t<std::is_floating_point_v<U> &&
                                       std::is_same_v<P, U>>,
           mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem> &invert();

  template<typename U  = P,
           typename    = std::enable_if_t<std::is_floating_point_v<U> &&
                                       std::is_same_v<P, U>>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  P determinant() const;

  //
  // basic queries to private data
  //
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  // for owners: stride == nrows
  // for views:  stride == owner's nrows
  int stride() const { return stride_; }
  int64_t size() const { return int64_t{nrows()} * ncols(); }

  bool empty() const { return size() == 0; }

  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  P const *data(int const i = 0, int const j = 0) const
  {
    // return data_ + i * stride() + j; // row-major
    return data_ + int64_t{j} * stride() + int64_t{i}; // column-major
  }
  //! \brief Non-const overload
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  P *data(int const i = 0, int const j = 0)
  {
    // return data_ + i * stride() + j; // row-major
    return data_ + int64_t{j} * stride() + int64_t{i}; // column-major
  }

  //
  // utility functions
  //
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>>
  matrix<P, mem, resrc> &
  update_col(int const, fk::vector<P, omem, resrc> const &);
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem> &update_col(int const, std::vector<P> const &);
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &update_row(int const, fk::vector<P, omem, resrc> const &);
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem> &update_row(int const, std::vector<P> const &);

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix<P, mem_type::owner, resrc> &clear_and_resize(int const, int const);

  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &set_submatrix(int const row_idx, int const col_idx,
                                fk::matrix<P, omem> const &submatrix);
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> extract_submatrix(int const row_idx, int const col_idx,
                              int const num_rows, int const num_cols) const;

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void print(std::string const label = "") const;

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void dump_to_octave(std::filesystem::path const &filename) const;

  using value_type     = P;
  using iterator       = matrix_iterator<P *, P &>;
  using const_iterator = matrix_iterator<P const *, P const &>;

  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  iterator begin()
  {
    return iterator(data(), stride(), nrows());
  }

  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
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
  // matrix view constructors (both const/nonconst) delegate to this private
  // constructor delegated is a dummy variable to assist in overload resolution
  template<mem_type m_ = mem, typename = enable_for_all_views<m_>,
           mem_type omem>
  explicit matrix(fk::matrix<P, omem, resrc> const &owner, int const start_row,
                  int const stop_row, int const start_col, int const stop_col,
                  bool const delegated);

  // matrix view from vector owner constructors (both const/nonconst) delegate
  // to this private constructor, also with a dummy variable
  template<mem_type omem, mem_type m_ = mem,
           typename = enable_for_all_views<m_>>
  explicit matrix(fk::vector<P, omem, resrc> const &source, int dummy,
                  int const num_rows, int const num_cols,
                  int const start_index);

  P *data_;    //< pointer to elements
  int nrows_;  //< row dimension
  int ncols_;  //< column dimension
  int stride_; //< leading dimension;
               // number of elements in memory between successive matrix
               // elements in a row
};

template<typename P, mem_type mem, resource resrc, mem_type omem,
         resource oresrc, typename = disable_for_const_view<mem>>
void copy_matrix(fk::matrix<P, mem, resrc> &dest,
                 fk::matrix<P, omem, oresrc> const &source);

} // namespace fk
} // namespace asgard

//
// This would otherwise be the start of the matrix.cpp, if we were still doing
// the explicit instantiations
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace asgard
{
template<typename P, mem_type mem, resource resrc, mem_type omem,
         resource oresrc, typename>
inline void fk::copy_matrix(fk::matrix<P, mem, resrc> &dest,
                            fk::matrix<P, omem, oresrc> const &source)
{
  expect(source.nrows() == dest.nrows());
  expect(source.ncols() == dest.ncols());
  if (!source.empty())
    fk::memcpy_2d<resrc, oresrc>(dest.data(), dest.stride(), source.data(),
                                 source.stride(), source.nrows(),
                                 source.ncols());
}

//-----------------------------------------------------------------------------
//
// fk::matrix class implementation starts here
//
//-----------------------------------------------------------------------------

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::matrix<P, mem, resrc>::matrix()
    : data_{nullptr}, nrows_{0}, ncols_{0}, stride_{nrows_}

{}

// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(int const m, int const n)
    : nrows_{m}, ncols_{n}, stride_{nrows_}

{
  expect(m >= 0);
  expect(n >= 0);
  allocate_resource<resrc>(data_, size());
}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(
    std::initializer_list<std::initializer_list<P>> llist)
    : nrows_{static_cast<int>(llist.size())},
      ncols_{static_cast<int>(llist.begin()->size())}, stride_{nrows_}
{
  if constexpr (resrc == resource::host)
  {
    data_       = new P[llist.size() * llist.begin()->size()]();
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
  else
  {
    fk::matrix<P, mem, resource::host> const wrap(llist);
    allocate_device(data_, llist.size() * llist.begin()->size());
    copy_matrix(*this, wrap);
  }
}

// create view from owner - const view version
// delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> const &owner,
                                  int const start_row, int const stop_row,
                                  int const start_col, int const stop_col)
    : matrix(owner, start_row, stop_row, start_col, stop_col, true)
{}

// create view from owner - modifiable view version
// delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> &owner,
                                  int const start_row, int const stop_row,
                                  int const start_col, int const stop_col)
    : matrix(owner, start_row, stop_row, start_col, stop_col, true)
{}

// overload for default case - whole matrix
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> const &owner)
    : matrix(owner, 0, std::max(0, owner.nrows() - 1), 0,
             std::max(0, owner.ncols() - 1), true)
{}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> &owner)
    : matrix(owner, 0, std::max(0, owner.nrows() - 1), 0,
             std::max(0, owner.ncols() - 1), true)
{}

// create matrix view of an existing vector
// const version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem, resrc>::matrix(fk::vector<P, omem, resrc> const &source,
                                  int const num_rows, int const num_cols,
                                  int const start_index)
    : matrix(source, 0, num_rows, num_cols, start_index)
{}

// create matrix view of existing vector
// modifiable view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(fk::vector<P, omem, resrc> &source,
                                  int const num_rows, int const num_cols,
                                  int const start_index)
    : matrix(source, 0, num_rows, num_cols, start_index)
{}

// destructor
template<typename P, mem_type mem, resource resrc>
#ifdef __clang__
fk::matrix<P, mem, resrc>::~matrix<P, mem, resrc>()
#else
fk::matrix<P, mem, resrc>::~matrix()
#endif
{
  if constexpr (mem == mem_type::owner)
  {
    delete_resource<resrc>(data_);
  }
}

//
// matrix copy constructor
//
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc>::matrix(matrix<P, mem, resrc> const &a)
    : nrows_{a.nrows()}, ncols_{a.ncols()}, stride_{a.stride()}

{
  if constexpr (mem == mem_type::owner)
  {
    allocate_resource<resrc>(data_, a.size());
    copy_matrix(*this, a);
  }
  else
  {
    data_ = const_cast<P *>(a.data());
  }
}

//
// matrix copy assignment
// this can probably be done better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc> &
fk::matrix<P, mem, resrc>::operator=(matrix<P, mem, resrc> const &a)
{
  static_assert(mem != mem_type::const_view,
                "cannot copy assign into const_view!");

  if (&a == this)
    return *this;

  expect((nrows() == a.nrows()) && (ncols() == a.ncols()));

  if constexpr (mem == mem_type::owner)
  {
    copy_matrix(*this, a);
  }
  else
  {
    data_ = const_cast<P *>(a.data());
  }

  return *this;
}

// copy construct owner from view values
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(matrix<P, omem, resrc> const &a)
    : nrows_{a.nrows()}, ncols_{a.ncols()}, stride_{a.nrows()}
{
  allocate_resource<resrc>(data_, size());
  copy_matrix(*this, a);
}

// assignment owner <-> view
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc> &
fk::matrix<P, mem, resrc>::operator=(matrix<P, omem, resrc> const &a)
{
  expect(nrows() == a.nrows());
  expect(ncols() == a.ncols());
  copy_matrix(*this, a);
  return *this;
}

//
// converting matrix copy constructor
//
template<typename P, mem_type mem, resource resrc>
template<typename PP, mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem, resrc>::matrix(matrix<PP, omem> const &a)
    : data_{new P[a.size()]()}, nrows_{a.nrows()}, ncols_{a.ncols()},
      stride_{a.nrows()}

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
template<typename P, mem_type mem, resource resrc>
template<typename PP, mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::operator=(matrix<PP, omem> const &a)
{
  expect((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  for (auto j = 0; j < a.ncols(); ++j)
    for (auto i = 0; i < a.nrows(); ++i)
    {
      (*this)(i, j) = static_cast<P>(a(i, j));
    }
  return *this;
}

// transfer functions
// host->dev, new matrix
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::matrix<P, mem_type::owner, resource::device>
fk::matrix<P, mem, resrc>::clone_onto_device() const

{
  fk::matrix<P, mem_type::owner, resource::device> a(nrows(), ncols());
  copy_matrix(a, *this);
  return a;
}

// host->dev copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P, mem, resrc> &fk::matrix<P, mem, resrc>::transfer_from(
    fk::matrix<P, omem, resource::host> const &a)
{
  expect(a.nrows() == nrows());
  expect(a.ncols() == ncols());

  copy_matrix(*this, a);

  return *this;
}

// dev->host, new matrix
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::matrix<P, mem_type::owner, resource::host>
fk::matrix<P, mem, resrc>::clone_onto_host() const

{
  fk::matrix<P> a(nrows(), ncols());
  copy_matrix(a, *this);
  return a;
}

// dev->host copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P, mem, resrc> &fk::matrix<P, mem, resrc>::transfer_from(
    matrix<P, omem, resource::device> const &a)
{
  expect(a.nrows() == nrows());
  expect(a.ncols() == ncols());
  copy_matrix(*this, a);
  return *this;
}

//
// matrix move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc>::matrix(matrix<P, mem, resrc> &&a)
    : data_{a.data_}, nrows_{a.nrows()}, ncols_{a.ncols()}, stride_{a.stride()}
{
  a.data_  = nullptr; // b/c a's destructor will be called
  a.nrows_ = 0;
  a.ncols_ = 0;
}

//
// matrix move assignment
//
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc> &
fk::matrix<P, mem, resrc>::operator=(matrix<P, mem, resrc> &&a)
{
  static_assert(mem != mem_type::const_view,
                "cannot move assign into const_view!");

  if (&a == this)
    return *this;

  std::swap(data_, a.data_);
  std::swap(nrows_, a.nrows_);
  std::swap(ncols_, a.ncols_);
  std::swap(stride_, a.stride_);

  return *this;
}

//
// copy out of fk::vector - assumes the vector is column-major
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::operator=(fk::vector<P, omem, resrc> const &v)
{
  expect(nrows() * ncols() == v.size());

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      (*this)(i, j) = v(j + i * ncols());

  return *this;
}

//
// matrix subscript operator - (row, col)
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
inline P &fk::matrix<P, mem, resrc>::operator()(int const i, int const j)
{
  expect(i < nrows() && j < ncols());
  return *(data(i, j));
}

template<typename P, mem_type mem, resource resrc>
template<resource, typename>
inline P fk::matrix<P, mem, resrc>::operator()(int const i, int const j) const
{
  expect(i < nrows() && j < ncols());
  return *(data(i, j));
}

// matrix comparison operators - set default tolerance above
// see https://stackoverflow.com/a/253874/6595797
// FIXME we may need to be more careful with these comparisons
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::matrix<P, mem, resrc>::operator==(matrix<P, omem> const &other) const
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
      if constexpr (std::is_floating_point_v<P>)
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

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::matrix<P, mem, resrc>::operator!=(matrix<P, omem> const &other) const
{
  return !(*this == other);
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::matrix<P, mem, resrc>::operator<(matrix<P, omem> const &other) const
{
  return std::lexicographical_compare(this->begin(), this->end(), other.begin(),
                                      other.end());
}

//
// matrix addition operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P>
fk::matrix<P, mem, resrc>::operator+(matrix<P, omem, resrc> const &right) const
{
  expect(nrows() == right.nrows() && ncols() == right.ncols());

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
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P>
fk::matrix<P, mem, resrc>::operator-(matrix<P, omem, resrc> const &right) const
{
  expect(nrows() == right.nrows() && ncols() == right.ncols());

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
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::matrix<P> fk::matrix<P, mem, resrc>::operator*(P const right) const
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
template<typename P, mem_type mem, resource resrc>
template<mem_type omem>
fk::vector<P, mem_type::owner, resrc> fk::matrix<P, mem, resrc>::operator*(
    fk::vector<P, omem, resrc> const &right) const
{
  // check dimension compatibility
  expect(ncols() == right.size());

  matrix<P, mem, resrc> const &A = (*this);
  vector<P, mem_type::owner, resrc> Y(A.nrows());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = A.stride();
  int one_i = 1;

  P one  = 1.0;
  P zero = 0.0;
  lib_dispatch::gemv<resrc>('n', m, n, one, A.data(), lda, right.data(), one_i,
                            zero, Y.data(), one_i);

  return Y;
}

//
// matrix*matrix multiplication operator C[m,n] = A[m,k] * B[k,n]
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P>
fk::matrix<P, mem, resrc>::operator*(matrix<P, omem, resrc> const &B) const
{
  expect(ncols() == B.nrows()); // k == k

  // just aliases for easier reading
  matrix const &A = (*this);
  matrix<P> C(A.nrows(), B.ncols());

  lib_dispatch::gemm('n', 'n', A.nrows(), B.ncols(), B.nrows(), P{1}, A.data(),
                     A.stride(), B.data(), B.stride(), P{0}, C.data(),
                     C.stride());

  return C;
}

/* in-place matrix transpose for column major data layout */
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::matrix<P, mem, resrc> &fk::matrix<P, mem, resrc>::transpose()
{
  /* empty matrix */
  if (size() == 0)
    return *this;

  /* vector pretending to be a matrix */
  if (nrows_ == 1 || ncols_ == 1)
  {
    std::swap(nrows_, ncols_);
    stride_ = nrows_;
    return *this;
  }

  /* square matrix */
  if (nrows_ == ncols_)
  {
    for (int r = 0; r < nrows_; ++r)
    {
      for (int c = 0; c < r; ++c)
      {
        std::swap(data_[c * nrows_ + r], data_[r * nrows_ + c]);
      }
    }
    return *this;
  }

  /* spot for each element, true ~ visited, false ~ unvisited */
  std::vector<bool> visited(size() - 2, false);

  /* Given index "pos" in a linear array interpreted as a matrix of "nrows_"
     rows, n_cols_ columns, and column-major data layout, return the linear
     index position of the element in the matrix's transpose */
  auto const remap_index = [this](int const pos) -> int {
    int const row         = pos % nrows_;
    int const col         = pos / nrows_;
    int const destination = row * ncols_ + col;
    return destination;
  };

  /* The first and last elements never change position and can be ignored */
  for (int pos = 1; pos < size() - 1; ++pos)
  {
    if (visited[pos])
      continue;

    P save = data_[pos];

    int next_pos = remap_index(pos);

    while (!visited[next_pos])
    {
      std::swap(save, data_[next_pos]);
      visited[next_pos] = true;
      next_pos          = remap_index(next_pos);
    }
  }

  std::swap(nrows_, ncols_);
  stride_ = nrows_;

  return *this;
}

// Simple quad-loop kron prod
// @return the product
//
// FIXME this is NOT optimized.
// we will use the other methods
// for performance-critical (large)
// krons
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P> fk::matrix<P, mem, resrc>::kron(matrix<P, omem> const &B) const
{
  fk::matrix<P> C(nrows() * B.nrows(), ncols() * B.ncols());

  auto const ie = nrows();
  auto const je = ncols();
  auto const ke = B.nrows();
  auto const le = B.ncols();

  //  Matrix data(i,j) assume column major ordering. So j and l should be the
  //  slowest iterating index for best cache utilization. Swap these for row
  //  major order.
  for (auto j = 0; j < je; ++j)
  {
    for (auto i = 0; i < ie; ++i)
    {
      for (auto l = 0; l < le; ++l)
      {
        for (auto k = 0; k < ke; ++k)
        {
          C((i * ke + k), (j * le + l)) += (*this)(i, j) * B(k, l);
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
template<typename P, mem_type mem, resource resrc>
template<typename U, typename, mem_type, typename, resource, typename>
fk::matrix<P, mem> &fk::matrix<P, mem, resrc>::invert()
{
  static_assert(resrc == resource::host);
  expect(nrows() == ncols());

  std::vector<int> ipiv(ncols());
  int lwork{static_cast<int>(size())};
  std::vector<P> work(size());

  int info =
      lib_dispatch::getrf(ncols_, ncols_, data(0, 0), stride(), ipiv.data());
  if (info != 0)
    throw std::runtime_error("Error returned from lib_dispatch::getrf: " +
                             std::to_string(info));
  info = lib_dispatch::getri(ncols_, data(0, 0), stride(), ipiv.data(),
                             work.data(), lwork);
  if (info != 0)
    throw std::runtime_error("Error returned from lib_dispatch::getri: " +
                             std::to_string(info));
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
template<typename P, mem_type mem, resource resrc>
template<typename U, typename, resource, typename>
P fk::matrix<P, mem, resrc>::determinant() const
{
  expect(nrows() == ncols());

  matrix<P, mem_type::owner> temp(*this); // get temp copy to do LU
  std::vector<int> ipiv(ncols());
  int info = lib_dispatch::getrf(temp.nrows(), temp.ncols(), temp.data(0, 0),
                                 temp.stride(), ipiv.data());
  if (info != 0)
    throw std::runtime_error("Error returned from lib_dispatch::getrf: " +
                             std::to_string(info));

  P det    = 1.0;
  int sign = 1;
  for (auto i = 0; i < nrows(); ++i)
  {
    if (ipiv[i] != i + 1)
      sign *= -1;
    det *= temp(i, i);
  }
  det *= static_cast<P>(sign);
  return det;
}

//
// Update a specific col of a matrix, given a fk::vector<P> (overwrites
// original)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc> &
fk::matrix<P, mem, resrc>::update_col(int const col_idx,
                                      fk::vector<P, omem, resrc> const &v)
{
  expect(nrows() == static_cast<int>(v.size()));
  expect(col_idx < ncols());

  lib_dispatch::copy<resrc>(v.size(), v.data(), data(0, col_idx));

  return *this;
}

//
// Update a specific col of a matrix, given a std::vector (overwrites original)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::update_col(int const col_idx,
                                      std::vector<P> const &v)
{
  expect(nrows() == static_cast<int>(v.size()));
  expect(col_idx < ncols());

  lib_dispatch::copy<resrc>(v.size(), v.data(), data(0, col_idx));

  return *this;
}

//
// Update a specific row of a matrix, given a fk::vector<P> (overwrites
// original)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::update_row(int const row_idx,
                                      fk::vector<P, omem, resrc> const &v)
{
  expect(ncols() == v.size());
  expect(row_idx < nrows());

  lib_dispatch::copy<resrc>(v.size(), v.data(), 1, data(row_idx, 0), stride());

  return *this;
}

//
// Update a specific row of a matrix, given a std::vector (overwrites original)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::update_row(int const row_idx,
                                      std::vector<P> const &v)
{
  expect(ncols() == static_cast<int>(v.size()));
  expect(row_idx < nrows());

  lib_dispatch::copy<resrc>(v.size(), v.data(), 1, data(row_idx, 0), stride());

  return *this;
}

//
// Resize, clearing all data
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::matrix<P, mem_type::owner, resrc> &
fk::matrix<P, mem, resrc>::clear_and_resize(int const rows, int const cols)
{
  expect(rows >= 0);
  expect(cols >= 0);
  if (rows == 0 || cols == 0)
    expect(cols == rows);

  delete_resource<resrc>(data_);
  allocate_resource<resrc>(data_, int64_t{rows} * cols);

  nrows_  = rows;
  ncols_  = cols;
  stride_ = nrows_;
  return *this;
}

//
// Set a submatrix within the matrix, given another (smaller) matrix
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::set_submatrix(int const row_idx, int const col_idx,
                                         matrix<P, omem> const &submatrix)
{
  expect(row_idx >= 0);
  expect(col_idx >= 0);
  expect(row_idx + submatrix.nrows() <= nrows());
  expect(col_idx + submatrix.ncols() <= ncols());

  matrix &mat = *this;
  for (auto j = 0; j < submatrix.ncols(); ++j)
  {
    for (auto i = 0; i < submatrix.nrows(); ++i)
    {
      mat(i + row_idx, j + col_idx) = submatrix(i, j);
    }
  }
  return mat;
}

//
// Extract a rectangular submatrix from within the matrix
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::matrix<P>
fk::matrix<P, mem, resrc>::extract_submatrix(int const row_idx,
                                             int const col_idx,
                                             int const num_rows,
                                             int const num_cols) const
{
  expect(row_idx >= 0);
  expect(col_idx >= 0);
  expect(row_idx + num_rows <= nrows());
  expect(col_idx + num_cols <= ncols());

  matrix<P> submatrix(num_rows, num_cols);
  matrix const &mat = *this;
  for (auto j = 0; j < num_cols; ++j)
  {
    for (auto i = 0; i < num_rows; ++i)
    {
      submatrix(i, j) = mat(i + row_idx, j + col_idx);
    }
  }

  return submatrix;
}

// Prints out the values of a matrix
// @return  Nothing
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::matrix<P, mem, resrc>::print(std::string label) const
{
  if constexpr (mem == mem_type::owner)
    std::cout << label << "(owner)" << '\n';

  else if constexpr (mem == mem_type::view)
    std::cout << label << "(view, "
              << "stride == " << std::to_string(stride()) << ")" << '\n';

  else if constexpr (mem == mem_type::const_view)
    std::cout << label << "(const view, "
              << "stride == " << std::to_string(stride()) << ")" << '\n';
  else
    expect(false); // above cases cover all implemented mem types

  //  Print these out as row major even though stored in memory as column major.
  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
    {
      if constexpr (std::is_floating_point_v<P>)
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
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::matrix<P, mem, resrc>::dump_to_octave(
    std::filesystem::path const &filename) const
{
  std::ofstream ofile(filename);
  auto coutbuf = std::cout.rdbuf(ofile.rdbuf());
  //  Print these out as row major even though stored in memory as column major.
  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
      std::cout << std::setprecision(12) << (*this)(i, j) << " ";

    std::cout << std::setprecision(4) << '\n';
  }
  std::cout.rdbuf(coutbuf);
}

// public const/nonconst view constructors delegate to this private
// constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> const &owner,
                                  int const start_row, int const stop_row,
                                  int const start_col, int const stop_col,
                                  bool const delegated)
{
  ignore(delegated);
  data_   = nullptr;
  nrows_  = 0;
  ncols_  = 0;
  stride_ = 0;

  int const view_rows = stop_row - start_row + 1;
  int const view_cols = stop_col - start_col + 1;
  if (owner.size() > 0)
  {
    expect(start_row >= 0);
    expect(start_col >= 0);
    expect(stop_col < owner.ncols());
    expect(stop_row < owner.nrows());
    expect(stop_row >= start_row);

    // OK to alias here, const is enforced by the "const_view" vs. "view"
    data_   = const_cast<P *>(owner.data(start_row, start_col));
    nrows_  = view_rows;
    ncols_  = view_cols;
    stride_ = owner.stride();
  }
}

// public const/nonconst matrix view from vector constructors delegate to
// this private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(fk::vector<P, omem, resrc> const &source, int,
                                  int const num_rows, int const num_cols,
                                  int const start_index)
{
  expect(start_index >= 0);
  expect(num_rows > 0);
  expect(num_cols > 0);

  int64_t const size = int64_t{num_rows} * num_cols;
  expect(start_index + size <= source.size());

  data_   = nullptr;
  nrows_  = 0;
  ncols_  = 0;
  stride_ = 0;

  if (size > 0)
  {
    // casting for the creation of a view (OK to alias)
    data_   = const_cast<P *>(source.data(start_index));
    nrows_  = num_rows;
    ncols_  = num_cols;
    stride_ = num_rows;
  }
}

template<typename P, mem_type mem, resource resrc>
template<typename T, typename R>
class fk::matrix<P, mem, resrc>::matrix_iterator
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

    if (!(next_pos % rows_))
    {
      start_ += stride_;
      return stride_ - rows_ + 1;
    }

    return 1;
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

template<typename P, mem_type left_mem, mem_type right_mem>
void debug_compare(fk::matrix<P, left_mem> const &left,
                   fk::matrix<P, right_mem> const &right)
{
  expect(left.nrows() == right.nrows());
  expect(left.ncols() == right.ncols());

  static std::string const red("\033[0;31m");
  static std::string const reset("\033[0m");

  //  Print these out as row major even though stored in memory as column major.
  for (auto i = 0; i < left.nrows(); ++i)
  {
    for (auto j = 0; j < left.ncols(); ++j)
    {
      if constexpr (std::is_floating_point_v<P>)
      {
        if (std::abs(left(i, j) - right(i, j)) > TOL)
        {
          std::cout << red;
        }

        std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                  << std::right << left(i, j) << reset;
      }
      else
      {
        if (left(i, j) != right(i, j))
        {
          std::cout << red;
        }
        std::cout << std::right << left(i, j) << reset << " ";
      }
    }

    std::cout << '\n';
  }
}
} // namespace asgard
