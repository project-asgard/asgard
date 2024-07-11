#pragma once

#include "asgard_resources.hpp"

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
class matrix;

/*! One-dimensional tensor managing memory allocation and destruction.
 */
template<typename P, mem_type mem = mem_type::owner,
         resource resrc = resource::host>
class vector
{
  // all types of vectors are mutual friends
  template<typename, mem_type, resource>
  friend class vector;

public:
  //! \brief Value type of the container
  using value_type = P;
  /*! constructor
   * \brief creates an empty vector.
   */
  vector();
  /*! constructor
   * \param size size of newly constructed vector.
   */
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  explicit vector(int const size);
  /*! constructor
   * \param list initial values of vector.
   */
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector(std::initializer_list<P> list);
  /*! copy constructor
   * \param other vector
   */
  template<mem_type m_ = mem, typename = enable_for_owner<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  vector(std::vector<P> const &other);
  /*! copy constructor
   * \param other vector
   */
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector(fk::matrix<P, mem_type::owner, resrc> const &other);
  /*! copy constructor returns non-owning view of data
   * \param vec vector owning data array
   * \param start_index first element contained in view (inclusive)
   * \param stop_index last element contained in view (inclusive)
   */
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit vector(fk::vector<P, omem, resrc> &vec, int const start_index,
                  int const stop_index);
  /*! copy constructor returns const non-owning view of data
   * \param vec vector owning data array
   * \param start_index first element contained in view (inclusive)
   * \param stop_index last element contained in view (inclusive)
   */
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit vector(fk::vector<P, omem, resrc> const &vec, int const start_index,
                  int const stop_index);
  /*! copy constructor
   * overloads for default case - whole vector
   * \param owner vector
   */
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit vector(fk::vector<P, omem, resrc> &owner);

  /*! copy constructor
   * overloads for default case - whole vector
   * \param owner vector
   */
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit vector(fk::vector<P, omem, resrc> const &owner);

  /*! create vector view from matrix
   *  const view version
   * \param source matrix owning data array
   * \param col_index column contained in view
   * \param row_start first element contained in view (inclusive)
   * \param row_stop last element contained in view (inclusive)
   */
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit vector(fk::matrix<P, omem, resrc> const &source, int const col_index,
                  int const row_start, int const row_stop);
  /*! create vector view from matrix
   * modifiable view version
   * \param source matrix owning data array
   * \param col_index column contained in view
   * \param row_start first element contained in view (inclusive)
   * \param row_stop last element contained in view (inclusive)
   */
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit vector(fk::matrix<P, omem, resrc> &source, int const col_index,
                  int const row_start, int const row_stop);

  /*! destructor
   */
  ~vector();

  /*! copy constructor/assignment
   * \param other required to be same to same type, T==T
   */
  vector(vector<P, mem, resrc> const &other);
  /*! copy constructor/assignment
   * \param other
   * cannot be templated per C++ spec 12.8
   * instead of disabling w/ sfinae for const_view,
   * static assert added to definition
   */
  vector<P, mem, resrc> &operator=(vector<P, mem, resrc> const &other);

  /*! move constructor/assignment
   * \param other required to be same to same type, T==T
   */
  vector(vector<P, mem, resrc> &&other);

  /*! move constructor/assignment
   * \param other
   * as with copy assignment, static assert added
   * to definition to prevent assignment into
   * const views
   */
  vector<P, mem, resrc> &operator=(vector<P, mem, resrc> &&other);

  /*! copy constructor creates owner from views
   * \param other view used to create new owner
   */
  template<mem_type omem, mem_type m_ = mem, typename = enable_for_owner<m_>,
           mem_type m__ = omem, typename = enable_for_all_views<m__>>
  explicit vector(vector<P, omem, resrc> const &other);

  /*! copy assignment creates owner from views
   * \param other view used to create new owner
   */
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>>
  vector<P, mem, resrc> &operator=(vector<P, omem, resrc> const &);

  /*! Copy from host memory to device memory
   *  \return new device vector
   */
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P, mem_type::owner, resource::device> clone_onto_device() const;
  /*! Copy from host memory to device memory
   *  \param other vector containing host memory
   *  \return vector containing device memory
   */
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_device<r_>>
  vector<P, mem, resrc> &
  transfer_from(vector<P, omem, resource::host> const &other);
  /*! Copy from device memory to host memory
   *  \return new host vector
   */
  template<resource r_ = resrc, typename = enable_for_device<r_>>
  vector<P, mem_type::owner, resource::host> clone_onto_host() const;
  /*! Copy from device memory to host memory
   *  \param other vector containing device memory
   *  \return vector containing host memory
   */
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  vector<P, mem, resrc> &
  transfer_from(vector<P, omem, resource::device> const &);

  /*! Copy data out of std::vector
   *  \param other input data
   *  \return ASGarD vector
   */
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P, mem> &operator=(std::vector<P> const &other);

  /*! Copy array into std::vector
   *  \return data copied to a std::vector
   */
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  std::vector<P> to_std() const;

  /*! subscript operator
   * \param i position of the element to return
   * \returns reference to the requested element.
   */
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  P &operator()(int const);
  /*! subscript operator
   * \param i position of the element to return
   * \returns const reference to the requested element.
   */
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  P const &operator()(int const) const;

  /*! array index operator
   * \param i position of the element to return
   * \returns reference to the requested element.
   */
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  P &operator[](int const i);
  /*! array index operator
   * \param i position of the element to return
   * \returns const reference to the requested element.
   */
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  P const &operator[](int const i) const;

  // comparison operators

  /*! Checks if the contents of this and other are equal. They must have the
   *  same number of elements and each element compares equal with the element
   *  at the same position.
   *  \param other vector this is to be compared against
   *  \return true if vectors are equal
   */
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator==(vector<P, omem> const &other) const;
  /*! Checks if the contents of this and other are not equal.
   *  \param other vector this is to be compared to
   *  \return true if vectors are not equal
   */
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator!=(vector<P, omem> const &other) const;
  /*! Compares the contents of this and other lexicographically.
   *  \param other vector this is being compared to
   *  \return result of lexicographical comparison
   */
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator<(vector<P, omem> const &other) const;

  // math operators

  /*! element-wise addition
   *  \param right elements on rhs for addition.
   *  \return vector with results of element-wise addition.
   */
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator+(vector<P, omem> const &right) const;
  /*! element-wise subtraction
   *  \param right elements on rhs for subtraction.
   *  \return vector with results of element-wise subtraction.
   */
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator-(vector<P, omem> const &right) const;
  /*! dot product of two vectors
   *  \param right elements on rhs for dot product.
   *  \return scalar result of dot product.
   */
  template<mem_type omem>
  P operator*(vector<P, omem, resrc> const &) const;
  /*! vector*matrix product
   *  \param right Matrix on RHS.
   *  \return result of vector*matrix multiplication .
   */
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator*(matrix<P, omem, resrc> const &) const;
  /*! vector*scalar product
   *  \param value value to multiply each element by
   *  \return result of vector*scalar product.
   */
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator*(P const) const;

  /*! perform the matrix kronecker product by interpreting vector
   *  operands/return vector as single column matrices.
   *  \param right RHS of kronecker product
   *  \return result of kronecker product
   */
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> single_column_kron(vector<P, omem> const &right) const;

  /*! inplace vector*scalar product
   *  \param x multiply each element by x
   *  \return reference to this vector
   */
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  vector<P, mem, resrc> &scale(P const x);

  // basic queries to private data

  /*! size of container
   * \return number of elements in the container
   */
  int size() const { return size_; }

  /*! size of container
   * \return number of elements in the container
   */
  bool empty() const { return size_ == 0; }

  /*! just get a pointer. cannot deref/assign. for e.g. blas
   *  Use subscript operators for general purpose access
   *  \param elem offset used for views.
   *  \return pointer to private data
   */
  P const *data(int const elem = 0) const { return data_ + elem; }
  //! \brief Non-const overload
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  P *data(int const elem = 0)
  {
    return data_ + elem;
  }

  // utility functions

  /*! Prints out the values of a vector
   *  \param label a string label printed with the output
   */
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void print(std::string_view const label = "") const;

  /*! Dumps to file a vector that can be read straight into octave
   *  \param filename name
   */
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void dump_to_octave(std::filesystem::path const &filename) const;

  /*! resize the vector
   *  \param size of vector after update.
   *  \return reference to this
   */
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  fk::vector<P, mem_type::owner, resrc> &resize(int const new_size)
  {
    expect(new_size >= 0);
    if (new_size == size_)
      return *this;

    P *old_data{data_};
    allocate_resource<resrc>(data_, new_size);
    if (size_ > 0 && new_size > 0)
      memcpy_1d<resrc, resrc>(data_, old_data, std::min(size_, new_size));
    delete_resource<resrc>(old_data);
    size_ = new_size;
    return *this;
  }

  /*! set a subvector beginning at provided index
   * \param index first element to set
   * \param sub_vector container of elements to assign
   * \return reference to this
   */
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  vector<P, mem> &
  set_subvector(int const index, vector<P, omem> const sub_vector);

  /*! extract subvector, indices inclusive
   * \param start first element to include
   * \param stop last element to include
   * \return container with elements [start, stop]
   */
  vector<P, mem_type::owner, resrc>
  extract(int const start, int const stop) const;

  /*! append vector to the end of the container
   * \param right elements added to the end of the container
   * \param return reference to this
   */
  template<mem_type omem, mem_type m_ = mem, typename = enable_for_owner<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P, mem> &concat(vector<P, omem> const &right);

  /*! Using raw pointer as iterator */
  typedef P *iterator;
  /*! Using raw pointer as iterator */
  typedef const P *const_iterator;

  /*!
   * \return iterator pointing to zeroth element of array.
   */
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  iterator begin()
  {
    return data();
  }

  /*!
   * \return iterator pointing to the end of the array.
   */
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  iterator end()
  {
    return data() + size();
  }

  /*!
   * \return const iterator pointing to zeroth element of array.
   */
  const_iterator begin() const { return data(); }
  /*!
   * \return const iterator pointing to the end of the array.
   */
  const_iterator end() const { return data() + size(); }

private:
  // const/nonconst view constructors delegate to this private constructor
  // delegated is a dummy variable to enable resolution
  template<mem_type m_ = mem, typename = enable_for_all_views<m_>,
           mem_type omem>
  explicit vector(fk::vector<P, omem, resrc> const &vec, int const start_index,
                  int const stop_index, bool const delegated);

  // vector view from matrix constructors (both const/nonconst) delegate
  // to this private constructor, also with a dummy variable
  template<mem_type omem, mem_type m_ = mem,
           typename = enable_for_all_views<m_>>
  explicit vector(fk::matrix<P, omem, resrc> const &source, int dummy,
                  int const column_index, int const row_start,
                  int const row_stop);

  //! pointer to elements
  P *data_;
  //! size of container
  int size_;
};

template<typename P, mem_type mem, resource resrc, mem_type omem,
         resource oresrc, typename = disable_for_const_view<mem>>
void copy_vector(fk::vector<P, mem, resrc> &dest,
                 fk::vector<P, omem, oresrc> const &source);

} // namespace fk
} // namespace asgard

//
// This would otherwise be the start of the tensors.cpp, if we were still doing
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
inline void fk::copy_vector(fk::vector<P, mem, resrc> &dest,
                            fk::vector<P, omem, oresrc> const &source)
{
  expect(source.size() == dest.size());
  if (!source.empty())
    fk::memcpy_1d<resrc, oresrc>(dest.data(), source.data(), source.size());
}

//-----------------------------------------------------------------------------
//
// fk::vector class implementation starts here
//
//-----------------------------------------------------------------------------
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::vector() : data_{nullptr}, size_{0}
{}
// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(int const size) : size_{size}
{
  expect(size >= 0);
  allocate_resource<resrc>(data_, size_);
}

// can also do this with variadic template constructor for constness
// https://stackoverflow.com/a/5549918
// but possibly this is "too clever" for our needs right now
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(std::initializer_list<P> list)
    : size_{static_cast<int>(list.size())}
{
  allocate_resource<resrc>(data_, size_, false);
  memcpy_1d<resrc, resource::host>(data_, list.begin(), size_);
}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::vector<P, mem, resrc>::vector(std::vector<P> const &v)
    : data_{new P[v.size()]}, size_{static_cast<int>(v.size())}
{
  std::copy(v.begin(), v.end(), data_);
}

//
// matrix conversion constructor linearizes the matrix, i.e. stacks the columns
// of the matrix into a single vector
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(
    fk::matrix<P, mem_type::owner, resrc> const &mat)
    : data_(nullptr), size_(mat.size())
{
  if (size_ != 0)
  {
    allocate_resource<resrc>(data_, size_, false);
    memcpy_1d<resrc, resrc>(data_, mat.data(), mat.size());
  }
}

// vector view constructor given a start and stop index
// modifiable view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> &vec,
                                  int const start_index, int const stop_index)
    : vector(vec, start_index, stop_index, true)
{}

// const view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &vec,
                                  int const start_index, int const stop_index)
    : vector(vec, start_index, stop_index, true)
{}

// delegating constructor to extract view from owner. overload for default case
// of viewing the entire owner
// const view version
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &a)
    : vector(a, 0, std::max(0, a.size() - 1), true)
{}

// modifiable view version
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> &a)
    : vector(a, 0, std::max(0, a.size() - 1), true)
{}

// create vector view of an existing matrix
// const version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> const &source,
                                  int const column_index, int const row_start,
                                  int const row_stop)
    : vector(source, 0, column_index, row_start, row_stop)
{}

// modifiable view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> &source,
                                  int const column_index, int const row_start,
                                  int const row_stop)
    : vector(source, 0, column_index, row_start, row_stop)
{}

template<typename P, mem_type mem, resource resrc>
#ifdef __clang__
fk::vector<P, mem, resrc>::~vector<P, mem, resrc>()
#else
fk::vector<P, mem, resrc>::~vector()
#endif
{
  if constexpr (mem == mem_type::owner)
  {
    delete_resource<resrc>(data_);
  }
}

//
// vector copy constructor for like types (like types only)
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::vector(vector<P, mem, resrc> const &a)
    : size_{a.size_}
{
  if constexpr (mem == mem_type::owner)
  {
    allocate_resource<resrc>(data_, a.size(), false);
    copy_vector(*this, a);
  }
  else
  {
    // working with view, OK to alias
    data_ = const_cast<P *>(a.data_);
  }
}

//
// vector copy assignment
// this can probably be optimized better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, mem, resrc> const &a)
{
  static_assert(mem != mem_type::const_view,
                "cannot copy assign into const_view!");

  if (&a == this)
    return *this;

  expect(size() == a.size());

  copy_vector(*this, a);

  return *this;
}

//
// vector move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::vector(vector<P, mem, resrc> &&a)
    : data_{a.data_}, size_{a.size_}
{
  a.data_ = nullptr; // b/c a's destructor will be called
  a.size_ = 0;
}

//
// vector move assignment
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, mem, resrc> &&a)
{
  static_assert(mem != mem_type::const_view,
                "cannot move assign into const_view!");

  if (&a == this)
    return *this;

  size_ = a.size_;
  P *const temp{data_};
  data_   = a.data_;
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

// copy construct owner from view values
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, mem_type, typename>
fk::vector<P, mem, resrc>::vector(vector<P, omem, resrc> const &a)
    : size_(a.size())
{
  allocate_resource<resrc>(data_, a.size(), false);
  copy_vector(*this, a);
}

// assignment owner <-> view
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, omem, resrc> const &a)
{
  expect(size() == a.size());
  copy_vector(*this, a);
  return *this;
}

// transfer functions
// host->dev, new vector
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P, mem_type::owner, resource::device>
fk::vector<P, mem, resrc>::clone_onto_device() const

{
  fk::vector<P, mem_type::owner, resource::device> a(size());
  copy_vector(a, *this);
  return a;
}

// host->dev copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem, resrc> &fk::vector<P, mem, resrc>::transfer_from(
    fk::vector<P, omem, resource::host> const &a)
{
  expect(a.size() == size());
  copy_vector(*this, a);
  return *this;
}

// dev -> host, new vector
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P, mem_type::owner, resource::host>
fk::vector<P, mem, resrc>::clone_onto_host() const

{
  fk::vector<P> a(size());
  copy_vector(a, *this);
  return a;
}

// dev -> host copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem, resrc> &fk::vector<P, mem, resrc>::transfer_from(
    vector<P, omem, resource::device> const &a)
{
  expect(a.size() == size());
  copy_vector(*this, a);
  return *this;
}

//
// copy out of std::vector
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::operator=(std::vector<P> const &v)
{
  expect(size() == static_cast<int>(v.size()));
  std::memcpy(data_, v.data(), v.size() * sizeof(P));
  return *this;
}

//
// copy into std::vector
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
std::vector<P> fk::vector<P, mem, resrc>::to_std() const
{
  return std::vector<P>(data(), data() + size());
}

// vector subscript operator
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
P &fk::vector<P, mem, resrc>::operator()(int i)
{
  expect(i < size_);
  return data_[i];
}

template<typename P, mem_type mem, resource resrc>
template<resource, typename>
P const &fk::vector<P, mem, resrc>::operator()(int i) const
{
  expect(i < size_);
  return data_[i];
}

// array index operators
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
P &fk::vector<P, mem, resrc>::operator[](int i)
{
  expect(i < size_);
  return data_[i];
}
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
P const &fk::vector<P, mem, resrc>::operator[](int i) const
{
  expect(i < size_);
  return data_[i];
}

// vector comparison operators - set default tolerance above
// see https://stackoverflow.com/a/253874/6595797
// FIXME do we need to be more careful with these fp comparisons?
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator==(vector<P, omem> const &other) const
{
  if constexpr (omem == mem)
    if (&other == this)
      return true;
  if (size() != other.size())
    return false;
  for (auto i = 0; i < size(); ++i)
    if constexpr (std::is_floating_point_v<P>)
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
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator!=(vector<P, omem> const &other) const
{
  return !(*this == other);
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator<(vector<P, omem> const &other) const
{
  return std::lexicographical_compare(begin(), end(), other.begin(),
                                      other.end());
}

//
// vector addition operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::operator+(vector<P, omem> const &right) const
{
  expect(size() == right.size());
  vector<P> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i) + right(i);
  return ans;
}

//
// vector subtraction operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::operator-(vector<P, omem> const &right) const
{
  expect(size() == right.size());
  vector<P> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i)-right(i);
  return ans;
}

//
// vector*vector multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem>
P fk::vector<P, mem, resrc>::operator*(
    vector<P, omem, resrc> const &right) const
{
  expect(size() == right.size());
  vector const &X = (*this);
  return lib_dispatch::dot<resrc>(size(), X.data(), 1, right.data(), 1);
}

//
// vector*matrix multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::operator*(fk::matrix<P, omem, resrc> const &A) const
{
  // check dimension compatibility
  expect(size() == A.nrows());

  vector const &X = (*this);
  vector<P> Y(A.ncols());
  lib_dispatch::gemv<resrc>('t', A.nrows(), A.ncols(), P{1.0}, A.data(),
                            A.stride(), X.data(), 1, P{0.0}, Y.data(), 1);
  return Y;
}

//
// vector*scalar multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P> fk::vector<P, mem, resrc>::operator*(P const x) const
{
  vector<P> a(*this);
  lib_dispatch::scal(a.size(), x, a.data(), 1);
  return a;
}

//
// perform the matrix kronecker product by
// interpreting vector operands/return vector
// as single column matrices.
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P> fk::vector<P, mem, resrc>::single_column_kron(
    vector<P, omem> const &right) const
{
  fk::vector<P> product((*this).size() * right.size());
  for (int j = 0; j < right.size(); ++j)
  {
    for (int i = 0; i < (*this).size(); ++i)
    {
      product(i * right.size() + j) = (*this)(i)*right(j);
    }
  }
  return product;
}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc> &fk::vector<P, mem, resrc>::scale(P const x)
{
  lib_dispatch::scal<resrc>(size(), x, data(), 1);
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
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::vector<P, mem, resrc>::print(std::string_view const label) const
{
  if constexpr (mem == mem_type::owner)
    std::cout << label << "(owner)" << '\n';
  else if constexpr (mem == mem_type::view)
    std::cout << label << "(view)" << '\n';
  else if constexpr (mem == mem_type::const_view)
    std::cout << label << "(const view)" << '\n';
  else
    expect(false); // above cases should cover all implemented types

  if constexpr (std::is_floating_point_v<P>)
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
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::vector<P, mem, resrc>::dump_to_octave(
    std::filesystem::path const &filename) const
{
  std::ofstream ofile(filename);
  auto coutbuf = std::cout.rdbuf(ofile.rdbuf());
  for (auto i = 0; i < size(); ++i)
    std::cout << std::setprecision(12) << (*this)(i) << " ";

  std::cout.rdbuf(coutbuf);
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::concat(vector<P, omem> const &right)
{
  int const old_size = this->size();
  int const new_size = this->size() + right.size();
  P *old_data{data_};
  data_ = new P[new_size];
  std::memcpy(data_, old_data, old_size * sizeof(P));
  std::memcpy(data(old_size), right.data(), right.size() * sizeof(P));
  size_ = new_size;
  delete[] old_data;
  return *this;
}

// set a subvector beginning at provided index
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::set_subvector(int const index,
                                         fk::vector<P, omem> const sub_vector)
{
  expect(index >= 0);
  expect((index + sub_vector.size()) <= this->size());
  std::memcpy(&(*this)(index), sub_vector.data(),
              sub_vector.size() * sizeof(P));
  return *this;
}

// extract subvector, indices inclusive
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem_type::owner, resrc>
fk::vector<P, mem, resrc>::extract(int const start, int const stop) const
{
  expect(start >= 0);
  expect(stop < this->size());
  expect(stop >= start);

  int const sub_size = stop - start + 1;
  fk::vector<P, mem_type::owner, resrc> sub_vector(sub_size);
  lib_dispatch::copy<resrc>(sub_size, data(start), sub_vector.data());
  return sub_vector;
}

// const/nonconst view constructors delegate to this private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &vec,
                                  int const start_index, int const stop_index,
                                  bool const delegated)
{
  // ignore dummy argument to avoid compiler warnings
  ignore(delegated);
  data_ = nullptr;
  size_ = 0;
  if (vec.size() > 0)
  {
    expect(start_index >= 0);
    expect(stop_index < vec.size());
    expect(stop_index >= start_index);

    data_ = vec.data_ + start_index;
    size_ = stop_index - start_index + 1;
  }
}

// public const/nonconst vector view from matrix constructors delegate to
// this private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> const &source, int,
                                  int const column_index, int const row_start,
                                  int const row_stop)
{
  expect(column_index >= 0);
  expect(column_index < source.ncols());
  expect(row_start >= 0);
  expect(row_start <= row_stop);
  expect(row_stop < source.nrows());

  data_ = nullptr;
  size_ = row_stop - row_start + 1;

  if (size_ > 0)
  {
    data_ = const_cast<P *>(source.data(
        int64_t{column_index} * source.stride() + int64_t{row_start}));
  }
}

#ifdef ASGARD_USE_CUDA

namespace gpu
{
/*!
 * \brief Simple container for GPU data, interoperable with std::vector
 *
 * This simple container allows for RAII memory management,
 * resizing (without relocating the data) and easy copy from/to std::vector
 */
template<typename T>
class vector
{
public:
  //! \brief The value type.
  using value_type = T;
  //! \brief Construct an empty vector.
  vector() : data_(nullptr), size_(0) {}
  //! \brief Free all resouces.
  ~vector()
  {
    if (data_ != nullptr)
      fk::delete_device(data_);
  }
  //! \brief Construct a vector with given size.
  vector(int64_t size) : data_(nullptr), size_(0)
  {
    this->resize(size);
  }
  //! \brief Move-constructor.
  vector(vector<T> &&other)
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0))
  {}
  //! \brief Move-assignment.
  vector &operator=(vector<T> &&other)
  {
    vector<T> temp(std::move(other));
    std::swap(data_, temp.data_);
    std::swap(size_, temp.size_);
    return *this;
  }
  //! \brief Copy-constructor.
  vector(vector<T> const &other) : vector()
  {
    *this = other;
  }
  //! \brief Copy-assignment.
  vector<T> &operator=(vector<T> const &other)
  {
    this->resize(other.size());
    fk::copy_on_device<T>(data_, other.data_, size_);
    return *this;
  }
  //! \brief Constructor that copies from an existing std::vector
  vector(std::vector<T> const &other) : vector()
  {
    *this = other;
  }
  //! \brief Copy the data from the std::vector
  vector<T> &operator=(std::vector<T> const &other)
  {
    this->resize(other.size());
    fk::copy_to_device<T>(data_, other.data(), size_);
    return *this;
  }
  //! \brief Does not rellocate the data, i.e., if size changes all old data is lost.
  void resize(int64_t new_size)
  {
    expect(new_size >= 0);
    if (new_size != size_)
    {
      if (data_ != nullptr)
        fk::delete_device<T>(data_);
      fk::allocate_device<T>(data_, new_size, false);
      size_ = new_size;
    }
  }
  //! \brief Returns the number of elements inside the vector.
  int64_t size() const { return size_; }
  //! \brief Returns true if the size is zero, false otherwise.
  bool empty() const { return (size_ == 0); }
  //! \brief Clears all content.
  void clear() { this->resize(0); }
  //! \brief Returns pointer to the first stored element.
  T *data() { return data_; }
  //! \brief Returns const pointer to the first stored element.
  T const *data() const { return data_; }
  //! \brief Copy to a device array, the destination must be large enough
  void copy_to_device(T *destination) const
  {
    fk::copy_on_device<T>(destination, data_, size_);
  }
  //! \brief Copy to a host array, the destination must be large enough
  void copy_to_host(T *destination) const
  {
    fk::copy_to_host<T>(destination, data_, size_);
  }
  //! \brief Copy to a std::vector on the host.
  std::vector<T> copy_to_host() const
  {
    std::vector<T> result(size_);
    this->copy_to_host(result.data());
    return result;
  }
  //! \brief Custom conversion, so we can assign to std::vector.
  operator std::vector<T>() const { return this->copy_to_host(); }

private:
  T *data_;
  int64_t size_;
};

} // namespace gpu
#endif

} // namespace asgard
