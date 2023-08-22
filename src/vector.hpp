#pragma once
#include "build_info.hpp"

#include "asgard_resources.hpp"
#include "lib_dispatch.hpp"
#include "tools.hpp"

#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <vector>

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
  fk::vector<P, mem_type::owner, resrc> &resize(int const size = 0);

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

#include "vector.tpp"
