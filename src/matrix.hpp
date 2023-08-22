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

#include "matrix.tpp"
