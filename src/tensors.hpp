#pragma once

#include <string>
#include <vector>

/* tolerance for answer comparisons */
#define TOL std::numeric_limits<P>::epsilon() * 2

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
template<typename P>
class matrix;

template<typename P, mem_type mem>
class vector
{
public:
  // FIXME disable all constructors for views
  // unless we decide to add one for views instead
  // of an extract function
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

  ~vector();

  // constructor/assignment (required to be same to same T==T)
  vector(vector<P, mem> const &);
  vector<P, mem> &operator=(vector<P, mem> const &);

  // move precision constructor/assignment (required to be same to same)
  // FIXME disable for view
  vector(vector<P, mem> &&);
  vector<P, mem> &operator=(vector<P, mem> &&);

  // converting constructor/assignment overloads
  // FIXME disable for view (constructor only)
  template<typename PP, mem_type omem = mem>
  vector(vector<PP, omem> const &);
  template<typename PP, mem_type omem = mem>
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
  bool operator==(vector<P, mem> const &) const;
  template<mem_type omem>
  bool operator==(vector<P, omem> const &) const;
  bool operator!=(vector<P, mem> const &) const;
  bool operator<(vector<P, mem> const &) const;
  //
  // math operators
  //
  // FIXME these return an owner - we need to allocate new space for results
  // could change to provide pre-allocated region for results

  template<mem_type omem>
  vector<P, mem> operator+(vector<P, omem> const &right) const;
  template<mem_type omem>
  vector<P, mem> operator-(vector<P, omem> const &right) const;
  template<mem_type omem>
  P operator*(vector<P, omem> const &)const;
  vector<P, mem> operator*(matrix<P> const &)const;
  vector<P, mem> operator*(P const) const;

  template<mem_type omem>
  vector<P, mem> single_column_kron(vector<P, omem> const &) const;

  //
  // basic queries to private data
  //
  int size() const { return size_; }
  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  // FIXME need to offset for view
  P *data(int const elem = 0) const { return &data_[elem]; }
  //
  // utility functions
  //
  void print(std::string const label = "") const;
  void dump_to_octave(char const *) const;
  // FIXME disable if views exist
  // FIXME owner only
  fk::vector<P, mem> &resize(int const size = 0);
  template<mem_type omem>
  vector<P, mem> &set(int const, vector<P, omem> const);
  // FIXME how does this play w extract view?
  // for now, this one returns an owner, extract view
  // extracts...a view
  vector<P, mem> extract(int const, int const) const;
  // FIXME concat only works for owners
  template<mem_type omem>
  vector<P, mem> &concat(vector<P, omem> const &right);

  typedef P *iterator;
  typedef const P *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }
  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + size(); }

private:
  // TODO template on ownership
  P *data_;  //< pointer to elements
  int size_; //< dimension
};

template<typename P>
class matrix
{
public:
  matrix();
  matrix(int rows, int cols);
  matrix(std::initializer_list<std::initializer_list<P>> list);

  ~matrix();

  matrix(matrix<P> const &);
  matrix<P> &operator=(matrix<P> const &);
  template<typename PP>
  matrix(matrix<PP> const &);
  template<typename PP>
  matrix<P> &operator=(matrix<PP> const &);

  matrix(matrix<P> &&);
  matrix<P> &operator=(matrix<P> &&);

  //
  // copy out of fk::vector
  //
  matrix<P> &operator=(fk::vector<P> const &);
  //
  // subscripting operators
  //
  P &operator()(int const, int const);
  P operator()(int const, int const) const;
  //
  // comparison operators
  //
  bool operator==(matrix<P> const &) const;
  bool operator!=(matrix<P> const &) const;
  bool operator<(matrix<P> const &) const;
  //
  // math operators
  //
  matrix<P> operator*(P const) const;
  vector<P> operator*(vector<P> const &)const;
  matrix<P> operator*(matrix<P> const &)const;
  matrix<P> operator+(matrix<P> const &) const;
  matrix<P> operator-(matrix<P> const &) const;

  matrix<P> &transpose();

  matrix<P> kron(matrix<P> const &) const;

  // clang-format off
  template<typename U = P>
  std::enable_if_t<
    std::is_floating_point<U>::value && std::is_same<P, U>::value,
  matrix<P> &> invert();


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
  int size() const { return nrows() * ncols(); }
  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  P *data(int const i = 0, int const j = 0) const
  {
    // return &data_[i * ncols() + j]; // row-major
    return &data_[j * nrows() + i]; // column-major
  }
  //
  // utility functions
  //

  matrix<P> &update_col(int const, fk::vector<P> const &);
  matrix<P> &update_col(int const, std::vector<P> const &);
  matrix<P> &update_row(int const, fk::vector<P> const &);
  matrix<P> &update_row(int const, std::vector<P> const &);
  matrix<P> &clear_and_resize(int const, int const);
  matrix<P> &set_submatrix(int const row_idx, int const col_idx,
                           fk::matrix<P> const &submatrix);
  matrix<P> extract_submatrix(int const row_idx, int const col_idx,
                              int const num_rows, int const num_cols) const;
  void print(std::string const label = "") const;
  void dump_to_octave(char const *name) const;

  typedef P *iterator;
  typedef const P *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }
  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + size(); }

private:
  P *data_;   //< pointer to elements
  int nrows_; //< row dimension
  int ncols_; //< column dimension
};

} // namespace fk

// suppress implicit instantiations later on
// implies fk::vector<double, mem_type::owner>
extern template class fk::vector<double>;
extern template class fk::vector<float>;
extern template class fk::vector<int>;
extern template class fk::matrix<double>;
extern template class fk::matrix<float>;
extern template class fk::matrix<int>;

extern template fk::vector<int>::vector(vector<float> const &);
extern template fk::vector<int>::vector(vector<double> const &);
extern template fk::vector<float>::vector(vector<int> const &);
extern template fk::vector<float>::vector(vector<double> const &);
extern template fk::vector<double>::vector(vector<int> const &);
extern template fk::vector<double>::vector(vector<float> const &);

extern template fk::vector<int>::vector(vector<float, mem_type::view> const &);
extern template fk::vector<int>::vector(vector<double, mem_type::view> const &);
extern template fk::vector<float>::vector(vector<int, mem_type::view> const &);
extern template fk::vector<float>::vector(
    vector<double, mem_type::view> const &);
extern template fk::vector<double>::vector(vector<int, mem_type::view> const &);
extern template fk::vector<double>::vector(
    vector<float, mem_type::view> const &);

extern template fk::vector<int> &fk::vector<int>::
operator=(vector<float> const &);
extern template fk::vector<int> &fk::vector<int>::
operator=(vector<double> const &);
extern template fk::vector<float> &fk::vector<float>::
operator=(vector<int> const &);
extern template fk::vector<float> &fk::vector<float>::
operator=(vector<double> const &);
extern template fk::vector<double> &fk::vector<double>::
operator=(vector<int> const &);
extern template fk::vector<double> &fk::vector<double>::
operator=(vector<float> const &);

extern template fk::matrix<int>::matrix(matrix<float> const &);
extern template fk::matrix<int>::matrix(matrix<double> const &);
extern template fk::matrix<float>::matrix(matrix<int> const &);
extern template fk::matrix<float>::matrix(matrix<double> const &);
extern template fk::matrix<double>::matrix(matrix<int> const &);
extern template fk::matrix<double>::matrix(matrix<float> const &);

extern template fk::matrix<int> &fk::matrix<int>::
operator=(matrix<float> const &);
extern template fk::matrix<int> &fk::matrix<int>::
operator=(matrix<double> const &);
extern template fk::matrix<float> &fk::matrix<float>::
operator=(matrix<int> const &);
extern template fk::matrix<float> &fk::matrix<float>::
operator=(matrix<double> const &);
extern template fk::matrix<double> &fk::matrix<double>::
operator=(matrix<int> const &);
extern template fk::matrix<double> &fk::matrix<double>::
operator=(matrix<float> const &);

// remove these when matrix::invert()/determinatn() is availble for ints
extern template fk::matrix<float> &fk::matrix<float>::invert();
extern template fk::matrix<double> &fk::matrix<double>::invert();
extern template float fk::matrix<float>::determinant() const;
extern template double fk::matrix<double>::determinant() const;

// added for mem_type support

extern template fk::vector<int, mem_type::owner>::vector(); // needed b/c of
                                                            // sfinae decl
extern template fk::vector<float, mem_type::owner>::vector();
extern template fk::vector<double, mem_type::owner>::vector();
extern template fk::vector<int, mem_type::owner>::vector(int const);
extern template fk::vector<float, mem_type::owner>::vector(int const);
extern template fk::vector<double, mem_type::owner>::vector(int const);
extern template fk::vector<int, mem_type::owner>::vector(
    std::initializer_list<int>);
extern template fk::vector<float, mem_type::owner>::vector(
    std::initializer_list<float>);
extern template fk::vector<double, mem_type::owner>::vector(
    std::initializer_list<double>);
extern template fk::vector<int, mem_type::owner>::vector(
    std::vector<int> const &);
extern template fk::vector<float, mem_type::owner>::vector(
    std::vector<float> const &);
extern template fk::vector<double, mem_type::owner>::vector(
    std::vector<double> const &);
extern template fk::vector<int, mem_type::owner>::vector(
    fk::matrix<int> const &);
extern template fk::vector<float, mem_type::owner>::vector(
    fk::matrix<float> const &);
extern template fk::vector<double, mem_type::owner>::vector(
    fk::matrix<double> const &);

extern template class fk::vector<double, mem_type::view>; // get the non-default
                                                          // mem_type::view
extern template class fk::vector<float, mem_type::view>;
extern template class fk::vector<int, mem_type::view>;

extern template fk::vector<int, mem_type::view>::vector(
    vector<float, mem_type::view> const &);
extern template fk::vector<int, mem_type::view>::vector(
    vector<double, mem_type::view> const &);
extern template fk::vector<float, mem_type::view>::vector(
    vector<int, mem_type::view> const &);
extern template fk::vector<float, mem_type::view>::vector(
    vector<double, mem_type::view> const &);
extern template fk::vector<double, mem_type::view>::vector(
    vector<int, mem_type::view> const &);
extern template fk::vector<double, mem_type::view>::vector(
    vector<float, mem_type::view> const &);

extern template fk::vector<int, mem_type::view>::vector(
    vector<float, mem_type::owner> const &);
extern template fk::vector<int, mem_type::view>::vector(
    vector<double, mem_type::owner> const &);
extern template fk::vector<float, mem_type::view>::vector(
    vector<int, mem_type::owner> const &);
extern template fk::vector<float, mem_type::view>::vector(
    vector<double, mem_type::owner> const &);
extern template fk::vector<double, mem_type::view>::vector(
    vector<int, mem_type::owner> const &);
extern template fk::vector<double, mem_type::view>::vector(
    vector<float, mem_type::owner> const &);

extern template fk::vector<int, mem_type::owner>::vector(
    vector<float, mem_type::view> const &);
extern template fk::vector<int, mem_type::owner>::vector(
    vector<double, mem_type::view> const &);
extern template fk::vector<float, mem_type::owner>::vector(
    vector<int, mem_type::view> const &);
extern template fk::vector<float, mem_type::owner>::vector(
    vector<double, mem_type::view> const &);
extern template fk::vector<double, mem_type::owner>::vector(
    vector<int, mem_type::view> const &);
extern template fk::vector<double, mem_type::owner>::vector(
    vector<float, mem_type::view> const &);

extern template fk::vector<int, mem_type::view>::vector(
    vector<int, mem_type::owner> const &);
extern template fk::vector<int, mem_type::owner>::vector(
    vector<int, mem_type::view> const &);
extern template fk::vector<float, mem_type::view>::vector(
    vector<float, mem_type::owner> const &);
extern template fk::vector<float, mem_type::owner>::vector(
    vector<float, mem_type::view> const &);
extern template fk::vector<double, mem_type::view>::vector(
    vector<double, mem_type::owner> const &);
extern template fk::vector<double, mem_type::owner>::vector(
    vector<double, mem_type::view> const &);

extern template fk::vector<int, mem_type::view> &
fk::vector<int, mem_type::view>::
operator=(vector<float, mem_type::owner> const &);
extern template fk::vector<int, mem_type::view> &
fk::vector<int, mem_type::view>::
operator=(vector<double, mem_type::owner> const &);
extern template fk::vector<float, mem_type::view> &
fk::vector<float, mem_type::view>::
operator=(vector<int, mem_type::owner> const &);
extern template fk::vector<float, mem_type::view> &
fk::vector<float, mem_type::view>::
operator=(vector<double, mem_type::owner> const &);
extern template fk::vector<double, mem_type::view> &
fk::vector<double, mem_type::view>::
operator=(vector<int, mem_type::owner> const &);
extern template fk::vector<double, mem_type::view> &
fk::vector<double, mem_type::view>::
operator=(vector<float, mem_type::owner> const &);

extern template fk::vector<int, mem_type::owner> &
fk::vector<int, mem_type::owner>::
operator=(vector<float, mem_type::view> const &);
extern template fk::vector<int, mem_type::owner> &
fk::vector<int, mem_type::owner>::
operator=(vector<double, mem_type::view> const &);
extern template fk::vector<float, mem_type::owner> &
fk::vector<float, mem_type::owner>::
operator=(vector<int, mem_type::view> const &);
extern template fk::vector<float, mem_type::owner> &
fk::vector<float, mem_type::owner>::
operator=(vector<double, mem_type::view> const &);
extern template fk::vector<double, mem_type::owner> &
fk::vector<double, mem_type::owner>::
operator=(vector<int, mem_type::view> const &);
extern template fk::vector<double, mem_type::owner> &
fk::vector<double, mem_type::owner>::
operator=(vector<float, mem_type::view> const &);

extern template fk::vector<int, mem_type::view> &
fk::vector<int, mem_type::view>::
operator=(vector<int, mem_type::owner> const &);
extern template fk::vector<int, mem_type::owner> &
fk::vector<int, mem_type::owner>::
operator=(vector<int, mem_type::view> const &);
extern template fk::vector<float, mem_type::view> &
fk::vector<float, mem_type::view>::
operator=(vector<float, mem_type::owner> const &);
extern template fk::vector<float, mem_type::owner> &
fk::vector<float, mem_type::owner>::
operator=(vector<float, mem_type::view> const &);
extern template fk::vector<double, mem_type::view> &
fk::vector<double, mem_type::view>::
operator=(vector<double, mem_type::owner> const &);
extern template fk::vector<double, mem_type::owner> &
fk::vector<double, mem_type::owner>::
operator=(vector<double, mem_type::view> const &);

extern template bool fk::vector<double, mem_type::view>::
operator==(vector<double, mem_type::owner> const &) const;
extern template bool fk::vector<float, mem_type::view>::
operator==(vector<float, mem_type::owner> const &) const;
extern template bool fk::vector<int, mem_type::view>::
operator==(vector<int, mem_type::owner> const &) const;

extern template bool fk::vector<double>::
operator==(vector<double, mem_type::view> const &) const;
extern template bool fk::vector<float>::
operator==(vector<float, mem_type::view> const &) const;
extern template bool fk::vector<int>::
operator==(vector<int, mem_type::view> const &) const;

extern template fk::vector<double> fk::vector<double>::
operator+(fk::vector<double, mem_type::view> const &right) const;
extern template fk::vector<float> fk::vector<float>::
operator+(fk::vector<float, mem_type::view> const &right) const;
extern template fk::vector<int> fk::vector<int>::
operator+(fk::vector<int, mem_type::view> const &right) const;

extern template fk::vector<double> fk::vector<double>::
operator+(fk::vector<double> const &right) const;
extern template fk::vector<float> fk::vector<float>::
operator+(fk::vector<float> const &right) const;
extern template fk::vector<int> fk::vector<int>::
operator+(fk::vector<int> const &right) const;

extern template fk::vector<double, mem_type::view>
fk::vector<double, mem_type::view>::
operator+(fk::vector<double, mem_type::view> const &right) const;
extern template fk::vector<float, mem_type::view>
fk::vector<float, mem_type::view>::
operator+(fk::vector<float, mem_type::view> const &right) const;
extern template fk::vector<int, mem_type::view>
fk::vector<int, mem_type::view>::
operator+(fk::vector<int, mem_type::view> const &right) const;

extern template fk::vector<double, mem_type::view>
fk::vector<double, mem_type::view>::
operator+(fk::vector<double> const &right) const;
extern template fk::vector<float, mem_type::view>
fk::vector<float, mem_type::view>::
operator+(fk::vector<float> const &right) const;
extern template fk::vector<int, mem_type::view>
fk::vector<int, mem_type::view>::operator+(fk::vector<int> const &right) const;

extern template fk::vector<double> fk::vector<double>::
operator-(fk::vector<double, mem_type::view> const &right) const;
extern template fk::vector<float> fk::vector<float>::
operator-(fk::vector<float, mem_type::view> const &right) const;
extern template fk::vector<int> fk::vector<int>::
operator-(fk::vector<int, mem_type::view> const &right) const;

extern template fk::vector<double> fk::vector<double>::
operator-(fk::vector<double> const &right) const;
extern template fk::vector<float> fk::vector<float>::
operator-(fk::vector<float> const &right) const;
extern template fk::vector<int> fk::vector<int>::
operator-(fk::vector<int> const &right) const;

extern template fk::vector<double, mem_type::view>
fk::vector<double, mem_type::view>::
operator-(fk::vector<double, mem_type::view> const &right) const;
extern template fk::vector<float, mem_type::view>
fk::vector<float, mem_type::view>::
operator-(fk::vector<float, mem_type::view> const &right) const;
extern template fk::vector<int, mem_type::view>
fk::vector<int, mem_type::view>::
operator-(fk::vector<int, mem_type::view> const &right) const;

extern template fk::vector<double, mem_type::view>
fk::vector<double, mem_type::view>::
operator-(fk::vector<double> const &right) const;
extern template fk::vector<float, mem_type::view>
fk::vector<float, mem_type::view>::
operator-(fk::vector<float> const &right) const;
extern template fk::vector<int, mem_type::view>
fk::vector<int, mem_type::view>::operator-(fk::vector<int> const &right) const;

extern template double fk::vector<double>::
operator*(fk::vector<double, mem_type::view> const &right) const;
extern template float fk::vector<float>::
operator*(fk::vector<float, mem_type::view> const &right) const;
extern template int fk::vector<int>::
operator*(fk::vector<int, mem_type::view> const &right) const;

extern template double fk::vector<double>::
operator*(fk::vector<double> const &right) const;
extern template float fk::vector<float>::
operator*(fk::vector<float> const &right) const;
extern template int fk::vector<int>::
operator*(fk::vector<int> const &right) const;

extern template double fk::vector<double, mem_type::view>::
operator*(fk::vector<double, mem_type::view> const &right) const;
extern template float fk::vector<float, mem_type::view>::
operator*(fk::vector<float, mem_type::view> const &right) const;
extern template int fk::vector<int, mem_type::view>::
operator*(fk::vector<int, mem_type::view> const &right) const;

extern template double fk::vector<double, mem_type::view>::
operator*(fk::vector<double> const &right) const;
extern template float fk::vector<float, mem_type::view>::
operator*(fk::vector<float> const &right) const;
extern template int fk::vector<int, mem_type::view>::
operator*(fk::vector<int> const &right) const;

extern template fk::vector<double> fk::vector<double>::single_column_kron(
    fk::vector<double, mem_type::view> const &right) const;
extern template fk::vector<float> fk::vector<float>::single_column_kron(
    fk::vector<float, mem_type::view> const &right) const;
extern template fk::vector<int> fk::vector<int>::single_column_kron(
    fk::vector<int, mem_type::view> const &right) const;

extern template fk::vector<double>
fk::vector<double>::single_column_kron(fk::vector<double> const &right) const;
extern template fk::vector<float>
fk::vector<float>::single_column_kron(fk::vector<float> const &right) const;
extern template fk::vector<int>
fk::vector<int>::single_column_kron(fk::vector<int> const &right) const;

extern template fk::vector<double> &
fk::vector<double>::concat(fk::vector<double> const &right);
extern template fk::vector<float> &
fk::vector<float>::concat(fk::vector<float> const &right);
extern template fk::vector<int> &
fk::vector<int>::concat(fk::vector<int> const &right);

extern template fk::vector<double> &
fk::vector<double>::concat(fk::vector<double, mem_type::view> const &right);
extern template fk::vector<float> &
fk::vector<float>::concat(fk::vector<float, mem_type::view> const &right);
extern template fk::vector<int> &
fk::vector<int>::concat(fk::vector<int, mem_type::view> const &right);

extern template fk::vector<double, mem_type::view> &
fk::vector<double, mem_type::view>::concat(fk::vector<double> const &right);
extern template fk::vector<float, mem_type::view> &
fk::vector<float, mem_type::view>::concat(fk::vector<float> const &right);
extern template fk::vector<int, mem_type::view> &
fk::vector<int, mem_type::view>::concat(fk::vector<int> const &right);

extern template fk::vector<double, mem_type::view> &
fk::vector<double, mem_type::view>::concat(
    fk::vector<double, mem_type::view> const &right);
extern template fk::vector<float, mem_type::view> &
fk::vector<float, mem_type::view>::concat(
    fk::vector<float, mem_type::view> const &right);
extern template fk::vector<int, mem_type::view> &
fk::vector<int, mem_type::view>::concat(
    fk::vector<int, mem_type::view> const &right);

extern template fk::vector<double> &
fk::vector<double>::set(int const, fk::vector<double> const);
extern template fk::vector<float> &
fk::vector<float>::set(int const, fk::vector<float> const);
extern template fk::vector<int> &
fk::vector<int>::set(int const, fk::vector<int> const);

extern template fk::vector<double> &
fk::vector<double>::set(int const, fk::vector<double, mem_type::view> const);
extern template fk::vector<float> &
fk::vector<float>::set(int const, fk::vector<float, mem_type::view> const);
extern template fk::vector<int> &
fk::vector<int>::set(int const, fk::vector<int, mem_type::view> const);

extern template fk::vector<double, mem_type::view> &
fk::vector<double, mem_type::view>::set(int const, fk::vector<double> const);
extern template fk::vector<float, mem_type::view> &
fk::vector<float, mem_type::view>::set(int const, fk::vector<float> const);
extern template fk::vector<int, mem_type::view> &
fk::vector<int, mem_type::view>::set(int const, fk::vector<int> const);

extern template fk::vector<double, mem_type::view> &
fk::vector<double, mem_type::view>::set(
    int const, fk::vector<double, mem_type::view> const);
extern template fk::vector<float, mem_type::view> &
fk::vector<float, mem_type::view>::set(int const,
                                       fk::vector<float, mem_type::view> const);
extern template fk::vector<int, mem_type::view> &
fk::vector<int, mem_type::view>::set(int const,
                                     fk::vector<int, mem_type::view> const);
