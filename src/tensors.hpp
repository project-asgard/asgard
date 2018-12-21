#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/* tolerance for answer comparisons */
#define TOL 1.0e-10
//#define TOL std::numeric_limits<double>::epsilon() * 8

namespace fk
{
// ==========================================================================
// external declarations for calling blas routines linked with -lblas
// ==========================================================================

/* --------------------------------------------------------------------------
   DCOPY copies a vector, x, to a vector, y.
   uses unrolled loops for increments equal to one.
   -------------------------------------------------------------------------- */
extern "C" void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern "C" void scopy_(int *n, float *x, int *incx, float *y, int *incy);
// --------------------------------------------------------------------------
// matrix-vector multiply
// y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
// --------------------------------------------------------------------------
extern "C" void dgemv_(char const *trans, int *m, int *n, double *alpha,
                       double *A, int *lda, double *x, int *incx, double *beta,
                       double *y, int *incy);
extern "C" void sgemv_(char const *trans, int *m, int *n, float *alpha,
                       float *A, int *lda, float *x, int *incx, float *beta,
                       float *y, int *incy);
// --------------------------------------------------------------------------
// matrix-matrix multiply
// C := alpha*A*B + beta*C
// --------------------------------------------------------------------------
extern "C" void dgemm_(char const *transa, char const *transb, int *m, int *n,
                       int *k, double *alpha, double *A, int *lda, double *B,
                       int *ldb, double *beta, double *C, int *ldc);
extern "C" void sgemm_(char const *transa, char const *transb, int *m, int *n,
                       int *k, float *alpha, float *A, int *lda, float *B,
                       int *ldb, float *beta, float *C, int *ldc);

//
// Simple matrix multiply for non-float types
// FIXME we will probably eventually need a version that does transpose
//
template<typename T>
static void igemm_(T *A, int const lda, T *B, int const ldb, T *C,
                   int const ldc, int const m, int const k, int const n)
{
  assert(m > 0);
  assert(k > 0);
  assert(n > 0);
  assert(lda > 0); // FIXME Tyler says these could be more thorough
  assert(ldb > 0);
  assert(ldc > 0);

  for (auto i = 0; i < m; ++i)
  {
    for (auto j = 0; j < n; ++j)
    {
      T result = 0.0;
      for (auto z = 0; z < k; ++z)
      {
        // result += A[i,k] * B[k,j]
        result += A[z * lda + i] * B[j * ldb + z];
      }
      // C[i,j] += result
      C[j * ldc + i] += result;
    }
  }
}

// --------------------------------------------------------------------------
// LU decomposition of a general matrix
// --------------------------------------------------------------------------
extern "C" void
dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);

extern "C" void
sgetrf_(int *m, int *n, float *A, int *lda, int *ipiv, int *info);

// --------------------------------------------------------------------------
// inverse of a matrix given its LU decomposition
// --------------------------------------------------------------------------
extern "C" void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work,
                        int *lwork, int *info);

extern "C" void sgetri_(int *n, float *A, int *lda, int *ipiv, float *work,
                        int *lwork, int *info);

// forward declarations
template<typename T>
class vector;
template<typename T>
class matrix;

template<typename T>
class vector
{
public:
  vector();
  vector(int const size);
  vector(std::initializer_list<T> list);
  vector(std::vector<T> const &);

  ~vector();

  vector(vector<T> const &);
  vector<T> &operator=(vector<T> const &);
  vector(vector<T> &&);
  vector<T> &operator=(vector<T> &&);

  //
  // copy out of std::vector
  //
  vector<T> &operator=(std::vector<T> const &);

  //
  // copy into std::vector
  //
  std::vector<T> to_std() const;

  //
  // subscripting operators
  //
  T &operator()(int const);
  T operator()(int const) const;
  //
  // comparison operators
  //
  bool operator==(vector<T> const &) const;
  bool operator!=(vector<T> const &) const;
  //
  // math operators
  //
  vector<T> operator+(vector<T> const &right) const;
  vector<T> operator-(vector<T> const &right) const;
  T operator*(vector<T> const &)const;
  vector<T> operator*(matrix<T> const &)const;
  //
  // basic queries to private data
  //
  int size() const { return size_; }
  T *data(int const elem = 0) const { return &data_[elem]; }
  //
  // utility functions
  //
  void print(std::string const label = "") const;
  void dump_to_octave(char const *) const;
  void resize(int const size = 0);

  typedef T *iterator;
  typedef const T *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }

private:
  T *data_;  //< pointer to elements
  int size_; //< dimension
};

template<typename T>
class matrix
{
public:
  matrix();
  matrix(int rows, int cols);
  matrix(std::initializer_list<std::initializer_list<T>> list);
  matrix(std::vector<T> const &);

  ~matrix();

  matrix(matrix<T> const &);
  matrix<T> &operator=(matrix<T> const &);
  matrix(matrix<T> &&);
  matrix<T> &operator=(matrix<T> &&);

  //
  // copy out of std::vector
  //
  matrix<T> &operator=(std::vector<T> const &);
  //
  // subscripting operators
  //
  T &operator()(int const, int const);
  T operator()(int const, int const) const;
  //
  // comparison operators
  //
  bool operator==(matrix<T> const &) const;
  bool operator!=(matrix<T> const &) const;
  //
  // math operators
  //
  matrix<T> operator*(matrix<T> const &)const;
  matrix<T> operator*(int const) const;
  matrix<T> operator+(matrix<T> const &) const;
  matrix<T> operator-(matrix<T> const &) const;

  matrix<T> &transpose();

  // clang-format off
  template<typename U = T>
  std::enable_if_t<
    std::is_floating_point<U>::value && std::is_same<T, U>::value, 
  matrix<T> &> invert();


  template<typename U = T>
  std::enable_if_t<
      std::is_floating_point<U>::value && std::is_same<T, U>::value, 
  T> determinant() const;
  // clang-format on

  //
  // basic queries to private data
  //
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  int size() const { return nrows() * ncols(); }
  T *data(int const i = 0, int const j = 0) const
  {
    // return &data_[i * ncols() + j]; // row-major
    return &data_[j * nrows() + i]; // column-major
  }
  //
  // utility functions
  //

  matrix<T> &update_col(int const, fk::vector<T> const &);
  matrix<T> &update_col(int const, std::vector<T> const &);
  matrix<T> &update_row(int const, fk::vector<T> const &);
  matrix<T> &update_row(int const, std::vector<T> const &);
  matrix<T> &set_submatrix(int const row_idx, int const col_idx,
                           fk::matrix<T> const &submatrix);
  matrix<T> extract_submatrix(int const row_idx, int const col_idx,
                              int const num_rows, int const num_cols) const;
  void print(std::string const label = "") const;
  void dump_to_octave(char const *name) const;

  typedef T *iterator;
  typedef const T *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }

private:
  T *data_;   //< pointer to elements
  int nrows_; //< row dimension
  int ncols_; //< column dimension
};

} // namespace fk

//-----------------------------------------------------------------------------
//
// fk::vector class implementation starts here
//
//-----------------------------------------------------------------------------
template<typename T>
fk::vector<T>::vector() : data_{nullptr}, size_{0}
{}
// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
template<typename T>
fk::vector<T>::vector(int const size) : data_{new T[size]()}, size_{size}
{}

// can also do this with variadic template constructor for constness
// https://stackoverflow.com/a/5549918
// but possibly this is "too clever" for our needs right now

template<typename T>
fk::vector<T>::vector(std::initializer_list<T> list)
    : data_{new T[list.size()]}, size_{static_cast<int>(list.size())}
{
  std::copy(list.begin(), list.end(), data_);
}

template<typename T>
fk::vector<T>::vector(std::vector<T> const &v)
    : data_{new T[v.size()]}, size_{static_cast<int>(v.size())}
{
  std::copy(v.begin(), v.end(), data_);
}

template<typename T>
fk::vector<T>::~vector()
{
  delete[] data_;
}

//
// vector copy constructor
//
template<typename T>
fk::vector<T>::vector(vector<T> const &a)
    : data_{new T[a.size_]}, size_{a.size_}
{
  std::memcpy(data_, a.data(), a.size() * sizeof(T));
}

//
// vector copy assignment
// this can probably be optimized better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename T>
fk::vector<T> &fk::vector<T>::operator=(vector<T> const &a)
{
  if (&a == this) return *this;

  assert(size() == a.size());

  size_ = a.size_;
  memcpy(data_, a.data(), a.size() * sizeof(T));

  return *this;
}

//
// vector move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
template<typename T>
fk::vector<T>::vector(vector<T> &&a) : data_{a.data_}, size_{a.size_}
{
  a.data_ = nullptr; // b/c a's destructor will be called
  a.size_ = 0;
}

//
// vector move assignment
//
template<typename T>
fk::vector<T> &fk::vector<T>::operator=(vector &&a)
{
  if (&a == this) return *this;

  assert(size() == a.size());

  size_ = a.size_;
  T *temp{data_};
  data_   = a.data_;
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// copy out of std::vector
//
template<typename T>
fk::vector<T> &fk::vector<T>::operator=(std::vector<T> const &v)
{
  assert(size() == static_cast<int>(v.size()));
  std::memcpy(data_, v.data(), v.size() * sizeof(T));
  return *this;
}

//
// copy into std::vector
//
template<typename T>
std::vector<T> fk::vector<T>::to_std() const
{
  return std::vector<T>(data(), data() + size());
}

// vector subscript operator
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename T>
T &fk::vector<T>::operator()(int i)
{
  assert(i < size_);
  return data_[i];
}

template<typename T>
T fk::vector<T>::operator()(int i) const
{
  assert(i < size_);
  return data_[i];
}

//
// vector comparison operators - set default tolerance above
//
template<typename T>
bool fk::vector<T>::operator==(vector<T> const &other) const
{
  if (&other == this) return true;
  if (size() != other.size()) return false;
  for (auto i = 0; i < size(); ++i)
    if (std::abs((*this)(i)) > TOL && std::abs(other(i)) > TOL)
      if (std::abs((*this)(i)-other(i)) > TOL) { return false; }
  return true;
}
template<typename T>
bool fk::vector<T>::operator!=(vector<T> const &other) const
{
  return !(*this == other);
}

//
// vector addition operator
//
template<typename T>
fk::vector<T> fk::vector<T>::operator+(vector<T> const &right) const
{
  assert(size() == right.size());
  vector<T> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i) + right(i);
  return ans;
}

//
// vector subtraction operator
//
template<typename T>
fk::vector<T> fk::vector<T>::operator-(vector<T> const &right) const
{
  assert(size() == right.size());
  vector<T> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i)-right(i);
  return ans;
}

//
// vector*vector multiplication operator
//
template<typename T>
T fk::vector<T>::operator*(vector<T> const &right) const
{
  assert(size() == right.size());
  T ans = 0.0;
  for (auto i = 0; i < size(); ++i)
    ans += (*this)(i)*right(i);
  return ans;
}

//
// vector*matrix multiplication operator
//
template<typename T>
fk::vector<T> fk::vector<T>::operator*(fk::matrix<T> const &A) const
{
  // check dimension compatibility
  assert(size() == A.nrows());

  vector const &X = (*this);
  vector<T> Y(A.ncols());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = m;
  int one_i = 1;

  if constexpr (std::is_same<T, double>::value)
  {
    T zero = 0.0;
    T one  = 1.0;
    dgemv_("t", &m, &n, &one, A.data(), &lda, X.data(), &one_i, &zero, Y.data(),
           &one_i);
  }
  else if constexpr (std::is_same<T, float>::value)
  {
    T zero = 0.0;
    T one  = 1.0;
    sgemv_("t", &m, &n, &one, A.data(), &lda, X.data(), &one_i, &zero, Y.data(),
           &one_i);
  }

  else
  {
    fk::matrix<T> At = A;
    At.transpose();

    // vectors don't have a leading dimension...
    int ldv = 1;
    n       = 1;

    // simple matrix multiply routine doesn't have a transpose (yet)
    // so the arguments are switched relative to the above BLAS calls
    lda   = At.nrows();
    m     = At.nrows();
    int k = At.ncols();
    igemm_(At.data(), lda, X.data(), ldv, Y.data(), ldv, m, k, n);
  }

  return Y;
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
template<typename T>
void fk::vector<T>::print(std::string const label) const
{
  std::cout << label << '\n';
  if constexpr (std::is_floating_point<T>::value)
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
// Dumps to file a vector that can be read data straight into octave
// Same as the matrix:: version
//
// @param[in]   label   a string label printed with the output
// @param[in]   b       the vector from the batch to print out
// @return      Nothing
//
template<typename T>
void fk::vector<T>::dump_to_octave(char const *filename) const
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
template<typename T>
void fk::vector<T>::resize(int const new_size)
{
  if (new_size == this->size()) return;
  T *old_data{data_};
  data_ = new T[new_size]();
  if (size() > 0 && new_size > 0)
    std::memcpy(data_, old_data, new_size * sizeof(T));
  size_ = new_size;
  delete[] old_data;
}

//-----------------------------------------------------------------------------
//
// fk::matrix class implementation starts here
//
//-----------------------------------------------------------------------------

template<typename T>
fk::matrix<T>::matrix() : data_{nullptr}, nrows_{0}, ncols_{0}
{}

// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region

template<typename T>
fk::matrix<T>::matrix(int M, int N)
    : data_{new T[M * N]()}, nrows_{M}, ncols_{N}
{}

template<typename T>
fk::matrix<T>::matrix(std::initializer_list<std::initializer_list<T>> llist)
    : data_{new T[llist.size() * llist.begin()->size()]},
      nrows_{static_cast<int>(llist.size())}, ncols_{static_cast<int>(
                                                  llist.begin()->size())}
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

//
// to enable conversions from std::vector to an assumed square matrix
// this isn't meant to be very robust; more of a convenience for testing
// purposes
//

template<typename T>
fk::matrix<T>::matrix(std::vector<T> const &v) : data_{new T[v.size()]}
{
  T iptr;
  assert(std::modf(std::sqrt(v.size()), &iptr) == 0);
  nrows_ = std::sqrt(v.size());
  ncols_ = std::sqrt(v.size());
  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      (*this)(i, j) = v[j + i * ncols()];
}

template<typename T>
fk::matrix<T>::~matrix()
{
  delete[] data_;
}

//
// matrix copy constructor
//
template<typename T>
fk::matrix<T>::matrix(matrix<T> const &a)
    : data_{new T[a.size()]}, nrows_{a.nrows()}, ncols_{a.ncols()}
{
  memcpy(data_, a.data(), a.size() * sizeof(T));
}

//
// matrix copy assignment
// this can probably be done better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename T>
fk::matrix<T> &fk::matrix<T>::operator=(matrix<T> const &a)
{
  if (&a == this) return *this;

  assert((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  memcpy(data_, a.data(), a.size() * sizeof(T));
  return *this;
}

//
// matrix move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//

template<typename T>
fk::matrix<T>::matrix(matrix<T> &&a)
    : data_{a.data()}, nrows_{a.nrows()}, ncols_{a.ncols()}
{
  a.data_  = nullptr; // b/c a's destructor will be called
  a.nrows_ = 0;
  a.ncols_ = 0;
}

//
// matrix move assignment
//
template<typename T>
fk::matrix<T> &fk::matrix<T>::operator=(matrix<T> &&a)
{
  if (&a == this) return *this;

  assert((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  T *temp{data_};
  data_   = a.data();
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// copy out of std::vector - assumes the std::vector is column-major
//
template<typename T>
fk::matrix<T> &fk::matrix<T>::operator=(std::vector<T> const &v)
{
  assert(nrows() * ncols() == static_cast<int>(v.size()));

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      (*this)(i, j) = v[j + i * ncols()];

  return *this;
}

//
// matrix subscript operator - row-major ordering
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename T>
T &fk::matrix<T>::operator()(int const i, int const j)
{
  assert(i < nrows() && j < ncols());
  return *(data(i, j));
}

template<typename T>
T fk::matrix<T>::operator()(int const i, int const j) const
{
  assert(i < nrows() && j < ncols());
  return *(data(i, j));
}

//
// matrix comparison operators - set default tolerance above
//
template<typename T>
bool fk::matrix<T>::operator==(matrix<T> const &other) const
{
  if (&other == this) return true;
  if (nrows() != other.nrows() || ncols() != other.ncols()) return false;
  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      if (std::abs((*this)(i, j)) > TOL && std::abs(other(i, j)) > TOL)
        if (std::abs((*this)(i, j) - other(i, j)) > TOL) { return false; }
  return true;
}

template<typename T>
bool fk::matrix<T>::operator!=(matrix<T> const &other) const
{
  return !(*this == other);
}

//
// matrix addition operator
//
template<typename T>
fk::matrix<T> fk::matrix<T>::operator+(matrix<T> const &right) const
{
  assert(nrows() == right.nrows() && ncols() == right.ncols());

  matrix<T> ans(nrows(), ncols());
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
template<typename T>
fk::matrix<T> fk::matrix<T>::operator-(matrix<T> const &right) const
{
  assert(nrows() == right.nrows() && ncols() == right.ncols());

  matrix<T> ans(nrows(), ncols());
  ans.nrows_ = nrows();
  ans.ncols_ = ncols();

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      ans(i, j) = (*this)(i, j) - right(i, j);

  return ans;
}

//
// matrix*integer multiplication operator
//
template<typename T>
fk::matrix<T> fk::matrix<T>::operator*(int const right) const
{
  matrix<T> ans(nrows(), ncols());
  ans.nrows_ = nrows();
  ans.ncols_ = ncols();

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      ans(i, j) = (*this)(i, j) * right;

  return ans;
}

//
// matrix*matrix multiplication operator C[m,n] = A[m,k] * B[k,n]
//
template<typename T>
fk::matrix<T> fk::matrix<T>::operator*(matrix<T> const &B) const
{
  assert(ncols() == B.nrows()); // k == k

  // just aliases for easier reading
  matrix const &A = (*this);
  int m           = A.nrows();
  int n           = B.ncols();
  int k           = B.nrows();

  matrix<T> C(m, n);

  int lda = m;
  int ldb = k;
  int ldc = lda;

  if constexpr (std::is_same<T, double>::value)
  {
    T one  = 1.0;
    T zero = 0.0;
    dgemm_("n", "n", &m, &n, &k, &one, A.data(), &lda, B.data(), &ldb, &zero,
           C.data(), &ldc);
  }
  else if constexpr (std::is_same<T, float>::value)
  {
    T one  = 1.0;
    T zero = 0.0;
    sgemm_("n", "n", &m, &n, &k, &one, A.data(), &lda, B.data(), &ldb, &zero,
           C.data(), &ldc);
  }
  else
  {
    igemm_(A.data(), lda, B.data(), ldb, C.data(), ldc, m, k, n);
  }
  return C;
}

//
// Transpose a matrix (overwrites original)
// @return  the transposed matrix
//
// FIXME could be worthwhile to optimize the matrix transpose
template<typename T>
fk::matrix<T> &fk::matrix<T>::transpose()
{
  matrix temp(ncols(), nrows());

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      temp(j, i) = (*this)(i, j);

  // inelegant manual "move assignment"
  nrows_     = temp.nrows();
  ncols_     = temp.ncols();
  data_      = temp.data();
  temp.data_ = nullptr;

  return *this;
}

//
// Invert a square matrix (overwrites original)
// disabled for non-fp types; haven't written a routine to do it
// @return  the inverted matrix
//
template<typename T>
template<typename U>
std::enable_if_t<std::is_floating_point<U>::value && std::is_same<T, U>::value,
                 fk::matrix<T> &>
fk::matrix<T>::invert()
{
  assert(nrows() == ncols());

  int *ipiv{new int[ncols()]};
  int lwork{nrows() * ncols()};
  int lda = ncols();
  T *work{new T[nrows() * ncols()]};
  int info;

  if constexpr (std::is_same<T, double>::value)
  {
    dgetrf_(&ncols_, &ncols_, data(0, 0), &lda, ipiv, &info);
    dgetri_(&ncols_, data(0, 0), &lda, ipiv, work, &lwork, &info);
  }
  else
  {
    sgetrf_(&ncols_, &ncols_, data(0, 0), &lda, ipiv, &info);
    sgetri_(&ncols_, data(0, 0), &lda, ipiv, work, &lwork, &info);
  }
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
template<typename T>
template<typename U>
std::enable_if_t<std::is_floating_point<U>::value && std::is_same<T, U>::value,
                 T>
fk::matrix<T>::determinant() const
{
  assert(nrows() == ncols());

  matrix temp{*this}; // get temp copy to do LU
  int *ipiv{new int[ncols()]};
  int info;
  int n   = ncols();
  int lda = ncols();

  if constexpr (std::is_same<T, double>::value)
  { dgetrf_(&n, &n, temp.data(0, 0), &lda, ipiv, &info); } else
  {
    sgetrf_(&n, &n, temp.data(0, 0), &lda, ipiv, &info);
  }

  T det    = 1.0;
  int sign = 1;
  for (auto i = 0; i < nrows(); ++i)
  {
    if (ipiv[i] != i + 1) sign *= -1;
    det *= temp(i, i);
  }
  det *= static_cast<T>(sign);
  delete[] ipiv;
  return det;
}

//
// Update a specific col of a matrix, given a fk::vector<T> (overwrites
// original)
//
template<typename T>
fk::matrix<T> &
fk::matrix<T>::update_col(int const col_idx, fk::vector<T> const &v)
{
  assert(nrows() == static_cast<int>(v.size()));
  assert(col_idx < ncols());

  int n{v.size()};
  int one{1};
  int stride = 1;

  if constexpr (std::is_same<T, double>::value)
  { dcopy_(&n, v.data(), &one, data(0, col_idx), &stride); }
  else if constexpr (std::is_same<T, float>::value)
  {
    scopy_(&n, v.data(), &one, data(0, col_idx), &stride);
  }
  else
  {
    for (auto i = 0; i < n; ++i)
    {
      (*this)(0 + i, col_idx) = v(i);
    }
  }
  return *this;
}

//
// Update a specific col of a matrix, given a std::vector (overwrites original)
//
template<typename T>
fk::matrix<T> &
fk::matrix<T>::update_col(int const col_idx, std::vector<T> const &v)
{
  assert(nrows() == static_cast<int>(v.size()));
  assert(col_idx < ncols());

  int n{static_cast<int>(v.size())};
  int one{1};
  int stride = 1;

  if constexpr (std::is_same<T, double>::value)
  { dcopy_(&n, const_cast<T *>(v.data()), &one, data(0, col_idx), &stride); }
  else if constexpr (std::is_same<T, float>::value)
  {
    scopy_(&n, const_cast<T *>(v.data()), &one, data(0, col_idx), &stride);
  }
  else
  {
    for (auto i = 0; i < n; ++i)
    {
      (*this)(0 + i, col_idx) = v[i];
    }
  }

  return *this;
}

//
// Update a specific row of a matrix, given a fk::vector<T> (overwrites
// original)
//
template<typename T>
fk::matrix<T> &
fk::matrix<T>::update_row(int const row_idx, fk::vector<T> const &v)
{
  assert(ncols() == v.size());
  assert(row_idx < nrows());

  int n{v.size()};
  int one{1};
  int stride = nrows();

  if constexpr (std::is_same<T, double>::value)
  { dcopy_(&n, v.data(), &one, data(row_idx, 0), &stride); }
  else if constexpr (std::is_same<T, float>::value)
  {
    scopy_(&n, v.data(), &one, data(row_idx, 0), &stride);
  }
  else
  {
    for (auto i = 0; i < n; i++)
    {
      (*this)(row_idx, 0 + i) = v(i);
    }
  }
  return *this;
}

//
// Update a specific row of a matrix, given a std::vector (overwrites original)
//
template<typename T>
fk::matrix<T> &
fk::matrix<T>::update_row(int const row_idx, std::vector<T> const &v)
{
  assert(ncols() == static_cast<int>(v.size()));
  assert(row_idx < nrows());

  int n{static_cast<int>(v.size())};
  int one{1};
  int stride = nrows();

  if constexpr (std::is_same<T, double>::value)
  { dcopy_(&n, const_cast<T *>(v.data()), &one, data(row_idx, 0), &stride); }
  else if constexpr (std::is_same<T, float>::value)
  {
    scopy_(&n, const_cast<T *>(v.data()), &one, data(row_idx, 0), &stride);
  }
  else
  {
    for (auto i = 0; i < n; i++)
    {
      (*this)(row_idx, 0 + i) = v[i];
    }
  }
  return *this;
}

//
// Set a submatrix within the matrix, given another (smaller) matrix
//
template<typename T>
fk::matrix<T> &
fk::matrix<T>::set_submatrix(int const row_idx, int const col_idx,
                             matrix<T> const &submatrix)
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
template<typename T>
fk::matrix<T>
fk::matrix<T>::extract_submatrix(int const row_idx, int const col_idx,
                                 int const num_rows, int const num_cols) const
{
  assert(row_idx >= 0);
  assert(col_idx >= 0);
  assert(row_idx + num_rows <= nrows());
  assert(col_idx + num_cols <= ncols());

  matrix submatrix(num_rows, num_cols);
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
template<typename T>
void fk::matrix<T>::print(std::string label) const
{
  std::cout << label << '\n';
  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
    {
      if constexpr (std::is_floating_point<T>::value)
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
template<typename T>
void fk::matrix<T>::dump_to_octave(char const *filename) const
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
