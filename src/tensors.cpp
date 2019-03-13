#include "tensors.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

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
// vector-vector multiply
// y := alpha*A*x + beta*y
// --------------------------------------------------------------------------
extern "C" double ddot_(int *n, double *X, int *incx, double *Y, int *incy);
extern "C" float sdot_(int *n, float *X, int *incx, float *Y, int *incy);

// --------------------------------------------------------------------------
// vector-scalar multiply
// y := x*alpha
// --------------------------------------------------------------------------
extern "C" double dscal_(int *n, double *alpha, double *X, int *incx);
extern "C" float sscal_(int *n, float *alpha, float *X, int *incx);

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
template<typename P>
static void igemm_(P *A, int const lda, P *B, int const ldb, P *C,
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
      P result = 0.0;
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

} // namespace fk

//-----------------------------------------------------------------------------
//
// fk::vector class implementation starts here
//
//-----------------------------------------------------------------------------
template<typename P, mem_type mem>
template<mem_type, typename>
fk::vector<P, mem>::vector() : data_{nullptr}, size_{0}
{}
// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
template<typename P, mem_type mem>
fk::vector<P, mem>::vector(int const size) : data_{new P[size]()}, size_{size}
{}

// can also do this with variadic template constructor for constness
// https://stackoverflow.com/a/5549918
// but possibly this is "too clever" for our needs right now

template<typename P, mem_type mem>
fk::vector<P, mem>::vector(std::initializer_list<P> list)
    : data_{new P[list.size()]}, size_{static_cast<int>(list.size())}
{
  std::copy(list.begin(), list.end(), data_);
}

template<typename P, mem_type mem>
fk::vector<P, mem>::vector(std::vector<P> const &v)
    : data_{new P[v.size()]}, size_{static_cast<int>(v.size())}
{
  std::copy(v.begin(), v.end(), data_);
}

//
// matrix conversion constructor linearizes the matrix, i.e. stacks the columns
// of the matrix into a single vector
//
template<typename P, mem_type mem>
fk::vector<P, mem>::vector(fk::matrix<P> const &mat) : data_{new P[mat.size()]}
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

template<typename P, mem_type mem>
fk::vector<P, mem>::~vector()
{
  delete[] data_;
}

//
// vector copy constructor for like types
//
template<typename P, mem_type mem>
fk::vector<P, mem>::vector(vector<P, mem> const &a)
    : data_{new P[a.size_]}, size_{a.size_}
{
  std::memcpy(data_, a.data(), a.size() * sizeof(P));
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

  size_ = a.size();
  memcpy(data_, a.data(), a.size() * sizeof(P));

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

  assert(size() == a.size());

  size_ = a.size_;
  P *temp{data_};
  data_   = a.data_;
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// converting vector copy constructor
//
template<typename P, mem_type mem>
template<typename PP, mem_type omem>
fk::vector<P, mem>::vector(vector<PP, omem> const &a)
    : data_{new P[a.size()]}, size_{a.size()}
{
  for (auto i = 0; i < a.size(); ++i)
  {
    (*this)(i) = static_cast<P>(a(i));
  }
}

//
// converting vector copy assignment
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
bool fk::vector<P, mem>::operator==(vector<P, mem> const &other) const
{
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
bool fk::vector<P, mem>::operator!=(vector<P, mem> const &other) const
{
  return !(*this == other);
}

template<typename P, mem_type mem>
bool fk::vector<P, mem>::operator<(vector<P, mem> const &other) const
{
  return std::lexicographical_compare(begin(), end(), other.begin(),
                                      other.end());
}

//
// vector addition operator
//
template<typename P, mem_type mem>
fk::vector<P, mem> fk::vector<P, mem>::
operator+(vector<P, mem> const &right) const
{
  assert(size() == right.size());
  vector<P, mem> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i) + right(i);
  return ans;
}

//
// vector subtraction operator
//
template<typename P, mem_type mem>
fk::vector<P, mem> fk::vector<P, mem>::
operator-(vector<P, mem> const &right) const
{
  assert(size() == right.size());
  vector<P, mem> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i)-right(i);
  return ans;
}

//
// vector*vector multiplication operator
//
template<typename P, mem_type mem>
P fk::vector<P, mem>::operator*(vector<P, mem> const &right) const
{
  assert(size() == right.size());
  int n           = size();
  int one         = 1;
  vector const &X = (*this);
  vector const &Y = right;

  if constexpr (std::is_same<P, double>::value)
  {
    return ddot_(&n, X.data(), &one, Y.data(), &one);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    return sdot_(&n, X.data(), &one, Y.data(), &one);
  }
  else
  {
    P ans = 0.0;
    for (auto i = 0; i < size(); ++i)
      ans += (*this)(i)*right(i);
    return ans;
  }
}

//
// vector*matrix multiplication operator
//
template<typename P, mem_type mem>
fk::vector<P, mem> fk::vector<P, mem>::operator*(fk::matrix<P> const &A) const
{
  // check dimension compatibility
  assert(size() == A.nrows());

  vector const &X = (*this);
  vector<P, mem> Y(A.ncols());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = m;
  int one_i = 1;

  if constexpr (std::is_same<P, double>::value)
  {
    P zero = 0.0;
    P one  = 1.0;
    dgemv_("t", &m, &n, &one, A.data(), &lda, X.data(), &one_i, &zero, Y.data(),
           &one_i);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    P zero = 0.0;
    P one  = 1.0;
    sgemv_("t", &m, &n, &one, A.data(), &lda, X.data(), &one_i, &zero, Y.data(),
           &one_i);
  }
  else
  {
    fk::matrix<P> At = A;
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
// vector*scalar multiplication operator
//
template<typename P, mem_type mem>
fk::vector<P, mem> fk::vector<P, mem>::operator*(P const x) const
{
  vector<P, mem> a(*this);
  int one_i = 1;
  int n     = a.size();
  P alpha   = x;
  if constexpr (std::is_same<P, double>::value)
  {
    dscal_(&n, &alpha, a.data(), &one_i);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sscal_(&n, &alpha, a.data(), &one_i);
  }
  else
  {
    for (int i = 0; i < n; ++i)
    {
      a(i) *= alpha;
    }
  }
  return a;
}

//
// perform the matrix kronecker product by
// interpreting vector operands/return vector
// as single column matrices.
//
template<typename P, mem_type mem>
fk::vector<P, mem>
fk::vector<P, mem>::single_column_kron(vector<P, mem> const &right) const
{
  fk::vector<P, mem> product((*this).size() * right.size());
  for (int i = 0; i < (*this).size(); ++i)
  {
    for (int j = 0; j < right.size(); ++j)
    {
      product(i * right.size() + j) = (*this)(i)*right(j);
    }
  }
  return product;
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
    std::cout << label << "(owner)" << '\n';
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
void fk::vector<P, mem>::resize(int const new_size)
{
  if (new_size == this->size())
    return;
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
}

template<typename P, mem_type mem>
fk::vector<P, mem> &fk::vector<P, mem>::concat(vector<P, mem> const &right)
{
  int const old_size = this->size();
  int const new_size = this->size() + right.size();
  data_ = static_cast<P *>(std::realloc(data(), new_size * sizeof(P)));
  size_ = new_size;
  std::memcpy(data(old_size), right.data(), right.size() * sizeof(P));
  return *this;
}

// set a subvector beginning at provided index
template<typename P, mem_type mem>
fk::vector<P, mem> &
fk::vector<P, mem>::set(int const index, fk::vector<P, mem> const sub_vector)
{
  assert(index >= 0);
  assert((index + sub_vector.size()) <= this->size());
  std::memcpy(&(*this)(index), sub_vector.data(),
              sub_vector.size() * sizeof(P));
  return *this;
}

// extract subvector, indices inclusive
template<typename P, mem_type mem>
fk::vector<P, mem>
fk::vector<P, mem>::extract(int const start, int const stop) const
{
  assert(start >= 0);
  assert(stop < this->size());
  assert(stop > start);

  int const sub_size = stop - start + 1;
  fk::vector<P, mem> sub_vector(sub_size);
  for (int i = 0; i < sub_size; ++i)
  {
    sub_vector(i) = (*this)(i + start);
  }
  return sub_vector;
}
//-----------------------------------------------------------------------------
//
// fk::matrix class implementation starts here
//
//-----------------------------------------------------------------------------

template<typename P>
fk::matrix<P>::matrix() : data_{nullptr}, nrows_{0}, ncols_{0}
{}

// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region

template<typename P>
fk::matrix<P>::matrix(int M, int N)
    : data_{new P[M * N]()}, nrows_{M}, ncols_{N}
{}

template<typename P>
fk::matrix<P>::matrix(std::initializer_list<std::initializer_list<P>> llist)
    : data_{new P[llist.size() * llist.begin()->size()]},
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

template<typename P>
fk::matrix<P>::~matrix()
{
  delete[] data_;
}

//
// matrix copy constructor
//
template<typename P>
fk::matrix<P>::matrix(matrix<P> const &a)
    : data_{new P[a.size()]}, nrows_{a.nrows()}, ncols_{a.ncols()}
{
  memcpy(data_, a.data(), a.size() * sizeof(P));
}

//
// matrix copy assignment
// this can probably be done better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P>
fk::matrix<P> &fk::matrix<P>::operator=(matrix<P> const &a)
{
  if (&a == this)
    return *this;

  assert((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  memcpy(data_, a.data(), a.size() * sizeof(P));
  return *this;
}

//
// converting matrix copy constructor
//
template<typename P>
template<typename PP>
fk::matrix<P>::matrix(matrix<PP> const &a)
    : data_{new P[a.size()]}, nrows_{a.nrows()}, ncols_{a.ncols()}
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
template<typename P>
template<typename PP>
fk::matrix<P> &fk::matrix<P>::operator=(matrix<PP> const &a)
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

template<typename P>
fk::matrix<P>::matrix(matrix<P> &&a)
    : data_{a.data()}, nrows_{a.nrows()}, ncols_{a.ncols()}
{
  a.data_  = nullptr; // b/c a's destructor will be called
  a.nrows_ = 0;
  a.ncols_ = 0;
}

//
// matrix move assignment
//
template<typename P>
fk::matrix<P> &fk::matrix<P>::operator=(matrix<P> &&a)
{
  if (&a == this)
    return *this;

  assert((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  P *temp{data_};
  data_   = a.data();
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// copy out of fk::vector - assumes the vector is column-major
//
template<typename P>
fk::matrix<P> &fk::matrix<P>::operator=(fk::vector<P> const &v)
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
template<typename P>
P &fk::matrix<P>::operator()(int const i, int const j)
{
  assert(i < nrows() && j < ncols());
  return *(data(i, j));
}

template<typename P>
P fk::matrix<P>::operator()(int const i, int const j) const
{
  assert(i < nrows() && j < ncols());
  return *(data(i, j));
}

// matrix comparison operators - set default tolerance above
// see https://stackoverflow.com/a/253874/6595797
// FIXME we may need to be more careful with these comparisons
template<typename P>
bool fk::matrix<P>::operator==(matrix<P> const &other) const
{
  if (&other == this)
    return true;
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

template<typename P>
bool fk::matrix<P>::operator!=(matrix<P> const &other) const
{
  return !(*this == other);
}

template<typename P>
bool fk::matrix<P>::operator<(matrix<P> const &other) const
{
  return std::lexicographical_compare(this->begin(), this->end(), other.begin(),
                                      other.end());
}

//
// matrix addition operator
//
template<typename P>
fk::matrix<P> fk::matrix<P>::operator+(matrix<P> const &right) const
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
template<typename P>
fk::matrix<P> fk::matrix<P>::operator-(matrix<P> const &right) const
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
template<typename P>
fk::matrix<P> fk::matrix<P>::operator*(P const right) const
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
template<typename P>
fk::vector<P> fk::matrix<P>::operator*(fk::vector<P> const &right) const
{
  // check dimension compatibility
  assert(ncols() == right.size());

  matrix<P> const &A = (*this);
  vector<P> Y(A.nrows());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = m;
  int one_i = 1;

  if constexpr (std::is_same<P, double>::value)
  {
    P one  = 1.0;
    P zero = 0.0;
    dgemv_("n", &m, &n, &one, A.data(), &lda, right.data(), &one_i, &zero,
           Y.data(), &one_i);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    P one  = 1.0;
    P zero = 0.0;
    sgemv_("n", &m, &n, &one, A.data(), &lda, right.data(), &one_i, &zero,
           Y.data(), &one_i);
  }
  else
  {
    igemm_(A.data(), lda, right.data(), right.size(), Y.data(), Y.size(), m, n,
           one_i);
  }

  return Y;
}

//
// matrix*matrix multiplication operator C[m,n] = A[m,k] * B[k,n]
//
template<typename P>
fk::matrix<P> fk::matrix<P>::operator*(matrix<P> const &B) const
{
  assert(ncols() == B.nrows()); // k == k

  // just aliases for easier reading
  matrix const &A = (*this);
  int m           = A.nrows();
  int n           = B.ncols();
  int k           = B.nrows();

  matrix<P> C(m, n);

  int lda = m;
  int ldb = k;
  int ldc = lda;

  if constexpr (std::is_same<P, double>::value)
  {
    P one  = 1.0;
    P zero = 0.0;
    dgemm_("n", "n", &m, &n, &k, &one, A.data(), &lda, B.data(), &ldb, &zero,
           C.data(), &ldc);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    P one  = 1.0;
    P zero = 0.0;
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
template<typename P>
fk::matrix<P> &fk::matrix<P>::transpose()
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

// Simple quad-loop kron prod
// @return the product
//
// FIXME this is NOT optimized.
// we will use the batch gemm method
// for performance-critical (large)
// krons
template<typename P>
fk::matrix<P> fk::matrix<P>::kron(matrix<P> const &B) const
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
template<typename P>
template<typename U>
std::enable_if_t<std::is_floating_point<U>::value && std::is_same<P, U>::value,
                 fk::matrix<P> &>
fk::matrix<P>::invert()
{
  assert(nrows() == ncols());

  int *ipiv{new int[ncols()]};
  int lwork{nrows() * ncols()};
  int lda = ncols();
  P *work{new P[nrows() * ncols()]};
  int info;

  if constexpr (std::is_same<P, double>::value)
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
template<typename P>
template<typename U>
std::enable_if_t<std::is_floating_point<U>::value && std::is_same<P, U>::value,
                 P>
fk::matrix<P>::determinant() const
{
  assert(nrows() == ncols());

  matrix temp{*this}; // get temp copy to do LU
  int *ipiv{new int[ncols()]};
  int info;
  int n   = ncols();
  int lda = ncols();

  if constexpr (std::is_same<P, double>::value)
  {
    dgetrf_(&n, &n, temp.data(0, 0), &lda, ipiv, &info);
  }
  else
  {
    sgetrf_(&n, &n, temp.data(0, 0), &lda, ipiv, &info);
  }

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
template<typename P>
fk::matrix<P> &
fk::matrix<P>::update_col(int const col_idx, fk::vector<P> const &v)
{
  assert(nrows() == static_cast<int>(v.size()));
  assert(col_idx < ncols());

  int n{v.size()};
  int one{1};
  int stride = 1;

  if constexpr (std::is_same<P, double>::value)
  {
    dcopy_(&n, v.data(), &one, data(0, col_idx), &stride);
  }
  else if constexpr (std::is_same<P, float>::value)
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
template<typename P>
fk::matrix<P> &
fk::matrix<P>::update_col(int const col_idx, std::vector<P> const &v)
{
  assert(nrows() == static_cast<int>(v.size()));
  assert(col_idx < ncols());

  int n{static_cast<int>(v.size())};
  int one{1};
  int stride = 1;

  if constexpr (std::is_same<P, double>::value)
  {
    dcopy_(&n, const_cast<P *>(v.data()), &one, data(0, col_idx), &stride);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    scopy_(&n, const_cast<P *>(v.data()), &one, data(0, col_idx), &stride);
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
// Update a specific row of a matrix, given a fk::vector<P> (overwrites
// original)
//
template<typename P>
fk::matrix<P> &
fk::matrix<P>::update_row(int const row_idx, fk::vector<P> const &v)
{
  assert(ncols() == v.size());
  assert(row_idx < nrows());

  int n{v.size()};
  int one{1};
  int stride = nrows();

  if constexpr (std::is_same<P, double>::value)
  {
    dcopy_(&n, v.data(), &one, data(row_idx, 0), &stride);
  }
  else if constexpr (std::is_same<P, float>::value)
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
template<typename P>
fk::matrix<P> &
fk::matrix<P>::update_row(int const row_idx, std::vector<P> const &v)
{
  assert(ncols() == static_cast<int>(v.size()));
  assert(row_idx < nrows());

  int n{static_cast<int>(v.size())};
  int one{1};
  int stride = nrows();

  if constexpr (std::is_same<P, double>::value)
  {
    dcopy_(&n, const_cast<P *>(v.data()), &one, data(row_idx, 0), &stride);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    scopy_(&n, const_cast<P *>(v.data()), &one, data(row_idx, 0), &stride);
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
// Resize, clearing all data FIXME when templating on ownership,
// this function will need to restrict callers to sizing up
//
template<typename P>
fk::matrix<P> &fk::matrix<P>::clear_and_resize(int const rows, int const cols)
{
  assert(rows >= 0);
  assert(cols >= 0);
  if (rows == 0 || cols == 0)
    assert(cols == rows);
  delete[] data_;
  data_  = new P[rows * cols]();
  nrows_ = rows;
  ncols_ = cols;
  return *this;
}

//
// Set a submatrix within the matrix, given another (smaller) matrix
//
template<typename P>
fk::matrix<P> &
fk::matrix<P>::set_submatrix(int const row_idx, int const col_idx,
                             matrix<P> const &submatrix)
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
template<typename P>
fk::matrix<P>
fk::matrix<P>::extract_submatrix(int const row_idx, int const col_idx,
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
template<typename P>
void fk::matrix<P>::print(std::string label) const
{
  std::cout << label << '\n';
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
template<typename P>
void fk::matrix<P>::dump_to_octave(char const *filename) const
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

// explicit instatiations
template class fk::vector<double>; // implies fk::vector<double,
                                   // mem_type::owner>
template class fk::vector<float>;
template class fk::vector<int>;
template class fk::matrix<double>;
template class fk::matrix<float>;
template class fk::matrix<int>;

template fk::vector<int>::vector(vector<float> const &);
template fk::vector<int>::vector(vector<double> const &);
template fk::vector<float>::vector(vector<int> const &);
template fk::vector<float>::vector(vector<double> const &);
template fk::vector<double>::vector(vector<int> const &);
template fk::vector<double>::vector(vector<float> const &);

template fk::vector<int> &fk::vector<int>::operator=(vector<float> const &);
template fk::vector<int> &fk::vector<int>::operator=(vector<double> const &);
template fk::vector<float> &fk::vector<float>::operator=(vector<int> const &);
template fk::vector<float> &fk::vector<float>::
operator=(vector<double> const &);
template fk::vector<double> &fk::vector<double>::operator=(vector<int> const &);
template fk::vector<double> &fk::vector<double>::
operator=(vector<float> const &);

template fk::matrix<int>::matrix(matrix<float> const &);
template fk::matrix<int>::matrix(matrix<double> const &);
template fk::matrix<float>::matrix(matrix<int> const &);
template fk::matrix<float>::matrix(matrix<double> const &);
template fk::matrix<double>::matrix(matrix<int> const &);
template fk::matrix<double>::matrix(matrix<float> const &);

template fk::matrix<int> &fk::matrix<int>::operator=(matrix<float> const &);
template fk::matrix<int> &fk::matrix<int>::operator=(matrix<double> const &);
template fk::matrix<float> &fk::matrix<float>::operator=(matrix<int> const &);
template fk::matrix<float> &fk::matrix<float>::
operator=(matrix<double> const &);
template fk::matrix<double> &fk::matrix<double>::operator=(matrix<int> const &);
template fk::matrix<double> &fk::matrix<double>::
operator=(matrix<float> const &);

// remove these when matrix::invert()/determinatn() is availble for ints
template fk::matrix<float> &fk::matrix<float>::invert();
template fk::matrix<double> &fk::matrix<double>::invert();
template float fk::matrix<float>::determinant() const;
template double fk::matrix<double>::determinant() const;

// added for mem_type support
template fk::vector<int, mem_type::owner>::vector();
template fk::vector<float, mem_type::owner>::vector();
template fk::vector<double, mem_type::owner>::vector();

template class fk::vector<double, mem_type::view>; // get the non-default
                                                   // mem_type::view
template class fk::vector<float, mem_type::view>;
template class fk::vector<int, mem_type::view>;
