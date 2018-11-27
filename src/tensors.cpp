
#include "tensors.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

//-----------------------------------------------------------------------------
//
// fk::vector class implementation starts here
//
//-----------------------------------------------------------------------------

fk::vector::vector() : data_{nullptr}, size_{0} {}

// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
fk::vector::vector(int const size) : data_{new double[size]()}, size_{size} {}

// can also do this with variadic template constructor for constness
// https://stackoverflow.com/a/5549918
// but possibly this is "too clever" for our needs right now
fk::vector::vector(std::initializer_list<double> list)
    : data_{new double[list.size()]}, size_{static_cast<int>(list.size())}
{
  std::copy(list.begin(), list.end(), data_);
}

fk::vector::vector(std::vector<double> const &v)
    : data_{new double[v.size()]}, size_{static_cast<int>(v.size())}
{
  std::copy(v.begin(), v.end(), data_);
}

fk::vector::~vector() { delete[] data_; }

//
// vector copy constructor
//
fk::vector::vector(vector const &a) : data_{new double[a.size_]}, size_{a.size_}
{
  std::memcpy(data_, a.data(), a.size() * sizeof(double));
}

//
// vector copy assignment
// this can probably be optimized better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
fk::vector &fk::vector::operator=(vector const &a)
{
  if (&a == this) return *this;

  assert(size() == a.size());

  size_ = a.size_;
  memcpy(data_, a.data(), a.size() * sizeof(double));

  return *this;
}

//
// vector move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
fk::vector::vector(vector &&a) : data_{a.data_}, size_{a.size_}
{
  a.data_ = nullptr; // b/c a's destructor will be called
  a.size_ = 0;
}

//
// vector move assignment
//
fk::vector &fk::vector::operator=(vector &&a)
{
  if (&a == this) return *this;

  assert(size() == a.size());

  size_ = a.size_;
  double *temp{data_};
  data_   = a.data_;
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// copy out of std::vector
//
fk::vector &fk::vector::operator=(std::vector<double> const &v)
{
  assert(size() == static_cast<int>(v.size()));
  std::memcpy(data_, v.data(), v.size() * sizeof(double));
  return *this;
}

//
// copy into std::vector
//
std::vector<double> fk::vector::to_std() const
{
  return std::vector<double>(data(), data() + size());
}

// vector subscript operator
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
double &fk::vector::operator()(int i)
{
  assert(i < size_);
  return data_[i];
}
double fk::vector::operator()(int i) const
{
  assert(i < size_);
  return data_[i];
}

//
// vector comparison operators - set default tolerance above
//
bool fk::vector::operator==(vector const &other) const
{
  if (&other == this) return true;
  if (size() != other.size()) return false;
  for (auto i = 0; i < size(); ++i)
    if (std::abs((*this)(i)) > TOL && std::abs(other(i)) > TOL)
      if (std::abs((*this)(i)-other(i)) > TOL) { return false; }
  return true;
}
bool fk::vector::operator!=(vector const &other) const
{
  return !(*this == other);
}

//
// vector addition operator
//
fk::vector fk::vector::operator+(vector const &right) const
{
  assert(size() == right.size());
  vector ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i) + right(i);
  return ans;
}

//
// vector subtraction operator
//
fk::vector fk::vector::operator-(vector const &right) const
{
  assert(size() == right.size());
  vector ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i)-right(i);
  return ans;
}

//
// vector*vector multiplication operator
//
double fk::vector::operator*(vector const &right) const
{
  assert(size() == right.size());
  double ans = 0.0;
  for (auto i = 0; i < size(); ++i)
    ans += (*this)(i)*right(i);
  return ans;
}

//
// vector*matrix multiplication operator
//
fk::vector fk::vector::operator*(matrix const &A) const
{
  // check dimension compatibility
  assert(size() == A.nrows());

  vector const &X = (*this);
  vector Y(A.ncols());

  int m       = A.nrows();
  int n       = A.ncols();
  int lda     = m;
  double one  = 1.0;
  int one_i   = 1;
  double zero = 0.0;

  dgemv_("t", &m, &n, &one, A.data(), &lda, X.data(), &one_i, &zero, Y.data(),
         &one_i);

  return Y;
}

//
// Prints out the values of a vector
//
// @param[in]   label   a string label printed with the output
// @param[in]   b       the vector from the batch to print out
// @return      Nothing
//
void fk::vector::print(std::string const label) const
{
  std::cout << label << '\n';
  for (auto i = 0; i < size(); ++i)
    std::cout << std::setw(12) << std::setprecision(4) << std::scientific
              << std::right << (*this)(i);

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
void fk::vector::dump_to_octave(char const *filename) const
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
void fk::vector::resize(int const new_size)
{
  if (new_size == this->size()) return;
  double *old_data{data_};
  data_ = new double[new_size]();
  if (size() > 0 && new_size > 0)
    std::memcpy(data_, old_data, new_size * sizeof(double));
  size_ = new_size;
  delete[] old_data;
}

//-----------------------------------------------------------------------------
//
// fk::matrix class implementation starts here
//
//-----------------------------------------------------------------------------

fk::matrix::matrix() : data_{nullptr}, nrows_{0}, ncols_{0} {}

// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
fk::matrix::matrix(int M, int N)
    : data_{new double[M * N]()}, nrows_{M}, ncols_{N}
{}

fk::matrix::matrix(std::initializer_list<std::initializer_list<double>> llist)
    : data_{new double[llist.size() * llist.begin()->size()]},
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
fk::matrix::matrix(std::vector<double> const &v) : data_{new double[v.size()]}
{
  double iptr;
  assert(std::modf(std::sqrt(v.size()), &iptr) == 0);
  nrows_ = std::sqrt(v.size());
  ncols_ = std::sqrt(v.size());
  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      (*this)(i, j) = v[j + i * ncols()];
}

fk::matrix::~matrix() { delete[] data_; }

//
// matrix copy constructor
//
fk::matrix::matrix(matrix const &a)
    : data_{new double[a.size()]}, nrows_{a.nrows()}, ncols_{a.ncols()}
{
  memcpy(data_, a.data(), a.size() * sizeof(double));
}

//
// matrix copy assignment
// this can probably be done better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
fk::matrix &fk::matrix::operator=(matrix const &a)
{
  if (&a == this) return *this;

  assert((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  memcpy(data_, a.data(), a.size() * sizeof(double));
  return *this;
}

//
// matrix move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
fk::matrix::matrix(matrix &&a)
    : data_{a.data()}, nrows_{a.nrows()}, ncols_{a.ncols()}
{
  a.data_  = nullptr; // b/c a's destructor will be called
  a.nrows_ = 0;
  a.ncols_ = 0;
}

//
// matrix move assignment
//
fk::matrix &fk::matrix::operator=(matrix &&a)
{
  if (&a == this) return *this;

  assert((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  double *temp{data_};
  data_   = a.data();
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// copy out of std::vector - assumes the std::vector is column-major
//
fk::matrix &fk::matrix::operator=(std::vector<double> const &v)
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
double &fk::matrix::operator()(int const i, int const j)
{
  assert(i < nrows() && j < ncols());
  return *(data(i, j));
}
double fk::matrix::operator()(int const i, int const j) const
{
  assert(i < nrows() && j < ncols());
  return *(data(i, j));
}

//
// matrix comparison operators - set default tolerance above
//
bool fk::matrix::operator==(matrix const &other) const
{
  if (&other == this) return true;
  if (nrows() != other.nrows() || ncols() != other.ncols()) return false;
  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      if (std::abs((*this)(i, j)) > TOL && std::abs(other(i, j)) > TOL)
        if (std::abs((*this)(i, j) - other(i, j)) > TOL) { return false; }
  return true;
}
bool fk::matrix::operator!=(matrix const &other) const
{
  return !(*this == other);
}

//
// matrix addition operator
//
fk::matrix fk::matrix::operator+(matrix const &right) const
{
  assert(nrows() == right.nrows() && ncols() == right.ncols());

  matrix ans(nrows(), ncols());
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
fk::matrix fk::matrix::operator-(matrix const &right) const
{
  assert(nrows() == right.nrows() && ncols() == right.ncols());

  matrix ans(nrows(), ncols());
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
fk::matrix fk::matrix::operator*(int const right) const
{
  matrix ans(nrows(), ncols());
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
fk::matrix fk::matrix::operator*(matrix const &B) const
{
  assert(ncols() == B.nrows()); // k == k

  // just aliases for easier reading
  matrix const &A = (*this);
  int m           = A.nrows();
  int n           = B.ncols();
  int k           = B.nrows();

  matrix C(m, n);

  int lda = m;
  int ldb = k;
  int ldc = lda;

  double one  = 1.0;
  double zero = 0.0;

  dgemm_("n", "n", &m, &n, &k, &one, A.data(), &lda, B.data(), &ldb, &zero,
         C.data(), &ldc);

  return C;
}

//
// Invert a square matrix (overwrites original)
// @return  the inverted matrix
//
fk::matrix &fk::matrix::invert()
{
  assert(nrows() == ncols());

  int *ipiv{new int[ncols()]};
  int lwork{nrows() * ncols()};
  int lda = ncols();
  double *work{new double[nrows() * ncols()]};
  int info;

  dgetrf_(&ncols_, &ncols_, data(0, 0), &lda, ipiv, &info);
  dgetri_(&ncols_, data(0, 0), &lda, ipiv, work, &lwork, &info);

  delete[] ipiv;
  delete[] work;
  return *this;
}

//
// Transpose a matrix (overwrites original)
// @return  the transposed matrix
//
// FIXME could be worthwhile to optimize the matrix transpose
fk::matrix &fk::matrix::transpose()
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
// Get the determinant of the matrix  (non destructive)
// (based on src/Numerics/DeterminantOperators.h)
// (note possible problems with over/underflow
// - see Ed's emails 12/5/16, 10/14/16, 10/10/16.
// how is this handled / is it necessary in production?
// possibly okay for small KxK matrices - can build in a check/warning)
//
// @param[in]   mat   integer matrix (walker) to get determinant from
// @return  the determinant (type double)
//
double fk::matrix::determinant() const
{
  assert(nrows() == ncols());

  matrix temp{*this}; // get temp copy to do LU
  int *ipiv{new int[ncols()]};
  int info;
  int n   = ncols();
  int lda = ncols();
  dgetrf_(&n, &n, temp.data(0, 0), &lda, ipiv, &info);
  double det = 1.0;
  int sign   = 1;
  for (auto i = 0; i < nrows(); ++i)
  {
    if (ipiv[i] != i + 1) sign *= -1;
    det *= temp(i, i);
  }
  det *= static_cast<double>(sign);
  delete[] ipiv;
  return det;
}

//
// Update a specific row of a matrix, given a fk::vector (overwrites original)
//
fk::matrix &fk::matrix::update_row(int const row_idx, fk::vector const &v)
{
  assert(ncols() == v.size());
  assert(row_idx < nrows());

  int n{v.size()};
  int one{1};
  int stride = nrows();
  dcopy_(&n, v.data(), &one, data(row_idx, 0), &stride);

  return *this;
}

//
// Update a specific row of a matrix, given a std::vector (overwrites original)
//
fk::matrix &
fk::matrix::update_row(int const row_idx, std::vector<double> const &v)
{
  assert(ncols() == static_cast<int>(v.size()));
  assert(row_idx < nrows());

  int n{static_cast<int>(v.size())};
  int one{1};
  int stride = nrows();
  // ugh, hate to do this. thanks blas
  dcopy_(&n, const_cast<double *>(v.data()), &one, data(row_idx, 0), &stride);

  return *this;
}
//
// Prints out the values of a matrix
// @return  Nothing
//
void fk::matrix::print(std::string label) const
{
  std::cout << label << '\n';
  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
      std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                << std::right << (*this)(i, j);

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
void fk::matrix::dump_to_octave(char const *filename) const
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
