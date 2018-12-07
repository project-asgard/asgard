
#ifndef __TENSORS_HPP__
#define __TENSORS_HPP__

#include <string>
#include <vector>

/* tolerance for answer comparisons */
#define TOL 1.0e-10
//#define TOL std::numeric_limits<double>::epsilon() * 50

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

// --------------------------------------------------------------------------
// matrix-vector multiply
// y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
// --------------------------------------------------------------------------
extern "C" void dgemv_(char const *trans, int *m, int *n, double *alpha,
                       double *A, int *lda, double *x, int *incx, double *beta,
                       double *y, int *incy);

// --------------------------------------------------------------------------
// matrix-matrix multiply
// C := alpha*A*B + beta*C
// --------------------------------------------------------------------------
extern "C" void dgemm_(char const *transa, char const *transb, int *m, int *n,
                       int *k, double *alpha, double *A, int *lda, double *B,
                       int *ldb, double *beta, double *C, int *ldc);

// --------------------------------------------------------------------------
// LU decomposition of a general matrix
// --------------------------------------------------------------------------
extern "C" void
dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);

// --------------------------------------------------------------------------
// inverse of a matrix given its LU decomposition
// --------------------------------------------------------------------------
extern "C" void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work,
                        int *lwork, int *info);

// forward declarations
class vector;
class matrix;

class vector
{
public:
  vector();
  vector(int const size);
  vector(std::initializer_list<double> list);
  vector(std::vector<double> const &);

  ~vector();

  vector(vector const &);
  vector &operator=(vector const &);
  vector(vector &&);
  vector &operator=(vector &&);

  //
  // copy out of std::vector
  //
  vector &operator=(std::vector<double> const &);

  //
  // copy into std::vector
  //
  std::vector<double> to_std() const;

  //
  // subscripting operators
  //
  double &operator()(int const);
  double operator()(int const) const;
  //
  // comparison operators
  //
  bool operator==(vector const &) const;
  bool operator!=(vector const &) const;
  //
  // math operators
  //
  vector operator+(vector const &right) const;
  vector operator-(vector const &right) const;
  double operator*(vector const &)const;
  vector operator*(matrix const &)const;
  //
  // basic queries to private data
  //
  int size() const { return size_; }
  double *data(int const elem = 0) const { return &data_[elem]; }
  //
  // utility functions
  //
  void print(std::string const label = "") const;
  void dump_to_octave(char const *) const;
  void resize(int const size = 0);

  typedef double *iterator;
  typedef const double *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }

private:
  double *data_; //< pointer to elements
  int size_;     //< dimension
};

class matrix
{
public:
  matrix();
  matrix(int rows, int cols);
  matrix(std::initializer_list<std::initializer_list<double>> list);
  matrix(std::vector<double> const &);

  ~matrix();

  matrix(matrix const &);
  matrix &operator=(matrix const &);
  matrix(matrix &&);
  matrix &operator=(matrix &&);

  //
  // copy out of std::vector
  //
  matrix &operator=(std::vector<double> const &);
  //
  // subscripting operators
  //
  double &operator()(int const, int const);
  double operator()(int const, int const) const;
  //
  // comparison operators
  //
  bool operator==(matrix const &) const;
  bool operator!=(matrix const &) const;
  //
  // math operators
  //
  matrix operator*(matrix const &)const;
  matrix operator*(int const) const;
  matrix operator+(matrix const &) const;
  matrix operator-(matrix const &) const;
  matrix &invert();
  matrix &transpose();
  double determinant() const;
  //
  // basic queries to private data
  //
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  int size() const { return nrows() * ncols(); }
  double *data(int const i = 0, int const j = 0) const
  {
    // return &data_[i * ncols() + j]; // row-major
    return &data_[j * nrows() + i]; // column-major
  }
  //
  // utility functions
  //
  matrix &update_row(int const, fk::vector const &);
  matrix &update_row(int const, std::vector<double> const &);
  matrix &set_submatrix(int const row_idx, int const col_idx,
                        fk::matrix const &submatrix);
  matrix extract_submatrix(int const row_idx, int const col_idx,
                           int const num_rows, int const num_cols) const;
  void print(std::string const label = "") const;
  void dump_to_octave(char const *name) const;

  typedef double *iterator;
  typedef const double *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }

private:
  double *data_; //< pointer to elements
  int nrows_;    //< row dimension
  int ncols_;    //< column dimension
};

} // namespace fk

#endif
