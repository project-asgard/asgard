#pragma once
#include "asgard_indexset.hpp"

namespace asgard
{
//! function format for 1d function, allows evaluations for large batches of funcitons
template<typename P>
using function_1d = std::function<void(std::vector<P> const &, std::vector<P> &)>;

/*!
 * \internal
 * \brief Stores a matrix in block format
 *
 * The entries of each block are stored contiguously in memory and logically
 * organized into an matrix with column major format.
 *
 * This is for testing and cross-reference purposes, i.e., the sparse block types
 * can be converted to this full-block format for earier introspection.
 * \endinternal
 */
template<typename P>
class block_matrix
{
public:
  //! make an empty matrix
  block_matrix() : nrows_(0), ncols_(0), data_(0, 0) {}
  //! initialize matrix with given block-size and number of rows/cols
  block_matrix(int block_size, int64_t num_rows, int64_t num_cols)
      : nrows_(num_rows), ncols_(num_cols), data_(block_size, num_rows * num_cols)
  {}

  //! block size
  int nblock() const { return data_.stride(); }
  //! number of rows
  int64_t nrows() const { return nrows_; }
  //! number of columns
  int64_t ncols() const { return ncols_; }

  //! gives the i,j-th block
  P *operator() (int64_t i, int64_t j) { return data_[j * nrows_ + i]; }
  //! gives the i,j-th block, const-overload
  P const *operator() (int64_t i, int64_t j) const { return data_[j * nrows_ + i]; }

  //! returns the raw internal data
  P *data() { return data_.data(); }
  //! returns the raw internal data, const-overload
  P const *data() const { return data_.data(); }

  //! fill with single entry
  void fill(P v) { std::fill_n(data_[0], data_.total_size(), v); }

  //! prints the block-matrix with given block row/cols, or assume block is square
  void print(std::ostream &os, int br = -1, int bc = -1, int oswidth = 12)
  {
    if (br == -1)
    {
      int const nb = data_.stride();
      br = 0;
      while (br < nb and br * br != nb)
        ++br;
      expect(br * br == nb);
      bc = br;
    }
    expect(br * bc == data_.stride());
    for (auto r : indexof(nrows_))
    {
      for (int i = 0; i < br; i++)
      {
        for (auto c : indexof(ncols_))
        {
          for (int j = 0; j < bc; j++)
            os << std::setw(oswidth) << data_[c * nrows_ + r][j * br + i];
          os << std::setw(oswidth / 2) << "  ";
        }
        os << '\n';
      }
      os << '\n';
    }
  }

  //! prints one column
  void printc(std::ostream &os, int c = 0, int oswidth = 12)
  {
    int const nb = data_.stride();
    int br = 0;
    while (br < nb and br * br != nb)
      ++br;
    expect(br * br == nb);
    int bc = br;
    expect(br * bc == data_.stride());
    for (auto r : indexof(nrows_))
    {
      for (int i = 0; i < br; i++)
      {

        for (int j = 0; j < bc; j++)
          os << std::setw(oswidth) << data_[c * nrows_ + r][j * br + i];
        os << std::setw(oswidth / 2) << "  ";

        os << '\n';
      }
      os << '\n';
    }
  }

  //! prints the block-matrix with given block row/cols, or assume block is square
  void printr(std::ostream &os, int r = 0, int oswidth = 12)
  {
    int const nb = data_.stride();
    int br = 0;
    while (br < nb and br * br != nb)
      ++br;
    expect(br * br == nb);
    int bc = br;
    expect(br * bc == data_.stride());
    for (int i = 0; i < br; i++)
    {
      for (auto c : indexof(ncols_))
      {
        for (int j = 0; j < bc; j++)
          os << std::setw(oswidth) << data_[c * nrows_ + r][j * br + i];
        os << std::setw(oswidth / 2) << "  ";
      }
      os << '\n';
    }
    os << '\n';
  }

private:
  int64_t nrows_, ncols_;
  vector2d<P> data_;
};

//! convert larger fk::matrix to block_matrix given the number of blocks and columns
template<typename P, mem_type mem>
block_matrix<P> block_convert(fk::matrix<P, mem> const &fm, int br, int bc)
{
  int64_t const nrows = fm.nrows() / br;
  int64_t const ncols = fm.ncols() / bc;
  expect(nrows * br == fm.nrows());
  expect(ncols * bc == fm.ncols());

  block_matrix<P> mat(br * bc, nrows, ncols);

  for (auto c : indexof(ncols))
  {
    for (auto r : indexof(nrows))
    {
      P *im = mat(r, c);
      for (int j : indexof<int>(bc))
        im = std::copy_n(fm.data(br * r, c * bc + j), br, im);
    }
  }

  return mat;
}

/*!
 * \internal
 * \brief Block-diagonal matrix, stores the factorized mass matrix
 *
 * The idea is almost the same as the diagonal matrix, the difference being
 * that we store the factorized form of the diagonal blocks so we can
 * quickly apply the inverse onto another matrix.
 * \endinternal
 */
template<typename P>
class mass_matrix
{
public:
  //! create an empty matrix
  mass_matrix() {}
  //! get the mass matrix for the given level
  mass_matrix(int const nblock, int const num_rows)
    : data_(nblock, num_rows)
  {
    expect(nblock > 0);
    expect(num_rows > 0);
  }
  //! size of the block
  int nblock() const { return data_.stride(); }
  //! number of rows
  int64_t nrows() const { return data_.num_strips(); }

  //! gives the diagonal block
  P *operator() (int64_t row) { return data_[row]; }
  //! gives the diagonal block, const-overload
  P const *operator() (int64_t row) const { return data_[row]; }

  //! gives the diagonal block
  P *operator[] (int64_t row) { return data_[row]; }
  //! gives the diagonal block, const-overload
  P const *operator[] (int64_t row) const { return data_[row]; }

  //! returns the raw internal data
  P *data() { return data_.data(); }
  //! returns the raw internal data, const-overload
  P const *data() const { return data_.data(); }

  //! returns true of the matrix is empty
  bool empty() const { return data_.empty(); }

  //! converts the matrix to a full one, mostly for testing/plotting
  block_matrix<P> to_full() const
  {
    int const n = nblock();
    block_matrix<P> full(n, nrows(), nrows());
    for (auto r : indexof(nrows()))
      std::copy_n(data_[r], n, full(r, r));
    return full;
  }

private:
  vector2d<P> data_;
};

/*!
 * \internal
 * \brief Stores mass-matrices per level
 *
 * The assumption is that all mass matrices do not change in time,
 * therefore, once we have computed the mass for a given level,
 * then we can sore and reuse it without the need to recompute.
 * \endinternal
 */
template<typename P>
struct level_mass_matrces
{
  //! allocate memory for the blocks
  void set_non_identity()
  {
    // 31 levels will overflow the int-index for the multi-index set
    // the supported delta-t is 1 / 2^31 ~ 4.6E-10
    if (mats_.size() != 31)
      mats_.resize(31);
  }
  //! return the matrix at the given level
  mass_matrix<P> &operator[] (int level) { return mats_[level]; }
  //! return the matrix at the given level (const)
  mass_matrix<P> const &operator[] (int level) const { return mats_[level]; }
  //! return the matrix at the given level
  mass_matrix<P> &at(int level) { return mats_[level]; }
  //! return the matrix at the given level (const)
  mass_matrix<P> const &at(int level) const { return mats_[level]; }

  //! return true if the matrix has been set for this level
  bool has_level(int l) const { return (not mats_.empty() and not mats_[l].empty()); }
  //! matrices
  std::vector<mass_matrix<P>> mats_;
};

/*!
 * \internal
 * \brief Stores a square block diagonal matrix
 *
 * \endinternal
 */
template<typename P>
class block_diag_matrix
{
public:
  //! make an empty matrix
  block_diag_matrix() : nrows_(0), data_(0, 0) {}
  //! initialize matrix with given block-size and number of rows/cols
  block_diag_matrix(int block_size, int64_t num_rows)
      : nrows_(num_rows), data_(block_size, num_rows)
  {}

  //! block size
  int nblock() const { return data_.stride(); }
  //! number of rows
  int64_t nrows() const { return nrows_; }

  //! gives the main diagonal block
  P *operator() (int64_t r) { return data_[r]; }
  //! gives the i,j-th block, const-overload
  P const *operator() (int64_t r) const { return data_[r]; }

  //! gives the main diagonal block
  P *operator[] (int64_t r) { return data_[r]; }
  //! gives the i,j-th block, const-overload
  P const *operator[] (int64_t r) const { return data_[r]; }

  //! returns the raw internal data
  P *data() { return data_.data(); }
  //! returns the raw internal data, const-overload
  P const *data() const { return data_.data(); }

  //! converts the matrix to a full one, mostly for testing/plotting
  block_matrix<P> to_full() const
  {
    int const n = nblock();
    block_matrix<P> full(n, nrows_, nrows_);
    for (auto r : indexof(nrows_))
      std::copy_n(data_[r], n, full(r, r));
    return full;
  }

  //! resizes the matrix and sets all entries to zero
  void resize_and_zero(int block_size, int64_t nrows)
  {
    nrows_ = nrows;
    data_.resize_and_zero(block_size, nrows);
  }
  //! resize to match the other matrix size
  void resize_and_zero(block_diag_matrix<P> const &other)
  {
    resize_and_zero(other.nblock(), other.nrows());
  }

private:
  int64_t nrows_;
  vector2d<P> data_;
};

/*!
 * \internal
 * \brief Stores a matrix in block tri-diagonal format
 *
 * The entries of each block are stored contiguously in memory and logically
 * organized into an tri-diagonal matrix (using row-major format).
 * The three diagonals have the names lower, diag, upper.
 * The entry for lower(0) is actually the top-right entry (0, n-1) corresponding
 * to periodic boundary condition. Similarly upper(n-1) is the bottom left entry.
 * Both entries are included in the pattern but could be numerically zero.
 * \endinternal
 */
template<typename P>
class block_tri_matrix
{
public:
  //! make an empty matrix
  block_tri_matrix() : nrows_(0), data_(0, 0) {}
  //! initialize matrix with given block-size and number of rows/cols
  block_tri_matrix(int block_size, int64_t num_rows)
      : nrows_(num_rows), data_(block_size, 3 * num_rows)
  {}

  //! copy diagonal to tri-diagonal matrix
  block_tri_matrix& operator = (block_diag_matrix<P> const &other)
  {
    resize_and_zero(other.nblock(), other.nrows());
    for (auto i : indexof(nrows_))
      std::copy_n(other[i], data_.stride(), diag(i));
    return *this;
  }

  //! block size
  int nblock() const { return data_.stride(); }
  //! number of rows
  int64_t nrows() const { return nrows_; }

  //! gives the main diagonal block
  P *operator() (int64_t r) { return data_[3 * r + 1]; }
  //! gives the i,j-th block, const-overload
  P const *operator() (int64_t r) const { return data_[3 * r + 1]; }

  //! gives the lower diagonal block
  P *lower(int64_t r) { return data_[3 * r]; }
  //! gives the lower diagonal block, const-overload
  P const *lower(int64_t r) const { return data_[3 * r]; }
  //! gives the main diagonal block
  P *diag(int64_t r) { return data_[3 * r + 1]; }
  //! gives the main diagonal block, const-overload
  P const *diag(int64_t r) const { return data_[3 * r + 1]; }
  //! gives the upper diagonal block
  P *upper(int64_t r) { return data_[3 * r + 2]; }
  //! gives the upper diagonal block, const-overload
  P const *upper(int64_t r) const { return data_[3 * r + 2]; }

  //! returns the raw internal data
  P *data() { return data_.data(); }
  //! returns the raw internal data, const-overload
  P const *data() const { return data_.data(); }

  //! fill with single entry
  void fill(P v) { std::fill_n(data_[0], data_.total_size(), v); }
  //! resizes the matrix and sets all entries to zero
  void resize_and_zero(int block_size, int64_t nrows)
  {
    nrows_ = nrows;
    data_.resize_and_zero(block_size, 3 * nrows);
  }
  //! resize to match the other matrix size
  void resize_and_zero(block_tri_matrix<P> const &other)
  {
    resize_and_zero(other.nblock(), other.nrows());
  }

  //! converts the matrix to a full one, mostly for testing/plotting
  block_matrix<P> to_full() const
  {
    int const n = nblock();
    block_matrix<P> full(n, nrows_, nrows_);
    std::copy_n(diag(0), n, full(0, 0));
    if (nrows_ == 1)
      return full;
    std::copy_n(lower(0), n, full(0, nrows_ - 1));
    std::copy_n(upper(0), n, full(0, 1));
    for (int64_t r = 1; r < nrows_ - 1; r++)
    {
      std::copy_n(lower(r), n, full(r, r - 1));
      std::copy_n(diag(r), n, full(r, r));
      std::copy_n(upper(r), n, full(r, r + 1));
    }
    std::copy_n(lower(nrows_ - 1), n, full(nrows_ - 1, nrows_ - 2));
    std::copy_n(diag(nrows_ - 1), n, full(nrows_ - 1, nrows_ - 1));
    std::copy_n(upper(nrows_ - 1), n, full(nrows_ - 1, 0));
    if (nrows_ == 2)
    {
      for (int i : indexof<int>(data_.stride()))
        full(0, 1)[i] += lower(0)[i];
      for (int i : indexof<int>(data_.stride()))
        full(1, 0)[i] += lower(nrows_ - 1)[i];
    }
    return full;
  };

private:
  int64_t nrows_;
  vector2d<P> data_;
};

/*!
 * \internal
 * \brief Block sparse matrix and holds the type-pattern (volume or edge)
 *
 * \endinternal
 */
template<typename P>
struct block_sparse_matrix
{
public:
  //! make an empty matrix
  block_sparse_matrix() : data_(0, 0) {}
  //! initialize matrix with given block-size and number of rows/cols
  block_sparse_matrix(int block_size, int64_t nnz, connect_1d::hierarchy htype)
      : htype_(htype), data_(block_size, nnz)
  {}
  //! block size
  int nblock() const { return data_.stride(); }
  //! num non-zero blocks
  int64_t nnz() const { return data_.num_strips(); }
  //! returns the type of the sparsity pattern
  operator connect_1d::hierarchy() const { return htype_; }
  //! returns the block at the given index
  P *operator[] (int64_t i) { return data_[i]; }
  //! returns the block at the given index
  P const *operator[] (int64_t i) const { return data_[i]; }

  //! converts the matrix to a full one, mostly for testing/plotting
  block_matrix<P> to_full(connection_patterns const &conns) const
  {
    connect_1d const &conn = conns(htype_);
    int const n     = nblock();
    int const nrows = conn.num_rows();
    int mcol = 0;
    for (int j = 0; j < conn.num_connections(); j++)
      mcol = std::max(mcol, conn[j]);
    block_matrix<P> full(n, nrows, mcol + 1);

    for (int r = 0; r < nrows; r++)
      for (int j = conn.row_begin(r); j < conn.row_end(r); j++)
        std::copy_n(data_[j], n, full(r, conn[j]));

    return full;
  }
  //! compute matrix vector product
  void gemv(int const n, int const level, connection_patterns const &conns, P const x[], P y[]) const;
  //! copy into external vector
  void copy_out(std::vector<P> &out) const
  {
    data_.copy_out(out);
  }
  //! convert the matrix to dense fk::matrix
  fk::matrix<P> to_fk_matrix(int const n, connection_patterns const &conns) const
  {
    connect_1d const &conn = conns(htype_);
    int const nrows = conn.num_rows();
    fk::matrix<P> mat(n * nrows, n * nrows);
    for (int r = 0; r < nrows; r++)
      for (int j = conn.row_begin(r); j < conn.row_end(r); j++)
        for (int k = 0; k < n; k++)
          std::copy_n(data_[j] + n * k , n, &mat(n * r, n * conn[j] + k));
    return mat;
  }
private:
  connect_1d::hierarchy htype_;
  vector2d<P> data_;
};

/*!
 * \internal
 * \brief Multiply block-tri-diagonal matrices
 *
 * The product of general tri-diagonal matrices is penta-diagonal matrix
 * and chaining multiple matrices together increases the bandwidth of the result.
 * However, if the matrices are logically tri-diagonal but numerically
 * lower/upper, then the result will be a general tridiagonal matrix.
 *
 * This is the upper times lower case, or upwind-downwind for terms 0 and 1
 * \endinternal
 */
template<typename P>
void gemm_block_tri_ul(int const n, block_tri_matrix<P> const &A, block_tri_matrix<P> const &B,
                       block_tri_matrix<P> &C);

/*!
 * \internal
 * \brief Multiply block-tri-diagonal matrices
 *
 * The product of general tri-diagonal matrices is penta-diagonal matrix
 * and chaining multiple matrices together increases the bandwidth of the result.
 * However, if the matrices are logically tri-diagonal but numerically
 * lower/upper, then the result will be a general tridiagonal matrix.
 *
 * This is the lower times upper case, or downwind-upwind for terms 0 and 1
 * \endinternal
 */
template<typename P>
void gemm_block_tri_lu(int const n, block_tri_matrix<P> const &A, block_tri_matrix<P> const &B,
                       block_tri_matrix<P> &C);

/*!
 * \internal
 * \brief Multiply block-diagonal by block-tri-triagonal matrix
 *
 * \endinternal
 */
template<typename P>
void gemm_diag_tri(int const n, block_diag_matrix<P> const &A,
                   block_tri_matrix<P> const &B, block_tri_matrix<P> &C);
/*!
 * \internal
 * \brief Multiply block-tri-diagonal by block-diagonal matrix
 *
 * \endinternal
 */
template<typename P>
void gemm_tri_diag(int const n, block_tri_matrix<P> const &A,
                   block_diag_matrix<P> const &B, block_tri_matrix<P> &C);

/*!
 * \internal
 * \brief Multiply block-diagonal matrices
 *
 * \endinternal
 */
template<typename P>
void gemm_block_diag(int const n, block_diag_matrix<P> const &A, block_diag_matrix<P> const &B,
                     block_diag_matrix<P> &C);

/*!
 * \internal
 * \brief Applies the inverse of a mass matrix times block tri-diagonal matrix
 *
 * \endinternal
 */
template<typename P>
void invert_mass(int const n, mass_matrix<P> const &mass, block_tri_matrix<P> &op);

/*!
 * \internal
 * \brief Applies the inverse of a mass matrix times block diagonal matrix
 *
 * \endinternal
 */
template<typename P>
void invert_mass(int const n, mass_matrix<P> const &mass, block_diag_matrix<P> &op);

/*!
 * \internal
 * \brief Applies the inverse of a mass matrix times vector
 *
 * \endinternal
 */
template<typename P>
void invert_mass(int const n, mass_matrix<P> const &mass, P x[]);

} // namespace asgard
