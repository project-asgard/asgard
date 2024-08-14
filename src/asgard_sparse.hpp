#pragma once

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"

namespace asgard
{
template<typename P>
struct dense_item
{
  int row;
  int col;
  P val;
};

namespace fk
{
template<typename P, resource resrc = resource::host>
class sparse
{
  template<typename, resource>
  friend class sparse;

public:
  sparse() : ncols_{0}, nrows_{0} {}

  template<mem_type mem>
  explicit sparse(int nrows, int ncols, int nnz,
                  fk::vector<int, mem, resrc> const &row_offsets,
                  fk::vector<int, mem, resrc> const &col_indices,
                  fk::vector<P, mem, resrc> const &values)
      : ncols_{ncols}, nrows_{nrows}, row_offsets_{row_offsets},
        col_indices_{col_indices}, values_{values}
  {
    expect(col_indices.size() == nnz);
  }

  template<mem_type mem>
  explicit sparse(int nrows, int ncols, int nnz,
                  fk::vector<int, mem, resrc> &&row_offsets,
                  fk::vector<int, mem, resrc> &&col_indices,
                  fk::vector<P, mem, resrc> &&values)
      : ncols_{ncols}, nrows_{nrows}, row_offsets_{row_offsets},
        col_indices_{col_indices}, values_{values}
  {
    expect(col_indices_.size() == nnz);
  }

  ~sparse() {}

  // create sparse matrix from dense matrix
  template<resource r_ = resrc, typename = enable_for_host<r_>, mem_type mem>
  sparse(fk::matrix<P, mem, resrc> const &m)
  {
    P constexpr tol = std::is_floating_point_v<P> ? 1.0e-10 : 0;
    // P constexpr tol = 2.0 * std::numeric_limits<P>::epsilon();

    ncols_ = m.ncols();
    nrows_ = m.nrows();
    int nz = 0;

    row_offsets_.resize(nrows_ + 1);
    row_offsets_[0] = 0;

    // Determine number of non-zeros to pre-allocate
    int nnz = 0;
    for (int row = 0; row < nrows_; row++)
    {
      for (int col = 0; col < ncols_; col++)
      {
        if (std::abs(m(row, col)) > tol)
        {
          nnz += 1;
        }
      }
    }

    values_      = fk::vector<P>(nnz);
    col_indices_ = fk::vector<int>(nnz);

    size_t index = 0;
    for (int row = 0; row < nrows_; row++)
    {
      size_t n_start = index;
      for (int col = 0; col < ncols_; col++)
      {
        if (std::abs(m(row, col)) > tol)
        {
          col_indices_[index] = col;
          values_[index]      = m(row, col);
          index += 1;
        }
        else
        {
          nz += 1;
        }
      }
      size_t n_end = index;

      row_offsets_[row + 1] = row_offsets_[row] + (n_end - n_start);
    }

    expect(col_indices_.size() == values_.size());
    expect(m.size() == values_.size() + nz);
  }

  // create sparse matrix from multimap
  sparse(std::multimap<int, dense_item<P>> const &items, int ncols, int nrows)
  {
    P constexpr tol = std::is_floating_point_v<P> ? 1.0e-10 : 0;
    // P constexpr tol = 2.0 * std::numeric_limits<P>::epsilon();

    ncols_ = ncols;
    nrows_ = nrows;

    row_offsets_.resize(nrows_ + 1);
    row_offsets_[0] = 0;

    std::vector<P> values;
    std::vector<int> col_indices_tmp;

    values.reserve(nrows_);
    for (int row = 0; row < nrows_; row++)
    {
      size_t n_start = values.size();
      auto range     = items.equal_range(row);
      for (auto col = range.first; col != range.second; ++col)
      {
        if (std::abs(col->second.val) > tol)
        {
          col_indices_tmp.push_back(col->second.col);
          values.push_back(col->second.val);
        }
      }
      size_t n_end = values.size();

      row_offsets_[row + 1] = row_offsets_[row] + (n_end - n_start);
    }

    col_indices_ = fk::vector<int, mem_type::owner, resrc>(col_indices_tmp);

    values_ = fk::vector<P>(values);
    expect(col_indices_.size() == values_.size());

    expect(col_indices_tmp.size() == static_cast<size_t>(values_.size()));
  }

  // create sparse matrix from a vector, filling elements on the diagonal
  template<resource r_ = resrc, typename = enable_for_host<r_>, mem_type mem>
  sparse(fk::vector<P, mem, resrc> const &diag)
  {
    nrows_ = diag.size();
    ncols_ = diag.size();

    values_      = fk::vector<P, mem, resrc>(diag);
    row_offsets_ = fk::vector<int, mem, resrc>(nrows_ + 1);
    col_indices_ = fk::vector<int, mem, resrc>(ncols_);

    row_offsets_[0] = 0;
    for (int i = 0; i < ncols_; i++)
    {
      col_indices_[i]     = i;
      row_offsets_[i + 1] = i + 1;
    }
  }

  // copy constructor
  sparse(fk::sparse<P, resrc> const &other)
      : ncols_{other.ncols_}, nrows_{other.nrows_},
        row_offsets_{other.row_offsets_},
        col_indices_{other.col_indices_}, values_{other.values_}
  {}

  // copy assignment
  sparse<P, resrc> &operator=(fk::sparse<P, resrc> const &a)
  {
    if (&a == this)
      return *this;

    nrows_ = a.nrows_;
    ncols_ = a.ncols_;

    row_offsets_ = fk::vector<int, mem_type::owner, resrc>(a.get_offsets());
    col_indices_ = fk::vector<int, mem_type::owner, resrc>(a.get_columns());
    values_      = fk::vector<P, mem_type::owner, resrc>(a.get_values());

    return *this;
  }

  // move constructor
  sparse(fk::sparse<P, resrc> &&other)
      : ncols_{other.ncols_}, nrows_{other.nrows_}, row_offsets_{std::move(
                                                        other.row_offsets_)},
        col_indices_{std::move(other.col_indices_)}, values_{std::move(
                                                         other.values_)}
  {}

  // move assignment
  sparse<P, resrc> &operator=(fk::sparse<P, resrc> &&a)
  {
    if (&a == this)
      return *this;

    this->ncols_ = a.ncols_;
    this->nrows_ = a.nrows_;

    this->row_offsets_ = std::move(a.row_offsets_);
    this->col_indices_ = std::move(a.col_indices_);
    this->values_      = std::move(a.values_);

    return *this;
  }

  // transfer functions
  // host->dev, new matrix
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  sparse<P, resource::device> clone_onto_device() const
  {
    return sparse<P, resource::device>(
        nrows_, ncols_, col_indices_.size(), row_offsets_.clone_onto_device(),
        col_indices_.clone_onto_device(), values_.clone_onto_device());
  }

  // transfer functions
  // dev->host, new matrix
  template<resource r_ = resrc, typename = enable_for_device<r_>>
  sparse<P, resource::host> clone_onto_host() const
  {
    return sparse<P, resource::host>(
        nrows_, ncols_, col_indices_.size(), row_offsets_.clone_onto_host(),
        col_indices_.clone_onto_host(), values_.clone_onto_host());
  }

  // convert this sparse matrix back to a dense matrix
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  fk::matrix<P, mem_type::owner, resrc> to_dense() const
  {
    // create dense, filled with 0 initially
    fk::matrix<P, mem_type::owner, resrc> dense(nrows_, ncols_);

    // populate entries of the dense matrix
    for (int row = 0; row < nrows_; row++)
    {
      for (int col = row_offsets_[row]; col < row_offsets_[row + 1]; col++)
      {
        dense(row, col_indices_[col]) = values_[col];
      }
    }

    return dense;
  }

  // checks if the dense element at (row, col) exists in the sparse matrix
  bool exists(int const row, int const col) const
  {
    expect(row >= 0);
    expect(row < nrows_);
    expect(col >= 0);
    expect(col < ncols_);

    for (int i = row_offsets_[row]; i < row_offsets_[row + 1]; i++)
    {
      if (col_indices_[i] == col)
      {
        return true;
      }
    }

    return false;
  }

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator==(fk::sparse<P> const &other) const
  {
    if (&other == this)
      return true;
    if (nnz() != other.nnz() || size() != other.size())
      return false;
    if (row_offsets_ == other.row_offsets_ &&
        col_indices_ == other.col_indices_ && values_ == other.values_)
      return true;
    return false;
  }
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator!=(sparse<P> const &other) const
  {
    return !(*this == other);
  }

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void print(std::string const label = "") const
  {
    std::cout << label << "(owner, nnz = " << nnz() << ")" << '\n';
    //  Print these out as row major even though stored in memory as column
    //  major.
    for (int row = 0; row < nrows_; row++)
    {
      for (int col = row_offsets_[row]; col < row_offsets_[row + 1]; col++)
      {
        std::cout << "(" << std::setw(2) << row << ", " << std::setw(2) << col
                  << ") ";
        P const val = values_[col];
        if constexpr (std::is_floating_point_v<P>)
        {
          std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                    << std::right << val;
        }
        else
        {
          std::cout << val << " ";
        }
        std::cout << '\n';
      }
    }
  }

  //
  // basic queries to private data
  //
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  int nnz() const { return col_indices_.size(); }
  int64_t size() const { return int64_t{nrows()} * ncols(); }
  int64_t sp_size() const { return int64_t{values_.size()}; }
  bool empty() const { return values_.size() == 0; }

  P const *data() const { return values_.data(); }
  P *data() { return values_.data(); }

  int const *offsets() const { return row_offsets_.data(); }
  int *offsets() { return row_offsets_.data(); }

  int const *columns() const { return col_indices_.data(); }
  int *columns() { return col_indices_.data(); }

  fk::vector<int, mem_type::owner, resrc> get_offsets() const
  {
    return row_offsets_;
  }
  fk::vector<int, mem_type::owner, resrc> get_columns() const
  {
    return col_indices_;
  }
  fk::vector<P, mem_type::owner, resrc> get_values() const { return values_; }

private:
  int ncols_;
  int nrows_;

  // CSR format
  fk::vector<int, mem_type::owner, resrc> row_offsets_;
  fk::vector<int, mem_type::owner, resrc> col_indices_;

  fk::vector<P, mem_type::owner, resrc> values_;
};

} // namespace fk
} // namespace asgard
