#pragma once
#include "build_info.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "lib_dispatch.hpp"
#include "tensors.hpp"
#include "tools.hpp"

#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <vector>

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
template<typename P, mem_type mem = mem_type::owner,
         resource resrc = resource::host>
class sparse
{
  template<typename, mem_type, resource>
  friend class sparse;

public:
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  sparse() : ncols_{0}, nrows_{0}
  {}

  explicit sparse(int nrows, int ncols, int nnz,
                  fk::vector<int, mem, resrc> const &row_offsets,
                  fk::vector<int, mem, resrc> const &col_indices,
                  fk::vector<P, mem, resrc> const &values)
      : ncols_{ncols}, nrows_{nrows}, row_offsets_{row_offsets},
        col_indices_{col_indices}, values_{values}
  {
    expect(col_indices.size() == nnz);
  }

  ~sparse() {}

  // create sparse matrix from dense matrix
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  sparse(fk::matrix<P, mem, resrc> const &m)
  {
    P constexpr tol = 1.0e-10;
    // P constexpr tol = 2.0 * std::numeric_limits<P>::epsilon();

    ncols_ = m.ncols();
    nrows_ = m.nrows();
    int nz = 0;

    row_offsets_.resize(nrows_ + 1);
    row_offsets_[0] = 0;

    std::vector<P> values;
    std::vector<int> col_indices_tmp;

    values.reserve(nrows_);
    for (int row = 0; row < nrows_; row++)
    {
      size_t n_start = values.size();
      for (int col = 0; col < ncols_; col++)
      {
        if (std::abs(m(row, col)) > tol)
        {
          col_indices_tmp.push_back(col);
          values.push_back(m(row, col));
        }
        else
        {
          nz += 1;
        }
      }
      size_t n_end = values.size();

      row_offsets_[row + 1] = row_offsets_[row] + (n_end - n_start);
    }

    col_indices_ = fk::vector<int, mem, resrc>(col_indices_tmp);

    values_ = fk::vector<P>(values);
    expect(col_indices_.size() == values_.size());

    expect(m.size() == values_.size() + nz);
  }

  // create sparse matrix from multimap
  sparse(std::multimap<int, dense_item<P>> const &items, int ncols, int nrows)
  {
    P constexpr tol = 1.0e-10;
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

    col_indices_ = fk::vector<int, mem, resrc>(col_indices_tmp);

    values_ = fk::vector<P>(values);
    expect(col_indices_.size() == values_.size());

    expect(col_indices_tmp.size() == static_cast<size_t>(values_.size()));
  }

  // create sparse matrix from a vector, filling elements on the diagonal
  template<resource r_ = resrc, typename = enable_for_host<r_>>
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
  sparse(fk::sparse<P, mem, resrc> const &other)
      : ncols_{other.ncols_}, nrows_{other.nrows_}
  {
    row_offsets_ = fk::vector<int, mem, resrc>(other.get_offsets());
    col_indices_ = fk::vector<int, mem, resrc>(other.get_columns());
    values_      = fk::vector<P, mem, resrc>(other.get_values());
  }

  // copy assignment
  sparse<P, mem, resrc> &operator=(fk::sparse<P, mem, resrc> const &a)
  {
    static_assert(mem != mem_type::const_view,
                  "cannot copy assign into const_view!");

    if (&a == this)
      return *this;

    nrows_ = a.nrows_;
    ncols_ = a.ncols_;

    row_offsets_ = fk::vector<int, mem, resrc>(a.get_offsets());
    col_indices_ = fk::vector<int, mem, resrc>(a.get_columns());
    values_      = fk::vector<P, mem, resrc>(a.get_values());

    return *this;
  }

  // move constructor
  sparse(fk::sparse<P, mem, resrc> &&other)
      : ncols_{other.ncols_}, nrows_{other.nrows_}, row_offsets_{std::move(
                                                        other.row_offsets_)},
        col_indices_{std::move(other.col_indices_)}, values_{std::move(
                                                         other.values_)}
  {}

  // move assignment
  // template<mem_type m_ = mem, typename = enable_for_owner<m_>,
  //         resource r_ = resrc, typename = enable_for_host<r_>>
  sparse<P, mem, resrc> &operator=(fk::sparse<P, mem, resrc> &&a)
  {
    static_assert(mem != mem_type::const_view,
                  "cannot move assign into const_view!");

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
  sparse<P, mem_type::owner, resource::device> clone_onto_device() const
  {
    auto offsets_dev = row_offsets_.clone_onto_device();
    auto col_dev     = col_indices_.clone_onto_device();
    auto val_dev     = values_.clone_onto_device();
    return sparse<P, mem_type::owner, resource::device>(
        nrows_, ncols_, col_dev.size(), offsets_dev, col_dev, val_dev);
  }

  // transfer functions
  // dev->host, new matrix
  template<resource r_ = resrc, typename = enable_for_device<r_>>
  sparse<P, mem_type::owner, resource::host> clone_onto_host() const
  {
    auto offsets_host = row_offsets_.clone_onto_host();
    auto col_host     = col_indices_.clone_onto_host();
    auto val_host     = values_.clone_onto_host();
    return sparse<P, mem_type::owner, resource::host>(
        nrows_, ncols_, col_host.size(), offsets_host, col_host, val_host);
  }

  // convert this sparse matrix back to a dense matrix
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  fk::matrix<P, mem, resrc> to_dense() const
  {
    // create dense, filled with 0 initially
    fk::matrix<P, mem, resrc> dense(nrows_, ncols_);

    // populate entries of the dense matrix
    int col_index_offset = 0;
    for (int row = 0; row < nrows_; row++)
    {
      int num_in_row = row_offsets_[row + 1] - row_offsets_[row];
      for (int col = 0; col < num_in_row; col++)
      {
        int index                       = col_index_offset + col;
        dense(row, col_indices_[index]) = values_[index];
      }

      col_index_offset += num_in_row;
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

  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator==(fk::sparse<P, omem> const &other) const
  {
    if constexpr (omem == mem)
      if (&other == this)
        return true;
    if (nnz() != other.nnz() || size() != other.size())
      return false;
    if (row_offsets_ == other.row_offsets_ &&
        col_indices_ == other.col_indices_ && values_ == other.values_)
      return true;
    return false;
  }
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator!=(sparse<P, omem> const &other) const
  {
    return !(*this == other);
  }

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void print(std::string const label = "") const
  {
    if constexpr (mem == mem_type::owner)
      std::cout << label << "(owner)" << '\n';
    else
      expect(false); // above cases cover all implemented mem types

    //  Print these out as row major even though stored in memory as column
    //  major.
    for (auto i = 0; i < nrows(); ++i)
    {
      for (auto j = 0; j < ncols(); ++j)
      {
        if (exists(i, j))
        {
          std::cout << "(" << std::setw(2) << i << ", " << std::setw(2) << j
                    << ") ";
          P const val = values_[col_indices_[row_offsets_[i]]];
          if constexpr (std::is_floating_point<P>::value)
          {
            std::cout << std::setw(12) << std::setprecision(4)
                      << std::scientific << std::right << val;
          }
          else
          {
            std::cout << val << " ";
          }
          std::cout << '\n';
        }
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

  fk::vector<int, mem, resrc> get_offsets() const { return row_offsets_; }
  fk::vector<int, mem, resrc> get_columns() const { return col_indices_; }
  fk::vector<P, mem, resrc> get_values() const { return values_; }

private:
  int ncols_;
  int nrows_;

  // CSR format
  fk::vector<int, mem, resrc> row_offsets_;
  fk::vector<int, mem, resrc> col_indices_;

  fk::vector<P, mem, resrc> values_;
};

} // namespace fk
} // namespace asgard