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
  P data;
};

namespace fk
{
template<typename P, mem_type mem, resource resrc>
class sparse
{
  template<typename, mem_type, resource>
  friend class sparse;

public:
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  sparse()
  {}

  explicit sparse(int nrows, int ncols, int nnz,
                  fk::vector<int, mem, resrc> &row_offsets,
                  fk::vector<int, mem, resrc> &col_indices,
                  fk::vector<P, mem, resrc> &values)
      : ncols_{ncols}, nrows_{nrows}, nnz_{nnz}, row_offsets_{row_offsets},
        col_indices_{col_indices}, values_{values}
  {}

  ~sparse() {}

  // create sparse matrix from dense matrix
  sparse(fk::matrix<P, mem, resrc> const &m)
  {
    P constexpr tol = 1.0e-10;
    // P constexpr tol = 2.0 * std::numeric_limits<P>::epsilon();

    ncols_ = m.ncols();
    nrows_ = m.nrows();
    nz_    = 0;

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
          nz_ += 1;
        }
      }
      size_t n_end = values.size();

      row_offsets_[row + 1] = row_offsets_[row] + (n_end - n_start);
    }

    col_indices_ = fk::vector<int, mem, resrc>(col_indices_tmp);

    values_ = fk::vector<P>(values);
    nnz_    = values_.size();

    expect(m.size() == values_.size() + nz_);
  }

  // create sparse matrix from multimap
  sparse(std::multimap<int, dense_item<P>> &items, int ncols, int nrows)
  {
    P constexpr tol = 1.0e-10;
    // P constexpr tol = 2.0 * std::numeric_limits<P>::epsilon();

    ncols_ = ncols;
    nrows_ = nrows;
    nz_    = 0;

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
        if (std::abs(col->second.data) > tol)
        {
          col_indices_tmp.push_back(col->second.col);
          values.push_back(col->second.data);
        }
      }
      size_t n_end = values.size();

      row_offsets_[row + 1] = row_offsets_[row] + (n_end - n_start);
    }

    col_indices_ = fk::vector<int, mem, resrc>(col_indices_tmp);

    values_ = fk::vector<P>(values);
    nnz_    = values_.size();

    expect(col_indices_tmp.size() == values_.size());
  }

  explicit sparse(sparse<P, mem, resrc> const &a)
      : ncols_{a.ncols_}, nrows_{a.nrows_}, nnz_{a.nnz_},
        row_offsets_{a.row_offsets_},
        col_indices_{a.col_indices_}, values_{a.values_}
  {}

  // move assignment
  sparse<P, mem, resrc> &operator=(fk::sparse<P, mem, resrc> &&a)
  {
    static_assert(mem != mem_type::const_view,
                  "cannot move assign into const_view!");

    if (&a == this)
      return *this;

    this->ncols_ = a.ncols_;
    this->nrows_ = a.nrows_;
    this->nnz_   = a.nnz_;

    this->row_offsets_ = std::move(a.get_offsets());
    this->col_indices_ = std::move(a.get_columns());
    this->values_      = std::move(a.get_values());

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
        nrows_, ncols_, nnz_, offsets_dev, col_dev, val_dev);
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
        nrows_, ncols_, nnz_, offsets_host, col_host, val_host);
  }

  //
  // basic queries to private data
  //
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  int nnz() const { return nnz_; }
  int64_t size() const { return int64_t{nrows()} * ncols(); }
  int64_t sp_size() const { return int64_t{values_.size()}; }

  const P *data() const { return values_.data(); }

  const int *offsets() const { return row_offsets_.data(); }

  const int *columns() const { return col_indices_.data(); }

  fk::vector<int, mem, resrc> &get_offsets() { return row_offsets_; }
  fk::vector<int, mem, resrc> &get_columns() { return col_indices_; }
  fk::vector<P, mem, resrc> &get_values() { return values_; }

private:
  int ncols_;
  int nrows_;
  int nnz_;
  int nz_;

  // CSR format
  fk::vector<int, mem, resrc> row_offsets_;
  fk::vector<int, mem, resrc> col_indices_;

  fk::vector<P, mem, resrc> values_;
};

} // namespace fk
} // namespace asgard