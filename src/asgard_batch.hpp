#pragma once
#include "asgard_adapt.hpp"

namespace asgard
{
class element_subgrid;

// wrapper around an array of pointers to matrices or
// vectors for a call to batch gemm/gemv; i.e., the class
// represents the information for a batch operand
template<typename P,
         resource resrc =
             resource::device> // default to device - batch building functions
                               // only support this type for now
class batch
{
public:
  batch(int const num_entries, int const nrows, int const ncols,
        int const stride, bool const do_trans);
  batch(batch<P, resrc> const &other);
  batch &operator=(batch<P, resrc> const &other);
  batch(batch<P, resrc> &&other);
  batch &operator=(batch<P, resrc> &&other);
  ~batch();

  bool operator==(batch<P, resrc> const &) const;
  P *operator()(int const) const;

  template<mem_type mem>
  void assign_entry(fk::matrix<P, mem, resrc> const &a, int const position);
  bool clear_entry(int const position);

  P **get_list() const;

  bool is_filled() const;
  batch &clear_all();

  int num_entries() const { return num_entries_; }
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  int get_stride() const { return stride_; }
  bool get_trans() const { return do_trans_; }

  // using P* const * because P*const *const because the
  // last const on non-class return would be ignored
  using const_iterator = P *const *;
  const_iterator begin() const { return batch_; }
  const_iterator end() const { return batch_ + num_entries(); }

private:
  int const num_entries_; // number of matrices/vectors for this chunk
  int const nrows_;  // number of rows in matrices/size of vectors in this batch
  int const ncols_;  // number of cols in matrices (1 for vectors) in this batch
  int const stride_; // leading dimension passed into BLAS call for matrices;
                     // stride of vectors
  bool const do_trans_; // transpose passed into BLAS call for matrices

  P **batch_; // array of pointers to pass into blas call

  // want these for convenience in the class
  // don't want to expose them publicly...
  using iterator = P **;
  iterator begin() { return batch_; }
  iterator end() { return batch_ + num_entries(); }
};

// execute a batched gemm given a, b, c batch lists
template<typename P, resource resrc>
void batched_gemm(batch<P, resrc> const &a, batch<P, resrc> const &b,
                  batch<P, resrc> const &c, P const alpha, P const beta);

// inline helper to calc workspace size for realspace batching, where dimensions
// of matrices not uniform
template<typename P, resource resrc>
inline int calculate_workspace_length(
    std::vector<fk::matrix<P, mem_type::const_view, resrc>> const &matrices,
    int const x_size)
{
  int greatest = x_size;
  int r_prod   = 1;
  int c_prod   = 1;
  typename std::vector<
      fk::matrix<P, mem_type::const_view, resrc>>::const_reverse_iterator iter;
  for (iter = matrices.rbegin(); iter != matrices.rend(); ++iter)
  {
    c_prod *= iter->ncols();
    expect(c_prod > 0);
    r_prod *= iter->nrows();
    int const size = x_size / c_prod * r_prod;
    greatest       = std::max(greatest, size);
  }
  return greatest;
}

// which of the kronecker-product based algorithms the batch chain supports
enum class chain_method
{
  realspace, // for realspace transform
  advance // for time advance //FIXME deprecated. once realspace is refactored,
          // this entire component will be deleted.
};

template<chain_method method>
using enable_for_realspace =
    std::enable_if_t<method == chain_method::realspace>;

// class that marshals pointers for batched gemm and calls into blas to execute.
// realspace method enqueues all gemms for one kron(A1,...,AN)*x
template<typename P, resource resrc,
         chain_method method = chain_method::realspace>
class batch_chain
{
public:
  // constructors allocate batches and assign data pointers
  // realspace transform constructor
  template<chain_method m_ = method, typename = enable_for_realspace<m_>>
  batch_chain(
      std::vector<fk::matrix<P, mem_type::const_view, resrc>> const &matrices,
      fk::vector<P, mem_type::const_view, resrc> const &x,
      std::array<fk::vector<P, mem_type::view, resrc>, 2> &workspace,
      fk::vector<P, mem_type::view, resrc> &final_output);
  void execute() const;

private:
  // A matrices for batched gemm
  std::vector<batch<P, resrc>> left_;
  // B matrices
  std::vector<batch<P, resrc>> right_;
  // C matrices
  std::vector<batch<P, resrc>> product_;
};

// helpers for converting linear coordinates into operator matrix indices
inline fk::vector<int> linearize(fk::vector<int> const &coords)
{
  fk::vector<int> linear(coords.size() / 2);
  for (int i = 0; i < linear.size(); ++i)
  {
    linear(i) = elements::get_1d_index(coords(i), coords(i + linear.size()));
  }
  return linear;
}

} // namespace asgard
