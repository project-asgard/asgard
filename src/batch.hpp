#pragma once
#include "chunk.hpp"
#include "element_table.hpp"
#include "pde/pde_base.hpp"
#include "tensors.hpp"
#include <array>
#include <numeric>

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
  void assign_raw(P *const a, int const position);
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

// execute a batched gemv given a, b, c batch lists
template<typename P, resource resrc>
void batched_gemv(batch<P, resrc> const &a, batch<P, resrc> const &b,
                  batch<P, resrc> const &c, P const alpha, P const beta);

// this could be named better - encode dimensions for a gemm
struct matrix_size_set
{
  int const rows_a;
  int const cols_a;
  int const rows_b;
  int const cols_b;
  matrix_size_set(int const rows_a, int const cols_a, int const rows_b,
                  int const cols_b)
      : rows_a(rows_a), cols_a(cols_a), rows_b(rows_b), cols_b(cols_b){};
};

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
    assert(c_prod > 0);
    r_prod *= iter->nrows();
    int const size = x_size / c_prod * r_prod;
    greatest       = std::max(greatest, size);
  }
  return greatest;
}

// workspace for the primary computation in time advance. along with
// the coefficient matrices, we need this space resident on whatever
// accelerator we are using
template<typename P, resource resrc>
class batch_workspace
{
public:
  batch_workspace(PDE<P> const &pde, element_subgrid const &subgrid,
                  std::vector<element_chunk> const &chunks);
  fk::vector<P, mem_type::owner, resrc> const &get_unit_vector() const;

  double size_MB() const
  {
    int64_t num_elems = input.size() + reduction_space.size() +
                        kron_intermediate.size() + output.size() +
                        unit_vector_.size();

    double const bytes     = static_cast<double>(num_elems) * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

  // input, output, workspace for batched gemm/reduction
  fk::vector<P, mem_type::owner, resrc> input;
  fk::vector<P, mem_type::owner, resrc> reduction_space;
  fk::vector<P, mem_type::owner, resrc> kron_intermediate;
  fk::vector<P, mem_type::owner, resrc> output;

private:
  fk::vector<P, mem_type::owner, resrc> unit_vector_;
};

// which of the kronecker-product based algorithms the batch chain supports
enum class chain_method
{
  realspace, // for realspace transform
  advance    // for time advance
};

template<chain_method method>
using enable_for_realspace =
    std::enable_if_t<method == chain_method::realspace>;

template<chain_method method>
using enable_for_advance = std::enable_if_t<method == chain_method::advance>;

// class that marshals pointers for batched gemm and calls into blas to execute.
// realspace method enqueues all gemms for one kron(A1,...,AN)*x,
// time advance method equeues gemms for many such krons
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
  // time advance constructor
  template<chain_method m_ = method, typename = enable_for_advance<m_>>
  batch_chain(PDE<P> const &pde, element_table const &elem_table,
              batch_workspace<P, resrc> const &workspace,
              element_subgrid const &subgrid, element_chunk const &chunk);
  void execute() const;

private:
  // enqueue gemms for one kronmult - time advance
  template<chain_method m_ = method, typename = enable_for_advance<m_>>
  void kronmult_to_batch_sets(P *const *const A, P *const x, P *const y,
                              P *const *const work, int const batch_offset,
                              PDE<P> const &pde);

  // compute gemm sizes for a given dimension for time advance batching
  template<chain_method m_ = method, typename = enable_for_advance<m_>>
  matrix_size_set
  compute_dimensions(int const degree, int const num_dims, int const dimension)
  {
    assert(dimension >= 0);
    assert(dimension < num_dims);
    assert(num_dims > 0);
    assert(degree > 0);
    if (dimension == 0)
    {
      return matrix_size_set(degree, degree, degree,
                             static_cast<int>(std::pow(degree, num_dims - 1)));
    }
    return matrix_size_set(static_cast<int>(std::pow(degree, dimension)),
                           degree, degree, degree);
  }

  // compute how many gemms required for kronmult at a given PDE dimension for
  // time advance batching
  template<chain_method m_ = method, typename = enable_for_advance<m_>>
  int compute_batch_size(int const degree, int const num_dims,
                         int const dimension)
  {
    assert(dimension >= 0);
    assert(dimension < num_dims);
    assert(num_dims > 0);
    assert(degree > 0);

    if (dimension == 0 || dimension == num_dims - 1)
    {
      return 1;
    }

    return std::pow(degree, (num_dims - dimension - 1));
  }

  // A matrices for batched gemm
  std::vector<batch<P, resrc>> left_;
  // B matrices
  std::vector<batch<P, resrc>> right_;
  // C matrices
  std::vector<batch<P, resrc>> product_;
};

// function to build system matrix for implicit stepping
// doesn't use batches, but does use many of the same helpers/structure
template<typename P>
void build_system_matrix(PDE<P> const &pde, element_table const &elem_table,
                         element_chunk const &chunk, fk::matrix<P> &A);
