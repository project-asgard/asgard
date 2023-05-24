
#pragma once

#include "adapt.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "tensors.hpp"

#include "./device/asgard_kronmult.hpp"

// this interface between the low level kernels in src/device
// and the higher level data-structures

namespace asgard
{
/*!
 * \brief Contains persistent data for a kronmult operation.
 *
 * Holds the data for one batch of kronmult operations
 * so multiple calls can be made without reloading data
 * onto the device or recomputing pointers.
 * Especially useful for iterative operations, such as GMRES.
 *
 * \tparam precision is double or float
 *
 * This is the dense implementation, assuming that the discretization
 * uses a dense matrix with number of columns equal to the number of
 * operator terms times the number of rows.
 * Each row/column entry corresponds to a Kronecker product.
 */
template<typename precision>
class kronmult_matrix
{
public:
  //! \brief Creates uninitialized matrix cannot be used except to be reinitialized.
  kronmult_matrix()
      : num_dimensions_(0), kron_size_(0), num_rows_(0), num_cols_(0),
        num_terms_(0), tensor_size_(0), flops_(0)
  {}
  /*!
   *\brief Creates a new matrix and copies the data into internal structures.
   *
   * \param num_dimensions is the number of dimensions
   * \param kron_size is the size of the matrices in the kron-product
   *        i.e., called n in the compute routines and tied to the polynomial
   *degree kron_size = 1 for constants, 2 for linears, 3 for quadratics and so
   *on \param num_rows is the number of output blocks \param num_columns is the
   *number of kron-products for each output block, namely num_columns = num_rows
   ** num_terms, where num_terms is the number of operator terms in the PDE
   * \param values_A is the set of matrices for the kron-products, each matrix
   *is is stored in column-major format in kron_size^2 consecutive entries
   * \param index_A is the index offset of the matrices for the different
   *kron-product, namely kron-product for output i with column j uses matrices
   *at index index_A[num_dimensions * (i * num_columns + j) ... num_dimensions *
   *(i * num_columns + j) + num_dimensions-1] \code int idx = num_dimensions *
   *(i * num_columns + j); T const *A_d = &( values_A[ index_A[idx] ] );
   *   ...
   *   T const *A_2 = &( values_A[ index_A[idx + num_dimensions-3] ] );
   *   T const *A_1 = &( values_A[ index_A[idx + num_dimensions-2] ] );
   *   T const *A_0 = &( values_A[ index_A[idx + num_dimensions-1] ] );
   * \endcode
   */
  kronmult_matrix(
      int num_dimensions, int kron_size, int num_rows, int num_cols,
      int num_terms,
      fk::vector<int, mem_type::const_view, resource::host> const &row_indx,
      fk::vector<int, mem_type::const_view, resource::host> const &col_indx,
      fk::vector<int, mem_type::const_view, resource::host> const &index_A,
      fk::vector<precision, mem_type::const_view, resource::host> const
          &values_A)
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(num_terms),
        tensor_size_(1), row_indx_(row_indx.size()), col_indx_(col_indx.size()),
        iA(index_A.size()), vA(values_A.size())
  {
    if constexpr (data_mode == resource::host)
    {
#ifndef ASGARD_USE_CUDA // workaround clang and c++-17
      iA = index_A;
      vA = values_A;
      row_indx_ = row_indx;
      col_indx_ = col_indx;
#endif
    }
    else
    {
#ifdef ASGARD_USE_CUDA // workaround clang and c++-17
      iA = index_A.clone_onto_device();
      vA = values_A.clone_onto_device();
      row_indx_ = row_indx.clone_onto_device();
      col_indx_ = col_indx.clone_onto_device();
#endif
    }

    finalize_variables();
  }
  /*!
   *\brief Creates a new matrix and accepts the data as a r-values.
   *
   * The template parameter has to be resource::device when the GPU capabilities
   * have been enabled and resource::host when running only on the CPU.
   */
  template<resource input_mode>
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols,
                  int num_terms,
                  fk::vector<int, mem_type::owner, input_mode> const &&row_indx,
                  fk::vector<int, mem_type::owner, input_mode> const &&col_indx,
                  fk::vector<int, mem_type::owner, input_mode> &&index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A
                  )
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(num_terms),
        tensor_size_(1), row_indx_(std::move(row_indx)),
        col_indx_(std::move(col_indx)), iA(std::move(index_A)), vA(std::move(values_A))
  {
#ifdef ASGARD_USE_CUDA
    static_assert(
        input_mode == resource::device,
        "the GPU is enabled, so r-value inputs must have resource::device");
#else
    static_assert(
        input_mode == resource::host,
        "the GPU is disabled, so r-value inputs must have resource::host");
#endif

    finalize_variables();
  }

  /*!
   * \brief Computes y = alpha * kronmult_matrix * x + beta * y
   *
   * This method is not thread-safe!
   */
  void apply(precision alpha, precision const x[], precision beta,
             precision y[]) const
  {
#ifdef ASGARD_USE_CUDA
    if (beta != 0)
      fk::copy_to_device(ydev.data(), y, ydev.size());
    fk::copy_to_device(xdev.data(), x, xdev.size());
    kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(), col_indx_.size(),
                         col_indx_.data(), row_indx_.data(), num_terms_,
                         iA.data(), vA.data(), alpha, xdev.data(), beta,
                         ydev.data());
    fk::copy_to_host(y, ydev.data(), ydev.size());
#else
    kronmult::cpu_sparse(num_dimensions_, kron_size_, num_rows_,
                         row_indx_.data(), col_indx_.data(), num_terms_,
                         iA.data(), vA.data(), alpha, x, beta, y);
#endif
  }

  //! \brief Returns the number of kron-products
  int num_batch() const { return num_rows_ * num_cols_; }

  //! \brief Returns the size of a tensor block, i.e., kron_size^num_dimensions
  int tensor_size() const { return tensor_size_; }

  //! \brief Returns the size of the input vector, i.e., num_cols * tensor_size()
  int input_size() const { return tensor_size_ * num_cols_; }

  //! \brief Returns the size of the output vector, i.e., num_rows * tensor_size()
  int output_size() const { return tensor_size_ * num_rows_; }

  //! \brief The matrix evaluates to true if it has been initialized and false otherwise.
  operator bool() const { return (num_dimensions_ > 0); }

  //! \brief Returns the number of flops in a single call to apply()
  int64_t flops() const { return flops_; }

protected:
  //! \brief After dimensions and sizes have been initialized, set flop count and temporaries.
  void finalize_variables()
  {
    for (int d = 0; d < num_dimensions_; d++)
      tensor_size_ *= kron_size_;

    flops_ = kron_size_;
    for (int i = 0; i < num_dimensions_; i++)
      flops_ *= kron_size_;
    flops_ *= 2 * iA.size();

#ifdef ASGARD_USE_CUDA
    xdev = fk::vector<precision, mem_type::owner, data_mode>(tensor_size_ *
                                                             num_cols_);
    ydev = fk::vector<precision, mem_type::owner, data_mode>(tensor_size_ *
                                                             num_rows_);
#endif
  }

private:
  int num_dimensions_;
  int kron_size_; // i.e., n - size of the matrices
  int num_rows_;
  int num_cols_;
  int num_terms_;
  int tensor_size_;
  int64_t flops_;

#ifdef ASGARD_USE_CUDA
  static constexpr resource data_mode = resource::device;
  mutable fk::vector<precision, mem_type::owner, data_mode> xdev, ydev;
#else
  static constexpr resource data_mode = resource::host;
#endif

  fk::vector<int, mem_type::owner, data_mode> row_indx_;
  fk::vector<int, mem_type::owner, data_mode> col_indx_;

  // index of the matrices for each kronmult product
  fk::vector<int, mem_type::owner, data_mode> iA;
  // list of the operators
  fk::vector<precision, mem_type::owner, data_mode> vA;
  // pointer and indexes for the sparsity
};

/*!
 * \brief Given the PDE an the discretization, creates a new kronmult matrix.
 *
 * This method will copy out the coefficient data from the PDE terms
 * into structures index_A and values_A, so the method should be called only
 * when the operator terms change, e.g., due to refinement update.
 * The main purpose of the method is to "glue" the data-structures together
 * and work-around the excessive leading dimension which breaks each matrix
 * into small block scattered across memory (and hard to cache).
 * If the PDE data-structures are updated, then only this method needs to
 * change.
 */
template<typename P>
kronmult_matrix<P>
make_kronmult_dense(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                    options const &program_options,
                    imex_flag const imex = imex_flag::unspecified);

/*!
 * \brief Given the PDE an the discretization, creates a new kronmult matrix.
 *
 * The method is similar to the dense variant, but the resulting matrix
 * will have a sparse format.
 */
template<typename P>
kronmult_matrix<P>
make_kronmult_sparse(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                     options const &program_options,
                     imex_flag const imex = imex_flag::unspecified);

} // namespace asgard
