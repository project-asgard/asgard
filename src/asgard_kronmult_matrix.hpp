
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
   * \brief Creates a new matrix and moves the data into internal structures.
   *
   * The constructor can be called directly but in most cases the matrix should
   * be constructed through the factory method make_kronmult_matrix().
   * The input parameters must reflect the PDE being used as well as the user
   * desired sparse or dense mode and whether the CPU or GPU are being used.
   *
   * \param num_dimensions is the number of dimensions
   * \param kron_size is the size of the matrices in the kron-product
   *        i.e., called n in the compute routines and tied to the polynomial
   *        degree kron_size = 1 for constants, 2 for linears, 3 for quadratics
   *        and so on
   * \param num_rows is the number of output blocks
   * \param num_columns is the number of kron-products for each output block,
   *        namely num_columns = num_rows num_terms, where num_terms is the
   *        number of operator terms in the PDE
   * \param row_indx is either empty, which indicates dense mode, or contains
   *        the row-indexes for the sparse matrix (see below)
   * \param col_indx is either empty, which indicates dense mode, or contains
   *        the column-indexes for the sparse matrix (see below)
   * \param values_A is the set of matrices for the kron-products, each matrix
   *        is is stored in column-major format in kron_size^2 consecutive
   *        entries
   * \param index_A is the index offset of the matrices for the different
   *         kron-product, namely kron-product for output i with column j uses
   *         matrices that start at index
   *         index_A[num_terms * num_dimensions * (i * num_columns + j)]
   *         and continue for num_terms * num_dimensions entries
   *
   * \code
   *   int idx = num_dimensions * (i * num_columns + j);
   *   for(int term = 0; term < num_terms; term++)
   *   {
   *     T const *A_d = &( values_A[ index_A[idx++] ] );
   *     ...
   *     T const *A_2 = &( values_A[ index_A[idx++] ] );
   *     T const *A_1 = &( values_A[ index_A[idx++] ] );
   *     T const *A_0 = &( values_A[ index_A[idx++] ] );
   *     ...
   *   }
   * \endcode
   *
   * \par Sparse matrix format
   * There are two formats that allow for better utilization of parallelism
   * when running on the CPU and GPU respectively. The CPU format uses standard
   * row-compressed sparse matrix format, where row_indx is size num_rows + 1
   * and the non-zeros for row i are stored in col_indx between entries
   * row_indx[i] and row_indx[i+1]. The actual tensors are offset at
   * i * tensor-size and col_indx[row_indx[i]] * tensor-size.
   * The GPU format row_indx.size() == col_indx.size() and each Kronecker
   * product uses tensors at row_indx[i] and col_indx[i].
   */
  template<resource input_mode>
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols,
                  int num_terms,
                  fk::vector<int, mem_type::owner, input_mode> const &&row_indx,
                  fk::vector<int, mem_type::owner, input_mode> const &&col_indx,
                  fk::vector<int, mem_type::owner, input_mode> &&index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A)
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(num_terms),
        tensor_size_(1), row_indx_(std::move(row_indx)),
        col_indx_(std::move(col_indx)), iA(std::move(index_A)),
        vA(std::move(values_A))
  {
#ifdef ASGARD_USE_CUDA
    static_assert(
        input_mode == resource::device,
        "the GPU is enabled, so input vectors must have resource::device");
#else
    static_assert(
        input_mode == resource::host,
        "the GPU is disabled, so input vectors must have resource::host");
#endif

    expect((row_indx_.size() == 0 and col_indx_.size() == 0) or
           (row_indx_.size() > 0 and col_indx_.size() > 0));

    tensor_size_ = compute_tensor_size(num_dimensions_, kron_size_);

    flops_ = int64_t(tensor_size_) * kron_size_ * iA.size();

#ifdef ASGARD_USE_CUDA
    if (row_indx_.size() > 0)
    {
      expect(row_indx_.size() == col_indx_.size());
      expect(iA.size() == col_indx_.size() * num_dimensions_ * num_terms_);
    }

    xdev = fk::vector<precision, mem_type::owner, data_mode>(tensor_size_ *
                                                             num_cols_);
    ydev = fk::vector<precision, mem_type::owner, data_mode>(tensor_size_ *
                                                             num_rows_);
#else
    if (row_indx_.size() > 0)
    {
      expect(row_indx_.size() == num_rows_ + 1);
      expect(iA.size() == col_indx_.size() * num_dimensions_ * num_terms_);
    }
#endif
  }

  /*!
   * \brief Creates a new dense matrix by skipping row_indx and col_indx.
   */
  template<resource input_mode>
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols,
                  int num_terms,
                  fk::vector<int, mem_type::owner, input_mode> &&index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A)
      : kronmult_matrix(num_dimensions, kron_size, num_rows, num_cols,
                        num_terms,
                        fk::vector<int, mem_type::owner, input_mode>(),
                        fk::vector<int, mem_type::owner, input_mode>(),
                        std::move(index_A), std::move(values_A))
  {}

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
    if (is_dense())
      kronmult::gpu_dense(num_dimensions_, kron_size_, output_size(),
                          num_batch(), num_cols_, num_terms_, iA.data(),
                          vA.data(), alpha, xdev.data(), beta, ydev.data());
    else
      kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(),
                           col_indx_.size(), col_indx_.data(), row_indx_.data(),
                           num_terms_, iA.data(), vA.data(), alpha, xdev.data(),
                           beta, ydev.data());
    fk::copy_to_host(y, ydev.data(), ydev.size());
#else
    if (is_dense())
      kronmult::cpu_dense(num_dimensions_, kron_size_, num_rows_, num_cols_,
                          num_terms_, iA.data(), vA.data(), alpha, x, beta, y);
    else
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

  //! \brief Helper, computes the size of a tensor for the given parameters.
  static int compute_tensor_size(int const num_dimensions, int const kron_size)
  {
    int tensor_size = kron_size;
    for (int d = 1; d < num_dimensions; d++)
      tensor_size *= kron_size;
    return tensor_size;
  }
  //! \brief Helper, computes the number of flops for each call to apply.
  static int64_t compute_flops(int const num_dimensions, int const kron_size,
                               int const num_terms, int const num_batch)
  {
    return int64_t(compute_tensor_size(num_dimensions, kron_size)) * kron_size *
           num_dimensions * num_terms * num_batch;
  }
  //! \brief Defined if the matrix is dense or sparse
  bool is_dense() const { return (row_indx_.size() == 0); }

  //! \brief Update coefficients
  template<resource input_mode>
  void update_stored_coefficients(
      fk::vector<precision, mem_type::owner, input_mode> &&values_A)
  {
#ifdef ASGARD_USE_CUDA
    static_assert(
        input_mode == resource::device,
        "the GPU is enabled, so input vectors must have resource::device");
#else
    static_assert(
        input_mode == resource::host,
        "the GPU is disabled, so input vectors must have resource::host");
#endif
    expect(num_dimensions_ > 0);
    expect(values_A.size() == vA.size());
    vA = std::move(values_A);
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
 * The main purpose of the method is to "glue" the data-structures together
 * from the definition of the PDE to the format used by the Kronecker product.
 *
 * This method will copy out the coefficient data from the PDE terms
 * into the matrix structure, so the method should be called only
 * when the operator terms change, e.g., due to refinement update.
 *
 * The format of the matrix will be either dense or sparse, depending on
 * the selected program options.
 *
 * \tparam P is either float or double
 *
 * \param pde is the instance of the PDE being simulated
 * \param grid is the current sparse grid for the discretization
 * \param program_options are the input options passed in by the user
 * \param imex indicates whether this is part of an imex time stepping scheme
 */
template<typename P>
kronmult_matrix<P>
make_kronmult_matrix(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                     options const &program_options,
                     imex_flag const imex = imex_flag::unspecified);

/*!
 * \brief Update the coefficients stored in the matrix without changing the rest
 *
 * Used when the coefficients change but the list of indexes, i.e., the rows
 * columns and potential sparsity pattern remains the same.
 *
 * Note that the number of terms and the imex flags must be the same as
 * the ones used in the construction of the matrix.
 * Best use the matrix_list as a helper class.
 */
template<typename P>
void update_kronmult_coefficients(PDE<P> const &pde,
                                  options const &program_options,
                                  imex_flag const imex,
                                  kronmult_matrix<P> &mat);

//! \brief Expressive indexing for the matrices
enum matrix_entry
{
  //! \brief Regular matrix for implicit or explicit time-stepping
  regular = 0,
  //! \brief IMEX explicit matrix
  imex_explicit = 1,
  //! \brief IMEX implicit matrix
  imex_implicit = 2
};

/*!
 * \brief Holds a list of matrices used for time-stepping.
 *
 * There are multiple types of matrices based on the time-stepping and the
 * different terms being used. Matrices are grouped in one object so they can go
 * as a set and reduce the number of matrix making.
 */
template<typename precision>
struct matrix_list
{
  //! \brief Makes a list of uninitialized matrices
  matrix_list() : matrices(3)
  {
    // make sure we have defined flags for all matrices
    expect(matrices.size() == flag_map.size());
  }

  //! \brief Returns an entry indexed by the enum
  kronmult_matrix<precision> &operator[](matrix_entry entry)
  {
    return matrices[static_cast<int>(entry)];
  }

  //! \brief Make the matrix for the given entry
  void make(matrix_entry entry, PDE<precision> const &pde,
            adapt::distributed_grid<precision> const &grid, options const &opts)
  {
    if (not(*this)[entry])
      (*this)[entry] = make_kronmult_matrix(pde, grid, opts, imex(entry));
  }
  /*!
   * \brief Either makes the matrix or if it exists, just updates only the
   *        coefficients
   */
  void reset_coefficients(matrix_entry entry, PDE<precision> const &pde,
                          adapt::distributed_grid<precision> const &grid,
                          options const &opts)
  {
    if (not(*this)[entry])
      (*this)[entry] = make_kronmult_matrix(pde, grid, opts, imex(entry));
    else
      update_kronmult_coefficients(pde, opts, imex(entry), (*this)[entry]);
  }

  //! \brief Clear the specified matrix
  void clear(matrix_entry entry)
  {
    if (matrices[static_cast<int>(entry)])
      matrices[static_cast<int>(entry)] = kronmult_matrix<precision>();
  }
  //! \brief Clear all matrices
  void clear_all()
  {
    for (auto &matrix : matrices)
      if (matrix)
        matrix = kronmult_matrix<precision>();
  }

  //! \brief Holds the matrices
  std::vector<kronmult_matrix<precision>> matrices;

private:
  static imex_flag imex(matrix_entry entry)
  {
    return flag_map[static_cast<int>(entry)];
  }

  static constexpr std::array<imex_flag, 3> flag_map = {
      imex_flag::unspecified, imex_flag::imex_explicit,
      imex_flag::imex_implicit};
};

} // namespace asgard
