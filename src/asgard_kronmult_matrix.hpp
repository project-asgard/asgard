#pragma once
#include "./device/asgard_kronmult.hpp"
#include "adapt.hpp"

// this interface between the low level kernels in src/device
// and the higher level data-structures

namespace asgard
{
/*!
 * \brief Returns a list of terms matching the imex index.
 */
template<typename precision>
std::vector<int> get_used_terms(PDE<precision> const &pde, imex_flag const imex);

/*!
 * \brief Converts the cells into a vector2d structure
 */
template<typename precision>
vector2d<int> get_cells(int num_dimensions,
                        adapt::distributed_grid<precision> const &dis_grid);

#ifndef KRON_MODE_GLOBAL
// using LOCAL kronmult, can be parallelised using MPI but much more expensive
// then the global modes below
// also has the ability to limit memory and do some operations out-of-core
// hence it is currently the most memory conserving mode

/*!
 * \brief Holds data for the pre-computed memory sizes.
 *
 * Also stores some kronmult data that can be easily reused in different
 * calls, specifically in the making of the matrices.
 */
struct memory_usage
{
  //! \brief Constructs uninitialized structure
  memory_usage() : initialized(false) {}
  //! \brief Indicates whether kronmult will be called in one or multiple calls
  enum kron_call_mode
  {
    //! \brief kronmult can be applied in one call
    one_call,
    //! \brief kronmult has to be applied in multiple calls
    multi_calls
  };
  /*!
   * \brief Indicates whether we are limited by the 32-bit index or allocated
   *        memory
   */
  enum size_limit_mode
  {
    //! \brief Limited by the index size, 32-bit
    overflow,
    //! \brief Limited by the user specified memory or device capabilities
    environment
  };
  //! \brief Keeps track if the memory compute has been initialized
  bool initialized;
  //! \brief Persistent size that cannot be any less, in MB.
  int baseline_memory;
  //! \brief Indicate whether one shot or multiple shots will be used.
  kron_call_mode kron_call;
  //! \brief Indicate how we are limited in size
  size_limit_mode mem_limit;
  //! \brief Index workspace size (does not include row/col indexes)
  int64_t work_size;
  //! \brief Index workspace size for the row/col indexes
  int64_t row_work_size;
  //! \brief Indicate whether it has been initialized
  operator bool() const { return initialized; }
  //! \brief Resets the memory parameters due to adapting the grid
  void reset() { initialized = false; }
};

/*!
 * \brief Holds data precomputed for the sparse mode of kronmult
 *
 * Ignored in the dense case.
 */
struct kron_sparse_cache
{
  //! \brief Constructor, makes and empty cache.
  kron_sparse_cache() : cells1d(2) {}

  // the cells1d should be moved to some discretization class
  // but that will be done when the sparse grids data-structs are updated
  //! \brief Contains the connectivity matrix for the 1D rule
  connect_1d cells1d;

  //! \brief Row-compressed style of an array that keeps the active connections
  std::vector<int> cconnect;
  //! \brief Number of non-zeros in the kronmult sparse matrix
  int64_t num_nonz;
};

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
class local_kronmult_matrix
{
public:
  //! \brief Creates uninitialized matrix cannot be used except to be reinitialized.
  local_kronmult_matrix()
      : num_dimensions_(0), kron_size_(0), num_rows_(0), num_cols_(0),
        num_terms_(0), tensor_size_(0), flops_(0), list_row_stride_(0),
        row_offset_(0), col_offset_(0), num_1d_blocks_(0)
  {}

  //! \brief Creates a zero/empty matrix no terms.
  local_kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols)
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(0),
        tensor_size_(fm::ipow(kron_size_, num_dimensions_)),
        flops_(0), list_row_stride_(0), row_offset_(0),
        col_offset_(0), num_1d_blocks_(0)
  {}

  template<resource input_mode>
  local_kronmult_matrix(
      int num_dimensions, int kron_size, int num_rows, int num_cols,
      int num_terms,
      std::vector<fk::vector<precision, mem_type::owner, input_mode>> &&terms,
      fk::vector<int, mem_type::owner, input_mode> &&elem, int row_offset,
      int col_offset, int num_1d_blocks, std::vector<precision> &&prec)
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(num_terms),
        tensor_size_(1), terms_(std::move(terms)), elem_(std::move(elem)),
        row_offset_(row_offset), col_offset_(col_offset),
        num_1d_blocks_(num_1d_blocks), pre_con_(std::move(prec))
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

    expect(terms_.size() == static_cast<size_t>(num_terms_));
    for (int t = 0; t < num_terms; t++)
      expect(terms_[t].size() == num_dimensions_ * num_1d_blocks_ *
                                     num_1d_blocks_ * kron_size_ * kron_size_);

#ifdef ASGARD_USE_CUDA
    fk::vector<precision *> cpu_term_pntr(num_terms_);
    for (int t = 0; t < num_terms; t++)
      cpu_term_pntr[t] = terms_[t].data();
    term_pntr_ = cpu_term_pntr.clone_onto_device();
#else
    term_pntr_ =
        fk::vector<precision *, mem_type::owner, data_mode>(num_terms_);
    for (int t = 0; t < num_terms; t++)
      term_pntr_[t] = terms_[t].data();
#endif

    tensor_size_ = fm::ipow(kron_size_, num_dimensions_);

    flops_ = tensor_size_ * kron_size_ * num_rows_ * num_cols_ *
             num_terms_ * num_dimensions_;
  }

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
   *        degree, kron_size = 1 for constants, 2 for linears, 3 for quadratics
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
   * This constructor will create a matrix in single call mode, if row_indx
   * and col_indx are empty vectors, the matrix will be dense. Otherwise,
   * it will be sparse.
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
  local_kronmult_matrix(
      int num_dimensions, int kron_size, int num_rows, int num_cols,
      int num_terms,
      fk::vector<int, mem_type::owner, input_mode> &&row_indx,
      fk::vector<int, mem_type::owner, input_mode> &&col_indx,
      fk::vector<int, mem_type::owner, input_mode> &&index_A,
      fk::vector<precision, mem_type::owner, input_mode> &&values_A,
      std::vector<precision> &&prec)
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(num_terms),
        tensor_size_(1), row_indx_(std::move(row_indx)),
        col_indx_(std::move(col_indx)), iA(std::move(index_A)),
        list_row_stride_(0), vA(std::move(values_A)), row_offset_(0),
        col_offset_(0), num_1d_blocks_(0), pre_con_(std::move(prec))
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

    expect(not row_indx_.empty() and not col_indx_.empty());

    tensor_size_ = fm::ipow(kron_size_, num_dimensions_);

    flops_ = tensor_size_ * kron_size_ * iA.size();

#ifdef ASGARD_USE_CUDA
    expect(row_indx_.size() == col_indx_.size());
    expect(iA.size() == col_indx_.size() * num_dimensions_ * num_terms_);
#else
    expect(row_indx_.size() == num_rows_ + 1);
    expect(iA.size() == col_indx_.size() * num_dimensions_ * num_terms_);
#endif
  }

  /*!
   * \brief Constructs a sparse matrix that will be processed in multiple calls.
   *
   * \tparam multi_mode must be set to the host if using only the CPU or if CUDA
   *         has out-of-core mode enabled, i.e., with ASGARD_USE_GPU_MEM_LIMIT
   *         set at compile time. Otherwise, the data for all calls will be
   *         loaded on the GPU and multi_mode must be set to device
   * \tparam input_mode the mode of the coefficient matrices is always host
   *         for the CPU and device when CUDA is enabled
   */
  template<resource multi_mode, resource input_mode>
  local_kronmult_matrix(
      int num_dimensions, int kron_size, int num_rows, int num_cols,
      int num_terms,
      std::vector<fk::vector<int, mem_type::owner, multi_mode>> &&row_indx,
      std::vector<fk::vector<int, mem_type::owner, multi_mode>> &&col_indx,
      std::vector<fk::vector<int, mem_type::owner, multi_mode>> &&list_index_A,
      fk::vector<precision, mem_type::owner, input_mode> &&values_A,
      std::vector<precision> &&prec)
      : local_kronmult_matrix(num_dimensions, kron_size, num_rows, num_cols,
                              num_terms, 0, std::move(row_indx), std::move(col_indx),
                              std::move(list_index_A), std::move(values_A),
                              std::move(prec))
  {
    expect(not(list_row_indx_.empty() and list_col_indx_.empty()));
  }

#ifdef ASGARD_USE_CUDA
  //! \brief Set the workspace memory for x and y
  void
  set_workspace(fk::vector<precision, mem_type::owner, resource::device> &x,
                fk::vector<precision, mem_type::owner, resource::device> &y)
  {
    xdev = fk::vector<precision, mem_type::view, resource::device>(x);
    ydev = fk::vector<precision, mem_type::view, resource::device>(y);
  }
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  //! \brief Set the workspace memory for loading the index list
  void set_workspace_ooc(fk::vector<int, mem_type::owner, resource::device> &a,
                         fk::vector<int, mem_type::owner, resource::device> &b,
                         cudaStream_t stream)
  {
    worka       = fk::vector<int, mem_type::view, resource::device>(a);
    workb       = fk::vector<int, mem_type::view, resource::device>(b);
    load_stream = stream;
  }
  //! \brief Set the workspace memory for loading the sparse row/col indexes
  void set_workspace_ooc_sparse(
      fk::vector<int, mem_type::owner, resource::device> &iya,
      fk::vector<int, mem_type::owner, resource::device> &iyb,
      fk::vector<int, mem_type::owner, resource::device> &ixa,
      fk::vector<int, mem_type::owner, resource::device> &ixb)
  {
    irowa = fk::vector<int, mem_type::view, resource::device>(iya);
    irowb = fk::vector<int, mem_type::view, resource::device>(iyb);
    icola = fk::vector<int, mem_type::view, resource::device>(ixa);
    icolb = fk::vector<int, mem_type::view, resource::device>(ixb);
  }
#endif

  /*!
   * \brief Computes y = alpha * kronmult_matrix * x + beta * y
   *
   * This method is not thread-safe!
   *
   * \tparam rec indicates whether the x and y buffers sit on the host or device
   */
  template<resource rec = resource::host>
  void apply(precision alpha, precision const x[], precision beta,
             precision y[]) const
  {
    if (num_terms_ == 0)
    {
      if (beta != 0)
        lib_dispatch::scal<rec>(output_size(), beta, y, 1);
      return;
    }
#ifdef ASGARD_USE_CUDA
    precision const *active_x = (rec == resource::host) ? xdev.data() : x;
    precision *active_y       = (rec == resource::host) ? ydev.data() : y;
    if constexpr (rec == resource::host)
    {
      if (beta != 0)
        fk::copy_to_device(ydev.data(), y, ydev.size());
      fk::copy_to_device(xdev.data(), x, xdev.size());
    }
    if (is_dense())
    {
      kronmult::gpu_dense(num_dimensions_, kron_size_, output_size(),
                          num_batch(), num_cols_, num_terms_, elem_.data(),
                          row_offset_, col_offset_, term_pntr_.data(),
                          num_1d_blocks_, alpha, active_x, beta, active_y);
    }
    else
    {
      if (iA.size() > 0)
      {
        kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(),
                             col_indx_.size(), col_indx_.data(),
                             row_indx_.data(), num_terms_, iA.data(), vA.data(),
                             alpha, active_x, beta, active_y);
      }
      else
      {
#ifdef ASGARD_USE_GPU_MEM_LIMIT
        int *load_buffer         = worka.data();
        int *compute_buffer      = workb.data();
        int *load_buffer_rows    = irowa.data();
        int *compute_buffer_rows = irowb.data();
        int *load_buffer_cols    = icola.data();
        int *compute_buffer_cols = icolb.data();

        auto stats1 = cudaMemcpyAsync(load_buffer, list_iA[0].data(),
                                      sizeof(int) * list_iA[0].size(),
                                      cudaMemcpyHostToDevice, load_stream);
        auto stats2 =
            cudaMemcpyAsync(load_buffer_rows, list_row_indx_[0].data(),
                            sizeof(int) * list_row_indx_[0].size(),
                            cudaMemcpyHostToDevice, load_stream);
        auto stats3 =
            cudaMemcpyAsync(load_buffer_cols, list_col_indx_[0].data(),
                            sizeof(int) * list_col_indx_[0].size(),
                            cudaMemcpyHostToDevice, load_stream);
        expect(stats1 == cudaSuccess);
        expect(stats2 == cudaSuccess);
        expect(stats3 == cudaSuccess);
        for (size_t i = 0; i < list_iA.size(); i++)
        {
          // sync load_stream to ensure that data has already been loaded
          cudaStreamSynchronize(load_stream);
          // ensure the last compute stage is done before swapping the buffers
          if (i > 0) // no need to sync at the very beginning
            cudaStreamSynchronize(nullptr);
          std::swap(load_buffer, compute_buffer);
          std::swap(load_buffer_rows, compute_buffer_rows);
          std::swap(load_buffer_cols, compute_buffer_cols);

          if (i + 1 < list_iA.size())
          {
            // begin loading the next chunk of data
            stats1 = cudaMemcpyAsync(load_buffer, list_iA[i + 1].data(),
                                     sizeof(int) * list_iA[i + 1].size(),
                                     cudaMemcpyHostToDevice, load_stream);
            stats2 =
                cudaMemcpyAsync(load_buffer_rows, list_row_indx_[i + 1].data(),
                                sizeof(int) * list_row_indx_[i + 1].size(),
                                cudaMemcpyHostToDevice, load_stream);
            stats3 =
                cudaMemcpyAsync(load_buffer_cols, list_col_indx_[i + 1].data(),
                                sizeof(int) * list_col_indx_[i + 1].size(),
                                cudaMemcpyHostToDevice, load_stream);
            expect(stats1 == cudaSuccess);
            expect(stats2 == cudaSuccess);
            expect(stats3 == cudaSuccess);
          }

          // num_batch is list_iA[i].size() / (num_dimensions_ * num_terms_)
          // note that the first call to gpu_dense with the given output_size()
          // will apply beta to the output y, thus follow on calls have to only
          // accumulate and beta should be set to 1
          kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(),
                               list_row_indx_[i].size(), compute_buffer_cols,
                               compute_buffer_rows, num_terms_, compute_buffer,
                               vA.data(), alpha, active_x, (i == 0) ? beta : 1,
                               active_y);
        }
#else
        for (size_t i = 0; i < list_iA.size(); i++)
        {
          kronmult::gpu_sparse(
              num_dimensions_, kron_size_, output_size(),
              list_row_indx_[i].size(), list_col_indx_[i].data(),
              list_row_indx_[i].data(), num_terms_, list_iA[i].data(),
              vA.data(), alpha, active_x, (i == 0) ? beta : 1, active_y);
        }
#endif
      }
    }
    if constexpr (rec == resource::host)
      fk::copy_to_host(y, ydev.data(), ydev.size());
#else
    static_assert(rec == resource::host,
                  "CUDA not enabled, only resource::host is allowed for "
                  "the kronmult_matrix::apply() template parameter");

    if (is_dense())
    {
      kronmult::cpu_dense(num_dimensions_, kron_size_, num_rows_, num_cols_,
                          num_terms_, elem_.data(), row_offset_, col_offset_,
                          term_pntr_.data(), num_1d_blocks_, alpha, x, beta, y);
    }
    else
    {
      int64_t row_offset = 0;
      for (size_t i = 0; i < list_row_indx_.size(); i++)
      {
        kronmult::cpu_sparse(num_dimensions_, kron_size_,
                             list_row_indx_[i].size() - 1,
                             list_row_indx_[i].data(), list_col_indx_[i].data(),
                             num_terms_, list_iA[i].data(), vA.data(), alpha, x,
                             beta, y + row_offset * tensor_size_);
        row_offset += list_row_indx_[i].size() - 1;
      }
    }
#endif
  }

  //! \brief Returns the number of kron-products
  int64_t num_batch() const { return int64_t{num_rows_} * int64_t{num_cols_}; }

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

  //! \brief Helper, computes the number of flops for each call to apply.
  static int64_t compute_flops(int const num_dimensions, int const kron_size,
                               int const num_terms, int64_t const num_batch)
  {
    return fm::ipow(kron_size, num_dimensions) * kron_size *
           num_dimensions * num_terms * num_batch * 2;
  }
  //! \brief Defined if the matrix is dense or sparse
  bool is_dense() const
  {
    return (row_indx_.empty() and list_row_indx_.empty());
  }

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
  //! \brief Update coefficients
  template<resource input_mode>
  void update_stored_coefficients(
      std::vector<fk::vector<precision, mem_type::owner, input_mode>>
          &&values_A)
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
    expect(values_A.size() == static_cast<size_t>(num_terms_));
    terms_ = std::move(values_A);

#ifdef ASGARD_USE_CUDA
    fk::vector<precision *> cpu_term_pntr(num_terms_);
    for (int t = 0; t < num_terms_; t++)
      cpu_term_pntr[t] = terms_[t].data();
    term_pntr_ = cpu_term_pntr.clone_onto_device();
#else
    for (int t = 0; t < num_terms_; t++)
      term_pntr_[t] = terms_[t].data();
#endif
  }

  //! \brief Returns the mode of the matrix, one call or multiple calls
  bool is_onecall()
  {
    if (is_dense())
      return true;
#ifdef ASGARD_USE_CUDA
    return (iA.size() > 0);
#else
    return (iA.size() > 0 or list_iA.size() == 1);
#endif
  }

  //! \brief Returns the preconditioner.
  template<resource rec>
  auto const &get_diagonal_preconditioner() const
  {
#ifdef ASGARD_USE_CUDA
    if constexpr (rec == resource::device)
    {
      if (gpu_pre_con_.empty())
        gpu_pre_con_ = pre_con_;
      return gpu_pre_con_;
    }
    else
      return pre_con_;
#else
    static_assert(rec == resource::host, "GPU not enabled");
    return pre_con_;
#endif
  }

private:
  //! \brief Multi-call constructors delegate to this one, handles list_row_stride_
  template<resource multi_mode, resource input_mode>
  local_kronmult_matrix(
      int num_dimensions, int kron_size, int num_rows, int num_cols,
      int num_terms, int list_row_stride,
      std::vector<fk::vector<int, mem_type::owner, multi_mode>> const
          &&row_indx,
      std::vector<fk::vector<int, mem_type::owner, multi_mode>> const
          &&col_indx,
      std::vector<fk::vector<int, mem_type::owner, multi_mode>> &&list_index_A,
      fk::vector<precision, mem_type::owner, input_mode> &&values_A,
      std::vector<precision> &&prec)
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(num_terms),
        tensor_size_(1), list_row_stride_(list_row_stride),
        list_row_indx_(std::move(row_indx)),
        list_col_indx_(std::move(col_indx)), list_iA(std::move(list_index_A)),
        vA(std::move(values_A)), pre_con_(std::move(prec))
  {
#ifdef ASGARD_USE_CUDA
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    static_assert(
        input_mode == resource::device,
        "the GPU is enabled, the coefficient vectors have resource::device");
    static_assert(
        multi_mode == resource::host,
        "the GPU memory usage has been limited, thus we are assuming that the "
        "problem data will not fit in GPU memory and the index vectors must "
        "have resource::host");
#else
    static_assert(input_mode == resource::device and
                      multi_mode == resource::device,
                  "the GPU is enabled, the vectors have resource::device");
#endif
#else
    static_assert(
        input_mode == resource::host and multi_mode == resource::host,
        "the GPU is enabled, the coefficient vectors have resource::host");
#endif

    expect(row_indx_.empty() == col_indx_.empty());

    tensor_size_ = fm::ipow(kron_size_, num_dimensions_);

    flops_ = 0;
    for (auto const &a : list_iA)
      flops_ += static_cast<int64_t>(a.size());
    flops_ *= int64_t{tensor_size_} * kron_size_;
  }

  int num_dimensions_  = 0;
  int kron_size_       = 0; // i.e., n - size of the matrices
  int num_rows_        = 0;
  int num_cols_        = 0;
  int num_terms_       = 0;
  int64_t tensor_size_ = 0;
  int64_t flops_       = 0;

#ifdef ASGARD_USE_CUDA
  // indicates that the input vectors for single-call-mode will be on the GPU
  static constexpr resource data_mode = resource::device;
  // cache vectors for the input and output
  mutable fk::vector<precision, mem_type::view, data_mode> xdev, ydev;
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  // if working out-of-code, multiple vectors will be handled from the host
  static constexpr resource multi_data_mode = resource::host;
  // worka, workb hold the iA indexes
  mutable fk::vector<int, mem_type::view, data_mode> worka, workb;
  // in sparse mode, irow/col contains the ix, iy indexes
  mutable fk::vector<int, mem_type::view, data_mode> irowa, irowb;
  mutable fk::vector<int, mem_type::view, data_mode> icola, icolb;
  // stream to load data while computing kronmult
  cudaStream_t load_stream;
#else
  // if memory is not limited, multiple vectors are all loaded on the GPU
  static constexpr resource multi_data_mode = resource::device;
#endif
#else
  static constexpr resource data_mode = resource::host;
  static constexpr resource multi_data_mode = resource::host;
#endif

  // sparse mode (single call), indexes for the rows and columns
  fk::vector<int, mem_type::owner, data_mode> row_indx_;
  fk::vector<int, mem_type::owner, data_mode> col_indx_;

  // single call, indexes of the kron matrices
  fk::vector<int, mem_type::owner, data_mode> iA;

  // multi call mode, multiple row/col and iA indexes
  int list_row_stride_; // for the dense case, how many rows fall in one list
  std::vector<fk::vector<int, mem_type::owner, multi_data_mode>> list_row_indx_;
  std::vector<fk::vector<int, mem_type::owner, multi_data_mode>> list_col_indx_;
  std::vector<fk::vector<int, mem_type::owner, multi_data_mode>> list_iA;

  // values of the kron matrices (loaded form the coefficients)
  fk::vector<precision, mem_type::owner, data_mode> vA;

  // new dense mode
  std::vector<fk::vector<precision, mem_type::owner, data_mode>> terms_;
  fk::vector<precision *, mem_type::owner, data_mode> term_pntr_;
  fk::vector<int, mem_type::owner, data_mode> elem_;
  int row_offset_    = 0;
  int col_offset_    = 0;
  int num_1d_blocks_ = 0;

  // preconditioner
  std::vector<precision> pre_con_;
#ifdef ASGARD_USE_CUDA
  mutable gpu::vector<precision> gpu_pre_con_; // gpu copy
#endif
};

/*!
 * \brief Given the PDE an the discretization, creates a new kronmult matrix.
 *
 * The main purpose of the method is to "glue" the data-structures together
 * from the definition of the PDE to the format used by the Kronecker product.
 * It also keeps a common workspace that will be used for all kron operations,
 * which means that virtually none of the operations here are thread-safe.
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
 * \param mem_stats is the cached information about memory usage
 * \param imex indicates whether this is part of an imex time stepping scheme
 * \param spcache cache holding sparse analysis meta-data
 * \param force_sparse (testing purposes only) forces a sparse matrix
 * \param verbosity_level indicates the level of output (noise)
 */
template<typename P>
local_kronmult_matrix<P>
make_local_kronmult_matrix(
    PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
    memory_usage const &mem_stats, imex_flag const imex, kron_sparse_cache &spcache,
    verbosity_level verb, bool force_sparse = false);

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
void update_kronmult_coefficients(PDE<P> const &pde, imex_flag const imex,
                                  kron_sparse_cache &spcache,
                                  local_kronmult_matrix<P> &mat);

/*!
 * \brief Compute the stats for the memory usage
 *
 * Computes how to avoid overflow or the use of more memory than
 * is available on the GPU device.
 *
 * \tparam P is float or double
 *
 * \param pde holds the problem data
 * \param grid is the discretization
 * \param program_options is the user provided options
 * \param imex is the flag indicating the IMEX mode
 * \param spcache holds precomputed data about the sparsity pattern which is
 *        used between multiple calls to avoid recomputing identical entries
 * \param memory_limit_MB can override the user specified limit (in MB),
 *        if set to zero the user selection will be used
 * \param index_limit should be set to the max value held by the 32-bit index,
 *        namely 2^31 -2 since 2^31 causes an overflow,
 *        a different value can be used for testing purposes
 */
template<typename P>
memory_usage
compute_mem_usage(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                  imex_flag const imex, kron_sparse_cache &spcache,
                  int memory_limit_MB = 0, int64_t index_limit = 2147483646,
                  bool force_sparse = false);

#endif // end the ifndef KRON_MODE_GLOBAL

#ifdef KRON_MODE_GLOBAL
#ifndef KRON_MODE_GLOBAL_BLOCK
// using GLOBAL kronmult where the explicitly form the sparse matrices
// can be used with the cuSparse library (in CUDA mode)
// will be replaced by the block global mode

// forward declaration that allows set/update methods to be friends
template<typename precision>
class global_kron_matrix;

/*!
 * \brief Sets the values and coefficients for a specific imex flag
 *
 * After the common part of the matrix patterns has been identified,
 * this sets the specific indexes and values for the given imex flag.
 * This MUST be called before an evaluate with the corresponding imex flag
 * can be issued.
 */
template<typename precision>
void set_specific_mode(PDE<precision> const &pde,
                       adapt::distributed_grid<precision> const &dis_grid,
                       imex_flag const imex,
                       global_kron_matrix<precision> &mat);

/*!
 * \brief Update the coefficients for the specific imex flag
 *
 * Once the coefficient data has changed, this will update the corresponding
 * coefficients in the matrix without recomputing the fixed indexing component
 * and while minimizing the number of allocations.
 */
template<typename precision>
void update_matrix_coefficients(PDE<precision> const &pde,
                                adapt::distributed_grid<precision> const &dis_grid,
                                imex_flag const imex,
                                global_kron_matrix<precision> &mat);

/*!
 * \brief Holds the data for a global Kronecker matrix
 */
template<typename precision>
class global_kron_matrix
{
public:
  //! \brief Several workspaces are needed
  enum class workspace
  {
    pad_x = 0, // padded entries for x/y
    pad_y,
    stage1, // global kron workspace (stages)
    stage2,
#ifdef ASGARD_USE_CUDA
    dev_x, // move CPU data to the GPU
    dev_y,
    gsparse, // cusparse workspace
#endif
    num_spaces
  };

#ifdef ASGARD_USE_CUDA
  //! \brief The default vector to use for data, gpu::vector for gpu
  template<typename T>
  using default_vector = gpu::vector<T>;
  //! \brief On the GPU, we split the patterns into triples
  static constexpr int patterns_per_dim = 3;
#else
  //! \brief The default vector to use for data, std::vector for the cpu
  template<typename T>
  using default_vector = std::vector<T>;
  //! \brief On the CPU, use a single pattern per dimension
  static constexpr int patterns_per_dim = 1;
#endif

  //! \brief Workspace buffers held externally to minimize allocations
  using workspace_type = std::array<default_vector<precision>,
                                    static_cast<int>(workspace::num_spaces)>;

  //! \brief Creates an empty matrix.
  global_kron_matrix() : num_dimensions_(0), num_active_(0), num_padded_(0),
                         flops_({0, 0, 0}), verbosity(verbosity_level::quiet) {}
  //! \brief Creates an empty matrix.
  global_kron_matrix(int num_dimensions, int64_t num_active, int64_t num_padded,
                     std::vector<kronmult::permutes> perms,
                     std::vector<std::vector<int>> gpntr,
                     std::vector<std::vector<int>> gindx,
                     std::vector<std::vector<int>> gdiag,
                     std::vector<std::vector<int>> givals,
                     verbosity_level verb_in)
      : num_dimensions_(num_dimensions), num_active_(num_active),
        num_padded_(num_padded), perms_(std::move(perms)), flops_({0, 0, 0}),
        gpntr_(std::move(gpntr)), gindx_(std::move(gindx)),
        gdiag_(std::move(gdiag)), givals_(std::move(givals)),
        gvals_(patterns_per_dim * perms_.size() * num_dimensions_),
        verbosity(verb_in)
  {
    expect(gdiag_.size() == static_cast<size_t>(num_dimensions_));
    if (num_dimensions_ > 1)
    {
      expect(gpntr_.size() == static_cast<size_t>(patterns_per_dim * num_dimensions_));
      expect(gindx_.size() == static_cast<size_t>(patterns_per_dim * num_dimensions_));
      expect(givals_.size() == static_cast<size_t>(patterns_per_dim * num_dimensions_));
    }
    else
    { // there is no split in a 1d context
      expect(gpntr_.size() == 1);
      expect(gindx_.size() == 1);
      expect(givals_.size() == 1);
    }

    expect(gpntr_.front().size() == static_cast<size_t>(num_padded + 1));
  }

#ifdef ASGARD_USE_CUDA
  //! \brief Set the GPU side of the global kron, needed for the handle
  void preset_gpu_gkron(gpu::sparse_handle const &hndl, imex_flag const imex);
#endif

  //! \brief Returns \b true if the matrix is empty, \b false otherwise.
  bool empty() const { return gvals_.empty(); }

  //! \brief Apply the operator, including expanding and remapping.
  template<resource rec = resource::host>
  void apply(imex_flag etype, precision alpha, precision const *x,
             precision beta, precision *y) const;

  //! \brief The matrix evaluates to true if it has been initialized and false otherwise.
  operator bool() const { return (not gvals_.empty()); }

  //! \brief Returns the preconditioner.
  template<resource rec>
  auto const &get_diagonal_preconditioner() const
  {
#ifdef ASGARD_USE_CUDA
    if constexpr (rec == resource::device)
    {
      if (gpu_pre_con_.empty())
        gpu_pre_con_ = pre_con_;
      return gpu_pre_con_;
    }
    else
      return pre_con_;
#else
    static_assert(rec == resource::host, "GPU not enabled");
    return pre_con_;
#endif
  }

  //! \brief Return the number of flops for the current matrix type
  int64_t flops(imex_flag etype) const
  {
    return flops_[static_cast<int>(etype)];
  }
  //! \brief Check if the corresponding lock pattern is set.
  bool local_unset(imex_flag etype) const
  {
    return (flops(etype) == 0);
  }
  //! \brief Set the four buffers
  void set_workspace_buffers(workspace_type *work)
  {
    work_ = work; // save a link to the pointers

    resize_buffer<workspace::pad_x>(num_padded_);
    resize_buffer<workspace::pad_y>(num_padded_);
    resize_buffer<workspace::stage1>(num_padded_);
    resize_buffer<workspace::stage2>(num_padded_);

#ifdef ASGARD_USE_CUDA
    resize_buffer<workspace::dev_x>(num_active_);
    resize_buffer<workspace::dev_y>(num_active_);
#endif
  }

  // made friends for two reasons
  // 1. Keeps the matrix API free from references to pde, which will allow an easier
  //    transition to a new API that does not require the PDE class
  // 2. Give the ability to modify the internal without encumbering the matrix API
  friend void set_specific_mode<precision>(
      PDE<precision> const &pde,
      adapt::distributed_grid<precision> const &dis_grid,
      imex_flag const imex,
      global_kron_matrix<precision> &mat);

  friend void update_matrix_coefficients<precision>(
      PDE<precision> const &pde,
      adapt::distributed_grid<precision> const &dis_grid,
      imex_flag const imex,
      global_kron_matrix<precision> &mat);

protected:
  // the workspace is kept externally to minimize allocations
  mutable workspace_type *work_ = nullptr;

  //! \brief Compile-time method to get a specific workspace
  template<workspace wid>
  precision *get_buffer() const
  {
    static_assert(wid != workspace::num_spaces,
                  "no buffer associated with the last entry (num_spaces)");
    return (*work_)[static_cast<int>(wid)].data();
  }
  //! \brief Method to resize the buffer, if the size is insufficient
  template<workspace wid>
  void resize_buffer(int64_t min_size)
  {
    static_assert(wid != workspace::num_spaces,
                  "no buffer associated with the last entry (num_spaces)");
    default_vector<precision> &w = (*work_)[static_cast<int>(wid)];
    if (static_cast<int64_t>(w.size()) < min_size)
      w.resize(min_size);
    // x must always be padded by zeros
    if constexpr (wid == workspace::pad_x)
      kronmult::set_buffer_to_zero(w);
  }

private:
  // description of the multi-indexes and the sparsity pattern
  // global case data
  int num_dimensions_ = 0;
  int64_t num_active_ = 0;
  int64_t num_padded_ = 0;
  std::vector<kronmult::permutes> perms_;
  std::array<int64_t, num_imex_variants> flops_;
  // data for the 1D tensors
  std::vector<std::vector<int>> gpntr_;
  std::vector<std::vector<int>> gindx_;
  std::vector<std::vector<int>> gdiag_;
  std::vector<std::vector<int>> givals_;
  std::vector<std::vector<precision>> gvals_;
  // collections of terms
  std::array<std::vector<int>, 3> term_groups;
  int degree_ = 0;
  // preconditioner
  std::vector<precision> pre_con_;
#ifdef ASGARD_USE_CUDA
  std::array<kronmult::global_gpu_operations<precision>, num_imex_variants> gpu_global;
  mutable gpu::vector<precision> gpu_pre_con_; // gpu copy
#endif
  verbosity_level verbosity = verbosity_level::high;
};

/*!
 * \brief Factory method for making a global kron matrix
 *
 * This sets up the common components of the matrix,
 */
template<typename precision>
global_kron_matrix<precision>
make_global_kron_matrix(PDE<precision> const &pde,
                        adapt::distributed_grid<precision> const &dis_grid,
                        verbosity_level verb);

/*!
 * \brief Holds a list of matrices used for time-stepping.
 *
 * There are multiple types of matrices based on the time-stepping and the
 * different terms being used. Matrices are grouped in one object so they can go
 * as a set and reduce the number of matrix making.
 */
template<typename precision>
struct kron_operators
{
  kron_operators(verbosity_level verb_in = verbosity_level::high)
      : verbosity(verb_in)
  {}

  template<resource rec = resource::host>
  void apply(imex_flag entry, precision alpha, precision const x[], precision beta, precision y[]) const
  {
    apply<rec>(entry, 0, alpha, x, beta, y);
  }

  //! \brief Apply the given matrix entry
  template<resource rec = resource::host>
  void apply(imex_flag entry, precision, precision alpha, precision const x[], precision beta, precision y[]) const
  {
    kglobal.template apply<rec>(entry, alpha, x, beta, y);
  }
  int64_t flops(imex_flag entry) const
  {
    return kglobal.flops(entry);
  }

  //! \brief Make the matrix for the given entry
  void make(imex_flag entry, PDE<precision> const &pde,
            adapt::distributed_grid<precision> const &grid)
  {
    if (not kglobal)
    {
      kglobal = make_global_kron_matrix(pde, grid, verbosity);
      // the buffers must be set before preset_gpu_gkron()
      kglobal.set_workspace_buffers(&workspaces);
    }

    if (kglobal.local_unset(entry))
    {
      set_specific_mode(pde, grid, entry, kglobal);
#ifdef ASGARD_USE_CUDA
      kglobal.preset_gpu_gkron(sp_handle, entry);
#endif
    }
  }
  /*!
   * \brief Either makes the matrix or if it exists, just updates only the
   *        coefficients
   */
  void reset_coefficients(imex_flag entry, PDE<precision> const &pde,
                          adapt::distributed_grid<precision> const &grid)
  {
    if (not kglobal)
      make(entry, pde, grid);
    else
    {
      if (kglobal.local_unset(entry))
      {
        set_specific_mode(pde, grid, entry, kglobal);
#ifdef ASGARD_USE_CUDA
        kglobal.preset_gpu_gkron(sp_handle, entry);
#endif
      }
      else
        update_matrix_coefficients(pde, grid, entry, kglobal);
    }
  }

  //! \brief Clear all matrices
  void clear()
  {
    if (kglobal)
      kglobal = global_kron_matrix<precision>();
  }

  //! \brief Returns the preconditioner.
  template<resource rec>
  auto const &get_diagonal_preconditioner() const
  {
    return kglobal.template get_diagonal_preconditioner<rec>();
  }

  //! controls the verbosity level
  verbosity_level verbosity = verbosity_level::high;

private:
  //! \brief Holds the global part of the kron product
  global_kron_matrix<precision> kglobal;

  typename global_kron_matrix<precision>::workspace_type workspaces;
#ifdef ASGARD_USE_CUDA
  gpu::sparse_handle sp_handle;
  gpu::vector<std::byte> gpu_sparse_buffer;
#endif
};
#endif // end ifndef KRON_MODE_GLOBAL_BLOCK
#endif // end ifdef KRON_MODE_GLOBAL

#ifdef KRON_MODE_GLOBAL_BLOCK
// using BLOCK-GLOBAL kronmult, the fastest method on the cpu and very memory conservative

template<typename precision>
class block_global_kron_matrix;

template<typename precision>
void set_specific_mode(
    PDE<precision> const &pde,
    adapt::distributed_grid<precision> const &dis_grid,
    imex_flag const imex,
    block_global_kron_matrix<precision> &mat);

template<typename precision>
class block_global_kron_matrix
{
public:
  block_global_kron_matrix() {}

  block_global_kron_matrix(int64_t num_active, int64_t num_padded,
                           int num_dimensions, int blockn, int64_t block_size,
                           vector2d<int> &&ilist, dimension_sort &&dsort,
                           std::vector<kronmult::permutes> &&perms, std::vector<int> &&flux_dir,
                           connect_1d const *conn_volumes, connect_1d const *conn_full,
                           kronmult::block_global_workspace<precision> *workspace,
                           verbosity_level verb_in)
      : num_active_(num_active), num_padded_(num_padded),
        num_dimensions_(num_dimensions), blockn_(blockn), block_size_(block_size),
        ilist_(std::move(ilist)), dsort_(std::move(dsort)), perms_(std::move(perms)),
        flux_dir_(std::move(flux_dir)), conn_volumes_(conn_volumes),
        conn_full_(conn_full), gvals_(flux_dir_.size() * num_dimensions_),
        workspace_(workspace), verb(verb_in)
  {
    for (auto &f : flops_)
      f = -1;
  }

  /*!
   * \brief Does the matrix vector product and appends into y
   *
   * The problme must be staged before this call, this is done by the
   * kron_operators apply method, in order to compbine the staging for
   * the interpolation and separable linear application (this method).
   *
   * The assumption are:
   *  - y has been scaled by beta
   *  - x is padded and stored in workspace->x
   *
   * This only computes y += alpha * A * workspace->x,
   * where A corresponds to the imex etype flag.
   */
  template<resource rec>
  void apply(imex_flag etype, precision alpha, precision *y) const;

  operator bool() const { return (num_dimensions_ > 0); }

  bool specific_is_set(imex_flag etype)
  {
    std::vector<int> const &terms = term_groups_[static_cast<int>(etype)];
    if (terms.empty())
      return true; // nothing to set, so we're OK

    for (int d = 0; d < num_dimensions_; d++)
      if (not gvals_[terms.front() * num_dimensions_ + d].empty())
        return true;

    return false;
  }

  //! \brief Allows overwriting of the loaded coefficients.
  template<resource rec>
  auto const &get_diagonal_preconditioner() const
  {
    static_assert(rec == resource::host, "GPU not enabled");
    return pre_con_;
  }

  //! \brief Return the number of flops for the current matrix type
  int64_t flops(imex_flag etype) const
  {
    int i = static_cast<int>(etype);
    if (flops_[i] == -1)
    {
      flops_[i] = kronmult::block_global_count_flops(
          num_dimensions_, block_size_, ilist_, dsort_, perms_,
          flux_dir_, *conn_volumes_, *conn_full_, term_groups_[i], *workspace_);
      if (verb == verbosity_level::high)
      {
        switch (etype)
        {
        case imex_flag::unspecified:
          std::cout << "regular block-global kronmult matrix\n";
          break;
        case imex_flag::imex_explicit:
          std::cout << "imex-explicit block-global kronmult matrix\n";
          break;
        case imex_flag::imex_implicit:
          std::cout << "imex-implicit block-global kronmult matrix\n";
          break;
        };
        std::cout << "   -- number of flops: " << flops_[i] * 1.E-9 << "Gflops\n";
      }
    }
    return flops_[i];
  }

  vector2d<int> const &get_cells() const { return ilist_; };
  dimension_sort const &get_dsort() const { return dsort_; };
  int64_t num_active() const { return num_active_; }

  bool is_active(imex_flag etype) const
  {
    return not term_groups_[static_cast<int>(etype)].empty();
  }

  // made friends for two reasons
  // 1. Keeps the matrix API free from references to pde, which will allow an easier
  //    transition to a new API that does not require the PDE class
  // 2. Give the ability to modify the internal without encumbering the matrix API
  friend void set_specific_mode<precision>(
      PDE<precision> const &pde,
      adapt::distributed_grid<precision> const &dis_grid,
      imex_flag const imex,
      block_global_kron_matrix<precision> &mat);

private:
  int64_t num_active_ = 0;
  int64_t num_padded_ = 0;
  int num_dimensions_ = 0;
  int blockn_         = 0;
  int64_t block_size_ = 0;
  vector2d<int> ilist_;
  dimension_sort dsort_;
  std::vector<kronmult::permutes> perms_;
  std::vector<int> flux_dir_;
  connect_1d const *conn_volumes_ = nullptr;
  connect_1d const *conn_full_    = nullptr;

  std::vector<std::vector<precision>> gvals_;
  std::array<std::vector<int>, 3> term_groups_;

  mutable kronmult::block_global_workspace<precision> *workspace_ = nullptr;

  mutable std::array<int64_t, num_imex_variants> flops_;

  // preconditioner
  std::vector<precision> pre_con_;

  verbosity_level verb = verbosity_level::quiet;
};

template<typename precision>
block_global_kron_matrix<precision>
make_block_global_kron_matrix(PDE<precision> const &pde,
                              adapt::distributed_grid<precision> const &dis_grid,
                              connect_1d const *volumes, connect_1d const *fluxes,
                              kronmult::block_global_workspace<precision> *workspace,
                              verbosity_level verb);

#endif

} // namespace asgard
