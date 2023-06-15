
#pragma once

#include "adapt.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "asgard_grid_1d.hpp"

#include "./device/asgard_kronmult.hpp"

// this interface between the low level kernels in src/device
// and the higher level data-structures

namespace asgard
{
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
  operator bool() const
  {
    return initialized;
  }
  //! \brief Resets the memory parameters due to adapting the grid
  void reset()
  {
    initialized = false;
  }
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
class kronmult_matrix
{
public:
  //! \brief Creates uninitialized matrix cannot be used except to be reinitialized.
  kronmult_matrix()
      : num_dimensions_(0), kron_size_(0), num_rows_(0), num_cols_(0),
        num_terms_(0), tensor_size_(0), flops_(0), list_row_stride_(0)
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
        list_row_stride_(0), vA(std::move(values_A))
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
#else
    if (row_indx_.size() > 0)
    {
      expect(row_indx_.size() == num_rows_ + 1);
      expect(iA.size() == col_indx_.size() * num_dimensions_ * num_terms_);
    }
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
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols,
                  int num_terms,
                  std::vector<fk::vector<int, mem_type::owner, multi_mode>> const &&row_indx,
                  std::vector<fk::vector<int, mem_type::owner, multi_mode>> const &&col_indx,
                  std::vector<fk::vector<int, mem_type::owner, multi_mode>> &&list_index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A)
      : kronmult_matrix(num_dimensions, kron_size, num_rows, num_cols,
                        num_terms, 0,
                        std::move(row_indx), std::move(col_indx),
                        std::move(list_index_A), std::move(values_A))
  {
    expect(list_row_indx_.size() > 0 and list_col_indx_.size() > 0);
  }

  /*!
   * \brief Creates a dense matrix in single call mode, skips row/col indexes.
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

  //! \brief Dense matrix in multi-call mode
  template<resource multi_mode, resource input_mode>
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols,
                  int num_terms, int list_row_stride,
                  std::vector<fk::vector<int, mem_type::owner, multi_mode>> &&list_index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A)
      : kronmult_matrix(num_dimensions, kron_size, num_rows, num_cols,
                        num_terms, list_row_stride,
                        std::vector<fk::vector<int, mem_type::owner, multi_data_mode>>(),
                        std::vector<fk::vector<int, mem_type::owner, multi_data_mode>>(),
                        std::move(list_index_A), std::move(values_A))
  {}

#ifdef ASGARD_USE_CUDA
  //! \brief Set the workspace memory for x and y
  void set_workspace(
            fk::vector<precision, mem_type::owner, resource::device> &x,
            fk::vector<precision, mem_type::owner, resource::device> &y) {
    xdev = fk::vector<precision, mem_type::view, resource::device>(x);
    ydev = fk::vector<precision, mem_type::view, resource::device>(y);
  }
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  //! \brief Set the workspace memory for loading the index list
  void set_workspace_ooc(
            fk::vector<int, mem_type::owner, resource::device> &a,
            fk::vector<int, mem_type::owner, resource::device> &b,
            cudaStream_t stream) {
    worka = fk::vector<int, mem_type::view, resource::device>(a);
    workb = fk::vector<int, mem_type::view, resource::device>(b);
    load_stream = stream;
  }
  //! \brief Set the workspace memory for loading the sparse row/col indexes
  void set_workspace_ooc_sparse(
            fk::vector<int, mem_type::owner, resource::device> &iya,
            fk::vector<int, mem_type::owner, resource::device> &iyb,
            fk::vector<int, mem_type::owner, resource::device> &ixa,
            fk::vector<int, mem_type::owner, resource::device> &ixb
            ) {
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
   */
  void apply(precision alpha, precision const x[], precision beta,
             precision y[]) const
  {
#ifdef ASGARD_USE_CUDA
    if (beta != 0)
      fk::copy_to_device(ydev.data(), y, ydev.size());
    fk::copy_to_device(xdev.data(), x, xdev.size());
    if (is_dense())
    {
      if (iA.size() > 0)
      {
        // single call to kronmult, all data is on the GPU
        kronmult::gpu_dense(num_dimensions_, kron_size_, output_size(),
                            num_batch(), num_cols_, num_terms_, iA.data(),
                            vA.data(), alpha, xdev.data(), beta, ydev.data());
      }
      else
      {
        std::cout << " multi-kron dense " << list_iA.size() << "\n";
#ifdef ASGARD_USE_GPU_MEM_LIMIT
        // multiple calls, need to move data, call kronmult, then move next data
        // data loading is done asynchronously using the load_stream
        int *load_buffer    = worka.data();
        int *compute_buffer = workb.data();
        auto stats = cudaMemcpyAsync(load_buffer, list_iA[0].data(), sizeof(int) * list_iA[0].size(), cudaMemcpyHostToDevice, load_stream);
        assert(stats == cudaSuccess);
        for(size_t i = 0; i < list_iA.size(); i++)
        {
          // sync load_stream to ensure that data has already been loaded
          cudaStreamSynchronize(load_stream);
          // ensure the last compute stage is done before swapping the buffers
          if (i > 0) // no need to sync at the very beginning
            cudaStreamSynchronize(nullptr);
          std::swap(load_buffer, compute_buffer);

          if (i+1 < list_iA.size())
          {
            // begin loading the next chunk of data
            stats = cudaMemcpyAsync(load_buffer, list_iA[i+1].data(), sizeof(int) * list_iA[i+1].size(), cudaMemcpyHostToDevice, load_stream);
            assert(stats == cudaSuccess);
          }

          // num_batch is list_iA[i].size() / (num_dimensions_ * num_terms_)
          // note that the first call to gpu_dense with the given output_size()
          // will apply beta to the output y, thus follow on calls have to only
          // accumulate and beta should be set to 1
          kronmult::gpu_dense(num_dimensions_, kron_size_, output_size(),
                              list_iA[i].size() / (num_dimensions_ * num_terms_), num_cols_, num_terms_, compute_buffer,
                              vA.data(), alpha, xdev.data(), (i == 0) ? beta : 1, ydev.data() + i * list_row_stride_ * tensor_size_);
        }
#else
        for(size_t i = 0; i < list_iA.size(); i++)
        {
          std::cerr << " multi-gpu " << i << "\n";
          kronmult::gpu_dense(num_dimensions_, kron_size_, output_size(), list_iA[i].size() / (num_dimensions_ * num_terms_), num_cols_,
                              num_terms_, list_iA[i].data(), vA.data(), alpha, xdev.data(), (i == 0) ? beta : 1, ydev.data() + i * list_row_stride_ * tensor_size_);
        }
#endif
      }
    }
    else
    {
      if (iA.size() > 0)
      {
        kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(),
                             col_indx_.size(), col_indx_.data(), row_indx_.data(),
                             num_terms_, iA.data(), vA.data(), alpha, xdev.data(),
                             beta, ydev.data());
      }
      else
      {
        std::cout << " multi-kron sparse" << list_iA.size() << "\n";
#ifdef ASGARD_USE_GPU_MEM_LIMIT
        int *load_buffer    = worka.data();
        int *compute_buffer = workb.data();
        int *load_buffer_rows    = irowa.data();
        int *compute_buffer_rows = irowb.data();
        int *load_buffer_cols    = icola.data();
        int *compute_buffer_cols = icolb.data();
        auto stats1 = cudaMemcpyAsync(load_buffer, list_iA[0].data(), sizeof(int) * list_iA[0].size(), cudaMemcpyHostToDevice, load_stream);
        auto stats2 = cudaMemcpyAsync(load_buffer_rows, list_row_indx_[0].data(), sizeof(int) * list_row_indx_[0].size(), cudaMemcpyHostToDevice, load_stream);
        auto stats3 = cudaMemcpyAsync(load_buffer_cols, list_col_indx_[0].data(), sizeof(int) * list_col_indx_[0].size(), cudaMemcpyHostToDevice, load_stream);
        assert(stats1 == cudaSuccess);
        assert(stats2 == cudaSuccess);
        assert(stats3 == cudaSuccess);
        for(size_t i = 0; i < list_iA.size(); i++)
        {
          // sync load_stream to ensure that data has already been loaded
          cudaStreamSynchronize(load_stream);
          // ensure the last compute stage is done before swapping the buffers
          if (i > 0) // no need to sync at the very beginning
            cudaStreamSynchronize(nullptr);
          std::swap(load_buffer, compute_buffer);
          std::swap(load_buffer_rows, compute_buffer_rows);
          std::swap(load_buffer_cols, compute_buffer_cols);

          if (i+1 < list_iA.size())
          {
            // begin loading the next chunk of data
            stats1 = cudaMemcpyAsync(load_buffer, list_iA[i+1].data(), sizeof(int) * list_iA[i+1].size(), cudaMemcpyHostToDevice, load_stream);
            stats2 = cudaMemcpyAsync(load_buffer_rows, list_row_indx_[i+1].data(), sizeof(int) * list_row_indx_[i+1].size(), cudaMemcpyHostToDevice, load_stream);
            stats3 = cudaMemcpyAsync(load_buffer_cols, list_col_indx_[i+1].data(), sizeof(int) * list_col_indx_[i+1].size(), cudaMemcpyHostToDevice, load_stream);
            assert(stats1 == cudaSuccess);
            assert(stats2 == cudaSuccess);
            assert(stats3 == cudaSuccess);
          }

          // num_batch is list_iA[i].size() / (num_dimensions_ * num_terms_)
          // note that the first call to gpu_dense with the given output_size()
          // will apply beta to the output y, thus follow on calls have to only
          // accumulate and beta should be set to 1
          kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(),
                               list_row_indx_[i].size(), compute_buffer_cols, compute_buffer_rows, num_terms_, compute_buffer,
                               vA.data(), alpha, xdev.data(), (i == 0) ? beta : 1, ydev.data());
        }
#else
        for(size_t i = 0; i < list_iA.size(); i++)
        {
          kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(), list_row_indx_[i].size(), list_col_indx_[i].data(), list_row_indx_[i].data(),
                              num_terms_, list_iA[i].data(), vA.data(), alpha, xdev.data(), (i == 0) ? beta : 1, ydev.data());
        }
#endif
      }
    }
    fk::copy_to_host(y, ydev.data(), ydev.size());
#else
    if (is_dense())
    {
      if (iA.size() > 0)
      {
        kronmult::cpu_dense(num_dimensions_, kron_size_, num_rows_, num_cols_,
                            num_terms_, iA.data(), vA.data(), alpha, x, beta, y);
      }
      else
      {
        for(size_t i = 0; i < list_iA.size(); i++)
        {
          kronmult::cpu_dense(num_dimensions_, kron_size_, list_iA[i].size() / (num_dimensions_ * num_terms_ * num_cols_), num_cols_,
                              num_terms_, list_iA[i].data(), vA.data(), alpha, x, beta, y + i * list_row_stride_ * tensor_size_);
        }
      }
    }
    else
    {
      int64_t row_offset = 0;
      for(size_t i = 0; i < list_row_indx_.size(); i++)
      {
        kronmult::cpu_sparse(num_dimensions_, kron_size_, list_row_indx_[i].size() - 1,
                             list_row_indx_[i].data(), list_col_indx_[i].data(),
                             num_terms_, list_iA[i].data(), vA.data(), alpha, x,
                             beta, y + row_offset * tensor_size_);
        row_offset += list_row_indx_[i].size() - 1;
      }
    }
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
  bool is_dense() const { return (row_indx_.size() == 0 and
                                  list_row_indx_.size() == 0); }

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

  //! \brief Returns the mode of the matrix, one call or multiple calls
  bool is_onecall()
  {
    return (iA.size() > 0);
  }

private:
  //! \brief Multi-call constructors delegate to this one, handles list_row_stride_
  template<resource multi_mode, resource input_mode>
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols,
                  int num_terms, int list_row_stride,
                  std::vector<fk::vector<int, mem_type::owner, multi_mode>> const &&row_indx,
                  std::vector<fk::vector<int, mem_type::owner, multi_mode>> const &&col_indx,
                  std::vector<fk::vector<int, mem_type::owner, multi_mode>> &&list_index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A)
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(num_terms),
        tensor_size_(1), list_row_stride_(list_row_stride),
        list_row_indx_(std::move(row_indx)),
        list_col_indx_(std::move(col_indx)), list_iA(std::move(list_index_A)),
        vA(std::move(values_A))
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
    static_assert(
        input_mode == resource::device and multi_mode == resource::device,
        "the GPU is enabled, the vectors have resource::device");
#endif
#else
    static_assert(
        input_mode == resource::host and multi_mode == resource::host,
        "the GPU is enabled, the coefficient vectors have resource::host");
#endif

    expect((row_indx_.size() == 0 and col_indx_.size() == 0) or
           (row_indx_.size() > 0 and col_indx_.size() > 0));

    tensor_size_ = compute_tensor_size(num_dimensions_, kron_size_);

    flops_ = 0;
    for(auto const &a : list_iA)
      flops_ += static_cast<int64_t>(a.size());
    flops_ *= int64_t(tensor_size_) * kron_size_;
  }

  int num_dimensions_;
  int kron_size_; // i.e., n - size of the matrices
  int num_rows_;
  int num_cols_;
  int num_terms_;
  int tensor_size_;
  int64_t flops_;

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
  static constexpr resource data_mode       = resource::host;
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
 */
template<typename P>
kronmult_matrix<P>
make_kronmult_matrix(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                     options const &program_options,
                     memory_usage const &mem_stats,
                     imex_flag const imex, kron_sparse_cache &spcache);

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
                                  kron_sparse_cache &spcache,
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
                  options const &program_options, imex_flag const imex,
                  kron_sparse_cache &spcache, int memory_limit_MB = 0,
                  int64_t index_limit = 2147483646);

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
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    load_stream = nullptr;
#endif
  }

  ~matrix_list()
  {
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    if (load_stream != nullptr)
    {
      auto status = cudaStreamDestroy(load_stream);
      assert(status == cudaSuccess);
    }
#endif
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
    if (not mem_stats)
      mem_stats = compute_mem_usage(pde, grid, opts, imex(entry), spcache);

    if (not(*this)[entry])
      (*this)[entry] = make_kronmult_matrix(pde, grid, opts, mem_stats, imex(entry), spcache);
#ifdef ASGARD_USE_CUDA
    if ((*this)[entry].input_size() != xdev.size()) {
        xdev = fk::vector<precision, mem_type::owner, resource::device>();
        xdev = fk::vector<precision, mem_type::owner, resource::device>((*this)[entry].input_size());
    }
    if ((*this)[entry].output_size() != ydev.size()) {
        ydev = fk::vector<precision, mem_type::owner, resource::device>();
        ydev = fk::vector<precision, mem_type::owner, resource::device>((*this)[entry].output_size());
    }
    (*this)[entry].set_workspace(xdev, ydev);
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    if (mem_stats.kron_call == memory_usage::multi_calls)
    {
      // doing multiple calls, prepare streams and workspaces
      if (load_stream == nullptr)
        cudaStreamCreate(&load_stream);
      if (worka.size() < static_cast<int>(mem_stats.work_size))
      {
        worka = fk::vector<int, mem_type::owner, resource::device>();
        workb = fk::vector<int, mem_type::owner, resource::device>();
        worka = fk::vector<int, mem_type::owner, resource::device>(mem_stats.work_size);
        workb = fk::vector<int, mem_type::owner, resource::device>(mem_stats.work_size);
        if (not (*this)[entry].is_dense())
        {
          irowa = fk::vector<int, mem_type::owner, resource::device>();
          irowb = fk::vector<int, mem_type::owner, resource::device>();
          icola = fk::vector<int, mem_type::owner, resource::device>();
          icolb = fk::vector<int, mem_type::owner, resource::device>();
          irowa = fk::vector<int, mem_type::owner, resource::device>(mem_stats.row_work_size);
          irowb = fk::vector<int, mem_type::owner, resource::device>(mem_stats.row_work_size);
          icola = fk::vector<int, mem_type::owner, resource::device>(mem_stats.row_work_size);
          icolb = fk::vector<int, mem_type::owner, resource::device>(mem_stats.row_work_size);
        }
      }
    }
    (*this)[entry].set_workspace_ooc(worka, workb, load_stream);
    (*this)[entry].set_workspace_ooc_sparse(irowa, irowb, icola, icolb);
#endif
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
      make(entry, pde, grid, opts);
    else
      update_kronmult_coefficients(pde, opts, imex(entry), spcache, (*this)[entry]);
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
    mem_stats.reset();
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

  memory_usage mem_stats;

  kron_sparse_cache spcache;

#ifdef ASGARD_USE_CUDA
  mutable fk::vector<precision, mem_type::owner, resource::device> xdev, ydev;
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  mutable fk::vector<int, mem_type::owner, resource::device> worka;
  mutable fk::vector<int, mem_type::owner, resource::device> workb;
  mutable fk::vector<int, mem_type::owner, resource::device> irowa;
  mutable fk::vector<int, mem_type::owner, resource::device> irowb;
  mutable fk::vector<int, mem_type::owner, resource::device> icola;
  mutable fk::vector<int, mem_type::owner, resource::device> icolb;
  cudaStream_t load_stream;
#endif

};

} // namespace asgard
