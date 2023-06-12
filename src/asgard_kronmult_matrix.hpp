
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
        col_indx_(std::move(col_indx)), list_row_stride_(0),
        iA(std::move(index_A)), vA(std::move(values_A))
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

  //! \brief Split computing mode, e.g., out-of-core mode
  template<resource input_mode>
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols,
                  int num_terms,
                  std::vector<fk::vector<int>> const &&row_indx,
                  std::vector<fk::vector<int>> const &&col_indx,
                  std::vector<fk::vector<int>> &&list_index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A)
      : num_dimensions_(num_dimensions), kron_size_(kron_size),
        num_rows_(num_rows), num_cols_(num_cols), num_terms_(num_terms),
        tensor_size_(1), list_row_stride_(0), list_iA(std::move(list_index_A)),
        list_row_indx_(std::move(row_indx)), list_col_indx_(std::move(col_indx)),
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

    flops_ = 0;
    for(auto const &a : list_iA)
      flops_ += static_cast<int64_t>(a.size());
    flops_ *= int64_t(tensor_size_) * kron_size_;
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

  //! \brief Split computing (e.g., out-of-core) dense case.
  template<resource input_mode>
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_cols,
                  int num_terms, int list_row_stride,
                  std::vector<fk::vector<int>> &&list_index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A)
      : kronmult_matrix(num_dimensions, kron_size, num_rows, num_cols,
                        num_terms,
                        std::vector<fk::vector<int>>(),
                        std::vector<fk::vector<int>>(),
                        std::move(list_index_A), std::move(values_A))
  {
    list_row_stride_ = list_row_stride;
  }
#ifdef ASGARD_USE_CUDA
  //! \brief Set the workspace memory for x and y
  void set_workspace(
            fk::vector<precision, mem_type::owner, resource::device> &x,
            fk::vector<precision, mem_type::owner, resource::device> &y) {
    xdev = fk::vector<precision, mem_type::view, resource::device>(x);
    ydev = fk::vector<precision, mem_type::view, resource::device>(y);
  }
  //! \brief Set the workspace memory for loading the index list
  void set_workspace_ooc(
            fk::vector<int, mem_type::owner, resource::device> &a,
            fk::vector<int, mem_type::owner, resource::device> &b,
            cudaStream_t stream) {
    worka = fk::vector<int, mem_type::view, resource::device>(a);
    workb = fk::vector<int, mem_type::view, resource::device>(b);
    load_stream = stream;
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
        // multiple calls, need to move data, call kronmult, then move next data
        // data loading is done asynchronously using the load_stream
        std::cout << " multi-mode with " << list_iA.size() << " chunks\n";
        int *load_buffer    = worka.data();
        int *compute_buffer = workb.data();
        auto stats = cudaMemcpyAsync(load_buffer, list_iA[0].data(), sizeof(int) * list_iA[0].size(), cudaMemcpyHostToDevice, load_stream);
        assert(stats == cudaSuccess);
        for(size_t i = 0; i < list_iA.size(); i++)
        {
          // std::cout << " i = " <<  i << "\n";
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
      }
    }
    else
      kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(),
                           col_indx_.size(), col_indx_.data(), row_indx_.data(),
                           num_terms_, iA.data(), vA.data(), alpha, xdev.data(),
                           beta, ydev.data());
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
      kronmult::cpu_sparse(num_dimensions_, kron_size_, num_rows_,
                           row_indx_.data(), col_indx_.data(), num_terms_,
                           iA.data(), vA.data(), alpha, x, beta, y);
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

private:
  int num_dimensions_;
  int kron_size_; // i.e., n - size of the matrices
  int num_rows_;
  int num_cols_;
  int num_terms_;
  int tensor_size_;
  int64_t flops_;

#ifdef ASGARD_USE_CUDA
  // load_stream is the stream to load data
  // compute is done on the default stream
  static constexpr resource data_mode = resource::device;
  mutable fk::vector<precision, mem_type::view, data_mode> xdev, ydev;
  mutable fk::vector<int, mem_type::view, data_mode> worka, workb;
  cudaStream_t load_stream;
#else
  static constexpr resource data_mode = resource::host;
#endif

  fk::vector<int, mem_type::owner, data_mode> row_indx_;
  fk::vector<int, mem_type::owner, data_mode> col_indx_;

  // break iA into a list, as well as indexing
  // used for out-of-core work and splitting work to prevent overflow
  // only one of iA or list_iA is used in an instance of the matrix
  int list_row_stride_; // for CPU dense case, how many rows fall in one list
  std::vector<fk::vector<int>> list_iA;
  std::vector<fk::vector<int>> list_row_indx_;
  std::vector<fk::vector<int>> list_col_indx_;

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
 */
template<typename P>
memory_usage
compute_mem_usage(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                  options const &program_options, imex_flag const imex,
                  kron_sparse_cache &spcache);

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
#ifdef ASGARD_USE_CUDA
    load_stream = nullptr;
#endif
  }

  ~matrix_list()
  {
#ifdef ASGARD_USE_CUDA
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
    {
      mem_stats = compute_mem_usage(pde, grid, opts, imex(entry), spcache);
      std::cout << " called compute_mem_usage() \n";
    }
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
      }
    }
    (*this)[entry].set_workspace(xdev, ydev);
    (*this)[entry].set_workspace_ooc(worka, workb, load_stream);
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
  mutable fk::vector<int, mem_type::owner, resource::device> worka;
  mutable fk::vector<int, mem_type::owner, resource::device> workb;
  cudaStream_t load_stream;
#endif

};

} // namespace asgard
