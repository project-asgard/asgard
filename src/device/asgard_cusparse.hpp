#pragma once

#ifdef ASGARD_USE_CUDA

#include "asgard_vector.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_60_atomic_functions.h>

#include <cusparse.h>

namespace asgard::gpu
{
/*!
 * \brief RAII wrapper around the cusparse handle
 *
 * Unlike the global structure that holds a single cusparse handle,
 * this one allows for more granular management of resource.
 * In the future, this will allow us to handle multiple streams
 * and multiple devices with multiple handles.
 */
struct sparse_handle
{
  //! \brief Constructor, creates the handle and sets the pointers to device mode
  sparse_handle()
  {
    auto status = cusparseCreate(&handle_);
    expect(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_DEVICE);
    expect(status == CUSPARSE_STATUS_SUCCESS);
  }
  //! \brief Free the handle
  ~sparse_handle()
  {
    if (handle_ != nullptr)
    {
      auto status = cusparseDestroy(handle_);
      expect(status == CUSPARSE_STATUS_SUCCESS);
    }
  }
  //! \brief The handle can be moved, via constructor or assignment
  sparse_handle(sparse_handle &&other)
    : handle_(std::exchange(other.handle_, nullptr))
  {}
  //! \brief The handle can be moved, via constructor or assignment
  sparse_handle& operator =(sparse_handle &&other)
  {
    sparse_handle tmp(std::move(other));
    std::swap(handle_, tmp.handle_);
    return *this;
  }

  //! \brief The handle cannot be copied, will lose unique ownership
  sparse_handle(sparse_handle const &other) = delete;
  //! \brief The handle cannot be copied, will lose unique ownership
  sparse_handle& operator =(sparse_handle &other) = delete;

  //! \brief Automatically converts to the cusparse handle
  operator cusparseHandle_t() const { return handle_; }
  //! \brief Declare the internal type, will have a rocSparse mode in the future
  using htype = cusparseHandle_t;

  //! \brief The handle variable
  mutable cusparseHandle_t handle_;
};

/*!
 * \brief Helper template, converts float/double to CUDA_R_32F/64F
 */
template<typename T>
struct cusparse_dtype
{};

//! \brief Float specialization for CUDA_R_32F
template<> struct cusparse_dtype<float>
{
  //! \brief The corresponding cuda type
  static const cudaDataType value = CUDA_R_32F;
};
//! \brief Float specialization for CUDA_R_64F
template<> struct cusparse_dtype<double>
{
  //! \brief The corresponding cuda type
  static const cudaDataType value = CUDA_R_64F;
};

/*!
 * \brief Wrapper around cusparse matrix
 *
 * Inside the cusparse library, a simple matrix vector product is associated
 * with multiple dynamically allocated data-structures.
 * In addition to the cusparse handle, we also have a structure describing
 * the sparse matrix, two structures describing the y/x vectors,
 * and a raw pointer that is a scratch space buffer.
 * During repeated calls to the same matrix vector operation, it is redundant
 * to keep allocating/deallocating the same opaque structures, so we put them
 * all in a single RAII container.
 *
 * The constructor will initialize the matrix structure with the compressed
 * row storage format. Then a call to set_vectors() is needed to set the x/y
 * vectors and the scalars for the matrix-vector product.
 * This structure will hold aliases to the x/y pointers and the scalars
 * will be pushed onto the GPU device, so they can be used directly from
 * GPU memory with less synchronization.
 * The vectors and scalars can be changed with a new call to set_vectors();
 * however, this can affect the size of the buffer.
 *
 * After the vectors have been set, a call to size_workspace() will return
 * the size of the workspace buffer. Note: that a change to the scalars in
 * the multiplication can change the size of the buffer.
 * The returned size is in units of <T>, e.g., the buffer can be allocated
 * by directly giving the number to the constructor of gpu::vector<T>.
 *
 * Finally, the actual matrix-vector operation is computed with the apply()
 * method, where we can pass a different handle (if using different streams
 * on the same device) and we have to pass in the scratch buffer,
 * that can be held externally and shared between multiple matrices.
 *
 * Here the input and the result appear as side-effects of the call.
 * This is the unfortunate behavior of the cuSparse library, conversely
 * if we accept x/y vectors during the call to apply() we will have to allocate
 * and free multiple temporary variables.
 *
 * The "proper" usage of the sparse_matrix class is to be wrapped in a higher
 * level structure, which in turn manages the input/output/scratch buffers
 * and the arrays with indexes and values.
 * This plays nicely with the global Kronecker approach, where we need to do
 * multiple matrix-vector products per term for each global kron operation.
 * Also, since the input and output have to be padded to complete
 * the hierarchical structure of the sparse grid multi-index set,
 * we are using the same input/output buffers for every sequence of calls
 * The external wrapper can also hold aliases (or more appropriately own)
 * the values array, which allow for the vals (i.e., matrix coefficients)
 * to be overwritten between calls to apply.
 *
 * Example usage:
 * \code
 *   sparse_handle cusp;
 *   sparse_matrix<double> sp_mat( ... );
 *   sp_mat.set_vectors( ... );
 *   gpu::vector<T> buffer( sp_mat.size_workspace(cusp) );
 *   sp_mat.apply( cusp, buffer.data() );
 * \endcode
 */
template<typename T>
class sparse_matrix
{
public:
  //! \brief Constructor, initialize the matrix portion of the container
  sparse_matrix(int64_t num_rows, int64_t num_cols, int64_t nnz,
                int *pntr, int *indx, T *vals)
    : desc_(nullptr), x_(nullptr), y_(nullptr)
  {
    auto status = cusparseCreateCsr(&desc_, num_rows, num_cols, nnz,
                                    pntr, indx, vals,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    cusparse_dtype<T>::value);
    expect(status == CUSPARSE_STATUS_SUCCESS);
  }
  //! \brief Container is movable, via constructor or assignment
  sparse_matrix(sparse_matrix<T> &&other)
    : desc_(std::exchange(other.desc_, nullptr)),
      x_(std::exchange(other.x_, nullptr)),
      y_(std::exchange(other.y_, nullptr)),
      scale_factors_(std::move(other.scale_factors_))
  {}
  //! \brief Container is movable, via constructor or assignment
  sparse_matrix<T> &operator =(sparse_matrix<T> &&other)
  {
    sparse_matrix<T> tmp(std::move(other));
    std::swap(desc_, tmp.desc_);
    std::swap(x_, tmp.x_);
    std::swap(y_, tmp.y_);
    std::swap(scale_factors_, tmp.scale_factors_);
    return *this;
  }
  //! \brief The handle cannot be copied, will lose unique ownership
  sparse_matrix(sparse_matrix<T> const &other) = delete;
  //! \brief The handle cannot be copied, will lose unique ownership
  sparse_matrix<T> &operator =(sparse_matrix<T> const &other) = delete;
  //! \brief Destructor, frees the resources
  ~sparse_matrix()
  {
    if (desc_ != nullptr)
      cusparseDestroySpMat(desc_);
    if (x_ != nullptr)
      cusparseDestroyDnVec(x_);
    if (y_ != nullptr)
      cusparseDestroyDnVec(y_);
  }

  //! \brief Sets the vectors and scalars for y = alpha * A * x + beta * y
  void set_vectors(int64_t n, T alpha, T *x, T beta, T *y)
  {
    if (x_ != nullptr)
      cusparseDestroyDnVec(x_);
    if (y_ != nullptr)
      cusparseDestroyDnVec(y_);

    auto status = cusparseCreateDnVec(&x_, n, x, cusparse_dtype<T>::value);
    expect(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseCreateDnVec(&y_, n, y, cusparse_dtype<T>::value);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    scale_factors_ = std::vector<T>{alpha, beta};
  }

  //! \brief The result is in units of the floating point type
  size_t size_workspace(cusparseHandle_t handle) const
  {
    size_t buffer_size = 0;

    auto status = cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, scale_factors_.data(), desc_,
        x_, scale_factors_.data() + 1, y_, cusparse_dtype<T>::value,
        CUSPARSE_SPMV_CSR_ALG1, &buffer_size);
    expect(status == CUSPARSE_STATUS_SUCCESS);
    return 1 + buffer_size / sizeof(T);
  }
  //! \brief Computes the matrix vector product with x/y as defined in set_vectors()
  void apply(cusparseHandle_t handle, void *workspace)
  {
    auto status = cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, scale_factors_.data(), desc_,
        x_, scale_factors_.data() + 1, y_, cusparse_dtype<T>::value,
        CUSPARSE_SPMV_CSR_ALG1, workspace);
    expect(status == CUSPARSE_STATUS_SUCCESS);
  }

private:
  //! \brief The matrix description
  cusparseSpMatDescr_t desc_;
  //! \brief The cuda versions of the x/y vectors
  cusparseDnVecDescr_t x_, y_;
  //! \brief Hold the scalars for the matrix-vector product
  gpu::vector<T> scale_factors_;
};

} // namespace asgard::gpu
#endif
