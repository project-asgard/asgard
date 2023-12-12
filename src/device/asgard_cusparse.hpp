#pragma once

#ifdef ASGARD_USE_CUDA

#include <iostream>
#include <vector>

#include "build_info.hpp"

#include "asgard_vector.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_60_atomic_functions.h>

#include <cusparse.h>

namespace asgard::gpu
{

struct sparse_handle
{
  sparse_handle()
  {
    auto status = cusparseCreate(&handle_);
    expect(status == CUSPARSE_STATUS_SUCCESS);
  }
  ~sparse_handle()
  {
    if (handle_ != nullptr)
    {
      auto status = cusparseDestroy(handle_);
      expect(status == CUSPARSE_STATUS_SUCCESS);
    }
  }

  sparse_handle(sparse_handle &&other)
    : handle_(std::exchange(other.handle_, nullptr))
  {}

  sparse_handle& operator =(sparse_handle &&other)
  {
    sparse_handle tmp(std::move(other));
    std::swap(handle_, tmp.handle_);
    return *this;
  }

  sparse_handle(sparse_handle const &other) = delete;
  sparse_handle& operator =(sparse_handle &other) = delete;

  operator cusparseHandle_t() const { return handle_; }

  using htype = cusparseHandle_t;

  mutable cusparseHandle_t handle_;
};

template<typename T>
struct cusparse_dtype
{};

template<> struct cusparse_dtype<float>
{
  static const cudaDataType value = CUDA_R_32F;
};
template<> struct cusparse_dtype<double>
{
  static const cudaDataType value = CUDA_R_64F;
};

template<typename T>
struct sparse_matrix
{
  sparse_matrix(int64_t num_rows, int64_t num_cols, int64_t nnz,
                int *pntr, int *indx, T *vals)
  {
    auto status = cusparseCreateCsr(&desc_, num_rows, num_cols, nnz,
                                    pntr, indx, vals,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    cusparse_dtype<T>::value);
    expect(status == CUSPARSE_STATUS_SUCCESS);
  }
  sparse_matrix(sparse_matrix<T> &&other)
    : desc_(std::exchange(other.desc_, nullptr)),
      x_(std::exchange(other.x_, nullptr)),
      y_(std::exchange(other.y_, nullptr))
  {}

  sparse_matrix<T> &operator =(sparse_matrix<T> &&other)
  {
    sparse_matrix<T> tmp(std::move(other));
    std::swap(desc_, tmp.desc_);
    std::swap(x_, tmp.x_);
    std::swap(y_, tmp.y_);
    return *this;
  }

  sparse_matrix(sparse_matrix<T> const &other) = delete;
  sparse_matrix<T> &operator =(sparse_matrix<T> const &other) = delete;

  ~sparse_matrix()
  {
    if (desc_ != nullptr)
      cusparseDestroySpMat(desc_);
    if (x_ != nullptr)
      cusparseDestroyDnVec(x_);
    if (y_ != nullptr)
      cusparseDestroyDnVec(y_);
  }

  void set_vectors(int64_t n, T alpha, T *x, T beta, T *y)
  {
    auto status = cusparseCreateDnVec(&x_, n, x, cusparse_dtype<T>::value);
    expect(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseCreateDnVec(&y_, n, y, cusparse_dtype<T>::value);
    expect(status == CUSPARSE_STATUS_SUCCESS);

    scale_factors_ = std::vector<T>{alpha, beta};
  }

  size_t size_workspace(cusparseHandle_t handle) const
  {
    size_t buffer_size = 0;

    auto status = cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, scale_factors_.data(), desc_, x_,
        scale_factors_.data() + 1, y_, cusparse_dtype<T>::value, CUSPARSE_SPMV_CSR_ALG1,
        &buffer_size);
    expect(status == CUSPARSE_STATUS_SUCCESS);
    return buffer_size;
  }

  void apply(cusparseHandle_t handle, void *workspace)
  {
    auto status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, scale_factors_.data(),
                               desc_, x_, scale_factors_.data() + 1, y_, cusparse_dtype<T>::value,
                               CUSPARSE_SPMV_CSR_ALG1, workspace);
    expect(status == CUSPARSE_STATUS_SUCCESS);
  }

  cusparseSpMatDescr_t desc_;
  cusparseDnVecDescr_t x_, y_;
  gpu::vector<T> scale_factors_;
};

} // namespace asgard::gpu::sparse
#endif
