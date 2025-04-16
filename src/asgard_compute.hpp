#pragma once

#include "asgard_tools.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

namespace asgard
{

/*!
 * \brief Default precision to use, double if enabled and float otherwise.
 */
#ifdef ASGARD_ENABLE_DOUBLE
using default_precision = double;
#else
using default_precision = float;
#endif

template<typename T>
constexpr bool is_double = std::is_same_v<double, T>;

#ifdef ASGARD_USE_CUDA

namespace gpu
{

//! converts CUDA error to a human readable string
std::string error_message(cudaError_t err);

#define gpu_check_error(_call_) \
  { cudaError_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != cudaSuccess) {\
      throw std::runtime_error(error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

/*!
 * \brief Simple container for GPU data, interoperable with std::vector
 *
 * This simple container allows for RAII memory management,
 * resizing (without relocating the data) and easy copy from/to std::vector
 */
template<typename T>
class vector
{
public:
  //! \brief The value type.
  using value_type = T;
  //! \brief Construct an empty vector.
  vector() : data_(nullptr), size_(0) {}
  //! \brief Free all resouces.
  ~vector() {
    if (data_ != nullptr)
      cudaFree(data_);
  }
  //! \brief Construct a vector with given size.
  vector(int64_t size) : data_(nullptr), size_(0)
  {
    this->resize(size);
  }
  //! \brief Move-constructor.
  vector(vector<T> &&other)
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0))
  {}
  //! \brief Move-assignment.
  vector &operator=(vector<T> &&other)
  {
    vector<T> temp(std::move(other));
    std::swap(data_, temp.data_);
    std::swap(size_, temp.size_);
    return *this;
  }
  //! \brief Copy-constructor.
  vector(vector<T> const &other) : vector()
  {
    *this = other;
  }
  //! \brief Copy-assignment.
  vector<T> &operator=(vector<T> const &other)
  {
    this->resize(other.size());
    gpu_check_error( cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice) );
    return *this;
  }
  //! \brief Constructor that copies from an existing std::vector
  vector(std::vector<T> const &other) : vector()
  {
    *this = other;
  }
  //! \brief Copy the data from the std::vector
  vector<T> &operator=(std::vector<T> const &other)
  {
    this->resize(other.size());
    gpu_check_error( cudaMemcpy(data_, other.data(), size_ * sizeof(T), cudaMemcpyHostToDevice) );
    return *this;
  }
  //! \brief Does not rellocate the data, i.e., if size changes all old data is lost.
  void resize(int64_t new_size)
  {
    expect(new_size >= 0);
    if (new_size != size_)
    {
      if (data_ != nullptr)
        gpu_check_error( cudaFree(data_) );
      gpu_check_error( cudaMalloc((void**)&data_, new_size * sizeof(T)) );
      size_ = new_size;
    }
  }
  //! \brief Returns the number of elements inside the vector.
  int64_t size() const { return size_; }
  //! \brief Returns true if the size is zero, false otherwise.
  bool empty() const { return (size_ == 0); }
  //! \brief Clears all content.
  void clear() { this->resize(0); }
  //! \brief Returns pointer to the first stored element.
  T *data() { return data_; }
  //! \brief Returns const pointer to the first stored element.
  T const *data() const { return data_; }
  //! \brief Copy to a device array, the destination must be large enough
  void copy_to_device(T *destination) const
  {
    gpu_check_error( cudaMemcpy(destination, data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice) );
  }
  //! \brief Copy to a host array, the destination must be large enough
  void copy_to_host(T *destination) const
  {
    gpu_check_error( cudaMemcpy(destination, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost) );
  }
  //! \brief Copy to a std::vector on the host.
  std::vector<T> copy_to_host() const
  {
    std::vector<T> result(size_);
    this->copy_to_host(result.data());
    return result;
  }
  //! \brief Custom conversion, so we can assign to std::vector.
  operator std::vector<T>() const { return this->copy_to_host(); }

private:
  T *data_;
  int64_t size_;
};

} // namespace gpu
#endif

/*!
 * \brief Holds general information about the compute resources
 *
 * Singleton class holding meta information about the CPU and GPU resources,
 * number of threads, number of GPUs, allows easy access to BLAS on both
 * CPU and GPU, etc.
 * The main goal of this class is to allow easy use of multiple GPUs handling
 * the corresponding streams and queues, managing memory, and so on.
 */
class compute_resources {
public:
  //! initialize the engine, call once per application
  compute_resources();

  //! PLU factorization of an M x M matrix
  template<typename P>
  void getrf(int M, std::vector<P> &A, std::vector<int> &ipiv);
  //! PLU solve of an M x M matrix
  template<typename P>
  void getrs(int M, std::vector<P> const &A, std::vector<int> const &ipiv, std::vector<P> &b);

private:
};

inline std::optional<compute_resources> compute;

inline void init_compute() {
  if (not compute)
    compute.emplace();
}

} // namespace asgard
