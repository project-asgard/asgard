#pragma once

#include "asgard_tools.hpp"

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

#ifdef ASGARD_USE_GPU

namespace gpu
{

#ifdef ASGARD_USE_CUDA
//! cuSolver uses 32-bit int
using direct_int = int;

//! converts CUDA error to a human readable string
std::string error_message(cudaError_t err);
std::string error_message(cusolverStatus_t err);

#define cuda_check_error(_call_) \
  { cudaError_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != cudaSuccess) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

#define cusolver_check_error(_call_) \
  { cusolverStatus_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != CUSOLVER_STATUS_SUCCESS) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

#else

using direct_int = rocblas_int;

//! converts ROCM error to a human readable string
std::string error_message(hipError_t err);
std::string error_message(rocblas_status err);

#define rocm_check_error(_call_) \
  { hipError_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != hipSuccess) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

#define rocblas_check_error(_call_) \
  { rocblas_status __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != rocblas_status_success) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

#endif

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
  vector() = default;
  //! \brief Free all resouces.
  ~vector() {
    if (data_ != nullptr)
      #ifdef ASGARD_USE_CUDA
      cudaFree(data_);
      #else
      (void) hipFree(data_);
      #endif
  }
  //! \brief Construct a vector with given size.
  vector(int64_t size)
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
    #ifdef ASGARD_USE_CUDA
    cuda_check_error( cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice) );
    #else
    rocm_check_error( hipMemcpy(data_, other.data_, size_ * sizeof(T), hipMemcpyDeviceToDevice) );
    #endif
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
    #ifdef ASGARD_USE_CUDA
    cuda_check_error( cudaMemcpy(data_, other.data(), size_ * sizeof(T), cudaMemcpyHostToDevice) );
    #else
    rocm_check_error( hipMemcpy(data_, other.data(), size_ * sizeof(T), hipMemcpyHostToDevice) );
    #endif
    return *this;
  }
  //! \brief Does not rellocate the data, i.e., if size changes all old data is lost.
  void resize(int64_t new_size)
  {
    expect(new_size >= 0);
    if (new_size != size_)
    {
      #ifdef ASGARD_USE_CUDA
      if (data_ != nullptr)
        cuda_check_error( cudaFree(data_) );
      cuda_check_error( cudaMalloc((void**)&data_, new_size * sizeof(T)) );
      #else
      if (data_ != nullptr)
        rocm_check_error( hipFree(data_) );
      rocm_check_error( hipMalloc((void**)&data_, new_size * sizeof(T)) );
      #endif
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
    #ifdef ASGARD_USE_CUDA
    cuda_check_error( cudaMemcpy(destination, data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice) );
    #else
    rocm_check_error( hipMemcpy(destination, data_, size_ * sizeof(T), hipMemcpyDeviceToDevice) );
    #endif
  }
  //! \brief Copy to a host array, the destination must be large enough
  void copy_to_host(T *destination) const
  {
    #ifdef ASGARD_USE_CUDA
    cuda_check_error( cudaMemcpy(destination, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost) );
    #else
    rocm_check_error( hipMemcpy(destination, data_, size_ * sizeof(T), hipMemcpyDeviceToHost) );
    #endif
  }
  //! \brief Copy to a std::vector on the host.
  void copy_to_host(std::vector<T> &destination) const
  {
    destination.resize(size_);
    this->copy_to_host(destination.data());
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
  T *data_ = nullptr;
  int64_t size_ = 0;
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
  //! initialize the compute engine, call once per application
  compute_resources();
  //! free all resources associated with the engine
  ~compute_resources();

  //! return the number of usable GPU devices
  int num_gpus() const { return num_gpus_; }
  //! returns true if there is an available GPU
  bool has_gpu() const { return (num_gpus_ > 0); }

  //! PLU factorization of an M x M matrix
  template<typename P>
  void getrf(int M, std::vector<P> &A, std::vector<int> &ipiv) const;
  //! PLU solve of an M x M matrix
  template<typename P>
  void getrs(int M, std::vector<P> const &A, std::vector<int> const &ipiv, std::vector<P> &b) const;

  #ifdef ASGARD_USE_GPU
  //! PLU factorization of an M x M matrix
  template<typename P>
  void getrf(int M, gpu::vector<P> &A, gpu::vector<int> &ipiv) const;
  //! PLU solve of an M x M matrix
  template<typename P>
  void getrs(int M, gpu::vector<P> const &A, gpu::vector<gpu::direct_int> const &ipiv,
             gpu::vector<P> &b) const;
  //! PLU solve of an M x M matrix
  template<typename P>
  void getrs(int M, gpu::vector<P> const &A, gpu::vector<gpu::direct_int> const &ipiv,
             std::vector<P> &b) const {
    gpu::vector<P> gpu_b = b;
    getrs(M, A, ipiv, gpu_b);
    gpu_b.copy_to_host(b);
  }
  #endif

  //! tri-diagonal solver, factorization stage
  template<typename P>
  void pttrf(std::vector<P> &diag, std::vector<P> &subdiag) const;
  //! tri-diagonal solver, solve using the factors
  template<typename P>
  void pttrs(std::vector<P> const &diag, std::vector<P> const &subdiag, std::vector<P> &b) const;

private:
  int num_gpus_ = 0;
  #ifdef ASGARD_USE_CUDA
  // std::array<cusolverDnHandle_t, max_num_gpus>
  cusolverDnHandle_t cusolverdn;
  #endif
  #ifdef ASGARD_USE_ROCM
  // std::array<cusolverDnHandle_t, max_num_gpus>
  rocblas_handle rocblas;
  #endif
};

inline std::optional<compute_resources> compute;

inline void init_compute() {
  if (not compute)
    compute.emplace();
}

/*!
 * \brief Math utilities for commonly used operations
 *
 * Many multi-index operations require the use of methods such as log()
 * and pow(), but use integer arithmetic instead.
 * This namesapce provides shorthand operations for methods that
 * compute the power of 2, power with integer component, integer log-2,
 * and several others.
 */
namespace fm {
//! computes 2^exponent using bit-shift operations, only for int-like types
template<typename T>
inline constexpr T ipow2(T const exponent)
{
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, unsigned> ||
                std::is_same_v<T, long> || std::is_same_v<T, unsigned long> ||
                std::is_same_v<T, long long> ||
                std::is_same_v<T, unsigned long long>);
  expect(exponent >= 0);
  expect(exponent < std::numeric_limits<T>::digits);
  return T{1} << exponent;
}

//! Raise the base to an integer power
template<typename T = int64_t>
inline constexpr T ipow(T base, int exponent)
{
  expect(exponent >= 1);
  T result = base;
  for (int e = 1; e < exponent; e++)
    result *= base;
  return result;
}

//! computes std::floor( std::log2(x) ), returns 0 for x = 0 using bit-wise shifts
inline constexpr int intlog2(int x)
{
  int result = 0;
  while (x >>= 1)
    result++;
  return result;
}
//! computes std::pow( 2, std::floor( std::log2(x) ) ) using bit-wise shifts
inline int ipow2_log2(int x)
{
  int result = 1;
  while (x >>= 1)
    result <<= 1;
  return result;
}
//! computes ipow2_log2(i) and std::pow(std::sqrt(2.0), intlog2(i))
inline void intlog2_pow2pows2(int x, int &i2l2, double &is2l2)
{
  i2l2  = 1;
  is2l2 = 1.0;
  while (x >>= 1)
  {
    i2l2 <<= 1;
    is2l2 *= 1.41421356237309505; // sqrt(2.0)
  }
}
//! computes base^p where p is in integer
template<typename P>
P powi(P base, int p) {
  P res = 1;
  while (--p > -1)
    res *= base;
  return res;
}

/*!
 * \brief Computes the l-inf norm of the difference between x and y
 *
 * This works with all std::vector, std::array and fk::vector.
 * Does not work with GPU vectors and does not check if the data is on the device.
 */
template<typename vecx, typename vecy>
auto diff_inf(vecx const &x, vecy const &y)
{
  using precision = typename vecx::value_type;
  using index     = decltype(x.size());
  expect(x.size() == static_cast<index>(y.size()));

  precision m{0};
  for (index i = index{0}; i < x.size(); i++)
    m = std::max(m, std::abs(x[i] - y[i]));
  return m;
}

/*!
 * \brief Computes the root-mean-square-error between two vectors
 *
 * This works with all std::vector, std::array and fk::vector.
 * Does not work with GPU vectors and does not check if the data is on the device.
 */
template<typename vecx, typename vecy>
auto rmserr(vecx const &x, vecy const &y)
{
  using precision = typename vecx::value_type;
  using index     = decltype(x.size());
  expect(x.size() == y.size());

  precision err{0};
  for (index i = index{0}; i < x.size(); i++)
  {
    precision const d = x[i] - y[i];
    err += d * d;
  }
  return std::sqrt(err / x.size());
}

} // namespace fm

} // namespace asgard
