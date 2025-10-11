#pragma once

#include "asgard_tools.hpp"

namespace asgard
{

/*!
 * \brief Indicates the upper/lower connectivity fill of a sparsity pattern
 *
 * In CPU mode, this is used only by the Kronmult module; however, the GPU algorithms
 * explicitly require the upper and lower connectivity patterns.
 * In GPU mode, this is used by the grid_1d module and asgard::gpu_connect_1d,
 * thus it is here in a common header.
 */
enum class conn_fill : int
{
  //! \brief Row r is connected only to self and the children of index r
  upper = 0,
  //! \brief All overlapping volume or edge support, regardless of child-parent relation
  both,
  //! \brief Row r is connected only to the parents of index r (no self-connection)
  lower,
  //! \brief Row r is connected only to the parents of index r, self-connection is identity
  lower_udiag,
};

/*!
 * \brief Default precision to use, double if enabled and float otherwise.
 */
#ifdef ASGARD_ENABLE_DOUBLE
using default_precision = double;
#else
using default_precision = float;
#endif

/*!
 * \brief Indicated if computing should be done suing the CPU or GPU
 */
enum class compute_mode {
  //! Using the CPU device
  cpu,
  //! Using the GPU device
  gpu
};

#ifdef ASGARD_USE_GPU

namespace gpu
{

#ifdef ASGARD_USE_CUDA
//! cuSolver uses 32-bit int
using direct_int = int;

//! converts CUDA error to a human readable string
std::string error_message(cudaError_t err);
std::string error_message(cublasStatus_t err);
std::string error_message(cusolverStatus_t err);

#define cuda_check_error(_call_) \
  { cudaError_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != cudaSuccess) {\
      throw std::runtime_error(::asgard::gpu::error_message(__asgard_intcudaerr__) \
                               + "\n        in file: " + __FILE__    \
                               + "\n           line: " + std::to_string(__LINE__) );  \
    } \
  } \

#define cublas_check_error(_call_) \
  { cublasStatus_t __asgard_intcudaerr__ = (_call_); \
    if (__asgard_intcudaerr__ != CUBLAS_STATUS_SUCCESS) {\
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
  //! \brief Copy from a host array, the source must contain enough data
  void copy_from_host(int64_t num, T const source[]) {
    #ifdef ASGARD_USE_CUDA
    cuda_check_error( cudaMemcpy(data_, source, num * sizeof(T), cudaMemcpyHostToDevice) );
    #else
    rocm_check_error( hipMemcpy(data_, source, num * sizeof(T), hipMemcpyHostToDevice) );
    #endif
  }
  //! \brief Custom conversion, so we can assign to std::vector.
  operator std::vector<T>() const { return this->copy_to_host(); }

private:
  T *data_ = nullptr;
  int64_t size_ = 0;
};

/*!
 * \brief Strong type to identify the GPU device ID.
 */
struct device {
  //! Make a new device identifier
  explicit device(int gpuid) : id(gpuid) {}
  //! Compare two devices and if they match
  bool operator == (device const &other) const { return (id == other.id); }
  //! The device ID, e.g., 0, 1, 2, 3, ...
  int id = -1; // default to an invalid ID, forces an error if used uninitialized
};

//! \brief Transfer data between devices, assumes that compute->set_device(dest_dev)
template<typename T>
void mcopy(device src_dev, vector<T> const &src, device dest_dev, vector<T> &dest) {
  expect(src.size() == dest.size());
  #ifdef ASGARD_USE_CUDA
  cuda_check_error( cudaMemcpyPeer(dest.data(), dest_dev.id, src.data(), src_dev.id,
                                   dest.size() * sizeof(T)) );
  #else
  ignore(src_dev); // ROCm automatically identifies the device for each pointer
  ignore(dest_dev);
  rocm_check_error( hipSetDevice(src_dev.id) );
  rocm_check_error( hipMemcpy(dest.data(), src.data(), dest.size() * sizeof(T), hipMemcpyDeviceToDevice) );
  rocm_check_error( hipSetDevice(dest_dev.id) );
  #endif
}
//! \brief Transfer data between devices, assumes that compute->set_device(dest_dev)
template<typename T>
void mcopy(device src_dev, T const src[], device dest_dev, vector<T> &dest) {
  #ifdef ASGARD_USE_CUDA
  cuda_check_error( cudaMemcpyPeer(dest.data(), dest_dev.id, src, src_dev.id,
                                   dest.size() * sizeof(T)) );
  #else
  ignore(src_dev); // ROCm automatically identifies the device for each pointer
  ignore(dest_dev);
  rocm_check_error( hipSetDevice(src_dev.id) );
  rocm_check_error( hipMemcpy(dest.data(), src, dest.size() * sizeof(T), hipMemcpyDeviceToDevice) );
  rocm_check_error( hipSetDevice(dest_dev.id) );
  #endif
}
//! \brief Transfer data between devices, assumes that compute->set_device(dest_dev)
template<typename T>
void mcopy(int64_t num_entries, device src_dev, T const src[], device dest_dev, T dest[]) {
  #ifdef ASGARD_USE_CUDA
  cuda_check_error( cudaMemcpyPeer(dest, dest_dev.id, src, src_dev.id,
                                   num_entries * sizeof(T)) );
  #else
  ignore(src_dev); // ROCm automatically identifies the device for each pointer
  ignore(dest_dev);
  rocm_check_error( hipSetDevice(src_dev.id) );
  rocm_check_error( hipMemcpy(dest, src, num_entries * sizeof(T), hipMemcpyDeviceToDevice) );
  rocm_check_error( hipSetDevice(dest_dev.id) );
  #endif
}
//! \brief Copy an array to the CPU vector
template<typename T>
void copy_to_host(int64_t num_entries, T const x[], std::vector<T> &y) {
  y.resize(num_entries);
  #ifdef ASGARD_USE_CUDA
  cuda_check_error( cudaMemcpy(y.data(), x, num_entries * sizeof(T), cudaMemcpyDeviceToHost) );
  #else
  rocm_check_error( hipMemcpy(y.data(), x, num_entries * sizeof(T), hipMemcpyDeviceToHost) );
  #endif
}
//! \brief Copy an CPU vector to a device array
template<typename T>
void copy_to_device(std::vector<T> const &x, T y[]) {
  size_t const num_entries = x.size();
  #ifdef ASGARD_USE_CUDA
  cuda_check_error( cudaMemcpy(y, x.data(), num_entries * sizeof(T), cudaMemcpyHostToDevice) );
  #else
  rocm_check_error( hipMemcpy(y, x.data(), num_entries * sizeof(T), hipMemcpyHostToDevice) );
  #endif
}

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

  #ifdef ASGARD_USE_GPU
  void set_device(gpu::device device) const {
    #ifdef ASGARD_USE_CUDA
    cuda_check_error( cudaSetDevice(device.id) );
    #endif
    #ifdef ASGARD_USE_ROCM
    rocm_check_error( hipSetDevice(device.id) );
    #endif
  }
  #endif

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
  //! fill the vector with zeros
  template<typename P>
  void fill_zeros(gpu::vector<P> &x) const { fill_zeros(x.size(), x.data()); }
  #endif

  //! tri-diagonal solver, factorization stage
  template<typename P>
  void pttrf(std::vector<P> &diag, std::vector<P> &subdiag) const;
  //! tri-diagonal solver, solve using the factors
  template<typename P>
  void pttrs(std::vector<P> const &diag, std::vector<P> const &subdiag, std::vector<P> &b) const;

  // few BLAS and BLAS-like methods used as helpers in multi-GPU setup
  #ifdef ASGARD_USE_CUDA
  //! synchronize the device
  void device_synchronize() const { cudaDeviceSynchronize();  }
  //! fill a gpu array with zeros
  template<typename P>
  void fill_zeros(int64_t num, P x[]) const { cuda_check_error( cudaMemset(x, 0, num * sizeof(P)) ); }
  //! increment add, assuming contiguous gpu arrays
  template<typename P>
  void axpy(int num, no_deduce<P> alpha, P const x[], P y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      cublas_check_error( cublasSaxpy(cublas, num, &alpha, x, 1, y, 1) );
    } else {
      cublas_check_error( cublasDaxpy(cublas, num, &alpha, x, 1, y, 1) );
    }
  }
  //! increment add using alpha = 1, assuming contiguous gpu arrays
  template<typename P>
  void axpy(int num, P const x[], P y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    cublas_check_error( cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE) );
    if constexpr (is_float<P>) {
      cublas_check_error( cublasSaxpy(cublas, num, fone.data(), x, 1, y, 1) );
    } else {
      cublas_check_error( cublasDaxpy(cublas, num, done.data(), x, 1, y, 1) );
    }
    cublas_check_error( cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST) );
  }
  //! sale an array, assuming contiguous gpu arrays
  template<typename P>
  void scal(int num, no_deduce<P> alpha, P x[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      cublas_check_error( cublasSscal(cublas, num, &alpha, x, 1) );
    } else {
      cublas_check_error( cublasDscal(cublas, num, &alpha, x, 1) );
    }
  }
  #endif
  #ifdef ASGARD_USE_ROCM
  //! synchronize the device
  void device_synchronize() const { rocm_check_error( hipDeviceSynchronize() ); }
  //! fill a gpu array with zeros
  template<typename P>
  void fill_zeros(int64_t num, P x[]) const { rocm_check_error( hipMemset(x, 0, num * sizeof(P)) ); }
  //! increment add, assuming contiguous gpu arrays
  template<typename P>
  void axpy(int num, no_deduce<P> alpha, P const x[], P y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      rocblas_check_error( rocblas_saxpy(rocblas, num, &alpha, x, 1, y, 1) );
    } else {
      rocblas_check_error( rocblas_daxpy(rocblas, num, &alpha, x, 1, y, 1) );
    }
  }
  //! increment add using alpha = 1, assuming contiguous gpu arrays
  template<typename P>
  void axpy(int num, P const x[], P y[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "axpy can be called only with floats and doubles");
    rocblas_check_error( rocblas_set_pointer_mode(rocblas, rocblas_pointer_mode_device) );
    if constexpr (is_float<P>) {
      rocblas_check_error( rocblas_saxpy(rocblas, num, fone.data(), x, 1, y, 1) );
    } else {
      rocblas_check_error( rocblas_daxpy(rocblas, num, done.data(), x, 1, y, 1) );
    }
    rocblas_check_error( rocblas_set_pointer_mode(rocblas, rocblas_pointer_mode_host) );
  }
  //! sale an array, assuming contiguous gpu arrays
  template<typename P>
  void scal(int num, no_deduce<P> alpha, P x[]) const {
    static_assert(is_float<P> or is_double<P>,
                  "scal can be called only with floats and doubles");
    if constexpr (is_float<P>) {
      rocblas_check_error( rocblas_sscal(rocblas, num, &alpha, x, 1) );
    } else {
      rocblas_check_error( rocblas_dscal(rocblas, num, &alpha, x, 1) );
    }
  }
  #endif

private:
  int num_gpus_ = 0;
  #ifdef ASGARD_USE_CUDA
  cublasHandle_t cublas = nullptr;
  cusolverDnHandle_t cusolverdn = nullptr;
  #endif
  #ifdef ASGARD_USE_ROCM
  rocblas_handle rocblas = nullptr;
  #endif
  #ifdef ASGARD_USE_GPU
  gpu::vector<float> fone;
  gpu::vector<double> done;
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

//! \brief returns the max norm of an array
template<typename P>
P nrm_inf(int n, P const x[]) {
  P r = 0;
  for (int i = 0; i < n; i++) r = std::max(r, std::abs(x[i]));
  return r;
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
