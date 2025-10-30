#pragma once

#include "asgard_compute_cpu.hpp"

#ifdef ASGARD_USE_CUDA
#include "asgard_compute_cuda.hpp"
#endif
#ifdef ASGARD_USE_ROCM
#include "asgard_compute_rocm.hpp"
#endif

namespace asgard
{

#ifdef ASGARD_USE_GPU
namespace gpu
{

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
      gpu::memfree(data_);
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
    gpu::memcopy_dev2dev(size_, other.data_, data_);
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
    gpu::memcopy_host2dev(size_, other.data(), data_);
    return *this;
  }
  //! \brief Does not rellocate the data, i.e., if size changes all old data is lost.
  void resize(int64_t new_size)
  {
    expect(new_size >= 0);
    if (new_size != size_)
    {
      if (data_ != nullptr)
        gpu::memfree(data_);
      data_ = gpu::memalloc<T>(new_size);
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
  void copy_to_device(T *destination) const {
    gpu::memcopy_dev2dev(size_, data_, destination);
  }
  //! \brief Copy to a host array, the destination must be large enough
  void copy_to_host(T *destination) const {
    gpu::memcopy_dev2host(size_, data_, destination);
  }
  //! \brief Copy number of entries to a host array, the destination must be large enough
  void copy_to_host(int64_t num, T *destination) const {
    gpu::memcopy_dev2host(num, data_, destination);
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
    expect(num <= size_);
    gpu::memcopy_host2dev(num, source, data_);
  }
  //! \brief Custom conversion, so we can assign to std::vector.
  operator std::vector<T>() const { return this->copy_to_host(); }

private:
  T *data_ = nullptr;
  int64_t size_ = 0;
};

//! \brief Transfer data between devices, assumes that compute->set_device(dest_dev)
template<typename T>
void mcopy(device src_dev, vector<T> const &src, device dest_dev, vector<T> &dest) {
  expect(src.size() == dest.size());
  mcopy(dest.size(), src_dev, src.data(), dest_dev, dest.data());
}
//! \brief Transfer data between devices, assumes that compute->set_device(dest_dev)
template<typename T>
void mcopy(device src_dev, T const src[], device dest_dev, vector<T> &dest) {
  mcopy(dest.size(), src_dev, src, dest_dev, dest.data());
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
class __signleton_compute_resources {
public:
  //! initialize the compute engine, call once per application
  __signleton_compute_resources();

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

  #ifdef ASGARD_USE_GPU
  //! set the active device
  void set_device(gpu::device device) const {
    gpu::set_device(device.id);
  }
  //! synchronize the device
  void device_synchronize() const { gpu::device_synchronize(); }
  //! fill the vector with zeros
  template<typename P>
  void fill_zeros(gpu::vector<P> &x) const { fill_zeros(x.size(), x.data()); }
  //! fill a gpu array with zeros
  template<typename P>
  void fill_zeros(int64_t num, P x[]) const { gpu::fill_zeros<P>(num, x); }
  //! increment add, assuming contiguous gpu arrays
  template<typename P>
  void axpy(int num, no_deduce<P> alpha, P const x[], P y[]) const {
    blas_.axpy(num, alpha, x, y);
  }
  //! increment add using alpha = 1, assuming contiguous gpu arrays
  template<typename P>
  void axpy(int num, P const x[], P y[]) const {
    blas_.axpy(num, x, y);
  }
  //! sale an array, assuming contiguous gpu arrays
  template<typename P>
  void scal(int num, no_deduce<P> alpha, P x[]) const {
    blas_.scal(num, alpha, x);
  }
  //! dot product between two vectors
  template<typename P>
  P dot(int num, P const x[], P const y[]) const {
    return blas_.dot(num, x, y);
  }
  //! dot product of a vector with itself
  template<typename P>
  P dot1(int num, P const x[]) const {
    P const n = blas_.nrm2(num, x);
    return n * n;
  }
  //! norm-2 of a vector
  template<typename P>
  P nrm2(int num, P const x[]) const {
    P const n = blas_.nrm2(num, x);
    return n;
  }
  //! transpose of a matrix, times a vector
  template<typename P>
  void gemtv(int m, int n, no_deduce<P> alpha, P const A[], P const x[],
             no_deduce<P> beta, P y[]) const {
    blas_.gemtv(m, n, alpha, A, x, beta, y);
  }
  //! matrix, times a vector
  template<typename P>
  void gemv(int m, int n, no_deduce<P> alpha, P const A[], P const x[],
             no_deduce<P> beta, P y[]) const {
    blas_.gemv(m, n, alpha, A, x, beta, y);
  }
  #endif

private:
  int num_gpus_ = 0;
  #ifdef ASGARD_USE_GPU
  gpu::blas_engine blas_;
  #endif
};

//! singleton, wrapper around some compute capabilities
inline std::optional<__signleton_compute_resources> compute;

/*!
 * \brief Initializes the compute environment, if not initialized already
 *
 * Called by the discretization manager, during the initial setup of the PDE
 * discretization. This is not technically not thread safe but multiple managers should not be
 * constructed in parallel, since the managers themselves use OpenMP in the background.
 */
inline void init_compute() {
  if (not compute)
    compute.emplace();
}

} // namespace asgard
