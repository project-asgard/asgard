#pragma once

#include "asgard_tools.hpp"

#ifdef ASGARD_USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_60_atomic_functions.h>

#define ASGARD_GPU_WARP_SIZE 32

#endif

// As of LLVM version 16, clang does not utlize #pragma omp simd, resulting in a
// pessimization
#if defined(__clang__)
#define ASGARD_PRAGMA_OMP_SIMD(x)
#define ASGARD_OMP_SIMD
#else
#define ASGARD_PRAGMA(x) _Pragma(#x)
#define ASGARD_OMP_SIMD ASGARD_PRAGMA(omp simd)
#define ASGARD_PRAGMA_OMP_SIMD(clause) ASGARD_PRAGMA(omp simd clause)
#endif

namespace asgard::kronmult
{
/*!
 * \brief Computes the number of CUDA blocks.
 *
 * \param work_size is the total amount of work, e.g., size of the batch
 * \param work_per_block is the work that a single thread block will execute
 * \param max_blocks is the maximum number of blocks
 */
inline int blocks(int64_t work_size, int work_per_block, int max_blocks)
{
  return std::min(
      max_blocks,
      static_cast<int>((work_size + work_per_block - 1) / work_per_block));
}

/*!
 * \brief Flag variable, indicates whether thread synchronization is necessary.
 *
 * Threads inside a warp are always synchronized, synchronization
 * in the kernel is not needed unless teams span more than one warp.
 */
enum class manual_sync
{
  //! \brief Use synchronization after updating the shared cache.
  enable,
  //! \brief No need for synchronization, thread teams are aligned to the warps.
  disable
};

/*!
 * \brief Template that computes n to power, e.g., ipow<2, 3>() returns constexpr 8.
 */
template<int n, int power>
constexpr int ipow()
{
  if constexpr (power == 1)
  {
    return n;
  }
  else if constexpr (power == 2)
  {
    return n * n;
  }
  else if constexpr (power == 3)
  {
    return n * n * n;
  }
  else if constexpr (power == 4)
  {
    return n * n * n * n;
  }
  else if constexpr (power == 5)
  {
    return n * n * n * n * n;
  }
  else if constexpr (power == 6)
  {
    return n * n * n * n * n * n;
  }
  static_assert(power >= 1 and power <= 6,
                "ipow() does not works with specified power");
  return 0;
}

//! \brief Helper methods that will zero out a buffer
template<resource rec, typename T>
struct buff_utils
{
  //! \brief Set the first num entries of x to zero
  static void set_to_zero(int64_t num, T *x)
  {
    static_assert(rec == resource::host);
    std::fill_n(x, num, T{0});
  }
};

//! \brief Method to zero out a buffer based on a resource type
template<resource rec, typename T>
void set_buffer_to_zero(int64_t num, T *x)
{
  buff_utils<rec, T>::set_to_zero(num, x);
}
//! \brief Helper method, fills the buffer with zeros
template<typename T>
void set_buffer_to_zero(std::vector<T> &x)
{
  std::fill(x.begin(), x.end(), T{0});
}

#ifdef ASGARD_USE_CUDA
//! \brief Sets a device buffer to zeros
template<typename T>
void set_gpu_buffer_to_zero(int64_t num, T *x);

//! \brief Specialization for the gpu case
template<typename T>
struct buff_utils<resource::device, T>
{
  //! \brief Set the first num entries of x to zero
  static void set_to_zero(int64_t num, T *x)
  {
    set_gpu_buffer_to_zero(num, x);
  }
};
//! \brief Helper method, fills the buffer with zeros
template<typename T>
void set_buffer_to_zero(gpu::vector<T> &x)
{
  set_gpu_buffer_to_zero(static_cast<int64_t>(x.size()), x.data());
}
#endif

} // namespace asgard::kronmult
