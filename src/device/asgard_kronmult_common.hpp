#pragma once

#include <iostream>
#include <vector>

#include "build_info.hpp"

#ifdef ASGARD_USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_60_atomic_functions.h>

#define ASGARD_GPU_WARP_SIZE 32

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
inline int blocks(int work_size, int work_per_block, int max_blocks)
{
  return std::min(max_blocks,
                  (work_size + work_per_block - 1) / work_per_block);
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
 * \brief Reference implementation use for testing.
 *
 * Explicitly constructs the Kronecker product of two matrices.
 */
template<typename T>
std::vector<T> kronecker(int m, T const A[], int n, T const B[])
{
  std::vector<T> result(n * n * m * m);
  for (int jm = 0; jm < m; jm++)
  {
    for (int jn = 0; jn < n; jn++)
    {
      for (int im = 0; im < m; im++)
      {
        for (int in = 0; in < n; in++)
        {
          result[(jm * n + jn) * (m * n) + im * n + in] =
              A[jm * m + im] * B[jn * n + in];
        }
      }
    }
  }
  return result;
}

/*!
 * \brief Reference implementation of gemv, compared to BLAS alpha = beta = 1.
 */
template<typename T>
void reference_gemv(int n, T const A[], T const x[], T y[])
{
  for (int j = 0; j < n; j++)
  {
    for (int i = 0; i < n; i++)
    {
      y[i] += A[j * n + i] * x[j];
    }
  }
}

/*!
 * \brief Reference implementation one Kronecker product.
 */
template<typename T>
void reference_kronmult_one(int dimensions, int n, T const *const pA[],
                            T const x[], T y[])
{
  std::vector<T> kron(pA[dimensions - 1], pA[dimensions - 1] + n * n);
  int total_size = n;
  for (int i = dimensions - 2; i >= 0; i--)
  {
    kron = kronecker(n, pA[i], total_size, kron.data());
    total_size *= n;
  }
  reference_gemv(total_size, kron.data(), x, y);
}

/*!
 * \brief Reference implementation of kronmult, do not use in production.
 */
template<typename T>
void reference_kronmult(int dimensions, int n, T const *const pA[],
                        T const *const pX[], T *pY[], int const num_batch)
{
  for (int i = 0; i < num_batch; i++)
  {
    reference_kronmult_one(dimensions, n, &pA[dimensions * i], pX[i], pY[i]);
  }
}

/*!
 * \brief Recursive template that computes n to power, e.g., ipow<2, 3>() returns constexpr 8.
 */
template<int n, int power>
constexpr int ipow()
{
  if constexpr (power == 1)
  {
    return n;
  }
  else
  {
    return n * ipow<n, power - 1>();
  }
}

} // namespace asgard::kronmult
