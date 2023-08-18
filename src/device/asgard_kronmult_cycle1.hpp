#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{
/*!
 * \brief Recursive template that computes n to power, e.g., ipow<2, 3>() returns constexpr 8.
 *
 * This is on-device implementation, not really a device function
 * since it will be evaluated at compile time, but it still
 * had to be marked as __device__
 */
template<int n, int power>
__device__ constexpr int int_pow()
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
                "int_pow() does not works with specified power");
  return 0;
}

/*!
 * \brief Kernel to scale a vector, i.e., y = beta * y
 *
 * \tparam T is float or double
 * \tparam beta_case reflect whether beta is 0, 1, -1, or something else
 *
 * \param num is the size of y
 * \param beta is the scalar
 * \param y is array to be scaled
 */
template<typename T, scalar_case beta_case>
__global__ void scale(int const num, T const beta, T y[])
{
  (void)beta;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < num)
  {
    if constexpr (beta_case == scalar_case::zero)
      y[i] = 0;
    else if constexpr (beta_case == scalar_case::neg_one)
      y[i] = -y[i];
    else if constexpr (beta_case == scalar_case::other)
      y[i] *= beta;

    i += gridDim.x * blockDim.x;
  }
}

/*!
 * \brief Kernel for the n==1 case.
 *
 * The algorithm for n==1, all tensors and matrices are in fact scalars.
 */
template<typename P, int dims, scalar_case alpha_case>
__global__ void
case_n1(int64_t const num_batch, int const num_cols, int const num_terms,
        int const elem[], int const row_offset, int const col_offset,
        P const *const vA[], int const num_1d_blocks, P const alpha,
        P const x[], P y[])
{
  (void)alpha;
  int64_t i = threadIdx.x + blockIdx.x * blockDim.x;

  while (i < num_batch)
  {
    int const rowy = i / num_cols;
    int const colx = i % num_cols;

    P X = x[colx];

    P sum = 0;

    int const *iy = elem + (rowy + row_offset) * dims;
    int const *ix = elem + (colx + col_offset) * dims;

    for (int t = 0; t < num_terms; t++)
    {
      P totalA = vA[t][ix[0] * num_1d_blocks + iy[0]];
      for (int d = 1; d < dims; d++)
        totalA *= vA[t][d * num_1d_blocks * num_1d_blocks +
                        ix[d] * num_1d_blocks + iy[d]];
      sum += totalA * X;
    }

    if constexpr (alpha_case == scalar_case::one)
      atomicAdd(&y[rowy], sum);
    else if constexpr (alpha_case == scalar_case::neg_one)
      atomicAdd(&y[rowy], -sum);
    else
      atomicAdd(&y[rowy], alpha * sum);

    i += gridDim.x * blockDim.x;
  }
}

/*!
 * \brief Kernel for the 1D case.
 *
 * The algorithm for 1D is slightly different,
 * most notably, the data is smaller than the matrix.
 * There is no reuse of the matrix entries,
 * so they are not stored in __shared__ memory.
 */
template<typename P, int n, int team_size, int num_teams,
         scalar_case alpha_case>
__global__ void
case_d1(int64_t const num_batch, int const num_cols, int const num_terms,
        int const elem[], int const row_offset, int const col_offset,
        P const *const vA[], int const num_1d_blocks, P const alpha,
        P const x[], P y[])
{
  // if thread teams span more than one warp, we must synchronize
  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
                                        ? manual_sync::enable
                                        : manual_sync::disable;

  constexpr int effective_team_size = n;

  static_assert(
      team_size <= n,
      "team is too small, size must equal the size of the matrices (n)");

  __shared__ P X[num_teams][team_size];

  int64_t i = threadIdx.y + blockIdx.x * blockDim.y;

  if constexpr (effective_team_size < team_size)
  {
    if (threadIdx.x >= effective_team_size)
      i = num_batch;
  }

  while (i < num_batch)
  {
    int const rowy = i / num_cols;
    int const colx = i % num_cols;

    X[threadIdx.y][threadIdx.x] = x[n * colx + threadIdx.x];
    if constexpr (sync_mode == manual_sync::enable)
      __syncthreads();

    int const ma =
        n * n *
        (elem[colx + col_offset] * num_1d_blocks + elem[rowy + row_offset]);

    P yinc = 0;
    for (int t = 0; t < num_terms; t++)
    {
      P const *A = vA[t] + ma;
      for (int k = 0; k < n; k++)
        yinc += A[k * n + threadIdx.x] * X[threadIdx.y][k];
    }

    if constexpr (alpha_case == scalar_case::one)
      atomicAdd(&y[n * rowy + threadIdx.x], yinc);
    else if constexpr (alpha_case == scalar_case::neg_one)
      atomicAdd(&y[n * rowy + threadIdx.x], -yinc);
    else
      atomicAdd(&y[n * rowy + threadIdx.x], alpha * yinc);

    i += gridDim.x * blockDim.y;

    if constexpr (sync_mode == manual_sync::enable)
      __syncthreads();
  }
}

/*!
 * \brief Kernel using one thread per data entry.
 *
 * \tparam T is float or double
 * \tparam dims is the number of dimensions of the tensors (dims <= 6)
 * \tparam n is the size of the matrices
 * \tparam team_size indicates the number of threads that will work on
 *         a single batch entry; team_size >= n^dims and can be set
 *         larger in order to align the thread team to the warp
 * \tparam num_teams indicates the number of thread teams that will work
 *         in a single thread block; num_teams * team_size = num_threads
 *
 * \b note: This kernel requires that the thread team is at least as
 *    large as the size of tensors (n^dims) and the maximum number
 *    of threads is limited to 1024.
 *    If the tensor size is more than that, each thread will be
 *    responsible for multiple tensor entries and computations will be
 *    done in more than one cycle.
 */
template<typename P, int dims, int n, int team_size, int num_teams,
         scalar_case alpha_case>
__global__ void
cycle1(int64_t const num_batch, int const num_cols, int const num_terms,
       int const elem[], int const row_offset, int const col_offset,
       P const *const vA[], int const num_1d_blocks, P const alpha, P const x[],
       P y[])
{
  (void)alpha;
  // if thread teams span more than one warp, we must synchronize
  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
                                        ? manual_sync::enable
                                        : manual_sync::disable;

  constexpr int effective_team_size = int_pow<n, dims>();

  static_assert(dims <= 6, "kernel won't work for more than 6 dimensions");
  static_assert(effective_team_size <= team_size,
                "team is too small, size must equal the size of the tensors");

  __shared__ P X[num_teams][team_size];
  __shared__ P A[num_teams][team_size];

  int64_t i = threadIdx.y + blockIdx.x * blockDim.y;

  int const ix5 =
      threadIdx.x % int_pow<n, 5>() +
      ((dims == 6) ? 0 : int_pow<n, 6>() * (threadIdx.x / int_pow<n, 6>()));
  int const ia5 = threadIdx.x / int_pow<n, 5>() -
                  ((dims == 6) ? 0 : n * (threadIdx.x / int_pow<n, 6>()));

  int const ix4 =
      threadIdx.x % int_pow<n, 4>() +
      ((dims == 5) ? 0 : int_pow<n, 5>() * (threadIdx.x / int_pow<n, 5>()));
  int const ia4 = threadIdx.x / int_pow<n, 4>() -
                  ((dims == 5) ? 0 : n * (threadIdx.x / int_pow<n, 5>()));

  int const ix3 =
      threadIdx.x % int_pow<n, 3>() +
      ((dims == 4) ? 0 : int_pow<n, 4>() * (threadIdx.x / int_pow<n, 4>()));
  int const ia3 = threadIdx.x / int_pow<n, 3>() -
                  ((dims == 4) ? 0 : n * (threadIdx.x / int_pow<n, 4>()));

  int const ix2 =
      threadIdx.x % int_pow<n, 2>() +
      ((dims == 3) ? 0 : int_pow<n, 3>() * (threadIdx.x / int_pow<n, 3>()));
  int const ia2 = threadIdx.x / int_pow<n, 2>() -
                  ((dims == 3) ? 0 : n * (threadIdx.x / int_pow<n, 3>()));

  int const ix1 =
      threadIdx.x % n +
      ((dims == 2) ? 0 : int_pow<n, 2>() * (threadIdx.x / int_pow<n, 2>()));
  int const ia1 =
      threadIdx.x / n - ((dims == 2) ? 0 : n * (threadIdx.x / int_pow<n, 2>()));

  int const ix0 = n * (threadIdx.x / n);
  int const ia0 = threadIdx.x % n;

  if constexpr (effective_team_size < team_size)
  {
    if (threadIdx.x >= effective_team_size)
      i = num_batch;
  }

#if (CUDART_VERSION < 11070)
  (void)alpha;
  (void)ix5;
  (void)ix4;
  (void)ix3;
  (void)ix2;
  (void)ia5;
  (void)ia4;
  (void)ia3;
  (void)ia2;
#endif

  while (i < num_batch)
  {
    int const rowy = i / num_cols;
    int const colx = i % num_cols;

    int const *iy = elem + (rowy + row_offset) * dims;
    int const *ix = elem + (colx + col_offset) * dims;

    int ii5, ii4, ii3, ii2, ii1, ii0;
#if (CUDART_VERSION < 11070)
    (void)ii5;
    (void)ii4;
    (void)ii3;
    (void)ii2;
#endif
    int ioff = 0;
    if constexpr (dims >= 6)
    {
      ii5 = n * n * ((*ix++) * num_1d_blocks + *iy++);
      ioff += num_1d_blocks * num_1d_blocks * n * n;
    }
    if constexpr (dims >= 5)
    {
      ii4 = ioff + n * n * ((*ix++) * num_1d_blocks + *iy++);
      ioff += num_1d_blocks * num_1d_blocks * n * n;
    }
    if constexpr (dims >= 4)
    {
      ii3 = ioff + n * n * ((*ix++) * num_1d_blocks + *iy++);
      ioff += num_1d_blocks * num_1d_blocks * n * n;
    }
    if constexpr (dims >= 3)
    {
      ii2 = ioff + n * n * ((*ix++) * num_1d_blocks + *iy++);
      ioff += num_1d_blocks * num_1d_blocks * n * n;
    }
    if constexpr (dims >= 2)
    {
      ii1 = ioff + n * n * ((*ix++) * num_1d_blocks + *iy++);
      ioff += num_1d_blocks * num_1d_blocks * n * n;
    }
    ii0 = ioff + n * n * ((*ix++) * num_1d_blocks + *iy++);

    P yinc = 0;
    P rawx = x[int_pow<n, dims>() * colx + threadIdx.x];

    for (int t = 0; t < num_terms; t++)
    {
      X[threadIdx.y][threadIdx.x] = rawx;

      P const *const pA = vA[t];

      if constexpr (dims >= 6)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = pA[ii5 + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        P sum = 0;
        for (int k = 0; k < n; k++)
          sum += X[threadIdx.y][ix5 + k * int_pow<n, 5>()] *
                 A[threadIdx.y][ia5 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum;
      }

      if constexpr (dims >= 5)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = pA[ii4 + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        P sum = 0;
        for (int k = 0; k < n; k++)
          sum += X[threadIdx.y][ix4 + k * int_pow<n, 4>()] *
                 A[threadIdx.y][ia4 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum;
      }

      if constexpr (dims >= 4)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = pA[ii3 + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        P sum = 0;
        for (int k = 0; k < n; k++)
          sum += X[threadIdx.y][ix3 + k * int_pow<n, 3>()] *
                 A[threadIdx.y][ia3 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum;
      }

      if constexpr (dims >= 3)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = pA[ii2 + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        P sum = 0;
        for (int k = 0; k < n; k++)
          sum += X[threadIdx.y][ix2 + k * int_pow<n, 2>()] *
                 A[threadIdx.y][ia2 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum;
      }

      if constexpr (dims >= 2)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = pA[ii1 + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        P sum = 0;
        for (int k = 0; k < n; k++)
          sum += X[threadIdx.y][ix1 + k * n] * A[threadIdx.y][ia1 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum;
      }

      if (threadIdx.x < n * n)
        A[threadIdx.y][threadIdx.x] = pA[ii0 + threadIdx.x];
      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();

      for (int k = 0; k < n; k++)
        yinc += A[threadIdx.y][ia0 + k * n] * X[threadIdx.y][ix0 + k];

      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();
    }

    if constexpr (alpha_case == scalar_case::one)
      atomicAdd(&y[int_pow<n, dims>() * rowy + threadIdx.x], yinc);
    else if constexpr (alpha_case == scalar_case::neg_one)
      atomicAdd(&y[int_pow<n, dims>() * rowy + threadIdx.x], -yinc);
    else
      atomicAdd(&y[int_pow<n, dims>() * rowy + threadIdx.x], alpha * yinc);

    i += gridDim.x * blockDim.y;
  }
}

} // namespace asgard::kronmult::kernel
