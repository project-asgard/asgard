#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{
/*!
 * \brief Kernel using one thread per two data entries.
 *
 * The team size is about n^dims / 2 which means that the tensor data
 * is processed in two cycles, increasing the compute per thread
 * and avoiding the situation when team_size is much bigger than
 * the matrix size n^2 and many threads have to wait for I/O.
 */
template<typename T, int dims, int n, int team_size, int num_teams,
         scalar_case alpha_case>
__global__ void
cycle2(int const num_batch, int const ix[], int const iy[], int const num_terms,
       int const iA[], T const vA[], T const alpha, T const x[], T y[])
{
  (void)alpha;
  static_assert(dims <= 6, "kernel won't work for more than 6 dimensions");
  static_assert(
      2 * team_size >= int_pow<n, dims>(),
      "team is too small, size must be at least half the size of the tensors");
  static_assert(team_size >= n * n,
                "the team size must be at least equal to the matrix size n^2");

  // if thread teams span more than one warp, we must synchronize
  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
                                        ? manual_sync::enable
                                        : manual_sync::disable;

  constexpr int num_second_cycle = int_pow<n, dims>() - team_size;
  static_assert(num_second_cycle > 0,
                "team is large enough for one cycle, no need for a second one");

  __shared__ T X[num_teams][2 * team_size];
  __shared__ T A[num_teams][team_size];

  int i = threadIdx.y + blockIdx.x * blockDim.y;

  int const ix50 =
      threadIdx.x % int_pow<n, 5>() +
      ((dims == 6) ? 0 : int_pow<n, 6>() * (threadIdx.x / int_pow<n, 6>()));
  int const ia50 = threadIdx.x / int_pow<n, 5>() -
                   ((dims == 6) ? 0 : n * (threadIdx.x / int_pow<n, 6>()));

  int const ix51 =
      (threadIdx.x + team_size) % int_pow<n, 5>() +
      ((dims == 6)
           ? 0
           : int_pow<n, 6>() * ((threadIdx.x + team_size) / int_pow<n, 6>()));
  int const ia51 =
      (threadIdx.x + team_size) / int_pow<n, 5>() -
      ((dims == 6) ? 0 : n * ((threadIdx.x + team_size) / int_pow<n, 6>()));

  int const ix40 =
      threadIdx.x % int_pow<n, 4>() +
      ((dims == 5) ? 0 : int_pow<n, 5>() * (threadIdx.x / int_pow<n, 5>()));
  int const ia40 = threadIdx.x / int_pow<n, 4>() -
                   ((dims == 5) ? 0 : n * (threadIdx.x / int_pow<n, 5>()));

  int const ix41 =
      (threadIdx.x + team_size) % int_pow<n, 4>() +
      ((dims == 5)
           ? 0
           : int_pow<n, 5>() * ((threadIdx.x + team_size) / int_pow<n, 5>()));
  int const ia41 =
      (threadIdx.x + team_size) / int_pow<n, 4>() -
      ((dims == 5) ? 0 : n * ((threadIdx.x + team_size) / int_pow<n, 5>()));

  int const ix30 =
      threadIdx.x % int_pow<n, 3>() +
      ((dims == 4) ? 0 : int_pow<n, 4>() * (threadIdx.x / int_pow<n, 4>()));
  int const ia30 = threadIdx.x / int_pow<n, 3>() -
                   ((dims == 4) ? 0 : n * (threadIdx.x / int_pow<n, 4>()));

  int const ix31 =
      (threadIdx.x + team_size) % int_pow<n, 3>() +
      ((dims == 4)
           ? 0
           : int_pow<n, 4>() * ((threadIdx.x + team_size) / int_pow<n, 4>()));
  int const ia31 =
      (threadIdx.x + team_size) / int_pow<n, 3>() -
      ((dims == 4) ? 0 : n * ((threadIdx.x + team_size) / int_pow<n, 4>()));

  int const ix20 =
      threadIdx.x % int_pow<n, 2>() +
      ((dims == 3) ? 0 : int_pow<n, 3>() * (threadIdx.x / int_pow<n, 3>()));
  int const ia20 = threadIdx.x / int_pow<n, 2>() -
                   ((dims == 3) ? 0 : n * (threadIdx.x / int_pow<n, 3>()));

  int const ix21 =
      (threadIdx.x + team_size) % int_pow<n, 2>() +
      ((dims == 3)
           ? 0
           : int_pow<n, 3>() * ((threadIdx.x + team_size) / int_pow<n, 3>()));
  int const ia21 =
      (threadIdx.x + team_size) / int_pow<n, 2>() -
      ((dims == 3) ? 0 : n * ((threadIdx.x + team_size) / int_pow<n, 3>()));

  int const ix10 =
      threadIdx.x % n +
      ((dims == 2) ? 0 : int_pow<n, 2>() * (threadIdx.x / int_pow<n, 2>()));
  int const ia10 =
      threadIdx.x / n - ((dims == 2) ? 0 : n * (threadIdx.x / int_pow<n, 2>()));

  int const ix11 =
      (threadIdx.x + team_size) % n +
      ((dims == 2)
           ? 0
           : int_pow<n, 2>() * ((threadIdx.x + team_size) / int_pow<n, 2>()));
  int const ia11 =
      (threadIdx.x + team_size) / n -
      ((dims == 2) ? 0 : n * ((threadIdx.x + team_size) / int_pow<n, 2>()));

  int const ix00 = n * (threadIdx.x / n);
  int const ia00 = threadIdx.x % n;

  int const ix01 = n * ((threadIdx.x + team_size) / n);
  int const ia01 = (threadIdx.x + team_size) % n;

#if (CUDART_VERSION < 11070)
  (void)ix50;
  (void)ix40;
  (void)ix30;
  (void)ix20;
  (void)ia50;
  (void)ia40;
  (void)ia30;
  (void)ia20;
  (void)ix51;
  (void)ix41;
  (void)ix31;
  (void)ix21;
  (void)ia51;
  (void)ia41;
  (void)ia31;
  (void)ia21;
#endif

  while (i < num_batch)
  {
    T yinc0 = 0;
    T yinc1 = 0;
    int ma  = i * num_terms * dims;

    T rawx0 = x[ix[i] + threadIdx.x];
    T rawx1 = 0;
    if constexpr (num_second_cycle == team_size)
    {
      rawx1 = x[ix[i] + threadIdx.x + team_size];
    }
    else
    {
      if (threadIdx.x < num_second_cycle)
        rawx1 = x[ix[i] + threadIdx.x + team_size];
    }

    for (int t = 0; t < num_terms; t++)
    {
      X[threadIdx.y][threadIdx.x]             = rawx0;
      X[threadIdx.y][threadIdx.x + team_size] = rawx1;

      if constexpr (dims >= 6)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = vA[iA[ma++] + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        T sum0 = 0;
        for (int k = 0; k < n; k++)
          sum0 += X[threadIdx.y][ix50 + k * int_pow<n, 5>()] *
                  A[threadIdx.y][ia50 + k * n];

        T sum1 = 0;
        for (int k = 0; k < n; k++)
          sum1 += X[threadIdx.y][ix51 + k * int_pow<n, 5>()] *
                  A[threadIdx.y][ia51 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum0;
        if constexpr (num_second_cycle == team_size)
        {
          X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
        else
        {
          if (threadIdx.x < num_second_cycle)
            X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
      }

      if constexpr (dims >= 5)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = vA[iA[ma++] + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        T sum0 = 0;
        for (int k = 0; k < n; k++)
          sum0 += X[threadIdx.y][ix40 + k * int_pow<n, 4>()] *
                  A[threadIdx.y][ia40 + k * n];

        T sum1 = 0;
        for (int k = 0; k < n; k++)
          sum1 += X[threadIdx.y][ix41 + k * int_pow<n, 4>()] *
                  A[threadIdx.y][ia41 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum0;
        if constexpr (num_second_cycle == team_size)
        {
          X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
        else
        {
          if (threadIdx.x < num_second_cycle)
            X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
      }

      if constexpr (dims >= 4)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = vA[iA[ma++] + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        T sum0 = 0;
        for (int k = 0; k < n; k++)
          sum0 += X[threadIdx.y][ix30 + k * int_pow<n, 3>()] *
                  A[threadIdx.y][ia30 + k * n];

        T sum1 = 0;
        for (int k = 0; k < n; k++)
          sum1 += X[threadIdx.y][ix31 + k * int_pow<n, 3>()] *
                  A[threadIdx.y][ia31 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum0;
        if constexpr (num_second_cycle == team_size)
        {
          X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
        else
        {
          if (threadIdx.x < num_second_cycle)
            X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
      }

      if constexpr (dims >= 3)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = vA[iA[ma++] + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        T sum0 = 0;
        for (int k = 0; k < n; k++)
          sum0 += X[threadIdx.y][ix20 + k * int_pow<n, 2>()] *
                  A[threadIdx.y][ia20 + k * n];

        T sum1 = 0;
        for (int k = 0; k < n; k++)
          sum1 += X[threadIdx.y][ix21 + k * int_pow<n, 2>()] *
                  A[threadIdx.y][ia21 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum0;
        if constexpr (num_second_cycle == team_size)
        {
          X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
        else
        {
          if (threadIdx.x < num_second_cycle)
            X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
      }

      if constexpr (dims >= 2)
      {
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = vA[iA[ma++] + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        T sum0 = 0;
        for (int k = 0; k < n; k++)
          sum0 += X[threadIdx.y][ix10 + k * n] * A[threadIdx.y][ia10 + k * n];

        T sum1 = 0;
        for (int k = 0; k < n; k++)
          sum1 += X[threadIdx.y][ix11 + k * n] * A[threadIdx.y][ia11 + k * n];

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        X[threadIdx.y][threadIdx.x] = sum0;
        if constexpr (num_second_cycle == team_size)
        {
          X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
        else
        {
          if (threadIdx.x < num_second_cycle)
            X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
      }

      if (threadIdx.x < n * n)
        A[threadIdx.y][threadIdx.x] = vA[iA[ma++] + threadIdx.x];
      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();

      for (int k = 0; k < n; k++)
        yinc0 += A[threadIdx.y][ia00 + k * n] * X[threadIdx.y][ix00 + k];

      for (int k = 0; k < n; k++)
        yinc1 += A[threadIdx.y][ia01 + k * n] * X[threadIdx.y][ix01 + k];

      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();
    }

    if constexpr (alpha_case == scalar_case::one)
    {
      atomicAdd(&y[iy[i] + threadIdx.x], yinc0);
      if constexpr (num_second_cycle == team_size)
      {
        atomicAdd(&y[iy[i] + threadIdx.x + team_size], yinc1);
      }
      else
      {
        if (threadIdx.x < num_second_cycle)
          atomicAdd(&y[iy[i] + threadIdx.x + team_size], yinc1);
      }
    }
    else if constexpr (alpha_case == scalar_case::neg_one)
    {
      atomicAdd(&y[iy[i] + threadIdx.x], -yinc0);
      if constexpr (num_second_cycle == team_size)
      {
        atomicAdd(&y[iy[i] + threadIdx.x + team_size], -yinc1);
      }
      else
      {
        if (threadIdx.x < num_second_cycle)
          atomicAdd(&y[iy[i] + threadIdx.x + team_size], -yinc1);
      }
    }
    else
    {
      atomicAdd(&y[iy[i] + threadIdx.x], alpha * yinc0);
      if constexpr (num_second_cycle == team_size)
      {
        atomicAdd(&y[iy[i] + threadIdx.x + team_size], alpha * yinc1);
      }
      else
      {
        if (threadIdx.x < num_second_cycle)
          atomicAdd(&y[iy[i] + threadIdx.x + team_size], alpha * yinc1);
      }
    }

    i += gridDim.x * blockDim.y;
  }
}

} // namespace asgard::kronmult::kernel
