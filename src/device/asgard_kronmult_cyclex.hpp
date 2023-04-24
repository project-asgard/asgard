#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{
/*!
 * \brief Variant of int_pow() that accepts the power as input, but the method is still constexpr.
 */
template<int n>
__device__ constexpr int int_pow(int p)
{
  return (p == 6)
             ? n * n * n * n * n * n
             : ((p == 5) ? n * n * n * n * n
                         : ((p == 4) ? n * n * n * n
                                     : ((p == 3) ? n * n * n
                                                 : ((p == 2) ? n * n : n))));
}

/*!
 * \brief Kernel using one thread per two data entries.
 *
 * Up to 4 cycle algorithm, the internal logic is different from the other
 * cases. The number of used register is reduced by implementing the index logic
 * via for loops, while the regsters are used to keep the intermediate
 * sums from the matrix-vector products.
 *
 * For cycles 1 or 2, use the dedicated algorithms.
 * This should be used only when more cycles are needed and/or the maximum
 * number of threads is not enough to hold a tensor in 1 or 2 cycles.
 */
template<typename T, int dims, int n, int team_size, int num_teams,
         int num_cycles>
__global__ void cyclex(T const *const pA[], int const lda, T const *const pX[],
                       T *pY[], int const num_batch)
{
  static_assert(dims <= 6, "kernel won't work for more than 6 dimensions");
  static_assert(num_cycles <= 4, "supporting up to 4 cycles");
  static_assert(num_cycles >= 1,
                "invalid number of cycles, must be at least 1");
  static_assert(
      num_cycles * team_size >= int_pow<n, dims>(),
      "team is too small, team_size X num_cycles must be at least n^dims");
  static_assert(team_size >= n * n,
                "the team size must be at least equal to the matrix size n^2");

  // if thread teams span more than one warp, we must synchronize
  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
                                        ? manual_sync::enable
                                        : manual_sync::disable;

  constexpr int num_last_cycle =
      int_pow<n, dims>() - (num_cycles - 1) * team_size;
  static_assert(num_last_cycle > 0,
                "too many cycles, the last one is pointless");

  __shared__ T X[num_teams][num_cycles * team_size];
  __shared__ T A[num_teams][team_size];

  int i = threadIdx.y + blockIdx.x * blockDim.y;

  int const matj = threadIdx.x % n + lda * (threadIdx.x / n);

  while (i < num_batch)
  {
    for (int c = 0; c < num_cycles - 1; c++)
    {
      X[threadIdx.y][threadIdx.x + c * team_size] =
          pX[i][threadIdx.x + c * team_size];
    }
    if constexpr (num_last_cycle == team_size)
    {
      X[threadIdx.y][threadIdx.x + (num_cycles - 1) * team_size] =
          pX[i][threadIdx.x + (num_cycles - 1) * team_size];
    }
    else
    {
      if (threadIdx.x < num_last_cycle)
        X[threadIdx.y][threadIdx.x + (num_cycles - 1) * team_size] =
            pX[i][threadIdx.x + (num_cycles - 1) * team_size];
    }

    for (int s = dims - 1; s > 0; s--)
    { // stages
      if (threadIdx.x < n * n)
        A[threadIdx.y][threadIdx.x] = pA[dims * i + dims - s - 1][matj];
      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();

      int ix = threadIdx.x % int_pow<n>(s) +
               int_pow<n>(s + 1) * (threadIdx.x / int_pow<n>(s + 1));
      int ia =
          threadIdx.x / int_pow<n>(s) - n * (threadIdx.x / int_pow<n>(s + 1));

      T sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
      for (int k = 0; k < n; k++)
        sum0 +=
            X[threadIdx.y][ix + k * int_pow<n>(s)] * A[threadIdx.y][ia + k * n];

      if constexpr (num_cycles >= 2)
      {
        if constexpr (num_cycles == 2 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
          {
            ix = (threadIdx.x + team_size) % int_pow<n>(s) +
                 int_pow<n>(s + 1) *
                     ((threadIdx.x + team_size) / int_pow<n>(s + 1));
            ia = (threadIdx.x + team_size) / int_pow<n>(s) -
                 n * ((threadIdx.x + team_size) / int_pow<n, s + 1>());

            for (int k = 0; k < n; k++)
              sum1 += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                      A[threadIdx.y][ia + k * n];
          }
        }
        else
        {
          ix = (threadIdx.x + team_size) % int_pow<n>(s) +
               int_pow<n>(s + 1) *
                   ((threadIdx.x + team_size) / int_pow<n>(s + 1));
          ia = (threadIdx.x + team_size) / int_pow<n>(s) -
               n * ((threadIdx.x + team_size) / int_pow<n>(s + 1));

          for (int k = 0; k < n; k++)
            sum1 += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                    A[threadIdx.y][ia + k * n];
        }
      }

      if constexpr (num_cycles >= 3)
      {
        if constexpr (num_cycles == 3 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
          {
            ix = (threadIdx.x + 2 * team_size) % int_pow<n>(s) +
                 int_pow<n>(s + 1) *
                     ((threadIdx.x + 2 * team_size) / int_pow<n>(s + 1));
            ia = (threadIdx.x + 2 * team_size) / int_pow<n>(s) -
                 n * ((threadIdx.x + 2 * team_size) / int_pow<n>(s + 1));

            for (int k = 0; k < n; k++)
              sum2 += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                      A[threadIdx.y][ia + k * n];
          }
        }
        else
        {
          ix = (threadIdx.x + 2 * team_size) % int_pow<n>(s) +
               int_pow<n>(s + 1) *
                   ((threadIdx.x + 2 * team_size) / int_pow<n>(s + 1));
          ia = (threadIdx.x + 2 * team_size) / int_pow<n>(s) -
               n * ((threadIdx.x + 2 * team_size) / int_pow<n>(s + 1));

          for (int k = 0; k < n; k++)
            sum2 += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                    A[threadIdx.y][ia + k * n];
        }
      }

      if constexpr (num_cycles >= 4)
      {
        if constexpr (num_cycles == 3 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
          {
            ix = (threadIdx.x + 3 * team_size) % int_pow<n>(s) +
                 int_pow<n>(s + 1) *
                     ((threadIdx.x + 3 * team_size) / int_pow<n>(s + 1));
            ia = (threadIdx.x + 3 * team_size) / int_pow<n>(s) -
                 n * ((threadIdx.x + 3 * team_size) / int_pow<n>(s + 1));

            for (int k = 0; k < n; k++)
              sum3 += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                      A[threadIdx.y][ia + k * n];
          }
        }
        else
        {
          ix = (threadIdx.x + 3 * team_size) % int_pow<n>(s) +
               int_pow<n>(s + 1) *
                   ((threadIdx.x + 3 * team_size) / int_pow<n>(s + 1));
          ia = (threadIdx.x + 3 * team_size) / int_pow<n>(s) -
               n * ((threadIdx.x + 3 * team_size) / int_pow<n>(s + 1));

          for (int k = 0; k < n; k++)
            sum3 += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                    A[threadIdx.y][ia + k * n];
        }
      }

      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();

      X[threadIdx.y][threadIdx.x] = sum0;
      if constexpr (num_cycles >= 2)
      {
        if constexpr (num_cycles == 2 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
            X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
        else
        {
          X[threadIdx.y][threadIdx.x + team_size] = sum1;
        }
      }

      if constexpr (num_cycles >= 3)
      {
        if constexpr (num_cycles == 3 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
            X[threadIdx.y][threadIdx.x + 2 * team_size] = sum2;
        }
        else
        {
          X[threadIdx.y][threadIdx.x + 2 * team_size] = sum2;
        }
      }

      if constexpr (num_cycles >= 4)
      {
        if constexpr (num_cycles == 4 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
            X[threadIdx.y][threadIdx.x + 3 * team_size] = sum3;
        }
        else
        {
          X[threadIdx.y][threadIdx.x + 3 * team_size] = sum3;
        }
      }
    }

    if (threadIdx.x < n * n)
      A[threadIdx.y][threadIdx.x] = pA[dims * i + dims - 1][matj];
    if constexpr (sync_mode == manual_sync::enable)
      __syncthreads();

    int ix = n * (threadIdx.x / n);
    int ia = threadIdx.x % n;

    T yinc = 0;
    for (int k = 0; k < n; k++)
      yinc += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
    atomicAdd(&pY[i][threadIdx.x], yinc);

    if constexpr (num_cycles >= 2)
    {
      if constexpr (num_cycles == 2 and num_last_cycle < team_size)
      {
        if (threadIdx.x < num_last_cycle)
        {
          ix   = n * ((threadIdx.x + team_size) / n);
          ia   = (threadIdx.x + team_size) % n;
          yinc = 0;
          for (int k = 0; k < n; k++)
            yinc += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
          atomicAdd(&pY[i][threadIdx.x + team_size], yinc);
        }
      }
      else
      {
        ix   = n * ((threadIdx.x + team_size) / n);
        ia   = (threadIdx.x + team_size) % n;
        yinc = 0;
        for (int k = 0; k < n; k++)
          yinc += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
        atomicAdd(&pY[i][threadIdx.x + team_size], yinc);
      }
    }
    if constexpr (num_cycles >= 3)
    {
      if constexpr (num_cycles == 3 and num_last_cycle < team_size)
      {
        if (threadIdx.x < num_last_cycle)
        {
          ix   = n * ((threadIdx.x + 2 * team_size) / n);
          ia   = (threadIdx.x + 2 * team_size) % n;
          yinc = 0;
          for (int k = 0; k < n; k++)
            yinc += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
          atomicAdd(&pY[i][threadIdx.x + 2 * team_size], yinc);
        }
      }
      else
      {
        ix   = n * ((threadIdx.x + 2 * team_size) / n);
        ia   = (threadIdx.x + 2 * team_size) % n;
        yinc = 0;
        for (int k = 0; k < n; k++)
          yinc += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
        atomicAdd(&pY[i][threadIdx.x + 2 * team_size], yinc);
      }
    }
    if constexpr (num_cycles >= 4)
    {
      if constexpr (num_cycles == 4 and num_last_cycle < team_size)
      {
        if (threadIdx.x < num_last_cycle)
        {
          ix   = n * ((threadIdx.x + 3 * team_size) / n);
          ia   = (threadIdx.x + 3 * team_size) % n;
          yinc = 0;
          for (int k = 0; k < n; k++)
            yinc += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
          atomicAdd(&pY[i][threadIdx.x + 3 * team_size], yinc);
        }
      }
      else
      {
        ix   = n * ((threadIdx.x + 3 * team_size) / n);
        ia   = (threadIdx.x + 3 * team_size) % n;
        yinc = 0;
        for (int k = 0; k < n; k++)
          yinc += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
        atomicAdd(&pY[i][threadIdx.x + 3 * team_size], yinc);
      }
    }

    i += gridDim.x * blockDim.y;

    if constexpr (sync_mode == manual_sync::enable)
      __syncthreads();
  }
}

} // namespace asgard::kronmult::kernel
