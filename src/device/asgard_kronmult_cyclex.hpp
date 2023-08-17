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

template<typename T, int dims, int n, int team_size, int num_teams,
         int num_cycles, scalar_case alpha_case>
__global__ void
cyclex(int const num_batch, int const num_cols, int const num_terms,
       int const iA[], T const vA[], T const alpha, T const x[], T y[])
{
#if (CUDART_VERSION < 11070)
  (void)alpha;
#endif
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

  while (i < num_batch)
  {
    int ma = i * num_terms * dims;

    const int xoffset = int_pow<n, dims>() * (i % num_cols);
    T rawx0, rawx1, rawx2, rawx3;
    rawx0 = x[xoffset + threadIdx.x];
    if constexpr (num_cycles >= 2)
      if constexpr (num_cycles == 2 and num_last_cycle == team_size)
        rawx1 = x[xoffset + threadIdx.x + team_size];
      else if (threadIdx.x < num_last_cycle)
        rawx1 = x[xoffset + threadIdx.x + team_size];
    if constexpr (num_cycles >= 3)
      if constexpr (num_cycles == 3 and num_last_cycle == team_size)
        rawx2 = x[xoffset + threadIdx.x + 2 * team_size];
      else if (threadIdx.x < num_last_cycle)
        rawx2 = x[xoffset + threadIdx.x + 2 * team_size];
    if constexpr (num_cycles >= 4)
      if constexpr (num_cycles == 4 and num_last_cycle == team_size)
        rawx3 = x[xoffset + threadIdx.x + 3 * team_size];
      else if (threadIdx.x < num_last_cycle)
        rawx3 = x[xoffset + threadIdx.x + 3 * team_size];

    T yinc0 = 0, yinc1 = 0, yinc2 = 0, yinc3 = 0;

    for (int t = 0; t < num_terms; t++)
    {
      X[threadIdx.y][threadIdx.x] = rawx0;

      if constexpr (num_cycles >= 2)
        if constexpr (num_cycles == 2 and num_last_cycle == team_size)
          X[threadIdx.y][threadIdx.x + team_size] = rawx1;
        else if (threadIdx.x < num_last_cycle)
          X[threadIdx.y][threadIdx.x + team_size] = rawx1;

      if constexpr (num_cycles >= 3)
        if constexpr (num_cycles == 3 and num_last_cycle == team_size)
          X[threadIdx.y][threadIdx.x + 2 * team_size] = rawx2;
        else if (threadIdx.x < num_last_cycle)
          X[threadIdx.y][threadIdx.x + 2 * team_size] = rawx2;

      if constexpr (num_cycles >= 4)
        if constexpr (num_cycles == 4 and num_last_cycle == team_size)
          X[threadIdx.y][threadIdx.x + 3 * team_size] = rawx3;
        else if (threadIdx.x < num_last_cycle)
          X[threadIdx.y][threadIdx.x + 3 * team_size] = rawx3;

      for (int s = dims - 1; s > 0; s--)
      { // stages
        if (threadIdx.x < n * n)
          A[threadIdx.y][threadIdx.x] = vA[iA[ma++] + threadIdx.x];
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        int ix = threadIdx.x % int_pow<n>(s) +
                 int_pow<n>(s + 1) * (threadIdx.x / int_pow<n>(s + 1));
        int ia =
            threadIdx.x / int_pow<n>(s) - n * (threadIdx.x / int_pow<n>(s + 1));

        T sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int k = 0; k < n; k++)
          sum0 += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                  A[threadIdx.y][ia + k * n];

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
        A[threadIdx.y][threadIdx.x] = vA[iA[ma++] + threadIdx.x];
      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();

      int ix = n * (threadIdx.x / n);
      int ia = threadIdx.x % n;

      for (int k = 0; k < n; k++)
        yinc0 += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];

      if constexpr (num_cycles >= 2)
        if constexpr (num_cycles == 2 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
          {
            ix = n * ((threadIdx.x + team_size) / n);
            ia = (threadIdx.x + team_size) % n;
            for (int k = 0; k < n; k++)
              yinc1 += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
          }
        }
        else
        {
          ix = n * ((threadIdx.x + team_size) / n);
          ia = (threadIdx.x + team_size) % n;
          for (int k = 0; k < n; k++)
            yinc1 += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
        }

      if constexpr (num_cycles >= 3)
        if constexpr (num_cycles == 3 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
          {
            ix = n * ((threadIdx.x + 2 * team_size) / n);
            ia = (threadIdx.x + 2 * team_size) % n;
            for (int k = 0; k < n; k++)
              yinc2 += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
          }
        }
        else
        {
          ix = n * ((threadIdx.x + 2 * team_size) / n);
          ia = (threadIdx.x + 2 * team_size) % n;
          for (int k = 0; k < n; k++)
            yinc2 += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
        }

      if constexpr (num_cycles >= 4)
        if constexpr (num_cycles == 4 and num_last_cycle < team_size)
        {
          if (threadIdx.x < num_last_cycle)
          {
            ix = n * ((threadIdx.x + 3 * team_size) / n);
            ia = (threadIdx.x + 3 * team_size) % n;
            for (int k = 0; k < n; k++)
              yinc3 += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
          }
        }
        else
        {
          ix = n * ((threadIdx.x + 3 * team_size) / n);
          ia = (threadIdx.x + 3 * team_size) % n;
          for (int k = 0; k < n; k++)
            yinc3 += A[threadIdx.y][ia + k * n] * X[threadIdx.y][ix + k];
        }

      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();
    } // end terms

    const int yoffset = int_pow<n, dims>() * (i / num_cols);

    if constexpr (alpha_case == scalar_case::one)
    {
      atomicAdd(&y[yoffset + threadIdx.x], yinc0);

      if constexpr (num_cycles >= 2)
        if constexpr (num_cycles == 2 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + team_size], yinc1);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + team_size], yinc1);

      if constexpr (num_cycles >= 3)
        if constexpr (num_cycles == 3 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + 2 * team_size], yinc2);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + 2 * team_size], yinc2);

      if constexpr (num_cycles >= 4)
        if constexpr (num_cycles == 4 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + 3 * team_size], yinc3);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + 3 * team_size], yinc3);
    }
    else if constexpr (alpha_case == scalar_case::neg_one)
    {
      atomicAdd(&y[yoffset + threadIdx.x], -yinc0);

      if constexpr (num_cycles >= 2)
        if constexpr (num_cycles == 2 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + team_size], -yinc1);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + team_size], -yinc1);

      if constexpr (num_cycles >= 3)
        if constexpr (num_cycles == 3 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + 2 * team_size], -yinc2);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + 2 * team_size], -yinc2);

      if constexpr (num_cycles >= 4)
        if constexpr (num_cycles == 4 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + 3 * team_size], -yinc3);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + 3 * team_size], -yinc3);
    }
    else
    {
      atomicAdd(&y[yoffset + threadIdx.x], alpha * yinc0);

      if constexpr (num_cycles >= 2)
        if constexpr (num_cycles == 2 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + team_size], alpha * yinc1);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + team_size], alpha * yinc1);

      if constexpr (num_cycles >= 3)
        if constexpr (num_cycles == 3 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + 2 * team_size], alpha * yinc2);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + 2 * team_size], alpha * yinc2);

      if constexpr (num_cycles >= 4)
        if constexpr (num_cycles == 4 and num_last_cycle == team_size)
          atomicAdd(&y[yoffset + threadIdx.x + 3 * team_size], alpha * yinc3);
        else if (threadIdx.x < num_last_cycle)
          atomicAdd(&y[yoffset + threadIdx.x + 3 * team_size], alpha * yinc3);
    }

    i += gridDim.x * blockDim.y;
  }
}


template<typename P, int dims, int n, int team_size, int num_teams,
         int num_cycles, scalar_case alpha_case>
__global__ void
cyclex(int const num_batch, int const num_cols, int const num_terms,
       int const elem[], int const row_offset, int const col_offset,
       P const * const vA[], int const num_1d_blocks, P const alpha,
       P const x[], P y[])
{
#if (CUDART_VERSION < 11070)
  (void)alpha;
#endif
  static_assert(dims <= 6, "kernel won't work for more than 6 dimensions");
  static_assert(num_cycles <= 8, "supporting up to 4 cycles");
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

  __shared__ P X[num_teams][num_cycles * team_size];
  __shared__ P A[num_teams][n * n];
  __shared__ P rawx[num_teams][team_size][num_cycles];
  __shared__ P yinc[num_teams][team_size][num_cycles];
  __shared__ P sums[num_teams][team_size][num_cycles];  
  


  int i = threadIdx.y + blockIdx.x * blockDim.y;

  while (i < num_batch)
  {
    int const rowy = i / num_cols;
    int const colx = i % num_cols;

    const int xoffset = int_pow<n, dims>() * (i % num_cols);
    for(int c=0; c<num_cycles-1; c++)
      rawx[threadIdx.y][threadIdx.x][c] = x[xoffset + threadIdx.x + c * team_size];
    if (threadIdx.x < num_last_cycle)
      rawx[threadIdx.y][threadIdx.x][num_cycles - 1] = x[xoffset + threadIdx.x + (num_cycles - 1) * team_size];

    //P rawx0, rawx1, rawx2, rawx3;
    //rawx0 = x[xoffset + threadIdx.x];
    //if constexpr (num_cycles >= 2)
    //  if constexpr (num_cycles == 2 and num_last_cycle == team_size)
    //    rawx1 = x[xoffset + threadIdx.x + team_size];
    //  else if (threadIdx.x < num_last_cycle)
    //    rawx1 = x[xoffset + threadIdx.x + team_size];
    //if constexpr (num_cycles >= 3)
    //  if constexpr (num_cycles == 3 and num_last_cycle == team_size)
    //    rawx2 = x[xoffset + threadIdx.x + 2 * team_size];
    //  else if (threadIdx.x < num_last_cycle)
    //    rawx2 = x[xoffset + threadIdx.x + 2 * team_size];
    //if constexpr (num_cycles >= 4)
    //  if constexpr (num_cycles == 4 and num_last_cycle == team_size)
    //    rawx3 = x[xoffset + threadIdx.x + 3 * team_size];
    //  else if (threadIdx.x < num_last_cycle)
    //    rawx3 = x[xoffset + threadIdx.x + 3 * team_size];

    //P yinc0 = 0, yinc1 = 0, yinc2 = 0, yinc3 = 0;
    //yinc = { 0 };
    for(int c=0; c<num_cycles; c++)
          yinc[threadIdx.y][threadIdx.x][c] = P{0};

    for (int t = 0; t < num_terms; t++)
    {
      //X[threadIdx.y][threadIdx.x] = rawx0;
      //
      //if constexpr (num_cycles >= 2)
      //  if constexpr (num_cycles == 2 and num_last_cycle == team_size)
      //    X[threadIdx.y][threadIdx.x + team_size] = rawx1;
      //  else if (threadIdx.x < num_last_cycle)
      //    X[threadIdx.y][threadIdx.x + team_size] = rawx1;
      //
      //if constexpr (num_cycles >= 3)
      //  if constexpr (num_cycles == 3 and num_last_cycle == team_size)
      //    X[threadIdx.y][threadIdx.x + 2 * team_size] = rawx2;
      //  else if (threadIdx.x < num_last_cycle)
      //    X[threadIdx.y][threadIdx.x + 2 * team_size] = rawx2;
      //
      //if constexpr (num_cycles >= 4)
      //  if constexpr (num_cycles == 4 and num_last_cycle == team_size)
      //    X[threadIdx.y][threadIdx.x + 3 * team_size] = rawx3;
      //  else if (threadIdx.x < num_last_cycle)
      //    X[threadIdx.y][threadIdx.x + 3 * team_size] = rawx3;
      for(int c=0; c<num_cycles; c++)
        X[threadIdx.y][threadIdx.x + c * team_size] = rawx[threadIdx.y][threadIdx.x][c];


      int const *iy = elem + (rowy + row_offset) * dims;
      int const *ix = elem + (colx + col_offset) * dims;

      int ma_stride = 0;
      int ma = n * n * ( (*ix) * num_1d_blocks + *iy );

      for (int s = dims - 1; s > 0; s--)
      { // stages
        if (threadIdx.x < n * n)
        {
          A[threadIdx.y][threadIdx.x] = vA[t][ma + threadIdx.x];
          ma_stride += num_1d_blocks * num_1d_blocks * n * n;
          ma = ma_stride + n * n * ( *(++ix) * num_1d_blocks + *(++iy));
        }
        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        int ix = threadIdx.x % int_pow<n>(s) +
                 int_pow<n>(s + 1) * (threadIdx.x / int_pow<n>(s + 1));
        int ia =
            threadIdx.x / int_pow<n>(s) - n * (threadIdx.x / int_pow<n>(s + 1));

        for(int c=0; c<num_cycles; c++)
          sums[threadIdx.y][threadIdx.x][c] = P{0};

        for(int c=0; c<num_cycles-1; c++)
        {
          int ix = (threadIdx.x + i * team_size) % int_pow<n>(s) +
                   int_pow<n>(s + 1) *
                       ((threadIdx.x + i * team_size) / int_pow<n>(s + 1));
          int ia = (threadIdx.x + i * team_size) / int_pow<n>(s) -
                   n * ((threadIdx.x + i * team_size) / int_pow<n>(s+1));

          for (int k = 0; k < n; k++)
            sums[threadIdx.y][threadIdx.x][c] += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                         A[threadIdx.y][ia + k * n];
        }
        if (threadIdx.x < num_last_cycle)
        {
          int ix = (threadIdx.x + (num_cycles-1) * team_size) % int_pow<n>(s) +
                   int_pow<n>(s + 1) *
                       ((threadIdx.x + (num_cycles-1) * team_size) / int_pow<n>(s + 1));
          int ia = (threadIdx.x + (num_cycles-1) * team_size) / int_pow<n>(s) -
                   n * ((threadIdx.x + (num_cycles-1) * team_size) / int_pow<n>(s + 1));

          for (int k = 0; k < n; k++)
            sums[threadIdx.y][threadIdx.x][num_cycles-1] += X[threadIdx.y][ix + k * int_pow<n>(s)] *
                         A[threadIdx.y][ia + k * n];
        }

        if constexpr (sync_mode == manual_sync::enable)
          __syncthreads();

        for(int c=0; c<num_cycles; c++)
          X[threadIdx.y][threadIdx.x + c * team_size] = sums[threadIdx.y][threadIdx.x][c];
      } // stages loop

      if (threadIdx.x < n * n)
        A[threadIdx.y][threadIdx.x] = vA[t][ma + threadIdx.x];
      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();

      for(int c=0; c<num_cycles-1; c++)
      {
        int idx = n * ((threadIdx.x + c * team_size) / n);
        int ia = (threadIdx.x + c * team_size) % n;
        for (int k = 0; k < n; k++)
          yinc[threadIdx.y][threadIdx.x][c] += A[threadIdx.y][ia + k * n] * X[threadIdx.y][idx + k];
      }
      if (threadIdx.x < num_last_cycle)
      {
        int idx = n * ((threadIdx.x + (num_cycles-1) * team_size) / n);
        int ia = (threadIdx.x + (num_cycles-1) * team_size) % n;
        for (int k = 0; k < n; k++)
          yinc[threadIdx.y][threadIdx.x][num_cycles-1] += A[threadIdx.y][ia + k * n] * X[threadIdx.y][idx + k];
      }

      if constexpr (sync_mode == manual_sync::enable)
        __syncthreads();
    } // end terms

    const int yoffset = int_pow<n, dims>() * (i / num_cols);

    if constexpr (alpha_case == scalar_case::one)
    {
      for(int c=0; c<num_cycles-1; c++)
        atomicAdd(&y[yoffset + threadIdx.x + c * team_size], yinc[threadIdx.y][threadIdx.x][c]);
      if (threadIdx.x < num_last_cycle)
        atomicAdd(&y[yoffset + threadIdx.x + (num_cycles - 1) * team_size], yinc[threadIdx.y][threadIdx.x][num_cycles - 1]);
    }
    else if constexpr (alpha_case == scalar_case::neg_one)
    {
      for(int c=0; c<num_cycles-1; c++)
        atomicAdd(&y[yoffset + threadIdx.x + c * team_size], -yinc[threadIdx.y][threadIdx.x][c]);
      if (threadIdx.x < num_last_cycle)
        atomicAdd(&y[yoffset + threadIdx.x + (num_cycles - 1) * team_size], -yinc[threadIdx.y][threadIdx.x][num_cycles - 1]);
    }
    else
    {
      for(int c=0; c<num_cycles-1; c++)
        atomicAdd(&y[yoffset + threadIdx.x + c * team_size], alpha * yinc[threadIdx.y][threadIdx.x][c]);
      if (threadIdx.x < num_last_cycle)
        atomicAdd(&y[yoffset + threadIdx.x + (num_cycles - 1) * team_size], alpha * yinc[threadIdx.y][threadIdx.x][num_cycles - 1]);
    }

    i += gridDim.x * blockDim.y;
  }
}


} // namespace asgard::kronmult::kernel
