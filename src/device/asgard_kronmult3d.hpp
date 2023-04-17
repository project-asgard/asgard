#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{
#ifdef USE_GPU

template<typename T, int num_threads, int n>
__global__ void gpu3d(T const *const pA[], int const lda, T const *const pX[],
                      T *pY[], int const num_batch)
{
  static_assert(n == 2 or n == 3, "kernel works only for n = 2, 3");
  static_assert(n != 3 or (n == 3 and num_threads == 32),
                "restriction on warp size limit this kernel to 32 threads");

  constexpr int data_per_proc = n * n * n;
  constexpr int mat_per_proc  = n * n;

  __shared__ T X[num_threads]; // cache for intermediate values
  __shared__ T A[num_threads];

  // do all integer logic once
  int locali =
      threadIdx.x / data_per_proc; // i is the index of the batch, locali is the
                                   // index within the thread-block
  int i = locali +
          blockIdx.x *
              (num_threads / data_per_proc); // global index within the batch
  int j    = threadIdx.x % data_per_proc;
  int matj = j % n + (j / mat_per_proc) * lda;

  int ix  = data_per_proc * locali;
  int ia2 = ix + j / mat_per_proc + n * ((j / n) % n);
  int iw  = ix + j % n + mat_per_proc * (j / mat_per_proc);
  int ia1 = ix + j / n;
  int iy  = ix + n * (j / n);
  ix += j % mat_per_proc;
  if constexpr (n == 3)
  {
    if (threadIdx.x >= 27)
      i = num_batch;
  }

  while (i < num_batch)
  {
    X[threadIdx.x] = pX[i][j];
    A[threadIdx.x] = pA[3 * i][matj];

    if constexpr (n == 2)
    {
      X[threadIdx.x] = X[ix] * A[ia2] + X[ix + 4] * A[ia2 + 4];
    }
    else
    {
      X[threadIdx.x] =
          X[ix] * A[ia2] + X[ix + 9] * A[ia2 + 9] + X[ix + 18] * A[ia2 + 18];
    }

    A[threadIdx.x] = pA[3 * i + 1][matj];

    if constexpr (n == 2)
    {
      X[threadIdx.x] = X[iw] * A[ia1] + X[iw + 2] * A[ia1 + 4];
    }
    else
    {
      X[threadIdx.x] =
          X[iw] * A[ia1] + X[iw + 3] * A[ia1 + 9] + X[iw + 6] * A[ia1 + 18];
    }

    A[threadIdx.x] = pA[3 * i + 2][matj];

    T yinc;
    if constexpr (n == 2)
    {
      yinc = A[ix] * X[iy] + A[ix + 4] * X[iy + 1];
    }
    else
    {
      yinc = A[ix] * X[iy] + A[ix + 9] * X[iy + 1] + A[ix + 18] * X[iy + 2];
    }

    atomicAdd(&pY[i][j], yinc);

    i += gridDim.x * (num_threads / data_per_proc);
  }
}

// 3d, n = 4 means data has 64 entries, matrices 16
// using one warp per batch entry, processing the data in two cycles
template<typename T, int num_threads>
__global__ void gpu3d_n4(T const *const pA[], int const lda, T const *const pX[],
                         T *pY[], int const num_batch)
{
  __shared__ T X[2][num_threads]; // cache for intermediate values
  __shared__ T W[2][num_threads]; // cache for intermediate values
  __shared__ T A[num_threads];

  // do all integer logic once
  int locali = threadIdx.x / 32;
  int i = locali + blockIdx.x * (num_threads / 32);
  int j = threadIdx.x % 32; // local index within the warp
  int matj = j % 4 + (j / 8) * lda;

  int ix = 32 * locali;
  int ia2 = ix + j / 16 + 4 * ( (j%16) / 8 );
  int iw = ix + j%4 + 16 * ( j/16 );
  int ia1 = ix + (j%16)/4 + 4 * ( j/16 );
  int iy = ix + 4 * ( j/4 );
  int ia0 = ix + j%8;
  ix += threadIdx.x % 16;

  while (i < num_batch)
  {
    X[0][threadIdx.x] = pX[i][j];
    X[1][threadIdx.x] = pX[i][j + 32];
    A[threadIdx.x] = pA[3 * i][matj];

    W[0][threadIdx.x] = X[0][ix] * A[ia2] + X[0][ix + 16] * A[ia2 + 8] + X[1][ix] * A[ia2 + 16] + X[1][ix + 16] * A[ia2 + 24];
    W[1][threadIdx.x] = X[0][ix] * A[ia2+2] + X[0][ix + 16] * A[ia2 + 10] + X[1][ix] * A[ia2 + 18] + X[1][ix + 16] * A[ia2 + 26];

    A[threadIdx.x] = pA[3 * i + 1][matj];

    X[0][threadIdx.x] = W[0][iw] * A[ia1] + W[0][iw + 4] * A[ia1 + 8] + W[0][iw + 8] * A[ia1 + 16] + W[0][iw + 12] * A[ia1 + 24];
    X[1][threadIdx.x] = W[1][iw] * A[ia1] + W[1][iw + 4] * A[ia1 + 8] + W[1][iw + 8] * A[ia1 + 16] + W[1][iw + 12] * A[ia1 + 24];

    A[threadIdx.x] = pA[3 * i + 2][matj];

    T yinc = A[ia0] * X[0][iy] + A[ia0+8] * X[0][iy+1] + A[ia0+16] * X[0][iy+2] + A[ia0+24] * X[0][iy+3];
    atomicAdd(&pY[i][j], yinc);

    yinc = A[ia0] * X[1][iy] + A[ia0+8] * X[1][iy+1] + A[ia0+16] * X[1][iy+2] + A[ia0+24] * X[1][iy+3];
    atomicAdd(&pY[i][j + 32], yinc);

    i += gridDim.x * (num_threads / 32);
  }
}

#endif

} // namespace asgard::kronmult::kernel
