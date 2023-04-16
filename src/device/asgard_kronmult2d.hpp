#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{
#ifdef USE_GPU

template<typename T, int num_threads, int n>
__global__ void gpu2d(T const *const pA[], int const lda, T const *const pX[],
                      T *pY[], int const num_batch)
{
  static_assert(n == 2 or n == 3 or n == 4,
                "kernel works only for n = 2, 3, 4");
  static_assert(n != 3 or (n == 3 and num_threads == 32),
                "restriction on warp size limit this kernel to 32 threads");

  constexpr int threads_per_i = n * n;

  __shared__ T X[num_threads]; // cache for intermediate values
  __shared__ T A[num_threads]; // cache for the matrices

  int locali =
      threadIdx.x / threads_per_i; // i is the index of the batch, locali is the
                                   // index within the thread-block
  int i = locali +
          blockIdx.x *
              (num_threads / threads_per_i); // global index within the batch
  int j    = threadIdx.x % threads_per_i;
  int matj = j % n + (j / n) * lda;

  int ix  = threads_per_i * locali;
  int iat = ix + j / n;
  int ia  = ix + n * (j / n);
  ix += threadIdx.x % n;
  if constexpr (n == 3)
  { // done at compile time since n is a template parameter
    // disable the last two threads of the warp since 32 does not divide into 3
    if (threadIdx.x >= 27)
    {
      i = num_batch;
    }
  }

  while (i < num_batch)
  {
    X[threadIdx.x] = pX[i][j];
    A[threadIdx.x] = pA[2 * i][matj];

    if constexpr (n == 2)
    {
      X[threadIdx.x] = X[ix] * A[iat] + X[ix + 2] * A[iat + 2];
    }
    else if constexpr (n == 3)
    {
      X[threadIdx.x] =
          X[ix] * A[iat] + X[ix + 3] * A[iat + 3] + X[ix + 6] * A[iat + 6];
    }
    else if constexpr (n == 4)
    {
      X[threadIdx.x] = X[ix] * A[iat] + X[ix + 4] * A[iat + 4] +
                       X[ix + 8] * A[iat + 8] + X[ix + 12] * A[iat + 12];
    }

    A[threadIdx.x] = pA[2 * i + 1][matj];

    T yinc;
    if constexpr (n == 2)
    {
      yinc = A[ix] * X[ia] + A[ix + 2] * X[ia + 1];
    }
    else if constexpr (n == 3)
    {
      yinc = A[ix] * X[ia] + A[ix + 3] * X[ia + 1] + A[ix + 6] * X[ia + 2];
    }
    else if constexpr (n == 4)
    {
      yinc = A[ix] * X[ia] + A[ix + 4] * X[ia + 1] + A[ix + 8] * X[ia + 2] +
             A[ix + 12] * X[ia + 3];
    }

    atomicAdd(&pY[i][j], yinc);

    i += gridDim.x * (num_threads / threads_per_i);
  }
}

#endif

} // namespace asgard::kronmult::kernel
