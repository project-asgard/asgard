#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{
#ifdef ASGARD_USE_CUDA

template<typename T, int num_threads, int n>
__global__ void gpu4d_n2(T const *const pA[], int const lda, T const *const pX[],
                         T *pY[], int const num_batch)
{
  static_assert(n == 2, "kernel works only for n = 2");

  constexpr int wrap_size = 32;
  constexpr int data_size = n * n * n * n;

  static_assert( data_size <= wrap_size, "kernel requires that a wrap has enough threads for the data");

  constexpr int team_size = data_size;
  constexpr int teams_per_wrap = wrap_size / team_size;
  constexpr int wraps_per_block = num_threads / wrap_size;
  constexpr int teams_per_block = teams_per_wrap * wraps_per_block;

  __shared__ T X[num_threads]; // cache for intermediate values
  __shared__ T A[num_threads];

  // j is the thread index within the team
  int const j = ( threadIdx.x % wrap_size ) % data_size;

  // i is the cell index for this team
  // index of this warp is (threadIdx.x / wrap_size)
  // index of thread within this wrap is (threadIdx.x % wrap_size)
  // index of team in this wrap is (threadIdx.x % wrap_size) / team_size
  int i = (threadIdx.x % wrap_size) / team_size
           + (threadIdx.x / wrap_size) * teams_per_wrap
              + blockIdx.x * teams_per_block;

  // matj is the index of the matrices that this thread will read
  // threads for j >= n * n will be masked
  int const matj = j % n + lda * (j / n);


  // 4D implies 4-stages, three transpose operations and one non-transpose
  // stages are counted backwards 3, 2, 1, 0
  // all cycles have the same root, every X and A index is offset from that
  // the local root is the first thread of the team
  int const rooti = wrap_size * (threadIdx.x / wrap_size) + team_size * ((threadIdx.x % wrap_size) / team_size);

  // each cycle has index for x and index for a, i3x means cycle 3 index x
  int const i3x = rooti + j % (n*n*n);
  int const i3a = rooti + j / (n*n*n);
  int const i2x = rooti + j % (n*n) + (n*n*n) * ( j / (n*n*n) );
  int const i2a = rooti + j / (n*n) - n * (j / (n*n*n));
  int const i1x = rooti + j % n + (n*n) * ( j / (n*n) );
  int const i1a = rooti + j / n - n * ( j / (n*n) );

  int const i0x = rooti + j / n;
  int const i0a = rooti + j % n;

  while (i < num_batch)
  {
    X[threadIdx.x] = pX[i][j];
    if (j < n * n) A[threadIdx.x] = pA[4*i][matj];

    X[threadIdx.x] = X[i3x] * A[i3a] + X[i3x + n*n*n] * A[i3a + n];

    if (j < n * n) A[threadIdx.x] = pA[4*i+1][matj];

    X[threadIdx.x] = X[i2x] * A[i2a] + X[i2x + n*n] * A[i2a + n];

    if (j < n * n) A[threadIdx.x] = pA[4*i+2][matj];

    X[threadIdx.x] = X[i1x] * A[i1a] + X[i1x + n] * A[i1a + n];

    if (j < n * n) A[threadIdx.x] = pA[4*i+1][matj];

    T yinc = A[i0a] * X[i0x] + A[i0a + n] * X[i0x + 1];

    atomicAdd(&pY[i][j], yinc);

    i += gridDim.x * teams_per_block;
  }
}

#endif

} // namespace asgard::kronmult::kernel
