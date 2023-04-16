#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel{

#ifdef USE_GPU

template<typename T, int num_threads, int n = 2>
__global__ void gpu3d(T const * const Aarray[],
                      int const lda,
                      T* pX[],
                      T* pY[],
                      int const num_batch){

  //static_assert( n == 2 or n == 3 or n == 4, "kernel works only for n = 2, 3, 4" );
  //static_assert( n != 3 or (n == 3 and num_threads == 32), "restriction on warp size limit this kernel to 32 threads" );

  // data entries have size n^3, but matrices are n^2
  // each thread reads/writes on one entry of either "vector" (x, y)
  // matrices are read redundantly

  __shared__ T X[num_threads]; // cache for intermediate values
  __shared__ T A[num_threads];

  // do all integer logic once
  int locali = threadIdx.x / 8; // i is the index of the batch, locali is the index within the thread-block
  int i = locali + blockIdx.x * num_threads / 8; // global index within the batch
  int j = threadIdx.x % 8;
  int matj = j % 2 + (j / 4) * lda;

  int ix = 8 * locali;
  int ia2 = ix + j/4 + 2 * ( (j/2) % 2 );
  int iw = ix + j%2 + 4 * ( j/4 );
  int ia1 = ix + j/2;
  int iy = ix + 2 * ( j/2 );
  ix += j % 4;

  while(i < num_batch){

    X[threadIdx.x] = pX[i][j];
    A[threadIdx.x] = Aarray[3*i][matj];

    X[threadIdx.x] = X[ix] * A[ia2] + X[ix+4] * A[ia2+4];

    A[threadIdx.x] = Aarray[3*i+1][matj];

    X[threadIdx.x] = X[iw] * A[ia1] + X[iw+2] * A[ia1+4];

    A[threadIdx.x] = Aarray[3*i+2][matj];

    T yinc =  A[ix] * X[iy] + A[ix+4] * X[iy+1];

    atomicAdd(&pY[i][j], yinc);

    i += gridDim.x * num_threads / 8;
  }
}

#endif

}
