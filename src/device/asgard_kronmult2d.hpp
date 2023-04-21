#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{

template<typename T, int num_threads, int n>
__global__ void gpu2d(T const *const pA[], int const lda, T const *const pX[],
                      T *pY[], int const num_batch)
{
  static_assert(n == 2 or n == 3 or n == 4,
                "kernel works only for n = 2, 3, 4");

  constexpr int team_size     = 32;
  constexpr int threads_per_i = n * n;
  constexpr int i_per_block   = (n == 3) ? (3 * (num_threads / team_size))
                                         : (num_threads / threads_per_i);

  __shared__ T X[num_threads]; // cache for intermediate values
  __shared__ T A[num_threads]; // cache for the matrices

  int locali;
  if constexpr (n == 3)
  {
    locali = 3 * (threadIdx.x / team_size) +
             (threadIdx.x % team_size) / threads_per_i;
  }
  else
  {
    locali =
        threadIdx.x / threads_per_i; // i is the index of the batch, locali is
                                     // the index within the thread-block
  }
  int i = locali + blockIdx.x * i_per_block; // global index within the batch
  int j;
  if constexpr (n == 3)
  {
    j = (threadIdx.x % team_size) % threads_per_i;
  }
  else
  {
    j = threadIdx.x % threads_per_i;
  }

  int matj = j % n + (j / n) * lda;

  int ix;
  if constexpr (n == 3)
  {
    ix = team_size * (threadIdx.x / team_size) +
         9 * ((threadIdx.x % team_size) / 9);
  }
  else
  {
    ix = threads_per_i * locali;
  }
  int iat = ix + j / n;
  int ia  = ix + n * (j / n);
  ix += j % n;
  if constexpr (n == 3)
  { // done at compile time since n is a template parameter
    // disable the last two threads of the warp since 32 does not divide into 3
    if (threadIdx.x % 32 >= 27)
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

    i += gridDim.x * i_per_block;
  }
}

template<int n, int power>
__device__ constexpr int int_pow(){
    if constexpr(power == 1){
        return n;
    }else{
        return n * int_pow<n, power-1>();
    }
}

template<typename T, int n, int team_size, int num_teams, manual_sync sync>
__global__ void gpu2d_v2(T const *const pA[], int const lda, T const *const pX[],
                      T *pY[], int const num_batch)
{
    constexpr int effective_team_size = int_pow<n, 2>();

    static_assert(effective_team_size <= team_size, "team is too small, size must equal the size of the tensors");

    __shared__ T X[num_teams][team_size];
    __shared__ T A[num_teams][team_size];

    int i = threadIdx.y + blockIdx.x * blockDim.y;

    int const matj = threadIdx.x % n + lda * ( threadIdx.x / n );

    int const ix1 = threadIdx.x % int_pow<n, 1>();
    int const ia1 = threadIdx.x / int_pow<n, 1>();

    int const ix0 = n * (threadIdx.x / n);
    int const ia0 = threadIdx.x % n;

    if constexpr (effective_team_size < team_size){
        if (threadIdx.x >= effective_team_size)
            i = num_batch;
    }

    while (i < num_batch)
    {
        X[threadIdx.y][threadIdx.x] = pX[i][threadIdx.x];

        if (threadIdx.x < n * n) A[threadIdx.y][threadIdx.x] = pA[2*i][matj];
        if constexpr (sync == manual_sync::enable){  __syncthreads(); }

        if constexpr (n==-2){
            X[threadIdx.y][threadIdx.x] = X[threadIdx.y][ix1] * A[threadIdx.y][ia1]
                                          + X[threadIdx.y][ix1 + n] * A[threadIdx.y][ia1 + n];
        }else if constexpr (n==-3){
            T sum = X[threadIdx.y][ix1] * A[threadIdx.y][ia1]
                    + X[threadIdx.y][ix1 + n] * A[threadIdx.y][ia1 + n]
                      + X[threadIdx.y][ix1 + 2*n] * A[threadIdx.y][ia1 + 2*n];

            if constexpr (sync == manual_sync::enable){ __syncthreads(); }
            X[threadIdx.y][threadIdx.x] = sum;
        }else if constexpr (n==-4){
            X[threadIdx.y][threadIdx.x] = X[threadIdx.y][ix1] * A[threadIdx.y][ia1]
                                          + X[threadIdx.y][ix1 + n] * A[threadIdx.y][ia1 + n]
                                            + X[threadIdx.y][ix1 + 2*n] * A[threadIdx.y][ia1 + 2*n]
                                              + X[threadIdx.y][ix1 + 3*n] * A[threadIdx.y][ia1 + 3*n];
        } else {
            T sum = 0;
            for(int k=0; k<n; k++)
                sum += X[threadIdx.y][ix1 + k * n] * A[threadIdx.y][ia1 + k * n];

            if constexpr (sync == manual_sync::enable){ __syncthreads(); }
            X[threadIdx.y][threadIdx.x] = sum;
        }

        if (threadIdx.x < n * n) A[threadIdx.y][threadIdx.x] = pA[2*i+1][matj];
        if constexpr (sync == manual_sync::enable){ __syncthreads(); }

        T yinc;
        if constexpr (n==-2){
            yinc = A[threadIdx.y][ia0] * X[threadIdx.y][ix0]
                   + A[threadIdx.y][ia0 + n] * X[threadIdx.y][ix0 + 1];
        }else if constexpr (n==-3){
            yinc = A[threadIdx.y][ia0] * X[threadIdx.y][ix0]
                   + A[threadIdx.y][ia0 + n] * X[threadIdx.y][ix0 + 1]
                     + A[threadIdx.y][ia0 + 2*n] * X[threadIdx.y][ix0 + 2];
        }else if constexpr (n==-4){
            yinc = A[threadIdx.y][ia0] * X[threadIdx.y][ix0]
                   + A[threadIdx.y][ia0 + n] * X[threadIdx.y][ix0 + 1]
                     + A[threadIdx.y][ia0 + 2*n] * X[threadIdx.y][ix0 + 2]
                       + A[threadIdx.y][ia0 + 3*n] * X[threadIdx.y][ix0 + 3];
        } else {
            yinc = 0;
            for(int k=0; k<n; k++)
                yinc += A[threadIdx.y][ia0 + k * n] * X[threadIdx.y][ix0 + k];
        }

        atomicAdd(&pY[i][threadIdx.x], yinc);

        i += gridDim.x * blockDim.y;

        if constexpr (sync == manual_sync::enable){  __syncthreads(); }
    }

}

template<typename T, int n, int team_size, int num_teams, manual_sync sync>
__global__ void case1D(T const *const pA[], int const lda, T const *const pX[],
                       T *pY[], int const num_batch)
{
    constexpr int effective_team_size = n;

    static_assert(effective_team_size <= n, "team is too small, size must equal the size of the matrices");

    __shared__ T X[num_teams][team_size];

    int i = threadIdx.y + blockIdx.x * blockDim.y;

    //int const matj = threadIdx.x % n + lda * ( threadIdx.x / n );

//     int const ix0 = n * (threadIdx.x / n);
//     int const ia0 = threadIdx.x % n;

    if constexpr (effective_team_size < team_size){
        if (threadIdx.x >= effective_team_size)
            i = num_batch;
    }

    while (i < num_batch)
    {
        X[threadIdx.y][threadIdx.x] = pX[i][threadIdx.x];
        if constexpr (sync == manual_sync::enable){  __syncthreads(); }

        T yinc = 0;
        for(int k=0; k<n; k++)
            yinc += pA[i][threadIdx.x + k * lda] * X[threadIdx.y][k];

        atomicAdd(&pY[i][threadIdx.x], yinc);

        i += gridDim.x * blockDim.y;

        if constexpr (sync == manual_sync::enable){  __syncthreads(); }
    }

}

template<typename T, int dims, int n, int team_size, int num_teams, manual_sync sync>
__global__ void cycle1(T const *const pA[], int const lda, T const *const pX[],
                       T *pY[], int const num_batch)
{
    constexpr int effective_team_size = int_pow<n, dims>();

    static_assert(dims <= 6, "kernel won't work for more than 5 dimensions");
    static_assert(effective_team_size <= team_size, "team is too small, size must equal the size of the tensors");

    __shared__ T X[num_teams][team_size];
    __shared__ T A[num_teams][team_size];

    int i = threadIdx.y + blockIdx.x * blockDim.y;

    int const matj = threadIdx.x % n + lda * ( threadIdx.x / n );

    int const ix5 = threadIdx.x % int_pow<n, 5>() + ((dims==6) ? 0 : int_pow<n, 6>() * (threadIdx.x / int_pow<n, 6>()));
    int const ia5 = threadIdx.x / int_pow<n, 5>() - ((dims==6) ? 0 : n * (threadIdx.x / int_pow<n, 6>()));

    int const ix4 = threadIdx.x % int_pow<n, 4>() + ((dims==5) ? 0 : int_pow<n, 5>() * (threadIdx.x / int_pow<n, 5>()));
    int const ia4 = threadIdx.x / int_pow<n, 4>() - ((dims==5) ? 0 : n * (threadIdx.x / int_pow<n, 5>()));

    int const ix3 = threadIdx.x % int_pow<n, 3>() + ((dims==4) ? 0 : int_pow<n, 4>() * (threadIdx.x / int_pow<n, 4>()));
    int const ia3 = threadIdx.x / int_pow<n, 3>() - ((dims==4) ? 0 : n * (threadIdx.x / int_pow<n, 4>()));

    int const ix2 = threadIdx.x % int_pow<n, 2>() + ((dims==3) ? 0 : int_pow<n, 3>() * (threadIdx.x / int_pow<n, 3>()));
    int const ia2 = threadIdx.x / int_pow<n, 2>() - ((dims==3) ? 0 : n * (threadIdx.x / int_pow<n, 3>()));

    int const ix1 = threadIdx.x % n + ((dims==2) ? 0 : int_pow<n, 2>() * (threadIdx.x / int_pow<n, 2>()));
    int const ia1 = threadIdx.x / n - ((dims==2) ? 0 : n * (threadIdx.x / int_pow<n, 2>()));

    int const ix0 = n * (threadIdx.x / n);
    int const ia0 = threadIdx.x % n;

    if constexpr (effective_team_size < team_size){
        if (threadIdx.x >= effective_team_size)
            i = num_batch;
    }

    while (i < num_batch)
    {
        X[threadIdx.y][threadIdx.x] = pX[i][threadIdx.x];

        if constexpr(dims >= 6){
          if (threadIdx.x < n * n) A[threadIdx.y][threadIdx.x] = pA[dims*i+dims-6][matj];
          if constexpr (sync == manual_sync::enable){  __syncthreads(); }
          T sum = 0;
          for(int k=0; k<n; k++)
            sum += X[threadIdx.y][ix5 + k * int_pow<n, 5>()] * A[threadIdx.y][ia5 + k * n];

          if constexpr (sync == manual_sync::enable){ __syncthreads(); }
          X[threadIdx.y][threadIdx.x] = sum;
        }

        if constexpr(dims >= 5){
          if (threadIdx.x < n * n) A[threadIdx.y][threadIdx.x] = pA[dims*i+dims-5][matj];
          if constexpr (sync == manual_sync::enable){  __syncthreads(); }
          T sum = 0;
          for(int k=0; k<n; k++)
            sum += X[threadIdx.y][ix4 + k * int_pow<n, 4>()] * A[threadIdx.y][ia4 + k * n];

          if constexpr (sync == manual_sync::enable){ __syncthreads(); }
          X[threadIdx.y][threadIdx.x] = sum;
        }

        if constexpr(dims >= 4){
          if (threadIdx.x < n * n) A[threadIdx.y][threadIdx.x] = pA[dims*i+dims-4][matj];
          if constexpr (sync == manual_sync::enable){  __syncthreads(); }
          T sum = 0;
          for(int k=0; k<n; k++)
            sum += X[threadIdx.y][ix3 + k * int_pow<n, 3>()] * A[threadIdx.y][ia3 + k * n];

          if constexpr (sync == manual_sync::enable){ __syncthreads(); }
          X[threadIdx.y][threadIdx.x] = sum;
        }

        if constexpr(dims >= 3){
          if (threadIdx.x < n * n) A[threadIdx.y][threadIdx.x] = pA[dims*i+dims-3][matj];
          if constexpr (sync == manual_sync::enable){  __syncthreads(); }
          T sum = 0;
          for(int k=0; k<n; k++)
            sum += X[threadIdx.y][ix2 + k * int_pow<n, 2>()] * A[threadIdx.y][ia2 + k * n];

          if constexpr (sync == manual_sync::enable){ __syncthreads(); }
          X[threadIdx.y][threadIdx.x] = sum;
        }

        if constexpr(dims >= 2){
          if (threadIdx.x < n * n) A[threadIdx.y][threadIdx.x] = pA[dims*i+dims-2][matj];
          if constexpr (sync == manual_sync::enable){  __syncthreads(); }
          T sum = 0;
          for(int k=0; k<n; k++)
            sum += X[threadIdx.y][ix1 + k * n] * A[threadIdx.y][ia1 + k * n];

          if constexpr (sync == manual_sync::enable){ __syncthreads(); }
          X[threadIdx.y][threadIdx.x] = sum;
        }

        if (threadIdx.x < n * n) A[threadIdx.y][threadIdx.x] = pA[dims*i+dims-1][matj];
        if constexpr (sync == manual_sync::enable){ __syncthreads(); }

        T yinc = 0;
        for(int k=0; k<n; k++)
            yinc += A[threadIdx.y][ia0 + k * n] * X[threadIdx.y][ix0 + k];

        atomicAdd(&pY[i][threadIdx.x], yinc);

        i += gridDim.x * blockDim.y;

        if constexpr (sync == manual_sync::enable){  __syncthreads(); }
    }

}


} // namespace asgard::kronmult::kernel
