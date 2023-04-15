#pragma once

#include <iostream>

#ifdef USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_60_atomic_functions.h>

template<typename T, int num_threads, int n>
__global__ void kernel_kronmult1_xbatched_gpu2(T const * const Aarray_[],
                       int const lda,
                       T* pX_[],
                       T* pY_[],
                       int const batchCount){
  // two threads operate on one entry of the batch

  __shared__ T X[num_threads];

  // do all integer logic once
  int locali = threadIdx.x / n;
  int i = locali + blockIdx.x * num_threads / n; // index within the batch
  int j = threadIdx.x % n; // indicated whether this is an even or odd thread
  int localx0 = n * locali; // the entry of x within the cache
  int localx1 = localx0 + 1;
  int localx2 = localx0 + 2;
  int localx3 = localx0 + 3;

  while(i < batchCount){

    X[threadIdx.x] = pX_[i][j]; // read the X, every 2 threads read consecutive entries and store in cache

    T yinc = (n == 2) ?
        Aarray_[i][j] * X[localx0] + Aarray_[i][lda + j] * X[localx1] :
        Aarray_[i][j] * X[localx0] + Aarray_[i][lda + j] * X[localx1] + Aarray_[i][2*lda + j] * X[localx2] + + Aarray_[i][3*lda + j] * X[localx3];

    atomicAdd(&pY_[i][j], yinc);

    i += gridDim.x * num_threads / n;
  }
}
template<typename T, int num_threads> // assumes we are using 32 threads
__global__ void kernel_kronmult1_xbatched_gpu3(T const * const Aarray_[],
                       int const lda,
                       T* pX_[],
                       T* pY_[],
                       int const batchCount){
  // two threads operate on one entry of the batch

  __shared__ T X[num_threads];

  // do all integer logic once
  int locali = threadIdx.x / 3;
  int i = locali + blockIdx.x * 10; // index within the batch
  int j = threadIdx.x % 3; // indicate the local index
  int localx0 = 3 * locali; // the index of entries within the cache
  int localx1 = localx0 + 1;
  int localx2 = localx0 + 2;
  if (threadIdx.x >= 30){
    i = batchCount;
  }

  while(i < batchCount){

    X[threadIdx.x] = pX_[i][j]; // read the X, every 3 threads read consecutive entries and store in cache

    T yinc = Aarray_[i][j] * X[localx0] + Aarray_[i][lda + j] * X[localx1] + Aarray_[i][2*lda+j] * X[localx2];

    if (threadIdx.x < 30) atomicAdd(&pY_[i][j], yinc);

    i += gridDim.x * 10;
  }
}

template<typename T, int n>
void kronmult1_xbatched_gpu(T const * const Aarray_[],
                            int const lda,
                            T* pX_[],
                            T* pY_[],
                            int const batchCount){
  if constexpr (n == 2){
    int constexpr num_threads = 1024;
    int num_blocks = std::min( 65535, (batchCount + num_threads / 2 + 1) / (num_threads / 2) ); // one operation takes two threads
    kernel_kronmult1_xbatched_gpu2<T, num_threads, 2><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, batchCount);
  }else if constexpr (n == 3){
    int constexpr num_threads = 32;
    int num_blocks = std::min( 65535, (batchCount + 11) / 10 ); // one operation takes two threads
    kernel_kronmult1_xbatched_gpu3<T, num_threads><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, batchCount);
  }else if constexpr (n == 4){
    int constexpr num_threads = 1024;
    int num_blocks = std::min( 65535, (batchCount + num_threads / 4 + 1) / (num_threads / 4) ); // one operation takes two threads
    kernel_kronmult1_xbatched_gpu2<T, num_threads, 4><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, batchCount);
  }else{
    static_assert( (n>=2) and (n<=4), "unimplemented size n (i.e., polynomial degree)");
  }
}



#endif


template<typename T, int n>
void kronmult1_xbatched_cpu(
                       T const * const Aarray_[],
                       int const lda,
                       T* pX_[],
                       T* pY_[],
                       int const batchCount)
{
  // CPU version, this is just Y = A * X in basic 2x2 matrix
  //#pragma omp parallel for
  for(int i=0; i<batchCount; i++){
    if constexpr (n == 2){
      pY_[i][0] += Aarray_[i][0] * pX_[i][0] + Aarray_[i][lda] * pX_[i][1];
      pY_[i][1] += Aarray_[i][1] * pX_[i][0] + Aarray_[i][lda+1] * pX_[i][1];
    }else if constexpr (n==3){
      pY_[i][0] += Aarray_[i][0] * pX_[i][0] + Aarray_[i][lda] * pX_[i][1] + Aarray_[i][2*lda] * pX_[i][2];
      pY_[i][1] += Aarray_[i][1] * pX_[i][0] + Aarray_[i][lda+1] * pX_[i][1] + Aarray_[i][2*lda+1] * pX_[i][2];
      pY_[i][2] += Aarray_[i][2] * pX_[i][0] + Aarray_[i][lda+2] * pX_[i][1] + Aarray_[i][2*lda+2] * pX_[i][2];
    }else if constexpr (n==4){
      pY_[i][0] += Aarray_[i][0] * pX_[i][0] + Aarray_[i][lda] * pX_[i][1] + Aarray_[i][2*lda] * pX_[i][2] + Aarray_[i][3*lda] * pX_[i][3];
      pY_[i][1] += Aarray_[i][1] * pX_[i][0] + Aarray_[i][lda+1] * pX_[i][1] + Aarray_[i][2*lda+1] * pX_[i][2] + Aarray_[i][3*lda+1] * pX_[i][3];
      pY_[i][2] += Aarray_[i][2] * pX_[i][0] + Aarray_[i][lda+2] * pX_[i][1] + Aarray_[i][2*lda+2] * pX_[i][2] + Aarray_[i][3*lda+2] * pX_[i][3];
      pY_[i][3] += Aarray_[i][3] * pX_[i][0] + Aarray_[i][lda+3] * pX_[i][1] + Aarray_[i][2*lda+3] * pX_[i][2] + Aarray_[i][3*lda+3] * pX_[i][3];
    }else{
      static_assert( (n>=2) and (n<=4), "unimplemented size n (i.e., polynomial degree)");
    }
  }
}


template<typename T, int n>
void kronmult2_xbatched_cpu(
                       T const * const Aarray_[],
                       int const lda,
                       T* pX_[],
                       T* pY_[],
                       int const batchCount){

  #define inline_kmult2_cpu3_nn( row )  ( A1[(row)] * w0 + A1[lda + (row)] * w1 + A1[2*lda + (row)] * w2 )

  // algorithm is basic, A1 * ( X * transpose(A2) )
  // construct column of X * transpose(A2) and multiply by A1, do this column by column
  for(int i=0; i<batchCount; i++){
    T const * const A2 = Aarray_[2*i];
    T const * const A1 = Aarray_[2*i+1]; // regular matrix multiplication is always on A1
    if constexpr (n == 2){
        T w0 = pX_[i][0] * A2[0] + pX_[i][2] * A2[lda];
        T w1 = pX_[i][1] * A2[0] + pX_[i][3] * A2[lda];
        pY_[i][0] += A1[0] * w0 + A1[lda] * w1;
        pY_[i][1] += A1[1] * w0 + A1[lda+1] * w1;
        w0 = pX_[i][0] * A2[1] + pX_[i][2] * A2[lda+1];
        w1 = pX_[i][1] * A2[1] + pX_[i][3] * A2[lda+1];
        pY_[i][2] += A1[0] * w0 + A1[lda] * w1;
        pY_[i][3] += A1[1] * w0 + A1[lda+1] * w1;
    } else if constexpr (n == 3){
        T w0 = pX_[i][0] * A2[0] + pX_[i][3] * A2[lda] + pX_[i][6] * A2[2*lda];
        T w1 = pX_[i][1] * A2[0] + pX_[i][4] * A2[lda] + pX_[i][7] * A2[2*lda];
        T w2 = pX_[i][2] * A2[0] + pX_[i][5] * A2[lda] + pX_[i][8] * A2[2*lda];
        pY_[i][0] += inline_kmult2_cpu3_nn(0);
        pY_[i][1] += inline_kmult2_cpu3_nn(1);
        pY_[i][2] += inline_kmult2_cpu3_nn(2);
        w0 = pX_[i][0] * A2[1] + pX_[i][3] * A2[lda+1] + pX_[i][6] * A2[2*lda+1];
        w1 = pX_[i][1] * A2[1] + pX_[i][4] * A2[lda+1] + pX_[i][7] * A2[2*lda+1];
        w2 = pX_[i][2] * A2[1] + pX_[i][5] * A2[lda+1] + pX_[i][8] * A2[2*lda+1];
        pY_[i][3] += inline_kmult2_cpu3_nn(0);
        pY_[i][4] += inline_kmult2_cpu3_nn(1);
        pY_[i][5] += inline_kmult2_cpu3_nn(2);
        w0 = pX_[i][0] * A2[2] + pX_[i][3] * A2[lda+2] + pX_[i][6] * A2[2*lda+2];
        w1 = pX_[i][1] * A2[2] + pX_[i][4] * A2[lda+2] + pX_[i][7] * A2[2*lda+2];
        w2 = pX_[i][2] * A2[2] + pX_[i][5] * A2[lda+2] + pX_[i][8] * A2[2*lda+2];
        pY_[i][6] += inline_kmult2_cpu3_nn(0);
        pY_[i][7] += inline_kmult2_cpu3_nn(1);
        pY_[i][8] += inline_kmult2_cpu3_nn(2);
    } else {
        static_assert( (n>=2) and (n<=3), "unimplemented size n (i.e., polynomial degree)" );
    }
  }
}
