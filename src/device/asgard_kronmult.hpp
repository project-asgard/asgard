#pragma once

#include <iostream>

#include "asgard_kronmult1d.hpp"
#include "asgard_kronmult2d.hpp"
#include "asgard_kronmult3d.hpp"

namespace asgard::kronmult{

#ifdef USE_GPU

template<typename T, int n>
void gpu1d(T const * const Aarray_[],
                            int const lda,
                            T* pX_[],
                            T* pY_[],
                            int const num_batch){
  constexpr int max_blocks = 300; // we want enough blocks to saturate the GPU, but note that each block repeats some integer ops.
  if constexpr (n == 2){
    int constexpr num_threads = 1024;
    int num_blocks = std::min( max_blocks, (num_batch + num_threads / 2 - 1) / (num_threads / 2) ); // one operation takes two threads
    kernel::gpu1d<T, num_threads, 2><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, num_batch);
  }else if constexpr (n == 3){
    int constexpr num_threads = 32;
    int num_blocks = std::min( max_blocks, (num_batch + 9) / 10 ); // one operation takes two threads
    kernel::gpu1d<T, num_threads, 3><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, num_batch);
  }else if constexpr (n == 4){
    int constexpr num_threads = 1024;
    int num_blocks = std::min( max_blocks, (num_batch + num_threads / 4 - 1) / (num_threads / 4) );
    kernel::gpu1d<T, num_threads, 4><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, num_batch);
  }else{
    static_assert( (n==2) or (n==3) or (n==4), "unimplemented size n (i.e., polynomial degree)");
  }
}

template<typename T, int n>
void gpu2d(T const * const Aarray_[],
           int const lda,
           T* pX_[],
           T* pY_[],
           int const num_batch){
  constexpr int max_blocks = 300;
  if constexpr (n == 2){
    int constexpr num_threads = 1024;
    int num_blocks = std::min( max_blocks, (num_batch + num_threads / 4 - 1) / (num_threads / 4) ); // one operation takes two threads
    kernel::gpu2d<T, num_threads, 2><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, num_batch);
  }else if constexpr (n == 3){
    int constexpr num_threads = 32;
    int num_blocks = std::min( max_blocks, (num_batch + num_threads / 9 - 1) / (num_threads / 9) ); // one operation takes two threads
    kernel::gpu2d<T, num_threads, 3><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, num_batch);
  }else if constexpr (n == 4){
    int constexpr num_threads = 1024;
    int num_blocks = std::min( max_blocks, (num_batch + num_threads / 16 - 1) / (num_threads / 16) ); // one operation takes two threads
    kernel::gpu2d<T, num_threads, 4><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, num_batch);
  }else{
    static_assert( (n>=2) and (n<=4), "unimplemented size n (i.e., polynomial degree)");
  }
}

template<typename T, int n>
void gpu3d(T const * const Aarray_[],
           int const lda,
           T* pX_[],
           T* pY_[],
           int const batchCount){
  constexpr int max_blocks = 300;
  if constexpr (n == 2){
    int constexpr num_threads = 1024;
    int num_blocks = std::min( max_blocks, (batchCount + num_threads / 8 - 1) / (num_threads / 8) ); // one operation takes two threads
    kernel::gpu3d<T, num_threads><<<num_blocks, num_threads>>>(Aarray_, lda, pX_, pY_, batchCount);
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

}
