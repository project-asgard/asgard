#include <iostream>
#include <set>

#include "build_info.hpp"

#ifdef ASGARD_USE_CUDA
#include "asgard_kronmult1d.hpp"
#include "asgard_kronmult2d.hpp"
#include "asgard_kronmult3d.hpp"
#include "asgard_kronmult4d.hpp"
#endif

namespace asgard::kronmult
{
#ifdef ASGARD_USE_CUDA

template<typename T, int n>
void gpu1d(T const *const pA[], int const lda, T const *const pX[], T *pY[],
           int const num_batch)
{
  static_assert(n == 2 or n == 3 or n == 4,
                "unimplemented size n (i.e., polynomial degree)");

  constexpr int max_blocks =
      300; // we want enough blocks to saturate the GPU, but note that each
           // block repeats some integer ops.

  constexpr int num_threads = 1024;
  constexpr int batch_per_block =
      (n == 3) ? 10 * num_threads / 32 : num_threads / n;

  int num_blocks = blocks(num_batch, batch_per_block, max_blocks);

  kernel::gpu1d<T, num_threads, n>
      <<<num_blocks, num_threads>>>(pA, lda, pX, pY, num_batch);
}

template<typename T, int n>
void gpu2d(T const *const pA[], int const lda, T const *const pX[], T *pY[],
           int const num_batch)
{
  constexpr int warp_size   = ASGARD_GPU_WARP_SIZE;
  constexpr int max_blocks  = 300;
  constexpr int max_threads = 1024;
  constexpr int team_size = n * n;
  constexpr int num_teams = max_threads / team_size;

  static_assert( max_threads >= team_size, "tensor size must be less than the max number of threads (1024)");

  constexpr manual_sync sync_mode = (team_size > warp_size or warp_size % team_size != 0) ? manual_sync::enable : manual_sync::disable;

  int num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  //kernel::gpu2d_v2<T, n, team_size, num_teams, sync_mode><<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
  kernel::cycle1<T, 2, n, team_size, num_teams, sync_mode><<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
}

template<typename T, int n>
void gpu3d(T const *const pA[], int const lda, T const *const pX[], T *pY[],
           int const num_batch)
{
//   static_assert(n == 2 or n == 3 or n == 4,
//                 "unimplemented size n (i.e., polynomial degree)");
//
//   constexpr int max_blocks      = 300;
//   constexpr int num_threads     = 1024;
//   constexpr int batch_per_block = num_threads / ((n == 2) ? 8 : 32);
//
//   int num_blocks = blocks(num_batch, batch_per_block, max_blocks);
//
//   if constexpr (n == 2 or n == 3)
//   {
//     kernel::gpu3d<T, num_threads, n>
//         <<<num_blocks, num_threads>>>(pA, lda, pX, pY, num_batch);
//   }
//   else if constexpr (n == 4)
//   {
//     kernel::gpu3d_n4<T, num_threads>
//         <<<num_blocks, num_threads>>>(pA, lda, pX, pY, num_batch);
//   }

  constexpr int warp_size   = ASGARD_GPU_WARP_SIZE;
  constexpr int max_blocks  = 300;
  constexpr int max_threads = 1024;
  constexpr int team_size = n * n * n;
  constexpr int num_teams = max_threads / team_size;

  static_assert( max_threads >= team_size, "tensor size must be less than the max number of threads (1024)");

  constexpr manual_sync sync_mode = (team_size > warp_size or warp_size % team_size != 0) ? manual_sync::enable : manual_sync::disable;

  int num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  //kernel::gpu2d_v2<T, n, team_size, num_teams, sync_mode><<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
  kernel::cycle1<T, 3, n, team_size, num_teams, sync_mode><<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);

}

template<typename T, int n>
void gpu4d(T const *const pA[], int const lda, T const *const pX[], T *pY[],
           int const num_batch)
{
  static_assert(n == 2 or n == 3 or n == 4,
                "unimplemented size n (i.e., polynomial degree)");

  constexpr int max_blocks      = 300;
  constexpr int num_threads     = 1024;
  constexpr int batch_per_block = num_threads / ((n == 2) ? 8 : 32);

  int num_blocks = blocks(num_batch, batch_per_block, max_blocks);

//  std::cerr << " calling her \n";
  kernel::gpu4d_n2<T, num_threads, n>
        <<<num_blocks, num_threads>>>(pA, lda, pX, pY, num_batch);

//   if constexpr (n == 2 or n == 3)
//   {
//     kernel::gpu3d<T, num_threads, n>
//         <<<num_blocks, num_threads>>>(pA, lda, pX, pY, num_batch);
//   }
//   else if constexpr (n == 4)
//   {
//     kernel::gpu3d_n4<T, num_threads>
//         <<<num_blocks, num_threads>>>(pA, lda, pX, pY, num_batch);
//   }
}


template<typename T>
void execute_gpu(int dimensions, int n,
                 T const *const pA[], int const lda, T *pX[], T *pY[],
                 int const num_batch){

  switch(dimensions){
    case 1:
      switch(n){
        case 2:
          gpu1d<T, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          gpu1d<T, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          gpu1d<T, 4>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    case 2:
      switch(n){
        case 2:
          gpu2d<T, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          gpu2d<T, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          gpu2d<T, 4>(pA, lda, pX, pY, num_batch);
          break;
        case 5:
          gpu2d<T, 5>(pA, lda, pX, pY, num_batch);
          break;
        case 6:
          gpu2d<T, 6>(pA, lda, pX, pY, num_batch);
          break;
        case 7:
          gpu2d<T, 7>(pA, lda, pX, pY, num_batch);
          break;
        case 8:
          gpu2d<T, 8>(pA, lda, pX, pY, num_batch);
          break;
        case 9:
          gpu2d<T, 9>(pA, lda, pX, pY, num_batch);
          break;
        case 10:
          gpu2d<T, 10>(pA, lda, pX, pY, num_batch);
          break;
        case 11:
          gpu2d<T, 11>(pA, lda, pX, pY, num_batch);
          break;
        case 12:
          gpu2d<T, 12>(pA, lda, pX, pY, num_batch);
          break;
        case 13:
          gpu2d<T, 13>(pA, lda, pX, pY, num_batch);
          break;
        case 14:
          gpu2d<T, 14>(pA, lda, pX, pY, num_batch);
          break;
        case 15:
          gpu2d<T, 15>(pA, lda, pX, pY, num_batch);
          break;
        case 16:
          gpu2d<T, 16>(pA, lda, pX, pY, num_batch);
          break;
        case 17:
          gpu2d<T, 17>(pA, lda, pX, pY, num_batch);
          break;
        case 18:
          gpu2d<T, 18>(pA, lda, pX, pY, num_batch);
          break;
        case 19:
          gpu2d<T, 19>(pA, lda, pX, pY, num_batch);
          break;
        case 20:
          gpu2d<T, 20>(pA, lda, pX, pY, num_batch);
          break;
        case 21:
          gpu2d<T, 21>(pA, lda, pX, pY, num_batch);
          break;
        case 22:
          gpu2d<T, 22>(pA, lda, pX, pY, num_batch);
          break;
        case 23:
          gpu2d<T, 23>(pA, lda, pX, pY, num_batch);
          break;
        case 24:
          gpu2d<T, 24>(pA, lda, pX, pY, num_batch);
          break;
        case 25:
          gpu2d<T, 25>(pA, lda, pX, pY, num_batch);
          break;
        case 26:
          gpu2d<T, 26>(pA, lda, pX, pY, num_batch);
          break;
        case 27:
          gpu2d<T, 27>(pA, lda, pX, pY, num_batch);
          break;
        case 28:
          gpu2d<T, 28>(pA, lda, pX, pY, num_batch);
          break;
        case 29:
          gpu2d<T, 29>(pA, lda, pX, pY, num_batch);
          break;
        case 30:
          gpu2d<T, 30>(pA, lda, pX, pY, num_batch);
          break;
        case 31:
          gpu2d<T, 31>(pA, lda, pX, pY, num_batch);
          break;
        case 32:
          gpu2d<T, 32>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    case 3:
      switch(n){
        case 2:
          gpu3d<T, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          gpu3d<T, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          gpu3d<T, 4>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    case 4:
      switch(n){
        case 2:
          gpu4d<T, 2>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    default:
      throw std::runtime_error("kronmult unimplemented number of dimensions for the gpu");
  }
}

template void execute_gpu<float>(int, int, float const *const[], int const, float *[], float *[], int const);
template void execute_gpu<double>(int, int, double const *const[], int const, double *[], double *[], int const);

#endif

template<typename T>
inline void omp_atomic_add(T *p, T inc_value)
{
#pragma omp atomic
  (*p) += inc_value;
}

/*
 * The CPU kernels can be better. The main issue is the atomic operations and
 * matrix lda, but we could speed this up with some intrinsics too.
 */
template<typename T, int n>
void cpu1d(T const *const Aarray_[], int const lda, T *pX_[], T *pY_[],
           int const num_batch)
{
// CPU version, this is just Y = A * X in basic 2x2 matrix
#pragma omp parallel for
  for (int i = 0; i < num_batch; i++)
  {
    if constexpr (n == 2)
    {
      omp_atomic_add(&pY_[i][0],
                     Aarray_[i][0] * pX_[i][0] + Aarray_[i][lda] * pX_[i][1]);
      omp_atomic_add(&pY_[i][1], Aarray_[i][1] * pX_[i][0] +
                                     Aarray_[i][lda + 1] * pX_[i][1]);
    }
    else if constexpr (n == 3)
    {
      omp_atomic_add(&pY_[i][0], Aarray_[i][0] * pX_[i][0] +
                                     Aarray_[i][lda] * pX_[i][1] +
                                     Aarray_[i][2 * lda] * pX_[i][2]);
      omp_atomic_add(&pY_[i][1], Aarray_[i][1] * pX_[i][0] +
                                     Aarray_[i][lda + 1] * pX_[i][1] +
                                     Aarray_[i][2 * lda + 1] * pX_[i][2]);
      omp_atomic_add(&pY_[i][2], Aarray_[i][2] * pX_[i][0] +
                                     Aarray_[i][lda + 2] * pX_[i][1] +
                                     Aarray_[i][2 * lda + 2] * pX_[i][2]);
    }
    else if constexpr (n == 4)
    {
      omp_atomic_add(&pY_[i][0], Aarray_[i][0] * pX_[i][0] +
                                     Aarray_[i][lda] * pX_[i][1] +
                                     Aarray_[i][2 * lda] * pX_[i][2] +
                                     Aarray_[i][3 * lda] * pX_[i][3]);
     omp_atomic_add(&pY_[i][1], Aarray_[i][1] * pX_[i][0] +
                                     Aarray_[i][lda + 1] * pX_[i][1] +
                                     Aarray_[i][2 * lda + 1] * pX_[i][2] +
                                     Aarray_[i][3 * lda + 1] * pX_[i][3]);
      omp_atomic_add(&pY_[i][2], Aarray_[i][2] * pX_[i][0] +
                                     Aarray_[i][lda + 2] * pX_[i][1] +
                                     Aarray_[i][2 * lda + 2] * pX_[i][2] +
                                     Aarray_[i][3 * lda + 2] * pX_[i][3]);
      omp_atomic_add(&pY_[i][3], Aarray_[i][3] * pX_[i][0] +
                                     Aarray_[i][lda + 3] * pX_[i][1] +
                                     Aarray_[i][2 * lda + 3] * pX_[i][2] +
                                     Aarray_[i][3 * lda + 3] * pX_[i][3]);
    }
    else
    {
      static_assert((n >= 2) and (n <= 4),
                    "unimplemented size n (i.e., polynomial degree)");
    }
  }
}

template<typename T, int n>
void cpu2d(T const *const Aarray_[], int const lda, T *pX_[], T *pY_[],
           int const num_batch)
{
#define inline_kmult2_cpu3_nn(row) \
  (A1[(row)] * w0 + A1[lda + (row)] * w1 + A1[2 * lda + (row)] * w2)

// algorithm is basic, A1 * ( X * transpose(A2) )
// construct column of X * transpose(A2) and multiply by A1, do this column by
// column
#pragma omp parallel for
  for (int i = 0; i < num_batch; i++)
  {
    T const *const A2 = Aarray_[2 * i];
    T const *const A1 =
        Aarray_[2 * i + 1]; // regular matrix multiplication is always on A1
    if constexpr (n == 2)
    {
      T w0 = pX_[i][0] * A2[0] + pX_[i][2] * A2[lda];
      T w1 = pX_[i][1] * A2[0] + pX_[i][3] * A2[lda];
      omp_atomic_add(&pY_[i][0], A1[0] * w0 + A1[lda] * w1);
      omp_atomic_add(&pY_[i][1], A1[1] * w0 + A1[lda + 1] * w1);
      w0 = pX_[i][0] * A2[1] + pX_[i][2] * A2[lda + 1];
      w1 = pX_[i][1] * A2[1] + pX_[i][3] * A2[lda + 1];
      omp_atomic_add(&pY_[i][2], A1[0] * w0 + A1[lda] * w1);
      omp_atomic_add(&pY_[i][3], A1[1] * w0 + A1[lda + 1] * w1);
    }
    else if constexpr (n == 3)
    {
      T w0 = pX_[i][0] * A2[0] + pX_[i][3] * A2[lda] + pX_[i][6] * A2[2 * lda];
      T w1 = pX_[i][1] * A2[0] + pX_[i][4] * A2[lda] + pX_[i][7] * A2[2 * lda];
      T w2 = pX_[i][2] * A2[0] + pX_[i][5] * A2[lda] + pX_[i][8] * A2[2 * lda];
      omp_atomic_add(&pY_[i][0], inline_kmult2_cpu3_nn(0));
      omp_atomic_add(&pY_[i][1], inline_kmult2_cpu3_nn(1));
      omp_atomic_add(&pY_[i][2], inline_kmult2_cpu3_nn(2));
      w0 = pX_[i][0] * A2[1] + pX_[i][3] * A2[lda + 1] +
           pX_[i][6] * A2[2 * lda + 1];
      w1 = pX_[i][1] * A2[1] + pX_[i][4] * A2[lda + 1] +
           pX_[i][7] * A2[2 * lda + 1];
      w2 = pX_[i][2] * A2[1] + pX_[i][5] * A2[lda + 1] +
           pX_[i][8] * A2[2 * lda + 1];
      omp_atomic_add(&pY_[i][3], inline_kmult2_cpu3_nn(0));
      omp_atomic_add(&pY_[i][4], inline_kmult2_cpu3_nn(1));
      omp_atomic_add(&pY_[i][5], inline_kmult2_cpu3_nn(2));
      w0 = pX_[i][0] * A2[2] + pX_[i][3] * A2[lda + 2] +
           pX_[i][6] * A2[2 * lda + 2];
      w1 = pX_[i][1] * A2[2] + pX_[i][4] * A2[lda + 2] +
           pX_[i][7] * A2[2 * lda + 2];
      w2 = pX_[i][2] * A2[2] + pX_[i][5] * A2[lda + 2] +
           pX_[i][8] * A2[2 * lda + 2];
      omp_atomic_add(&pY_[i][6], inline_kmult2_cpu3_nn(0));
      omp_atomic_add(&pY_[i][7], inline_kmult2_cpu3_nn(1));
      omp_atomic_add(&pY_[i][8], inline_kmult2_cpu3_nn(2));
    }
    else
    {
      static_assert((n >= 2) and (n <= 3),
                    "unimplemented size n (i.e., polynomial degree)");
    }
  }
}

template<typename T>
void execute_cpu(int dimensions, int n,
                 T const *const pA[], int const lda, T *pX[], T *pY[],
                 int const num_batch){

  switch(dimensions){
    case 1:
      switch(n){
        case 2:
          cpu1d<T, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          cpu1d<T, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          cpu1d<T, 4>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the cpu");
      }
      break;
    case 2:
      switch(n){
        case 2:
          cpu2d<T, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          cpu2d<T, 3>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the cpu");
      }
      break;
    default:
      throw std::runtime_error("kronmult unimplemented number of dimensions for the cpu");
  }
}

template void execute_cpu<float>(int, int, float const *const[], int const, float *[], float *[], int const);
template void execute_cpu<double>(int, int, double const *const[], int const, double *[], double *[], int const);

} // namespace asgard::kronmult
