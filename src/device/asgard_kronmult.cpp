#include <iostream>
#include <set>

#include "build_info.hpp"

#ifdef ASGARD_USE_CUDA
#include "asgard_kronmult_cycle1.hpp"
#endif

namespace asgard::kronmult
{
#ifdef ASGARD_USE_CUDA

/*!
 * \brief Run a GPU kernel for the specified problem.
 *
 * Instantiates a GPU kernel, computes the appropriate grid and executes the kernel.
 * \tparam precision is either float or double
 * \tparam dims is the number of dimensions of the tensors
 * \tparam n is the number of degrees of freedom of the tensors,
 *         i.e., polynomial order + 1, linear n=2, quadratic n=3, etc.
 *
 * \param pA pointer array to the matrices associated with the kron products,
 *        the matrices for the i-th entry of the batch are located at
 *        pA[dims * i] ... pA[dims * i + (dims-1)]
 *        where pA[dims * i + (dims-1)] is the last matrix and
 *        is applied in non-transpose format
 * \param lda is the leading dimension of A (TODO: fix the layout so lda is always n)
 * \param pX is the pointer to the input tensors
 * \param pY is the pointer to the output tensors
 * \param num_batch is the number of kron entries in this batch
 */
template<typename precision, int dims, int n>
void run_kernel(precision const *const pA[], int const lda, precision const *const pX[], precision *pY[],
                int const num_batch)
{
  constexpr int max_blocks  = 300;
  constexpr int max_threads = 1024;
  constexpr int team_size = ipow<n, dims>();
  constexpr int num_teams = max_threads / team_size;

  static_assert( max_threads >= team_size, "tensor size must be less than the max number of threads (1024)");

  int num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  if constexpr(dims == 1){
    kernel::case1D<precision, n, team_size, num_teams><<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
  }else if constexpr(n == 1){
    kernel::case1N<precision, dims, max_threads><<<std::min(max_blocks, (num_batch + max_threads-1) /max_threads), max_threads>>>(pA, lda, pX, pY, num_batch);
  }else{
    kernel::cycle1<precision, dims, n, team_size, num_teams><<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
  }
}

template<typename T>
void execute_gpu(int dimensions, int n,
                 T const *const pA[], int const lda, T *pX[], T *pY[],
                 int const num_batch){

  switch(dimensions){
    case 1:
      switch(n){
        case 1:
          run_kernel<T, 1, 1>(pA, lda, pX, pY, num_batch);
          break;
        case 2:
          run_kernel<T, 1, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          run_kernel<T, 1, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          run_kernel<T, 1, 4>(pA, lda, pX, pY, num_batch);
          break;
        case 5:
          run_kernel<T, 1, 5>(pA, lda, pX, pY, num_batch);
          break;
        case 6:
          run_kernel<T, 1, 6>(pA, lda, pX, pY, num_batch);
          break;
        case 7:
          run_kernel<T, 1, 7>(pA, lda, pX, pY, num_batch);
          break;
        case 8:
          run_kernel<T, 1, 8>(pA, lda, pX, pY, num_batch);
          break;
        case 9:
          run_kernel<T, 1, 9>(pA, lda, pX, pY, num_batch);
          break;
        case 10:
          run_kernel<T, 1, 10>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    case 2:
      switch(n){
        case 1:
          run_kernel<T, 2, 1>(pA, lda, pX, pY, num_batch);
          break;
        case 2:
          run_kernel<T, 2, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          run_kernel<T, 2, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          run_kernel<T, 2, 4>(pA, lda, pX, pY, num_batch);
          break;
        case 5:
          run_kernel<T, 2, 5>(pA, lda, pX, pY, num_batch);
          break;
        case 6:
          run_kernel<T, 2, 6>(pA, lda, pX, pY, num_batch);
          break;
        case 7:
          run_kernel<T, 2, 7>(pA, lda, pX, pY, num_batch);
          break;
        case 8:
          run_kernel<T, 2, 8>(pA, lda, pX, pY, num_batch);
          break;
        case 9:
          run_kernel<T, 2, 9>(pA, lda, pX, pY, num_batch);
          break;
        case 10:
          run_kernel<T, 2, 10>(pA, lda, pX, pY, num_batch);
          break;
        case 11:
          run_kernel<T, 2, 11>(pA, lda, pX, pY, num_batch);
          break;
        case 12:
          run_kernel<T, 2, 12>(pA, lda, pX, pY, num_batch);
          break;
        case 13:
          run_kernel<T, 2, 13>(pA, lda, pX, pY, num_batch);
          break;
        case 14:
          run_kernel<T, 2, 14>(pA, lda, pX, pY, num_batch);
          break;
        case 15:
          run_kernel<T, 2, 15>(pA, lda, pX, pY, num_batch);
          break;
        case 16:
          run_kernel<T, 2, 16>(pA, lda, pX, pY, num_batch);
          break;
        case 17:
          run_kernel<T, 2, 17>(pA, lda, pX, pY, num_batch);
          break;
        case 18:
          run_kernel<T, 2, 18>(pA, lda, pX, pY, num_batch);
          break;
        case 19:
          run_kernel<T, 2, 19>(pA, lda, pX, pY, num_batch);
          break;
        case 20:
          run_kernel<T, 2, 20>(pA, lda, pX, pY, num_batch);
          break;
        case 21:
          run_kernel<T, 2, 21>(pA, lda, pX, pY, num_batch);
          break;
        case 22:
          run_kernel<T, 2, 22>(pA, lda, pX, pY, num_batch);
          break;
        case 23:
          run_kernel<T, 2, 23>(pA, lda, pX, pY, num_batch);
          break;
        case 24:
          run_kernel<T, 2, 24>(pA, lda, pX, pY, num_batch);
          break;
        case 25:
          run_kernel<T, 2, 25>(pA, lda, pX, pY, num_batch);
          break;
        case 26:
          run_kernel<T, 2, 26>(pA, lda, pX, pY, num_batch);
          break;
        case 27:
          run_kernel<T, 2, 27>(pA, lda, pX, pY, num_batch);
          break;
        case 28:
          run_kernel<T, 2, 28>(pA, lda, pX, pY, num_batch);
          break;
        case 29:
          run_kernel<T, 2, 29>(pA, lda, pX, pY, num_batch);
          break;
        case 30:
          run_kernel<T, 2, 30>(pA, lda, pX, pY, num_batch);
          break;
        case 31:
          run_kernel<T, 2, 31>(pA, lda, pX, pY, num_batch);
          break;
        case 32:
          run_kernel<T, 2, 32>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    case 3:
      switch(n){
        case 1:
          run_kernel<T, 3, 1>(pA, lda, pX, pY, num_batch);
          break;
        case 2:
          run_kernel<T, 3, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          run_kernel<T, 3, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          run_kernel<T, 3, 4>(pA, lda, pX, pY, num_batch);
          break;
        case 5:
          run_kernel<T, 3, 5>(pA, lda, pX, pY, num_batch);
          break;
        case 6:
          run_kernel<T, 3, 6>(pA, lda, pX, pY, num_batch);
          break;
        case 7:
          run_kernel<T, 3, 7>(pA, lda, pX, pY, num_batch);
          break;
        case 8:
          run_kernel<T, 3, 8>(pA, lda, pX, pY, num_batch);
          break;
        case 9:
          run_kernel<T, 3, 9>(pA, lda, pX, pY, num_batch);
          break;
        case 10:
          run_kernel<T, 3, 10>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    case 4:
      switch(n){
        case 1:
          run_kernel<T, 4, 1>(pA, lda, pX, pY, num_batch);
          break;
        case 2:
          run_kernel<T, 4, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          run_kernel<T, 4, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          run_kernel<T, 4, 4>(pA, lda, pX, pY, num_batch);
          break;
        case 5:
          run_kernel<T, 4, 5>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    case 5:
      switch(n){
        case 1:
          run_kernel<T, 5, 1>(pA, lda, pX, pY, num_batch);
          break;
        case 2:
          run_kernel<T, 5, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          run_kernel<T, 5, 3>(pA, lda, pX, pY, num_batch);
          break;
        case 4:
          run_kernel<T, 5, 4>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    case 6:
      switch(n){
        case 1:
          run_kernel<T, 6, 1>(pA, lda, pX, pY, num_batch);
          break;
        case 2:
          run_kernel<T, 6, 2>(pA, lda, pX, pY, num_batch);
          break;
        case 3:
          run_kernel<T, 6, 3>(pA, lda, pX, pY, num_batch);
          break;
        default:
          throw std::runtime_error("kronmult unimplemented n for the gpu");
      }
      break;
    default:
      throw std::runtime_error("kronmult unimplemented number of dimensions for the gpu " + std::to_string(dimensions));
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
