#include "kronmult_cuda.hpp"
#include "build_info.hpp"

// TODO tidy these includes
#ifdef ASGARD_USE_CUDA
#define USE_GPU
#endif
#include "kronmult1_pbatched.hpp"
#include "kronmult2_pbatched.hpp"
#include "kronmult3_pbatched.hpp"
#include "kronmult4_pbatched.hpp"
#include "kronmult5_pbatched.hpp"
#include "kronmult6_pbatched.hpp"

// helper - call the right kronmult
// --------------------------------------------
// note  the input memory referenced by x_ptrs will be over-written
// --------------------------------------------
template<typename P>
void call_kronmult(int const n, P *x_ptrs[], P *output_ptrs[], P *work_ptrs[],
                   P *operator_ptrs[], int const lda, int const num_krons,
                   int const num_dims)
{
#ifdef ASGARD_USE_CUDA
  {
    P **x_d;
    P **work_d;
    P **output_d;
    P **operators_d;
    auto const list_size = num_krons * sizeof(P *);

    auto stat = cudaMalloc((void **)&x_d, list_size);
    assert(stat == 0);
    stat = cudaMalloc((void **)&work_d, list_size);
    assert(stat == 0);
    stat = cudaMalloc((void **)&output_d, list_size);
    assert(stat == 0);
    stat = cudaMalloc((void **)&operators_d, list_size * num_dims);
    assert(stat == 0);

    stat = cudaMemcpy(x_d, x_ptrs, list_size, cudaMemcpyHostToDevice);
    assert(stat == 0);
    stat = cudaMemcpy(work_d, work_ptrs, list_size, cudaMemcpyHostToDevice);
    assert(stat == 0);
    stat = cudaMemcpy(output_d, output_ptrs, list_size, cudaMemcpyHostToDevice);
    assert(stat == 0);
    stat = cudaMemcpy(operators_d, operators, list_size * num_dims,
                      cudaMemcpyHostToDevice);
    assert(stat == 0);

    int constexpr warpsize    = 32;
    int constexpr nwarps      = 8;
    int constexpr num_threads = nwarps * warpsize;

    switch (num_dims)
    {
    case 1:
      kronmult1_pbatched<P><<<num_krons, num_threads>>>(
          n, lda, operators_d, x_d, output_d, work_d, num_krons);
      break;
    case 2:
      kronmult2_pbatched<P><<<num_krons, num_threads>>>(
          n, lda, operators_d, x_d, output_d, work_d, num_krons);
      break;
    case 3:
      kronmult3_pbatched<P><<<num_krons, num_threads>>>(
          n, lda, operators_d, x_d, output_d, work_d, num_krons);
      break;
    case 4:
      kronmult4_pbatched<P><<<num_krons, num_threads>>>(
          n, lda, operators_d, x_d, output_d, work_d, num_krons);
      break;
    case 5:
      kronmult5_pbatched<P><<<num_krons, num_threads>>>(
          n, lda, operators_d, x_d, output_d, work_d, num_krons);
      break;
    case 6:
      kronmult6_pbatched<P><<<num_krons, num_threads>>>(
          n, lda, operators_d, x_d, output_d, work_d, num_krons);
      break;
    default:
      assert(false);
    };

    // -------------------------------------------
    // note important to wait for kernel to finish
    // -------------------------------------------
    cudaError_t const istat = cudaDeviceSynchronize();
    assert(istat == cudaSuccess);
  }
#else

  {
    switch (num_dims)
    {
    case 1:
      kronmult1_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 2:
      kronmult2_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 3:
      kronmult3_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 4:
      kronmult4_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 5:
      kronmult5_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 6:
      kronmult6_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    default:
      assert(false);
    };
  }
#endif
}

template void call_kronmult(int const n, float const operators[],
                            float *x_ptrs[], float *output_ptrs[],
                            float *work_ptrs[], int const num_krons,
                            int const num_dims);

template void call_kronmult(int const n, double const operators[],
                            double *x_ptrs[], double *output_ptrs[],
                            double *work_ptrs[], int const num_krons,
                            int const num_dims);
