#include "kronmult_cuda.hpp"
#include "build_info.hpp"

// TODO tidy these includes
#ifdef ASGARD_USE_CUDA
#define USE_GPU
#endif
#include "kronmult1_xbatched.hpp"
#include "kronmult2_xbatched.hpp"
#include "kronmult3_xbatched.hpp"
#include "kronmult4_xbatched.hpp"
#include "kronmult5_xbatched.hpp"
#include "kronmult6_xbatched.hpp"

// helper - call the right kronmult
// --------------------------------------------
// note  the input memory referenced by x_ptrs will be over-written
// --------------------------------------------
template<typename P>
void call_kronmult(int const n, P *x_ptrs[], P *output_ptrs[], P *work_ptrs[],
                   P const * const operator_ptrs[], int const lda, int const num_krons,
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
    stat = cudaMemcpy(operators_d, operator_ptrs, list_size * num_dims,
                      cudaMemcpyHostToDevice);
    assert(stat == 0);

    int constexpr warpsize    = 32;
    int constexpr nwarps      = 8;
    int constexpr num_threads = nwarps * warpsize;

    switch (num_dims)
    {
    case 1:
      kronmult1_xbatched<P><<<num_krons, num_threads>>>(
          n, operators_d, lda, x_d, output_d, work_d, num_krons);
      break;
    case 2:
      kronmult2_xbatched<P><<<num_krons, num_threads>>>(
          n, operators_d, lda, x_d, output_d, work_d, num_krons);
      break;
    case 3:
      kronmult3_xbatched<P><<<num_krons, num_threads>>>(
          n, operators_d, lda, x_d, output_d, work_d, num_krons);
      break;
    case 4:
      kronmult4_xbatched<P><<<num_krons, num_threads>>>(
          n, operators_d, lda, x_d, output_d, work_d, num_krons);
      break;
    case 5:
      kronmult5_xbatched<P><<<num_krons, num_threads>>>(
          n, operators_d, lda, x_d, output_d, work_d, num_krons);
      break;
    case 6:
      kronmult6_xbatched<P><<<num_krons, num_threads>>>(
          n, operators_d, lda, x_d, output_d, work_d, num_krons);
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
      kronmult1_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 2:
      kronmult2_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 3:
      kronmult3_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 4:
      kronmult4_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 5:
      kronmult5_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 6:
      kronmult6_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    default:
      assert(false);
    };
  }
#endif
}

template
void call_kronmult(int const n, float *x_ptrs[], float *output_ptrs[], float *work_ptrs[],
                   float const * const operator_ptrs[], int const lda, int const num_krons,
                   int const num_dims);

template
void call_kronmult(int const n, double *x_ptrs[], double *output_ptrs[], double *work_ptrs[],
                   double const * const operator_ptrs[], int const lda, int const num_krons,
                   int const num_dims);
