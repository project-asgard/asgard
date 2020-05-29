#include "kronmult_cuda.hpp"
#include "build_info.hpp"
#include <iostream>
#ifdef ASGARD_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#define USE_GPU
#endif
// todo merge
#ifdef ASGARD_USE_CUDA
#define GLOBAL_FUNCTION __global__
#define SYNCTHREADS __syncthreads()
#define SHARED_MEMORY __shared__
#define DEVICE_FUNCTION __device__
#define HOST_FUNCTION __host__
#else
#define GLOBAL_FUNCTION
#define SYNCTHREADS
#define SHARED_MEMORY
#define DEVICE_FUNCTION
#define HOST_FUNCTION
#endif

#include "kronmult1_xbatched.hpp"
#include "kronmult2_xbatched.hpp"
#include "kronmult3_xbatched.hpp"
#include "kronmult4_xbatched.hpp"
#include "kronmult5_xbatched.hpp"
#include "kronmult6_xbatched.hpp"

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

template<typename P>
GLOBAL_FUNCTION void
stage_inputs_kronmult_kernel(P const *const x, P *const workspace,
                             int const num_elems, int const num_copies)
{
#ifdef ASGARD_USE_CUDA

  assert(blockIdx.y == 0);
  assert(blockIdx.z == 0);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  auto const id          = blockIdx.x * blockDim.x + threadIdx.x;
  auto const num_threads = blockDim.x * gridDim.x;
  auto const start       = id;
  auto const increment   = num_threads;
  auto const stop        = num_elems * num_copies;

  for (auto i = start; i < stop; i += increment)
  {
    workspace[i] = x[i % num_elems];
  }

#else

  auto const start     = 0;
  auto const increment = 1;
  auto const stop      = num_copies;

#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
  for (auto i = start; i < stop; i += increment)
  {
    auto const dest = workspace + i * num_elems;
    std::copy(x, x + num_elems, dest);
  }

#endif
}

template<typename P>
void stage_inputs_kronmult(P const *const x, P *const workspace,
                           int const num_elems, int const num_copies)
{
#ifdef ASGARD_USE_CUDA

  auto constexpr warp_size   = 32;
  auto constexpr num_warps   = 8;
  auto constexpr num_threads = num_warps * warp_size;
  auto const num_blocks      = (num_copies / num_threads) + 1;

  stage_inputs_kronmult_kernel<P>
      <<<num_blocks, num_threads>>>(x, workspace, num_elems, num_copies);

  auto const stat = cudaDeviceSynchronize();
  assert(stat == cudaSuccess);
#else
  stage_inputs_kronmult_kernel(x, workspace, num_elems, num_copies);
#endif
}

// helper - given a cell and level coordinate, return a 1-dimensional index
DEVICE_FUNCTION
inline int get_1d_index(int const level, int const cell)

{
  assert(level >= 0);
  assert(cell >= 0);

  if (level == 0)
  {
    return 0;
  }
  return static_cast<int>(pow((double)2, (double)(level - 1))) + cell;
}

// helper - calculate element coordinates -> operator matrix indices
DEVICE_FUNCTION
void get_indices(int const *const coords, int indices[], int const degree,
                 int const num_dims)
{
  assert(degree > 0);

  for (int i = 0; i < num_dims; ++i)
  {
    indices[i] = get_1d_index(coords[i], coords[i + num_dims]) * degree;
  }
}

template<typename P>
GLOBAL_FUNCTION void
prepare_kronmult_kernel(int const *const flattened_table,
                        P *const *const operators, int const operator_lda,
                        P *const element_x, P *const element_work, P *const fx,
                        P **const operator_ptrs, P **const work_ptrs,
                        P **const input_ptrs, P **const output_ptrs,
                        int const degree, int const num_terms,
                        int const num_dims, int const elem_row_start,
                        int const elem_row_stop, int const elem_col_start,
                        int const elem_col_stop)
{
  auto const num_cols = elem_col_stop - elem_col_start + 1;
  auto const num_rows = elem_row_stop - elem_row_start + 1;

  auto const deg_to_dim = static_cast<int>(pow((float)degree, (float)num_dims));

  auto const x_size     = num_cols * deg_to_dim;
  auto const coord_size = num_dims * 2;
  auto const num_elems  = static_cast<int64_t>(num_cols) * num_rows;

#ifdef ASGARD_USE_CUDA

  assert(blockIdx.y == 0);
  assert(blockIdx.z == 0);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  auto const id          = blockIdx.x * blockDim.x + threadIdx.x;
  auto const num_threads = blockDim.x * gridDim.x;
  auto const start       = id;
  auto const increment   = num_threads;

#else
  auto const start     = 0;
  auto const increment = 1;
#endif

#ifndef ASGARD_USE_CUDA
#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
#endif
  for (auto i = start; i < num_elems; i += increment)
  {
    auto const row = i / num_cols + elem_row_start;
    auto const col = i % num_cols + elem_col_start;

    int constexpr max_dims = 6;
    assert(num_dims <= max_dims);

    // calculate and store operator row indices for this element
    int operator_row[max_dims];
    int const *const row_coords = flattened_table + coord_size * row;
    get_indices(row_coords, operator_row, degree, num_dims);

    // calculate and store operator col indices for this element
    int operator_col[max_dims];
    int const *const col_coords = flattened_table + coord_size * col;
    get_indices(col_coords, operator_col, degree, num_dims);

    auto const x_start =
        element_x + ((row - elem_row_start) * num_terms * x_size +
                     (col - elem_col_start) * deg_to_dim);

    for (auto t = 0; t < num_terms; ++t)
    {
      // get preallocated vector position for this kronmult
      auto const num_kron = (row - elem_row_start) * num_cols * num_terms +
                            (col - elem_col_start) * num_terms + t;

      // point to inputs
      input_ptrs[num_kron] = x_start + t * x_size;

      // point to work/output
      work_ptrs[num_kron]   = element_work + num_kron * deg_to_dim;
      output_ptrs[num_kron] = fx + (row - elem_row_start) * deg_to_dim;

      // point to operators
      auto const operator_start = num_kron * num_dims;
      for (auto d = 0; d < num_dims; ++d)
      {
        P *const coeff = operators[t * num_dims + d];
        operator_ptrs[operator_start + d] =
            coeff + operator_row[d] + operator_col[d] * operator_lda;
      }
    }
  }
}

// build batch lists for kronmult from simple
// arrays. built on device if cuda-enabled.
template<typename P>
void prepare_kronmult(int const *const flattened_table,
                      P *const *const operators, int const operator_lda,
                      P *const element_x, P *const element_work, P *const fx,
                      P **const operator_ptrs, P **const work_ptrs,
                      P **const input_ptrs, P **const output_ptrs,
                      int const degree, int const num_terms, int const num_dims,
                      int const elem_row_start, int const elem_row_stop,
                      int const elem_col_start, int const elem_col_stop)
{
  // TODO asserts

#ifdef ASGARD_USE_CUDA

  auto constexpr warp_size   = 32;
  auto constexpr num_warps   = 8;
  auto constexpr num_threads = num_warps * warp_size;
  auto const num_krons =
      static_cast<int64_t>(elem_col_stop - elem_col_start + 1) *
      (elem_row_stop - elem_row_start + 1);
  auto const num_blocks = (num_krons / num_threads) + 1;
  prepare_kronmult_kernel<P><<<num_blocks, num_threads>>>(
      flattened_table, operators, operator_lda, element_x, element_work, fx,
      operator_ptrs, work_ptrs, input_ptrs, output_ptrs, degree, num_terms,
      num_dims, elem_row_start, elem_row_stop, elem_col_start, elem_col_stop);
  auto const stat = cudaDeviceSynchronize();
  assert(stat == cudaSuccess);
#else
  prepare_kronmult_kernel(
      flattened_table, operators, operator_lda, element_x, element_work, fx,
      operator_ptrs, work_ptrs, input_ptrs, output_ptrs, degree, num_terms,
      num_dims, elem_row_start, elem_row_stop, elem_col_start, elem_col_stop);
#endif
}

// call kronmult as function or kernel invocation
// --------------------------------------------
// note  the input memory referenced by x_ptrs will be over-written
// --------------------------------------------
template<typename P>
void call_kronmult(int const n, P *x_ptrs[], P *output_ptrs[], P *work_ptrs[],
                   P const *const operator_ptrs[], int const lda,
                   int const num_krons, int const num_dims)
{
#ifdef ASGARD_USE_CUDA
  {
    int constexpr warpsize    = 32;
    int constexpr nwarps      = 8;
    int constexpr num_threads = nwarps * warpsize;

    switch (num_dims)
    {
    case 1:
      kronmult1_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 2:
      kronmult2_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 3:
      kronmult3_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 4:
      kronmult4_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 5:
      kronmult5_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 6:
      kronmult6_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    default:
      assert(false);
    };

    // -------------------------------------------
    // note important to wait for kernel to finish
    // -------------------------------------------
    auto const stat = cudaDeviceSynchronize();
    assert(stat == cudaSuccess);
  }
#else

  {
    switch (num_dims)
    {
    case 1:
      kronmult1_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 2:
      kronmult2_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 3:
      kronmult3_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 4:
      kronmult4_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 5:
      kronmult5_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 6:
      kronmult6_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    default:
      assert(false);
    };
  }
#endif
}

template void stage_inputs_kronmult(float const *const x,
                                    float *const workspace, int const num_elems,
                                    int const num_copies);

template void stage_inputs_kronmult(double const *const x,
                                    double *const workspace,
                                    int const num_elems, int const num_copies);

template void prepare_kronmult(
    int const *const flattened_table, float *const *const operators,
    int const operator_lda, float *const element_x, float *const element_work,
    float *const fx, float **const operator_ptrs, float **const work_ptrs,
    float **const input_ptrs, float **const output_ptrs, int const degree,
    int const num_terms, int const num_dims, int const elem_row_start,
    int const elem_row_stop, int const elem_col_start, int const elem_col_stop);

template void prepare_kronmult(
    int const *const flattened_table, double *const *const operators,
    int const operator_lda, double *const element_x, double *const element_work,
    double *const fx, double **const operator_ptrs, double **const work_ptrs,
    double **const input_ptrs, double **const output_ptrs, int const degree,
    int const num_terms, int const num_dims, int const elem_row_start,
    int const elem_row_stop, int const elem_col_start, int const elem_col_stop);

template void call_kronmult(int const n, float *x_ptrs[], float *output_ptrs[],
                            float *work_ptrs[],
                            float const *const operator_ptrs[], int const lda,
                            int const num_krons, int const num_dims);

template void call_kronmult(int const n, double *x_ptrs[],
                            double *output_ptrs[], double *work_ptrs[],
                            double const *const operator_ptrs[], int const lda,
                            int const num_krons, int const num_dims);
