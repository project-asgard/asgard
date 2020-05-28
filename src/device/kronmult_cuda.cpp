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

// helper - given a cell and level coordinate, return a 1-dimensional index
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
inline void
get_indices(int const * const coords, int indices[], int const degree, int const num_dims)
{
  assert(degree > 0);

  for (int i = 0; i < num_dims; ++i)
  {
    indices[i] = get_1d_index(coords[i], coords[i + num_dims]) * degree;
  }
}

// build batch lists for kronmult from simple
// arrays. built on device if cuda-enabled.
template<typename P>
void prepare_kronmult(int const * const flattened_table, 
		     P * const * const operators, 
		     int const operator_lda,
		     P * const element_x,
		     P * const element_work,
		     P * const fx,
		     P **const operator_ptrs, P  **const work_ptrs,
		     P **const input_ptrs, P  ** const output_ptrs,  
                     int const degree,
		     int const num_terms, 
		     int const num_dims,
		     int const elem_row_start, int const elem_row_stop,
		     int const elem_col_start, int const elem_col_stop) { 

auto const num_cols = elem_col_stop - elem_col_start + 1;
auto const deg_to_dim = static_cast<int>(pow((double)degree, (double)num_dims));
auto const x_size = num_cols * deg_to_dim;
auto const coord_size = num_dims * 2;

#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (auto i = elem_row_start; i <= elem_row_stop; ++i)
  {
    for (auto j = elem_col_start; j <= elem_col_stop; ++j)
    {
      // calculate and store operator row indices for this element
      static int constexpr max_dims = 6;
      assert(num_dims <= max_dims);
      int operator_row[max_dims];
      int const * const row_coords = flattened_table + coord_size * i;
      get_indices(row_coords, operator_row, degree, num_dims);


      // calculate and store operator col indices for this element
      int operator_col[max_dims];
      int const * const col_coords = flattened_table + coord_size * j;
      get_indices(col_coords, operator_col, degree, num_dims);

      auto const x_start =
          element_x + ((i-elem_row_start) * num_terms * x_size +
                       (j-elem_col_start) * deg_to_dim);

      for (auto t = 0; t < num_terms; ++t)
      {

	// get preallocated vector positions for this kronmult
        auto const num_kron =
            (i-elem_row_start) * num_cols * num_terms +
            (j-elem_col_start) * num_terms + t;

        // point to inputs
        input_ptrs[num_kron] = x_start + t * x_size;

        // point to work/output
        work_ptrs[num_kron] = element_work + num_kron * deg_to_dim;
        output_ptrs[num_kron] =
            fx + (i-elem_row_start) * deg_to_dim;

        // point to operators
        auto const operator_start = num_kron * num_dims;
        for (auto d = 0; d < num_dims; ++d)
        {
          P * const coeff = operators[t * num_dims + d];
          operator_ptrs[operator_start + d] =
              coeff + operator_row[d] + operator_col[d]*operator_lda;
        }
      }
    }
  }


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
    P **x_d;
    P **work_d;
    P **output_d;
    P const **operators_d;
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

    stat = cudaFree(x_d);
    assert(stat == 0);
    stat = cudaFree(operators_d);
    assert(stat == 0);
    stat = cudaFree(output_d);
    assert(stat == 0);
    stat = cudaFree(work_d);
    assert(stat == 0);
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

template
void prepare_kronmult(int const * const flattened_table, 
		     float  * const * const operators, 
		     int const operator_lda,
		     float * const element_x,
		     float * const element_work,
		     float * const fx,
		     float **const operator_ptrs, float  **const work_ptrs,
		     float ** const input_ptrs, float  ** const output_ptrs,  
                     int const degree,
		     int const num_terms, 
		     int const num_dims,
		     int const elem_row_start, int const elem_row_stop,
		     int const elem_col_start, int const elem_col_stop);

template
void prepare_kronmult(int const * const flattened_table, 
		     double * const * const operators, 
		     int const operator_lda,
		     double  * const element_x,
		     double  * const element_work,
		     double  * const fx,
		     double **const operator_ptrs, double  **const work_ptrs,
		     double ** const input_ptrs, double  ** const output_ptrs,  
                     int const degree,
		     int const num_terms, 
		     int const num_dims,
		     int const elem_row_start, int const elem_row_stop,
		     int const elem_col_start, int const elem_col_stop);


template void call_kronmult(int const n, float *x_ptrs[], float *output_ptrs[],
                            float *work_ptrs[],
                            float const *const operator_ptrs[], int const lda,
                            int const num_krons, int const num_dims);

template void call_kronmult(int const n, double *x_ptrs[],
                            double *output_ptrs[], double *work_ptrs[],
                            double const *const operator_ptrs[], int const lda,
                            int const num_krons, int const num_dims);
