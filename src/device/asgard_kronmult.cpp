#include <iostream>
#include <set>

#include "build_info.hpp"

#include "asgard_kronmult.hpp"

#ifdef ASGARD_USE_CUDA
#include "asgard_kronmult_cycle1.hpp"
#include "asgard_kronmult_cycle2.hpp"
#include "asgard_kronmult_cyclex.hpp"
#endif

namespace asgard::kronmult
{
#ifdef ASGARD_USE_CUDA

/*!
 * \brief Computes the team size for the given dims and n.
 *
 * Normally, the team size is n^dims, but in some cases
 * performance can be improved by increasing the team size to align
 * the teams to the warps, e.g., dims=n=3 means effective team size 27,
 * but that would require thread synchronization and padding the team
 * with idle treads actually increases performance.
 */
template<int dims, int n>
constexpr int compute_team_size()
{
  // TODO account for warp size != 32
  // case dims == 2 and n == 9, rounding to team size 16 is too much
  if constexpr (dims == 1)
  {
    if constexpr (n == 3 or n == 7)
    {
      return n + 1;
    }
    else
    {
      return n;
    }
  }
  else if constexpr ((dims == 3 and n == 3) or (dims == 2 and n == 5))
  {
    return ASGARD_GPU_WARP_SIZE;
  }
  else
  {
    return ipow<n, dims>();
  }
}

/*!
 * \brief Run a GPU kernel for the specified problem.
 *
 * Instantiates a GPU kernel, computes the appropriate grid and executes the
 * kernel. Handles the one cycle case including all instances of 1D and n=1.
 *
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
void run_kernel(precision const *const pA[], int const lda,
                precision const *const pX[], precision *pY[],
                int const num_batch)
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
  constexpr int team_size   = compute_team_size<dims, n>();
  constexpr int num_teams   = max_threads / team_size;

  static_assert(max_threads >= team_size,
                "tensor size must be less than the max number of threads");

  int num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  if constexpr (dims == 1)
  {
    kernel::case1D<precision, n, team_size, num_teams>
        <<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
  }
  else if constexpr (n == 1)
  {
    kernel::case1N<precision, dims, max_threads>
        <<<std::min(max_blocks, (num_batch + max_threads - 1) / max_threads),
           max_threads>>>(pA, lda, pX, pY, num_batch);
  }
  else
  {
    kernel::cycle1<precision, dims, n, team_size, num_teams>
        <<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
  }
}

/*!
 * \brief Run a GPU kernel for the specified problem, two cycle case.
 *
 * Same as run_kernel() but uses logic for 2 cycles.
 */
template<typename precision, int dims, int n>
void run_kernel2(precision const *const pA[], int const lda,
                 precision const *const pX[], precision *pY[],
                 int const num_batch)
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
  constexpr int team_size   = (ipow<n, dims>() + 1) / 2;
  constexpr int num_teams   = max_threads / team_size;

  int const num_blocks = blocks(num_batch, num_teams, max_blocks);

  static_assert(max_threads >= team_size,
                "tensor size must be less than the max number of threads");

  dim3 grid(team_size, num_teams);
  kernel::cycle2<precision, dims, n, team_size, num_teams>
      <<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
}

/*!
 * \brief Run a GPU kernel for the specified problem, two cycle case.
 *
 * Same as run_kernel() but uses logic for up to 4 kernels,
 * the extra input num_cycles has to be 1 - 4,
 * but the 1 and 2 case should use run_kernel() or run_kernel2().
 */
template<typename precision, int dims, int n, int num_cycles>
void run_kernelx(precision const *const pA[], int const lda,
                 precision const *const pX[], precision *pY[],
                 int const num_batch)
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
  constexpr int team_size   = (ipow<n, dims>() + 1) / num_cycles;
  constexpr int num_teams   = max_threads / team_size;

  static_assert(max_threads >= team_size,
                "tensor size must be less than the max number of threads");

  int const num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  kernel::cyclex<precision, dims, n, team_size, num_teams, num_cycles>
      <<<num_blocks, grid>>>(pA, lda, pX, pY, num_batch);
}

template<typename T>
void execute_gpu(int dimensions, int n, T const *const pA[], int const lda,
                 T *pX[], T *pY[], int const num_batch, int const)
{
  switch (dimensions)
  {
  case 1:
    switch (n)
    {
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
    switch (n)
    {
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
    switch (n)
    {
    case 1:
      run_kernel<T, 3, 1>(pA, lda, pX, pY, num_batch);
      break;
    case 2:
      run_kernel2<T, 3, 2>(pA, lda, pX, pY, num_batch);
      break;
    case 3:
      run_kernel<T, 3, 3>(pA, lda, pX, pY, num_batch);
      break;
    case 4:
      run_kernel2<T, 3, 4>(pA, lda, pX, pY, num_batch);
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
    switch (n)
    {
    case 1:
      run_kernel<T, 4, 1>(pA, lda, pX, pY, num_batch);
      break;
    case 2:
      run_kernel<T, 4, 2>(pA, lda, pX, pY, num_batch);
      break;
    case 3:
      run_kernel2<T, 4, 3>(pA, lda, pX, pY, num_batch);
      break;
    case 4:
      run_kernel2<T, 4, 4>(pA, lda, pX, pY, num_batch);
      break;
    case 5:
      run_kernel2<T, 4, 5>(pA, lda, pX, pY, num_batch);
      break;
    default:
      throw std::runtime_error("kronmult unimplemented n for the gpu");
    }
    break;
  case 5:
    switch (n)
    {
    case 1:
      run_kernel<T, 5, 1>(pA, lda, pX, pY, num_batch);
      break;
    case 2:
      run_kernel<T, 5, 2>(pA, lda, pX, pY, num_batch);
      break;
    case 3:
      run_kernel2<T, 5, 3>(pA, lda, pX, pY, num_batch);
      break;
    case 4:
      run_kernel2<T, 5, 4>(pA, lda, pX, pY, num_batch);
      break;
    default:
      throw std::runtime_error("kronmult unimplemented n for the gpu");
    }
    break;
  case 6:
    switch (n)
    {
    case 1:
      run_kernel<T, 6, 1>(pA, lda, pX, pY, num_batch);
      break;
    case 2:
      run_kernel2<T, 6, 2>(pA, lda, pX, pY, num_batch);
      break;
    case 3:
      run_kernel2<T, 6, 3>(pA, lda, pX, pY, num_batch);
      break;
    case 4:
      run_kernelx<T, 6, 4, 4>(pA, lda, pX, pY, num_batch);
      break;
    default:
      throw std::runtime_error("kronmult unimplemented n for the gpu");
    }
    break;
  default:
    throw std::runtime_error(
        "kronmult unimplemented number of dimensions for the gpu " +
        std::to_string(dimensions));
  }
}

template void execute_gpu<float>(int, int, float const *const[], int const,
                                 float *[], float *[], int const, int const);
template void execute_gpu<double>(int, int, double const *const[], int const,
                                  double *[], double *[], int const, int const);

#endif

template<typename T>
void execute_cpu(int dimensions, int n, T const *const pA[], int const lda,
                 T const *const pX[], T *pY[], int const num_batch,
                 int const output_stride)
{
  switch (dimensions)
  {
  case 1:
    switch (n)
    {
    case 1:
      run_cpu_variant0(dimensions, pA, pX, pY, num_batch, output_stride);
      break;
    case 2:
      run_cpu_variant<T, 1, 2>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 3:
      run_cpu_variant<T, 1, 3>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 4:
      run_cpu_variant<T, 1, 4>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    default:
      run_cpu_variant<T, 1>(n, pA, lda, pX, pY, num_batch, output_stride);
    }
    break;
  case 2:
    switch (n)
    {
    case 1:
      run_cpu_variant0(dimensions, pA, pX, pY, num_batch, output_stride);
      break;
    case 2:
      run_cpu_variant<T, 2, 2>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 3:
      run_cpu_variant<T, 2, 3>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 4:
      run_cpu_variant<T, 2, 4>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    default:
      run_cpu_variant<T, 2>(n, pA, lda, pX, pY, num_batch, output_stride);
    }
    break;
  case 3:
    switch (n)
    {
    case 1:
      run_cpu_variant0(dimensions, pA, pX, pY, num_batch, output_stride);
      break;
    case 2:
      run_cpu_variant<T, 3, 2>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 3:
      run_cpu_variant<T, 3, 3>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 4:
      run_cpu_variant<T, 3, 4>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    default:
      run_cpu_variant<T, 3>(n, pA, lda, pX, pY, num_batch, output_stride);
    }
    break;
  case 4:
    switch (n)
    {
    case 1:
      run_cpu_variant0(dimensions, pA, pX, pY, num_batch, output_stride);
      break;
    case 2:
      run_cpu_variant<T, 4, 2>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 3:
      run_cpu_variant<T, 4, 3>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 4:
      run_cpu_variant<T, 4, 4>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    default:
      run_cpu_variant<T, 4>(n, pA, lda, pX, pY, num_batch, output_stride);
    }
    break;
  case 5:
    switch (n)
    {
    case 1:
      run_cpu_variant0(dimensions, pA, pX, pY, num_batch, output_stride);
      break;
    case 2:
      run_cpu_variant<T, 5, 2>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 3:
      run_cpu_variant<T, 5, 3>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 4:
      run_cpu_variant<T, 5, 4>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    default:
      run_cpu_variant<T, 5>(n, pA, lda, pX, pY, num_batch, output_stride);
    }
    break;
  case 6:
    switch (n)
    {
    case 1:
      run_cpu_variant0(dimensions, pA, pX, pY, num_batch, output_stride);
      break;
    case 2:
      run_cpu_variant<T, 6, 2>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 3:
      run_cpu_variant<T, 6, 3>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    case 4:
      run_cpu_variant<T, 6, 4>(pA, lda, pX, pY, num_batch, output_stride);
      break;
    default:
      run_cpu_variant<T, 6>(n, pA, lda, pX, pY, num_batch, output_stride);
    }
    break;
  default:
    throw std::runtime_error(
        "kronmult unimplemented number of dimensions for the cpu");
  }
}

template void execute_cpu<float>(int, int, float const *const[], int const,
                                 float const *const[], float *[], int const, int const);
template void execute_cpu<double>(int, int, double const *const[], int const,
                                  double const *const[], double *[], int const, int const);

} // namespace asgard::kronmult
