#include "asgard_kronmult.hpp"

#ifndef KRON_MODE_GLOBAL

#ifdef ASGARD_USE_CUDA
#include "asgard_spkronmult_cycle1.hpp"
#include "asgard_spkronmult_cycle2.hpp"
#include "asgard_spkronmult_cyclex.hpp"
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
  // case dims == 2 and n == 3, rounding to team size 16 is too much
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
  return ipow<n, dims>();
}

//! \brief Helper to instantiate and call the scaling kernel.
template<typename T>
void scale(int const num, T const beta, T y[])
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;

  int num_blocks = std::min(max_blocks, (num + max_threads - 1) / max_threads);
  if (beta == 0)
    kernel::scale<T, scalar_case::zero>
        <<<num_blocks, max_threads>>>(num, beta, y);
  else if (beta == 1)
    return;
  else if (beta == -1)
    kernel::scale<T, scalar_case::neg_one>
        <<<num_blocks, max_threads>>>(num, beta, y);
  else
    kernel::scale<T, scalar_case::other>
        <<<num_blocks, max_threads>>>(num, beta, y);
}
//! \brief Helper to instantiate and call the kernel for n=1.
template<typename T, int dims>
void case_n1(int const num_batch, int const ix[], int const iy[],
             int const num_terms, int const iA[], T const vA[], T const alpha,
             T const x[], T y[])
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;

  int num_blocks = blocks(num_batch, max_threads, max_blocks);

  if (alpha == 1)
    kernel::case_n1<T, dims, scalar_case::one><<<num_blocks, max_threads>>>(
        num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
  else if (alpha == -1)
    kernel::case_n1<T, dims, scalar_case::neg_one><<<num_blocks, max_threads>>>(
        num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
  else
    kernel::case_n1<T, dims, scalar_case::other><<<num_blocks, max_threads>>>(
        num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
}
//! \brief Helper to instantiate and call the kernel for d=1.
template<typename T, int n>
void case_d1(int const num_batch, int const ix[], int const iy[],
             int const num_terms, int const iA[], T const vA[], T const alpha,
             T const x[], T y[])
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
  constexpr int team_size   = n;
  constexpr int num_teams   = max_threads / team_size;

  int num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  if (alpha == 1)
    kernel::case_d1<T, n, team_size, num_teams, scalar_case::one>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
  else if (alpha == -1)
    kernel::case_d1<T, n, team_size, num_teams, scalar_case::neg_one>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
  else
    kernel::case_d1<T, n, team_size, num_teams, scalar_case::other>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
}
//! \brief Helper to instantiate and call the kernel for cycle1.
template<typename T, int dims, int n>
void case_cycle1(int const num_batch, int const ix[], int const iy[],
                 int const num_terms, int const iA[], T const vA[],
                 T const alpha, T const x[], T y[])
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
  constexpr int team_size   = compute_team_size<dims, n>();
  constexpr int num_teams   = max_threads / team_size;

  static_assert(max_threads >= team_size,
                "tensor size must be less than the max number of threads");

  int const num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  if (alpha == 1)
    kernel::cycle1<T, dims, n, team_size, num_teams, scalar_case::one>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
  else if (alpha == -1)
    kernel::cycle1<T, dims, n, team_size, num_teams, scalar_case::neg_one>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
  else
    kernel::cycle1<T, dims, n, team_size, num_teams, scalar_case::other>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
}
//! \brief Helper to instantiate and call the kernel for cycle2.
template<typename T, int dims, int n>
void case_cycle2(int const num_batch, int const ix[], int const iy[],
                 int const num_terms, int const iA[], T const vA[],
                 T const alpha, T const x[], T y[])
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
  constexpr int team_size   = (ipow<n, dims>() + 1) / 2;
  constexpr int num_teams   = max_threads / team_size;

  static_assert(max_threads >= team_size,
                "tensor size must be less than the max number of threads");

  int const num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  if (alpha == 1)
    kernel::cycle2<T, dims, n, team_size, num_teams, scalar_case::one>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
  else if (alpha == -1)
    kernel::cycle2<T, dims, n, team_size, num_teams, scalar_case::neg_one>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
  else
    kernel::cycle2<T, dims, n, team_size, num_teams, scalar_case::other>
        <<<num_blocks, grid>>>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                               y);
}
/*!
 * \brief Helper to instantiate and call the kernel for cyclex.
 */
template<typename T, int dims, int n, int num_cycles>
void case_cyclex(int const num_batch, int const ix[], int const iy[],
                 int const num_terms, int const iA[], T const vA[],
                 T const alpha, T const x[], T y[])
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
  constexpr int team_size   = (ipow<n, dims>() + 1) / num_cycles;
  constexpr int num_teams   = max_threads / team_size;

  static_assert(max_threads >= team_size,
                "tensor size must be less than the max number of threads");

  int const num_blocks = blocks(num_batch, num_teams, max_blocks);

  dim3 grid(team_size, num_teams);
  if (alpha == 1)
    kernel::cyclex<T, dims, n, team_size, num_teams, num_cycles,
                   scalar_case::one><<<num_blocks, grid>>>(
        num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
  else if (alpha == -1)
    kernel::cyclex<T, dims, n, team_size, num_teams, num_cycles,
                   scalar_case::neg_one><<<num_blocks, grid>>>(
        num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
  else
    kernel::cyclex<T, dims, n, team_size, num_teams, num_cycles,
                   scalar_case::other><<<num_blocks, grid>>>(
        num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
}

template<typename T>
void gpu_sparse(int const dimensions, int const n, int const output_size,
                int const num_batch, int const ix[], int const iy[],
                int const num_terms, int const iA[], T const vA[],
                T const alpha, T const x[], T const beta, T y[])
{
  // apply the scaling to y and assume beta == 1 for the other kernels
  scale(output_size, beta, y);

  switch (dimensions)
  {
  case 1:
    switch (n)
    {
    case 1:
      case_n1<T, 1>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 2:
      case_d1<T, 2>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 3:
      case_d1<T, 3>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 4:
      case_d1<T, 4>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 5:
      case_d1<T, 5>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 6:
      case_d1<T, 6>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 7:
      case_d1<T, 7>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 8:
      case_d1<T, 8>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 9:
      case_d1<T, 9>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 10:
      case_d1<T, 10>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    default:
      throw std::runtime_error("kronmult unimplemented n for the gpu");
    }
    break;
  case 2:
    switch (n)
    {
    case 1:
      case_n1<T, 2>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 2:
      case_cycle1<T, 2, 2>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 3:
      case_cycle1<T, 2, 3>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 4:
      case_cycle1<T, 2, 4>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 5:
      case_cycle1<T, 2, 5>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 6:
      case_cycle1<T, 2, 6>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 7:
      case_cycle1<T, 2, 7>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 8:
      case_cycle1<T, 2, 8>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 9:
      case_cycle1<T, 2, 9>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 10:
      case_cycle1<T, 2, 10>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 11:
      case_cycle1<T, 2, 11>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 12:
      case_cycle1<T, 2, 12>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 13:
      case_cycle1<T, 2, 13>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 14:
      case_cycle1<T, 2, 14>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 15:
      case_cycle1<T, 2, 15>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 16:
      case_cycle1<T, 2, 16>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 17:
      case_cycle1<T, 2, 17>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 18:
      case_cycle1<T, 2, 18>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 19:
      case_cycle1<T, 2, 19>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 20:
      case_cycle1<T, 2, 20>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 21:
      case_cycle1<T, 2, 21>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 22:
      case_cycle1<T, 2, 22>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 23:
      case_cycle1<T, 2, 23>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 24:
      case_cycle1<T, 2, 24>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 25:
      case_cycle1<T, 2, 25>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 26:
      case_cycle1<T, 2, 26>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 27:
      case_cycle1<T, 2, 27>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 28:
      case_cycle1<T, 2, 28>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 29:
      case_cycle1<T, 2, 29>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 30:
      case_cycle1<T, 2, 30>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 31:
      case_cycle1<T, 2, 31>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 32:
      case_cycle1<T, 2, 32>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    default:
      throw std::runtime_error("kronmult unimplemented n for the gpu");
    }
    break;
  case 3:
    switch (n)
    {
    case 1:
      case_n1<T, 3>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 2:
      case_cycle2<T, 3, 2>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 3:
      case_cycle1<T, 3, 3>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 4:
      case_cycle2<T, 3, 4>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 5:
      case_cycle1<T, 3, 5>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 6:
      case_cycle1<T, 3, 6>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 7:
      case_cycle1<T, 3, 7>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 8:
      case_cycle1<T, 3, 8>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 9:
      case_cycle1<T, 3, 9>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 10:
      case_cycle1<T, 3, 10>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    default:
      throw std::runtime_error("kronmult unimplemented n for the gpu");
    }
    break;
  case 4:
    switch (n)
    {
    case 1:
      case_n1<T, 4>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 2:
      case_cycle1<T, 4, 2>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 3:
      case_cycle2<T, 4, 3>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 4:
      case_cycle2<T, 4, 4>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 5:
      case_cycle2<T, 4, 5>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    default:
      throw std::runtime_error("kronmult unimplemented n for the gpu");
    }
    break;
  case 5:
    switch (n)
    {
    case 1:
      case_n1<T, 5>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 2:
      case_cycle1<T, 5, 2>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 3:
      case_cycle2<T, 5, 3>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 4:
      case_cycle2<T, 5, 4>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    default:
      throw std::runtime_error("kronmult unimplemented n for the gpu");
    }
    break;
  case 6:
    switch (n)
    {
    case 1:
      case_n1<T, 6>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 2:
      case_cycle2<T, 6, 2>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 3:
      case_cycle2<T, 6, 3>(num_batch, ix, iy, num_terms, iA, vA, alpha, x, y);
      break;
    case 4:
      case_cyclex<T, 6, 4, 4>(num_batch, ix, iy, num_terms, iA, vA, alpha, x,
                              y);
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

#ifdef ASGARD_ENABLE_DOUBLE

template void gpu_sparse<double>(int const, int const, int const, int const,
                                 int const[], int const[], int const,
                                 int const[], double const[], double const,
                                 double const[], double const, double[]);

#endif

#ifdef ASGARD_ENABLE_FLOAT

template void gpu_sparse<float>(int const, int const, int const, int const,
                                int const[], int const[], int const,
                                int const[], float const[], float const,
                                float const[], float const, float[]);
#endif

#endif

} // namespace asgard::kronmult

#endif // KRON_MODE_GLOBAL
