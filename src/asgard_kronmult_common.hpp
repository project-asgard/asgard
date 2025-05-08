#pragma once

#include "asgard_tools.hpp"

#ifdef ASGARD_USE_CUDA

// #include <sm_60_atomic_functions.h> // TODO: move to private headers

#define ASGARD_GPU_WARP_SIZE 32

#endif

// As of LLVM version 16, clang does not utilize #pragma omp simd
// resulting in under-performing code
#define ASGARD_PRAGMA(x) _Pragma(#x)
#if defined(__clang__)
#define ASGARD_PRAGMA_OMP_SIMD(x)
#define ASGARD_OMP_SIMD
#define ASGARD_OMP_PARFOR_SIMD
#define ASGARD_OMP_PARFOR_SIMD_EXTRA(x)
#else
#define ASGARD_OMP_SIMD ASGARD_PRAGMA(omp simd)
#define ASGARD_PRAGMA_OMP_SIMD(clause) ASGARD_PRAGMA(omp simd clause)
#define ASGARD_OMP_PARFOR_SIMD ASGARD_PRAGMA(omp parallel for simd)
#define ASGARD_OMP_PARFOR_SIMD_EXTRA(clause) ASGARD_PRAGMA(omp parallel for simd clause)
#endif

namespace asgard::kronmult
{
/*!
 * \brief Computes the number of CUDA blocks.
 *
 * \param work_size is the total amount of work, e.g., size of the batch
 * \param work_per_block is the work that a single thread block will execute
 * \param max_blocks is the maximum number of blocks
 */
inline int blocks(int64_t work_size, int work_per_block, int max_blocks)
{
  return std::min(
      max_blocks,
      static_cast<int>((work_size + work_per_block - 1) / work_per_block));
}

/*!
 * \brief Flag variable, indicates whether thread synchronization is necessary.
 *
 * Threads inside a warp are always synchronized, synchronization
 * in the kernel is not needed unless teams span more than one warp.
 */
enum class manual_sync
{
  //! \brief Use synchronization after updating the shared cache.
  enable,
  //! \brief No need for synchronization, thread teams are aligned to the warps.
  disable
};

/*!
 * \internal
 * \brief (internal use only) Indicates how to interpret the alpha/beta scalars.
 *
 * Matrix operations include scalar parameters, e.g., \b beta \b y.
 * Flops can be saved in special cases and those are in turn
 * handled with template parameters and if-constexpr clauses.
 * \endinternal
 */
enum class scalar_case
{
  //! \brief Overwrite the existing output
  zero,
  //! \brief Ignore \b beta and just add to the existing output
  one,
  //! \brief Ignore \b beta and subtract from the existing output
  neg_one,
  //! \brief Scale by \b beta and add the values
  other
};

/*!
 * \brief Template that computes n to power, e.g., ipow<2, 3>() returns constexpr 8.
 */
template<int n, int power>
constexpr int ipow()
{
  if constexpr (power == 1)
  {
    return n;
  }
  else if constexpr (power == 2)
  {
    return n * n;
  }
  else if constexpr (power == 3)
  {
    return n * n * n;
  }
  else if constexpr (power == 4)
  {
    return n * n * n * n;
  }
  else if constexpr (power == 5)
  {
    return n * n * n * n * n;
  }
  else if constexpr (power == 6)
  {
    return n * n * n * n * n * n;
  }
  static_assert(power >= 1 and power <= 6,
                "ipow() does not works with specified power");
  return 0;
}

#ifdef ASGARD_USE_CUDA
//! \brief Sets a device buffer to zeros
template<typename T>
void set_gpu_buffer_to_zero(int64_t num, T *x);

//! \brief Helper method, fills the buffer with zeros
template<typename T>
void set_buffer_to_zero(gpu::vector<T> &x)
{
  set_gpu_buffer_to_zero(static_cast<int64_t>(x.size()), x.data());
}
#endif

/*!
  * \brief Compute the permutations (upper/lower) for global kronecker operations
  *
  * This computes all the permutations for the given dimensions
  * and sets up the fill and direction vector-of-vectors.
  * Direction 0 will be set to full and all others will alternate
  * between upper and lower.
  *
  * By default, the directions are in order (0, 1, 2, 3); however, if a term has
  * entries (identity, term, identity, term), then the effective dimension is 2
  * and first the permutation should be set for dimension 2,
  * then we should call .remap_directions({1, 3}) to remap (0, 1) into the active
  * directions of 1 and 3 (skipping the call to the identity.
 */
struct permutes
{
  //! \brief Indicates the fill of the matrix.
  enum class matrix_fill
  {
    upper,
    both,
    lower
  };
  //! \brief Matrix fill for each operation.
  std::vector<std::vector<matrix_fill>> fill;
  //! \brief Direction for each matrix operation.
  std::vector<std::vector<int>> direction;
  //! \brief Direction of the flux, if any
  int flux_dir = -1;
  //! \brief Empty permutation list.
  permutes() = default;
  //! \brief Initialize the permutations.
  permutes(int num_dimensions)
  {
    if (num_dimensions < 1) // could happen with identity operator term
      return;

    int num_permute = 1;
    for (int d = 0; d < num_dimensions - 1; d++)
      num_permute *= 2;

    direction.resize(num_permute);
    fill.resize(num_permute);
    for (int perm = 0; perm < num_permute; perm++)
    {
      direction[perm].resize(num_dimensions, 0);
      fill[perm].resize(num_dimensions);
      int t = perm;
      for (int d = 1; d < num_dimensions; d++)
      {
        // negative dimension means upper fill, positive for lower fill
        direction[perm][d] = (t % 2 == 0) ? d : -d;
        t /= 2;
      }
      // sort puts the upper matrices first
      std::sort(direction[perm].begin(), direction[perm].end());
      for (int d = 0; d < num_dimensions; d++)
      {
        fill[perm][d] = (direction[perm][d] < 0) ? matrix_fill::upper : ((direction[perm][d] > 0) ? matrix_fill::lower : matrix_fill::both);

        direction[perm][d] = std::abs(direction[perm][d]);
      }
    }
  }
  permutes(std::vector<int> const &active_dirs, int fdir = -1)
      : permutes(static_cast<int>(active_dirs.size()))
  {
    remap_directions(active_dirs);
    flux_dir = fdir;
  }
  //! \brief Convert the fill to a string (for debugging).
  std::string_view fill_name(int perm, int stage) const
  {
    switch (fill[perm][stage])
    {
    case matrix_fill::upper:
      return "upper";
    case matrix_fill::lower:
      return "lower";
    default:
      return "full";
    }
  }
  //! \brief Shows the number of dimensions considered in the permutation
  int num_dimensions() const
  {
    return (direction.empty()) ? 0 : static_cast<int>(direction.front().size());
  }
  //! \brief Reindexes the dimensions to match the active (non-identity) dimensions
  void remap_directions(std::vector<int> const &active_dirs)
  {
    for (auto &dirs : direction) // for all permutations
      for (auto &d : dirs)       // for all directions
        d = active_dirs[d];
  }
  //! \brief Indicates if the permutation has been set
  operator bool () const { return not direction.empty(); }
};

} // namespace asgard::kronmult
