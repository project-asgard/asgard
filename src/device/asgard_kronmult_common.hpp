#pragma once

#include <iostream>

#ifdef USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_60_atomic_functions.h>

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
inline int blocks(int work_size, int work_per_block, int max_blocks)
{
  return std::min(max_blocks,
                  (work_size + work_per_block - 1) / work_per_block);
}

} // namespace asgard::kronmult

// TODO add intrinsics here too for the CPU
