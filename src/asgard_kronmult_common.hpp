#pragma once

#include "asgard_compute.hpp"




namespace asgard::kronmult
{
// /*!
//  * \brief Computes the number of CUDA blocks.
//  *
//  * \param work_size is the total amount of work, e.g., size of the batch
//  * \param work_per_block is the work that a single thread block will execute
//  * \param max_blocks is the maximum number of blocks
//  */
// inline int blocks(int64_t work_size, int work_per_block, int max_blocks)
// {
//   return std::min(
//       max_blocks,
//       static_cast<int>((work_size + work_per_block - 1) / work_per_block));
// }

// /*!
//  * \brief Flag variable, indicates whether thread synchronization is necessary.
//  *
//  * Threads inside a warp are always synchronized, synchronization
//  * in the kernel is not needed unless teams span more than one warp.
//  */
// enum class manual_sync
// {
//   //! \brief Use synchronization after updating the shared cache.
//   enable,
//   //! \brief No need for synchronization, thread teams are aligned to the warps.
//   disable
// };
//
// /*!
//  * \internal
//  * \brief (internal use only) Indicates how to interpret the alpha/beta scalars.
//  *
//  * Matrix operations include scalar parameters, e.g., \b beta \b y.
//  * Flops can be saved in special cases and those are in turn
//  * handled with template parameters and if-constexpr clauses.
//  * \endinternal
//  */
// enum class scalar_case
// {
//   //! \brief Overwrite the existing output
//   zero,
//   //! \brief Ignore \b beta and just add to the existing output
//   one,
//   //! \brief Ignore \b beta and subtract from the existing output
//   neg_one,
//   //! \brief Scale by \b beta and add the values
//   other
// };



// #ifdef ASGARD_USE_CUDA
// //! \brief Sets a device buffer to zeros
// template<typename T>
// void set_gpu_buffer_to_zero(int64_t num, T *x);
//
// //! \brief Helper method, fills the buffer with zeros
// template<typename T>
// void set_buffer_to_zero(gpu::vector<T> &x)
// {
//   set_gpu_buffer_to_zero(static_cast<int64_t>(x.size()), x.data());
// }
// #endif


} // namespace asgard::kronmult
