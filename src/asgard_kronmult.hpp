#pragma once

#include "asgard_block_matrix.hpp"
#include "asgard_indexset.hpp"
#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult
{
/*!
 * \internal
 * \brief Persistent workspace for kronmult operations
 *
 * The methods will use resize on the vectors, thus adjusting the memory
 * being used, but also minimizing the new allocations.
 * \endinternal
 */
template<typename precision>
struct workspace
{
  std::vector<precision> w1, w2;
  std::vector<std::vector<int64_t>> row_map;
  #ifdef ASGARD_USE_GPU
  std::array<gpu::vector<precision>, max_num_gpus> gpu_w1, gpu_w2;
  #endif
};

/*!
 * \brief Computes the action of a sparse Kronecker onto a vector
 *
 * Computes y = alpha * A * x + beta * y, where A is a sparse Kronecker matrix
 * defined on a sparse grid by a set of possibly different matrices.
 *
 * \tparam precision is float or double
 *
 * \param n is the size of the block, e.g., 1 for degree 0, 2 for linear basis and so on.
 * \param grid is the current sparse grid
 * \param conns is the connection patter of the 1D operators
 * \param cmats define a matrix for each dimension
 * \param alpha scale parameter
 * \param x is the vector that A should act upon
 * \param beta scale parameter
 * \param y is the output vector
 * \param work is initialized workspace
 */
template<typename precision>
void block_cpu(int n, sparse_grid const &grid, connection_patterns const &conns,
               permutes const &perm,
               std::array<block_sparse_matrix<precision>, max_num_dimensions> const &cmats,
               precision alpha, precision const x[], precision beta, precision y[],
               workspace<precision> &work);

/*!
 * \brief Computes the action of a sparse Kronecker onto a vector
 *
 * Uses the same matrix across all dimensions, otherwise identical
 * to kronmult::block_cpu
 */
template<typename precision>
void block_cpu(int n, sparse_grid const &grid, connection_patterns const &conns,
               permutes const &perm, block_sparse_matrix<precision> const &cmats,
               precision alpha, precision const x[], precision beta, precision y[],
               workspace<precision> &work);

#ifdef ASGARD_USE_FLOPCOUNTER
//! counts the flops for the specific kronmult operation
template<typename precision>
int64_t block_cpu(int n, sparse_grid const &grid, connection_patterns const &conns,
                  permutes const &perm, precision alpha, precision beta, workspace<precision> &work);
#endif

#ifdef ASGARD_USE_GPU
/*!
 * \brief GPU implementation for the block-cpu evaluate
 *
 * Computes y = alpha * A * x + beta * y, where A is a sparse Kronecker matrix
 * defined on a sparse grid by a set of possibly different matrices.
 * The input and output arrays are located on the GPU device and compute->set_device()
 * has been correctly set for the current thread, i.e., this method uses only one thread
 * but launches multiple kernel on the set GPU device.
 *
 * The device gpu::device is used to identify the workspace and the correctly cached
 * sparse_grid and connection_patterns values.
 */
template<typename precision>
void block_gpu(gpu::device dev, int n, sparse_grid const &grid,
               connection_patterns const &conns, permutes const &perm,
               std::array<gpu::vector<precision *>, max_num_dimensions> const &coeffs,
               precision alpha, precision const x[], precision beta, precision y[],
               workspace<precision> &work,
               std::array<block_sparse_matrix<precision>, max_num_dimensions> const &);

/*!
 * \brief GPU implementation for the block-cpu evaluate
 *
 * Uses the same matrix across all dimensions
 */
template<typename precision>
void block_gpu(gpu::device dev, int n, sparse_grid const &grid,
               connection_patterns const &conns, permutes const &perm,
               gpu::vector<precision *> const &coeffs,
               precision alpha, precision const x[], precision beta, precision y[],
               workspace<precision> &work, block_sparse_matrix<precision> const &cmat);
#endif

} // namespace asgard::kronmult
