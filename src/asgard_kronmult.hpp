#pragma once

#include <algorithm>
#include <iostream>
#include <set>

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

/*!
 * \brief Computes the inverse-action of a sparse Kronecker onto a vector
 *
 * Computes y = inv(A) * y, where A is a sparse Kronecker matrix.
 *
 * \tparam precision is float or double
 *
 * \param n is the size of the block, e.g., 1 for degree 0, 2 for linear basis and so on.
 * \param grid is the current sparse grid
 * \param volume_conn is the 1d volume connection pattern
 * \param gvlas defines the matrix to invert, gvlas is unit-block-lower-triangular
 *              and A is defined by the negative of gvlas
 * \param y is the vector to apply the inverse onto
 * \param work is initialized workspace
 */
template<typename precision>
void blocksv_cpu(int n, sparse_grid const &grid,
                 connect_1d const &volume_conn,
                 block_sparse_matrix<precision> const &gvals,
                 precision y[], workspace<precision> &work);

} // namespace asgard::kronmult
