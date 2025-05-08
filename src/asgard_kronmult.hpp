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
 * \brief Perform global Kronecked product
 *
 * Reference algorithm using the multi-index data-structures directly.
 *
 * The permutations between upper/lower parts and the order of the directions
 * is stored in \b kron_permute.
 *
 * The definition of the sparsity pattern and sets is the same as in
 * global_kron_1d().
 * The vals contains a vector for each dimension.
 *
 * The result is y += sum_{t in terms} alpha * mat_t * x
 * i.e., one such operation has to be applied for each term.
 *
 * The size of the workspace must be twice the size of x/y,
 * i.e., it must match 2 * iset.num_indexes()
 */
template<typename precision>
void global_cpu(permutes const &perms,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                connect_1d const &conn, std::vector<int> const &terms,
                std::vector<std::vector<precision>> const &vals,
                precision alpha, precision const *x, precision *y,
                precision *worspace1, precision *worspace2);

/*!
 * \brief Perform global Kronecked product
 *
 * Fast algorithm, using a sparsity pattern loaded into the vectors.
 *
 * The index vector lists gpntr, gindx, gdiag hold a vector for each dimension,
 * this is the common part of the sparse matrices.
 * The values gvals are number-of-terms X number-of-dimensions.
 *
 * terms gives the subset of terms to use for this operation
 *
 * computes y += A * x
 */
template<typename precision>
void global_cpu(int num_dimensions,
                std::vector<permutes> const &perms,
                std::vector<std::vector<int>> const &gpntr,
                std::vector<std::vector<int>> const &gindx,
                std::vector<std::vector<int>> const &gdiag,
                std::vector<std::vector<precision>> const &gvals,
                std::vector<int> const &terms, precision const *x, precision *y,
                precision *worspace1, precision *worspace2);

template<typename precision>
struct workspace
{
  std::vector<precision> x, y; // TODO: rename for the v2
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
