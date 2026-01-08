#pragma once

#include "asgard_gpu_algorithms.hpp"

// Additional GPU kernels, related to more advanced tensor operations, e.g.,
// associated with sources and moments

namespace asgard::gpu
{

/*!
 * \brief Tensors the vectors into the output according to the multi-indexes
 *
 * Used in the construction of the sources, the constant vectors (separable) in
 * each direction are given by c1 ... c6 (only the first num_dim are used).
 * The sparse grid is given by num_indexes and indexes[] and n is the polynomial
 * degrees of freedom, e.g., n = 1 for constant basis, n = 3 for quadratic basis, etc.
 * The output is stored in x.
 *
 * The size of indexes[] must be num_dims * num_indexes.
 * The sizes of the c1 ... c(num_dims) must be n * max-level, the rest can be nullptr.
 * The size of x must be fm::ipow(n, num_dims) * num_indexes.
 */
template<typename P>
void tensor_by_index(int n, int num_dims, int num_indexes, int const indexes[],
                     P const c1[], P const c2[], P const c3[], P const c4[], P const c5[], P const c6[],
                     P x[]);

/*!
 * \brief Computes the moments from the state into vals, assumes level-zero is sufficient
 *
 * Assuming the polynomials order is sufficient to compute the moment from only level 0,
 * computes the moment.
 *
 * \tparam P is float or double
 *
 * \param pdof is the polynomial degrees of freedom
 * \param pos_block is the position block, e.g., fm::ipow(pdof, position-dimensions)
 * \param full_block is the full block, e.g., fm::ipow(pdof, num-all-dimensions)
 * \param vdims is the velocity dimensions
 * \param rij is the map for blocks, (i, j) = rij(2 * k, 2 * k + 1),
 *        then state block j corresponds to the vals block i
 * \param integ0 is the integrals for velocity dimension 0
 * \param integ1 is the integrals for velocity dimension 1, used only if vdims >= 2
 * \param integ2 is the integrals for velocity dimension 2, used only if vdims >= 3
 *
 * \param state is the current state using both position and velocity dimensions
 * \param vals is the output using only position dimensions
 */
template<typename P>
void moment_reduce_zero(int pdof, int pos_block, int full_block, int vdims,
                        gpu::vector<int> const &rij,
                        std::array<P const *, max_mom_dims> const &integ,
                        gpu::vector<P> const &state, gpu::vector<P> &vals);

/*!
 * \brief Computes the moments from the state into vals, assumes some dims don't use level 0
 *
 * Assuming at least in one direction the polynomial order is not sufficient to compute
 * the moments from the level 0 data.
 *
 * \tparam P is float or double
 *
 * \param pdof is the polynomial degrees of freedom
 * \param pos_block is the position block, e.g., fm::ipow(pdof, position-dimensions)
 * \param full_block is the full block, e.g., fm::ipow(pdof, num-all-dimensions)
 * \param pdims is the position dimensions
 * \param vdims is the velocity dimensions
 * \param lzero is an array with true/false in each direction indicating if going above level 0
 * \param indexes are the sparse grid multi-indexes, size is (pdims + vdims) * max-i-index-in-rij
 * \param rij is the map for blocks, (i, j) = rij(2 * k, 2 * k + 1),
 *        then state block j corresponds to the vals block i
 * \param integ0 is the integrals for velocity dimension 0
 * \param integ1 is the integrals for velocity dimension 1, used only if vdims >= 2
 * \param integ2 is the integrals for velocity dimension 2, used only if vdims >= 3
 *
 * \param state is the current state using both position and velocity dimensions
 * \param vals is the output using only position dimensions
 */
template<typename P>
void moment_reduce(int pdof, int pos_block, int full_block, int pdims, int vdims,
                   std::array<bool, max_mom_dims> lzero, int const *indexes,
                   gpu::vector<int> const &rij,
                   std::array<P const *, max_mom_dims> const &integ,
                   gpu::vector<P> const &state, gpu::vector<P> &vals);

/*!
 * \brief Expands the point-wise values of the moment to the full grid
 *
 * Given the nodal data at the position grid, expands the moment to the full grid.
 *
 * \tparam P is float or double
 *
 * \param pdof is the polynomial degrees of freedom
 * \param num_pos position dimensions
 * \param num_vel velocity dimensions
 * \param rij is the map for blocks, (i, j) = rij(2 * k, 2 * k + 1),
 *        then state block j corresponds to the vals block i
 * \param pos_data is defined on the position grid
 * \param vals is the result on the full grid
 */
template<typename P>
void moment_expand(int pdof, int num_pos, int num_vel, gpu::vector<int> const &rij,
                   gpu::vector<P> const &pos_data, gpu::vector<P> &vals);

}
