#pragma once

#include "asgard_block_matrix.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_algorithms.hpp"
#endif

namespace asgard::kronmult
{
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
  //! \brief Matrix fill for each operation.
  std::vector<std::vector<conn_fill>> fill;
  //! \brief Direction for each matrix operation.
  std::vector<std::vector<int>> direction;
  //! \brief Direction of the flux, if any
  int flux_dir = -1;
  //! \brief Empty permutation list.
  permutes() = default;
  //! \brief Initialize the permutations.
  permutes(int num_dimensions);
  //! \brief Create uniform transformation, only lower or upper.
  permutes(int num_dimensions, conn_fill same_fill);
  //! Creates a transformation with the specified active and flux directions.
  permutes(std::vector<int> const &active_dirs, int fdir = -1)
      : permutes(static_cast<int>(active_dirs.size()))
  {
    remap_directions(active_dirs);
    flux_dir = fdir;
  }
  //! \brief (debugging) Convert the fill to a string.
  std::string_view fill_name(int perm, int stage) const;
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
  //! \brief Pads all permutations with the given dimensions and assuming upper matrices
  void prepad_upper(std::vector<int> const &additional);
  //! \brief Indicates if the permutation has been set
  operator bool () const { return not direction.empty(); }
};

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
                  permutes const &perm, workspace<precision> &work);
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

#ifdef ASGARD_GPU_GREEDY
/*!
 * \brief Uses the CPU to compute the connection pattern for all perms
 *
 * While this is executed on the CPU, it caches the connection patter for the greedy
 * GPU kernels.
 */
template<typename precision>
void connect_cpu(gpu::device dev, sparse_grid const &grid, connection_patterns const &conns,
                 permutes const &perm, workspace<precision> &work);
/*!
 * \brief GPU implementation that indexes the blocks
 */
template<typename precision>
void block_gpu(gpu::device dev, sparse_grid const &grid,
               connection_patterns const &conns, permutes const &perm,
               std::array<gpu::vector<precision>, max_num_dimensions> const &coeffs,
               workspace<precision> &work,
               std::array<block_sparse_matrix<precision>, max_num_dimensions> const &);
#endif

#endif

} // namespace asgard::kronmult
