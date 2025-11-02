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
  //! \brief Holds a permutation step, single matrix applied across a direction.
  struct step {
    //! The matrix fill for the operation.
    conn_fill fill = conn_fill::both;
    //! The direction of the operation.
    int direction = 0;
  };
  //! Holds the permutation steps.
  vector2d<step> ops;
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
    return ops.stride();
  }
  //! \brief Reindexes the dimensions to match the active (non-identity) dimensions
  void remap_directions(std::vector<int> const &active_dirs)
  {
    expect(static_cast<size_t>(ops.stride()) == active_dirs.size());
    for (int i = 0; i < ops.num_strips(); i++) {
      step *sweep = ops[i];
      for (int d = 0; d < ops.stride(); d++) {
        int dir = sweep[d].direction;
        sweep[d].direction = active_dirs[dir];
      }
    }
  }
  //! \brief Pads all permutations with the given dimensions and assuming upper matrices
  void prepad_upper(std::vector<int> const &additional);
  //! \brief Indicates if the permutation has been set
  operator bool () const { return (ops.stride() > 0); }
  //! \brief Get the stage of the i-th step
  step operator() (int i, int stage) const { return ops[i][stage]; }
  //! \brief Return the number of permutation steps
  int64_t size() const { return ops.num_strips(); }
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
 * \tparam coeff_type is either a single block_sparse_matrix indicating identical matrix
 *                    applied to all dimensions or array of block_sparse_matrix with size
 *                    max_num_dimensions
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
template<typename precision, typename coeff_type>
void block_cpu(int n, sparse_grid const &grid, connection_patterns const &conns,
               permutes const &perm, coeff_type const &cmats,
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
 *
 * The GPU algorithm has two modes, memory-greedy (default) and low-memory.
 * The greedy approach uses explicit indexing, which exhaust the memory for any
 * sufficiently large problem, but the low-memory is obviously slower.
 * For the greedy method, coeff_type is either a single gpu vector of precision entries
 * holding the matrix coefficients, or an array of one vector per dimension,
 * conversely the low-memory variant holds a gpu vector of pointers to the matrices
 * at different levels. In both cases, the backup_type is either a single block-sparse matrix
 * or an array of one matrix per dimension.
 */
template<typename precision, typename coeff_type, typename backup_type>
void block_gpu(gpu::device dev, int n, sparse_grid const &grid,
               connection_patterns const &conns, permutes const &perm,
               coeff_type const &coeffs,
               precision alpha, precision const x[], precision beta, precision y[],
               workspace<precision> &work, backup_type const &);

#ifdef ASGARD_GPU_MEMGREEDY
/*!
 * \brief Uses the CPU to compute the connection pattern for all perms
 *
 * While this is executed on the CPU, it caches the connection patter for the greedy
 * GPU kernels.
 */
template<typename precision>
void connect_cpu(gpu::device dev, sparse_grid const &grid, connection_patterns const &conns,
                 permutes const &perm, workspace<precision> &work);

#endif

#endif

} // namespace asgard::kronmult
