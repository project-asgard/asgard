#pragma once

#include <algorithm>
#include <iostream>
#include <set>

#include "asgard_indexset.hpp"
#include "asgard_interpolation1d.hpp"
#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult
{
/*!
 * \brief Applies the diagonal Euler preconditioner onto x.
 */
template<typename T>
void gpu_precon_jacobi(int64_t size, T dt, T const prec[], T x[]);

#ifndef KRON_MODE_GLOBAL
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
 * \brief Performs a batch of kronmult operations using a dense CPU matrix.
 *
 * This is the CPU implementation of the dense case.
 *
 * Takes a matrix where each entry is a Kronecker product and multiplies
 * it by a vector.
 * The matrix has size num_rows by num_rows times num_terms,
 * each row outputs in a tensor represented by a contiguous block within
 * y with size n^d, similarly x is comprised by linearized tensor blocks
 * with size n^d and each consecutive num_terms entries operate on the same
 * block in x.
 *
 * The short notation is that:
 * y[i * n^d ... (i+1) * n^d - 1] = beta * y[i * n^d ... (i+1) * n^d - 1]
 *      + alpha * sum_j sum_k
 *          kron(vA[t][n * n * (elem[j * dimensions] * num_1d_blocks
 *                              + elem[i * dimensions])]
 *               ...
 *              kron(vA[t][(dims - 1) * num_1d_blocks^2 * n * n +
 *                    n * n * (elem[j * dimensions + (dims - 1)] * num_1d_blocks
 *                                  + elem[i * dimensions + (dims - 1)])]
 *          * x[j * n^d ... (j+1) * n^d - 1]
 *
 * i indexes the tensors in y, j the tensors in x,
 * both go from 0 to num_rows - 1
 * t indexes the operator terms (0 to num_terms - 1)
 * vA[t] is the list of coefficients for this term, i.e., the n by n matrices
 * all such matrices are stored in column-major format and stacked by
 * rows inside vA (i.e., there is one row of matrices)
 *
 * \tparam P is float or double
 *
 * \param dimensions must be between 1D and 6D (included)
 * \param n is the size of the problem, e.g., for linear basis n=2
 *        and cubic basis n=4
 *
 * \param num_rows is the number of rows of the matrix
 * \param num_cols is the number of rows of the matrix
 * \param num_terms is the number of operator terms
 * \param elem is the list multi-indexes
 * \param row_offset is the offset inside elem of the first row multi-index
 * \param col_offset is the offset inside elem of the first row multi-index
 * \param vA is an array of arrays that holds all coefficients
 * \param num_1d_blocks is the number of cells in one-dimension
 */
template<typename T>
void cpu_dense(int const dimensions, int const n, int const num_rows,
               int const num_cols, int const num_terms, int const elem[],
               int const row_offset, int const col_offset, T const *const vA[],
               int const num_1d_blocks, T const alpha, T const x[],
               T const beta, T y[]);

/*!
 * \brief Sparse variant for the CPU.
 *
 * The inputs are the same with the exception of the pntr and indx
 * that describe a standard sparse matrix in row-compressed format.
 * The indexes cover the tensor, i.e., for the pair i, indx[pntr[i]]
 * the Y offset is i * tensor-size and the X one is indx[pntr[i]] * tensor-size
 * The length of pntr is num_rows+1 and indx is pntr[num_rows]
 */
template<typename T>
void cpu_sparse(int const dimensions, int const n, int const num_rows,
                int const pntr[], int const indx[], int const num_terms,
                int const iA[], T const vA[], T const alpha, T const x[],
                T const beta, T y[]);

#ifdef ASGARD_USE_CUDA
/*!
 * \brief Performs a batch of kronmult operations using a dense GPU matrix.
 *
 * The arrays iA, vA, x and y are stored on the GPU device.
 * The indexes and scalars alpha and beta are stored on the CPU.
 *
 * \b output_size is the total size of y, i.e., num_rows * n^dimensions
 *
 * \b num_batch is the product num_cols times num_rows
 */
template<typename P>
void gpu_dense(int const dimensions, int const n, int const output_size,
               int64_t const num_batch, int const num_cols, int const num_terms,
               int const elem[], int const row_offset, int const col_offset,
               P const *const vA[], int const num_1d_blocks, P const alpha,
               P const x[], P const beta, P y[]);

/*!
 * \brief Sparse variant for the GPU.
 *
 * The inputs are the same with the exception of the ix and iy that hold the
 * offsets of the tensors for each product in the batch.
 * The tensors for the i-th product are at ix[i] and iy[i] and there no need
 * for multiplication by the tensor-size, also the length of ix[] and iy[]
 * matches and equals num_batch.
 */
template<typename T>
void gpu_sparse(int const dimensions, int const n, int const output_size,
                int const num_batch, int const ix[], int const iy[],
                int const num_terms, int const iA[], T const vA[],
                T const alpha, T const x[], T const beta, T y[]);
#endif
#endif

#ifdef KRON_MODE_GLOBAL
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
  permutes(std::vector<int> const &active_dirs)
      : permutes(static_cast<int>(active_dirs.size()))
  {
    remap_directions(active_dirs);
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
};

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
struct block_global_workspace
{
  std::vector<precision> x, y;
  std::vector<precision> w1, w2;
  std::vector<std::vector<int64_t>> row_map;
};

// block-size = n^num_dimensions but saves work recomputing
template<typename precision>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                std::vector<permutes> const &perms,
                std::vector<int> const &flux_dir,
                connect_1d const &conn_volumes, connect_1d const &conn_full,
                std::vector<std::vector<precision>> const &gvals,
                std::vector<int> const &terms,
                precision const x[], precision y[],
                block_global_workspace<precision> &workspace);

template<typename precision>
int64_t block_global_count_flops(
    int num_dimensions, int64_t block_size,
    vector2d<int> const &ilist, dimension_sort const &dsort,
    std::vector<permutes> const &perms,
    std::vector<int> const &flux_dir,
    connect_1d const &conn_volumes, connect_1d const &conn_full,
    std::vector<int> const &terms,
    block_global_workspace<precision> &workspace);

template<typename precision>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                permutes const &perm, connect_1d const &vconn,
                precision const gvals[], precision alpha, precision const x[],
                precision y[], block_global_workspace<precision> &workspace);

template<typename precision>
void global_cpu(int num_dimensions, int n, int64_t block_size,
                vector2d<int> const &ilist, dimension_sort const &dsort,
                permutes const &perm, connect_1d const &vconn,
                precision const gvals[], precision const x[], precision y[],
                block_global_workspace<precision> &workspace);

template<typename precision>
void globalsv_cpu(int num_dimensions, int n, vector2d<int> const &ilist,
                  dimension_sort const &dsort, connect_1d const &vconn,
                  precision const gvals[], precision y[],
                  block_global_workspace<precision> &workspace);

#endif

} // namespace asgard::kronmult
