#pragma once

#include <iostream>
#include <set>

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult
{

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
 *          kron(vA[n * n * iA[i * d * T * num_rows + j * d * T + k * d]]
 *               ...
 *               vA[iA[i * d * T * num_rows + j * d * T + k * d + d - 1])
 *          * x[j * n^d ... (j+1) * n^d - 1]
 * T is the number of terms (num_terms)
 * i indexes the tensors in y, j the tensors in x,
 * both go from 0 to num_rows - 1
 * k indexes the operator terms (0 to T - 1)
 * and iA is the index of the small matrices of size n * n
 * all such matrices are stored in column-major format and stacked by
 * rows inside vA (i.e., there is one row of matrices)
 *
 * \tparam T is float or double
 *
 * \param dimensions must be between 1D and 6D (included)
 * \param n is the size of the problem, e.g., for linear basis n=2
 *        and cubic basis n=4
 *
 * \param num_rows is the number of rows of the matrix with
 */
template<typename T>
void cpu_dense(int const dimensions, int const n, int const num_rows,
               int const num_terms, int const iA[], T const vA[], T const alpha,
               T const x[], T const beta, T y[]);

#ifdef ASGARD_USE_CUDA
/*!
 * \brief Performs a batch of kronmult operations using a dense GPU matrix.
 *
 * The arrays iA, vA, x and y are stored on the GPU device.
 * The indexes and scalars alpha and beta are stored on the CPU.
 *
 * \b total_size is the total size of x and y, i.e., num_rows * n^dimensions
 */
template<typename T>
void gpu_dense(int const dimensions, int const n, int const total_size,
               int const num_rows, int const num_terms, int const iA[],
               T const vA[], T const alpha, T const x[], T const beta, T y[]);
#endif

} // namespace asgard::kronmult
