#pragma once

#include <iostream>
#include <set>

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult
{

/*!
 * \brief Indicates how to interpret the beta parameter.
 *
 * Matrix operations include scalar parameters, e.g., \b beta \b y.
 * Flops can be saved in special cases and those are in turn
 * handled with template parameters and if-constexpr clauses.
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

#ifdef ASGARD_USE_CUDA


#endif


/*!
 * \brief Handles the special case of n=1
 *
 * When n=1 the Kronecker product is just a product of constants,
 * we would be much better off expressing the problem as a sparse matrix
 * and using optimized sparse linear algebra libraries.
 * Nevertheless, this implementation is provided for completeness,
 * where the case n=1 is handled in the same data structures as n>1.
 *
 * TODO: can fix that in the kronmult_matrix factory and switch to dense
 *       matrix-matrix implementation.
 */
template<typename T, scalar_case alpha_case, scalar_case beta_case>
void cpu_n0(int const dimensions, int const num_rows, int const num_terms,
            int const iA[], T const vA[], T const alpha, T const x[],
            T const beta, T y[]);

/*!
 * \brief Baseline kronmult algorithm on the CPU (dense case).
 *
 * The template handles up to 6D problems and could handle arbitrary n,
 * however, only n = 2, 3, 4 is instantiated in asgard_kronmult_cpu.cpp.
 * Fixing n as a template parameter allows for automatic unroll of
 * the for-loop of the general algorithm, but also significantly
 * increases the compile times.
 *
 * \tparam T is either float or double
 * \tparam dimensions is between 1 and 6
 * \tparam n is the size of the small matrices
 * \tparam alpha_case must be one if alpha is 1 and neg_one if alpha is -1,
 *         otherwise it must be scalar_case::other
 *         alpha_case cannot be scalar_case::zero since that means
 *         no multiplication
 * \tparam beta_case must match beta, one for beta = 1, neg_one for beta = -1,
 *         zero for beta = 0 and other in all other cases
 */
template<typename T, int dimensions, int n,
         scalar_case alpha_case, scalar_case beta_case>
void cpu_dense(int const num_rows, int const num_terms, int const iA[],
               T const vA[], T const alpha, T const x[], T const beta,
               T y[]);

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
 * all such matrices are stored in row-major format and stacked in columns
 * inside vA
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

} // namespace asgard::kronmult
