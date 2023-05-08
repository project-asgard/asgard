#pragma once

#include <iostream>
#include <set>

#include "asgard_kronmult_common.hpp"
#include "asgard_kronmult_v2.hpp"

namespace asgard::kronmult
{
#ifdef ASGARD_USE_CUDA

/*!
 * \brief Performs a batch of kronmult operations using the GPU.
 *
 * See execute_cpu() for details.
 */
template<typename T>
void execute_gpu(int dimensions, int n, T const *const pA[], int const lda,
                 T *pX[], T *pY[], int const num_batch,
                 int const output_stride);
#endif

/*!
 * \brief Handles the special case of n=1
 *
 * When n=1 the Kronecker product is just a product of constants,
 * we would be much better off expressing the problem as a sparse matrix
 * and using optimized sparse linear algebra libraries.
 * Nevertheless, this implementation is provided for completeness,
 * where the case n=1 is handled in the same data structures as n>1.
 */
template<typename T>
void run_cpu_variant0(int dimensions, T const *const pA[], T const *const pX[],
                      T *pY[], int const num_batch, int const output_length);

/*!
 * \brief Baseline kronmult algorithm on the CPU.
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
 *
 * The rest of the inputs are the same as in execute_cpu()
 */
template<typename T, int dimensions, int n>
void run_cpu_variant(T const *const pA[], int const lda, T const *const pX[],
                     T *pY[], int const num_batch, int const output_length);

/*!
 * \brief Kronmult algorithm that allows for arbitrary n.
 *
 * Overload to run_cpu_variant but \n n is defined at runtime,
 * while the algorithm is more general in that sense,
 * it comes with significant slowdown due to runtime indexing and
 * for-loops that cannot be unrolled.
 */
template<typename T, int dimensions>
void run_cpu_variant(int n, T const *const pA[], int const lda,
                     T const *const pX[], T *pY[], int const num_batch,
                     int const output_length);

/*!
 * \brief Performs a batch of kronmult operations using the CPU.
 *
 * This is the CPU implementation, the only difference in the GPU version
 * is that the pointer arrays must be loaded on the GPU device.
 *
 * \tparam T is float or double
 *
 * \param dimensions must be between 1D and 6D (included)
 * \param n is the size of the problem, e.g., for linear basis n=2
 *        and cubic basis n=4
 *
 * \param pA pointer array to the matrices associated with the kron products,
 *        the matrices for the i-th entry of the batch are located at
 *        pA[dims * i] ... pA[dims * i + (dims-1)]
 *        where pA[dims * i + (dims-1)] is the last matrix and
 *        is applied in non-transpose format
 * \param lda is the leading dimension of A
 * \param pX is the pointer to the input tensors
 * \param pY is the pointer to the output tensors
 * \param num_batch is the number of kron entries in this batch
 * \param output_stride number of consecutive outputs in pY
 */
template<typename T>
void execute_cpu(int dimensions, int n, T const *const pA[], int const lda,
                 T const *const pX[], T *pY[], int const num_batch,
                 int const output_stride);

} // namespace asgard::kronmult
