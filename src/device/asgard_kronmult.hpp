#pragma once

#include <iostream>
#include <set>

#include "asgard_kronmult_common.hpp"

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
 * \brief Indicates whether specific kernel has been implemented.
 *
 * Will be removed when we switch to only one kronmult implementation.
 */
struct is_implemented
{
  static bool gpu(int dimensions, int n)
  {
    std::set<int> available(
        {201,  301,  401,  501,  601,  701,  801,  901,  1001, 202,  302,  402,
         502,  602,  702,  802,  902,  1002, 1102, 1202, 1302, 1402, 1502, 1602,
         1702, 1802, 1902, 2002, 2102, 2202, 2302, 2402, 2502, 2602, 2702, 2802,
         2902, 3002, 3102, 3202, 203,  303,  403,  503,  603,  703,  803,  903,
         1003, 204,  304,  404,  504,  205,  305,  405,  206,  306,  406});
    return (available.find(100 * n + dimensions) != available.end());
  }
};

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
