#pragma once

#include <iostream>
#include <set>

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult
{
#ifdef ASGARD_USE_CUDA

template<typename T>
void execute_gpu(int dimensions, int n, T const *const pA[], int const lda,
                 T *pX[], T *pY[], int const num_batch);
#endif

/*!
 * \brief Indicates whether specific kernel has been implemented.
 *
 * Will be removed when we switch to only one kronmult implementation.
 */
struct is_implemented
{
  static bool cpu(int dimensions, int n)
  {
    std::set<int> available({201, 301, 401, 202, 302, 402, 203, 303, 403,
                             204, 304, 404, 205, 305, 405, 206, 306, 406});
    return (available.find(100 * n + dimensions) != available.end());
  }
  static bool gpu(int dimensions, int n)
  {
    std::set<int> available({
        201,  301,  401,  501,  601,  701,  801,  901,  1001, 202,  302,  402,
        502,  602,  702,  802,  902,  1002, 1102, 1202, 1302, 1402, 1502, 1602,
        1702, 1802, 1902, 2002, 2102, 2202, 2302, 2402, 2502, 2602, 2702, 2802,
        2902, 3002, 3102, 3202, 203,  303,  403,  503,  603,  703,  803,  903,
        1003, 204,  304,  404,  504,  205,  305,  405,  206,  306,
    });
    return (available.find(100 * n + dimensions) != available.end());
  }
};

template<typename T, int dimensions, int n>
void run_cpu_variant(T const *const pA[], int const lda,
                     T *pX[], T *pY[], int const num_batch);

template<typename T>
void execute_cpu(int dimensions, int n, T const *const pA[], int const lda,
                 T *pX[], T *pY[], int const num_batch);

} // namespace asgard::kronmult
