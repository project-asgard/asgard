#pragma once

#include <iostream>
#include <set>

#include "asgard_kronmult_common.hpp"


namespace asgard::kronmult
{
#ifdef ASGARD_USE_CUDA

template<typename T>
void execute_gpu(int dimensions, int n,
                 T const *const pA[], int const lda, T *pX[], T *pY[],
                 int const num_batch);
#endif

struct is_implemented{
    static bool cpu(int dimensions, int n){
        std::set<int> available({201, 301, 401, 202, 302});
        return (available.find(100 * n + dimensions) != available.end());
    }
    static bool gpu(int dimensions, int n){
        std::set<int> available(
            {201, 301, 401,
             202, 302, 402, 502, 602, 702, 802, 902,
             1002, 1102, 1202, 1302, 1402, 1502, 1602, 1702, 1802, 1902,
             2002, 2102, 2202, 2302, 2402, 2502, 2602, 2702, 2802, 2902, 3002, 3102, 3202,
             203, 303, 403, 204});
        return (available.find(100 * n + dimensions) != available.end());
    }
};

template<typename T>
void execute_cpu(int dimensions, int n,
                 T const *const pA[], int const lda, T *pX[], T *pY[],
                 int const num_batch);

} // namespace asgard::kronmult
