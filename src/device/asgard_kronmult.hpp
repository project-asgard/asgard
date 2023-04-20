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
        std::set<int> available({201, 301, 401, 202, 302, 402, 203, 303, 403, 204});
        return (available.find(100 * n + dimensions) != available.end());
    }
};

template<typename T>
void execute_cpu(int dimensions, int n,
                 T const *const pA[], int const lda, T *pX[], T *pY[],
                 int const num_batch);

} // namespace asgard::kronmult
