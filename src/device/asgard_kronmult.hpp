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

template<typename T>
void execute_cpu(int dimensions, int n,
                 T const *const pA[], int const lda, T *pX[], T *pY[],
                 int const num_batch);

} // namespace asgard::kronmult
