#pragma once

#include <iostream>
#include <vector>

#ifdef USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_60_atomic_functions.h>

#endif

namespace asgard::kronmult
{
/*!
 * \brief Computes the number of CUDA blocks.
 *
 * \param work_size is the total amount of work, e.g., size of the batch
 * \param work_per_block is the work that a single thread block will execute
 * \param max_blocks is the maximum number of blocks
 */
inline int blocks(int work_size, int work_per_block, int max_blocks)
{
  return std::min(max_blocks,
                  (work_size + work_per_block - 1) / work_per_block);
}

template<typename T>
std::vector<T> kronecker(int m, std::vector<T> const &A, int n, std::vector<T> const &B){
  std::vector<T> result(n*n*m*m);
  for(int jm=0; jm<m; jm++){
    for(int jn=0; jn<n; jn++){
      for(int im=0; im<m; im++){
        for(int in=0; in<n; in++){
          result[ (jm * n + jn) * (m*n) + im * n + in ]
            = A[jm * m + im] * B[jn * n + in];
        }
      }
    }
  }
  return result;
}

template<typename T>
void reference_gemv(int n, std::vector<T> const &A, std::vector<T> const &x, std::vector<T> &y){
  for(int j=0; j<n; j++){
    for(int i=0; i<n; i++){
      y[i] += A[j*n + i] * x[j];
    }
  }
}

} // namespace asgard::kronmult

// TODO add intrinsics here too for the CPU
