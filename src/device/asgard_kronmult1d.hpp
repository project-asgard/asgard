#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{
#ifdef USE_GPU

/*
 * The idea behind the algorithm is similar for all kernels.
 * Data is moved from RAM to the __shared__ cache,
 * data is operated on in the cache and written back.
 * Most of the integer logic is done once per thread-block.
 *
 * Common nomenclature:
 * \tparam T is the type, float or double
 * \tparam num_threads is the number of threads that the kernel will launch,
 *         making it a template parameter allows some operations to happen at
 * compile time.
 * \tparam n is the size of the small matrix which corresponds to the parameters
 *         of the polynomial basis, 2 for linear, 3 for quadratic, 4 for cubic.
 *
 * \param pA is pointer to all matrices for the products in the batch
 * \param lda is the leading dimension of the matrices
 *        (this leads to a massive bottleneck, the matrices should be
 *         stored in a format where lda is equal to n)
 * \param pX are pointer to the X entries
 * \param pY are pointers to the Y entries
 * \param num_batch is the number of entries of the batch
 *
 * Internally, some indexes are always the same:
 * - i is the global index of the kron product within the batch
 * - j is index of this thread within the group of threads that work on the
 * batch
 * - locali is the same but within the thread-block
 *          used to find the entries in the cache
 * - matj is the index of the matrix that will be read from memory
 *        note that the 1D kernel does not read the matrix
 */
template<typename T, int num_threads, int n>
__global__ void gpu1d(T const *const pA[], int const lda, T const *const pX[],
                      T *pY[], int const num_batch)
{
  static_assert(n == 2 or n == 3 or n == 4,
                "kernel works only for n = 2, 3, 4");
  static_assert(n != 3 or (n == 3 and num_threads == 32),
                "restriction on warp size limit this kernel to 32 threads");

  __shared__ T X[num_threads];

  int locali = threadIdx.x / n;
  int i = locali + blockIdx.x * (num_threads / n); // index within the batch
  int j = threadIdx.x % n;  // indicated whether this is an even or odd thread
  int localx0 = n * locali; // the entry of x within the cache
  int locala1 = lda + j;
  int locala2 = locala1 + lda;
  int locala3 = locala2 + lda;
  if constexpr (n == 3)
  { // done at compile time since n is a template parameter
    // disable the last two threads of the warp since 32 does not divide into 3
    if (threadIdx.x >= 30)
    {
      i = num_batch;
    }
  }

  while (i < num_batch)
  {
    X[threadIdx.x] = pX[i][j]; // read the X, every 2 threads read consecutive
                               // entries and store in cache

    T yinc;
    if constexpr (n == 2)
    { // done at compile time
      yinc = pA[i][j] * X[localx0] + pA[i][locala1] * X[localx0 + 1];
    }
    else if constexpr (n == 3)
    {
      yinc = pA[i][j] * X[localx0] + pA[i][locala1] * X[localx0 + 1] +
             pA[i][locala2] * X[localx0 + 2];
    }
    else if constexpr (n == 4)
    {
      yinc = pA[i][j] * X[localx0] + pA[i][locala1] * X[localx0 + 1] +
             pA[i][locala2] * X[localx0 + 2] + pA[i][locala3] * X[localx0 + 3];
    }

    atomicAdd(&pY[i][j], yinc);

    i += gridDim.x * (num_threads / n);
  }
}

#endif

} // namespace asgard::kronmult::kernel
