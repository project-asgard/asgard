#pragma once

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult::kernel
{

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

  constexpr int team_size = 32;
  constexpr int i_per_block =
      (n == 3) ? (10 * (num_threads / team_size)) : (num_threads / n);

  __shared__ T X[num_threads];

  // i is the index of the batch, locali is the index within the thread-block
  int locali;
  if constexpr (n == 3)
  {
    locali = 10 * (threadIdx.x / team_size) + (threadIdx.x % team_size) / n;
  }
  else
  {
    locali = threadIdx.x / n;
  }

  int i = locali + blockIdx.x * i_per_block; // index within the batch
  int j; // indicated whether this is an even or odd thread
  if constexpr (n == 3)
  {
    j = (threadIdx.x % team_size) % n;
  }
  else
  {
    j = threadIdx.x % n;
  }
  int localx0; // the entry of x within the cache
  if constexpr (n == 3)
  {
    localx0 = team_size * (threadIdx.x / team_size) +
              n * ((threadIdx.x % team_size) / n);
  }
  else
  {
    localx0 = n * locali;
  }

  int locala1 = lda + j;
  int locala2 = locala1 + lda;
  int locala3 = locala2 + lda;
  if constexpr (n == 3)
  { // done at compile time since n is a template parameter
    // disable the last two threads of the warp since 32 does not divide into 3
    if (threadIdx.x % 32 >= 30)
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

    i += gridDim.x * i_per_block;
  }
}

} // namespace asgard::kronmult::kernel
