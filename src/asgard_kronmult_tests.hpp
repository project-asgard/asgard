
#pragma once

#include "./asgard_kronmult_matrix.hpp"
#include "./device/asgard_kronmult.hpp"
#include "lib_dispatch.hpp"
#include "tensors.hpp"

#include <iostream>
#include <random>

using namespace asgard::kronmult;

/*!
 * \brief Contains random inputs formatted for a call to kronmult.
 *
 * Contains mock up matrices and tensors that can directly feed into kronmult.
 */
template<typename T>
struct kronmult_intputs
{
  std::vector<int> pointer_map;

  std::vector<T> matrices;
  std::vector<T> input_x;
  std::vector<T> output_y;
  std::vector<T> reference_y;

  // vectors of pointers for reference solution
  std::vector<T *> pA;
  std::vector<T *> pX;
  std::vector<T *> pY;
};

/*!
 * \brief Reference implementation use for testing.
 *
 * Explicitly constructs the Kronecker product of two matrices.
 */
template<typename T>
std::vector<T> kronecker(int m, T const A[], int n, T const B[])
{
  std::vector<T> result(n * n * m * m);
  for (int jm = 0; jm < m; jm++)
  {
    for (int jn = 0; jn < n; jn++)
    {
      for (int im = 0; im < m; im++)
      {
        for (int in = 0; in < n; in++)
        {
          result[(jm * n + jn) * (m * n) + im * n + in] =
              A[jm * m + im] * B[jn * n + in];
        }
      }
    }
  }
  return result;
}

/*!
 * \brief Reference implementation one Kronecker product.
 */
template<typename T>
void reference_kronmult_one(int dimensions, int n, T const *const pA[],
                            T const x[], T y[])
{
  std::vector<T> kron(pA[dimensions - 1], pA[dimensions - 1] + n * n);
  int total_size = n;
  for (int i = dimensions - 2; i >= 0; i--)
  {
    kron = kronecker(n, pA[i], total_size, kron.data());
    total_size *= n;
  }
  char charN = 'n';
  T alpha = 1.0, beta = 1.0;
  int int_one = 1;
  asgard::lib_dispatch::gemv(charN, total_size, total_size, alpha, kron.data(),
                             total_size, x, int_one, beta, y, int_one);
}

/*!
 * \brief Reference implementation of kronmult, do not use in production.
 */
template<typename T>
void reference_kronmult(int dimensions, int n, T const *const pA[],
                        T const *const pX[], T *pY[], int const num_batch)
{
  for (int i = 0; i < num_batch; i++)
  {
    reference_kronmult_one(dimensions, n, &pA[dimensions * i], pX[i], pY[i]);
  }
}

/*!
 * \brief Generates data for a kronmult call using random inputs.
 *
 * \tparam T is double or float
 * \tparam compute_reference can be false to skip computing the reference solution,
 *         this is useful for benchmarking.
 *
 * \param dimensions is the number of dimensions of the tensors
 * \param n is the size of the matrix, e.g., 2 for linear and 3 for cubic basis
 * \param num_batch is the number of Kronecker products to compute
 * \param num_matrices is the total number of unique matrices,
 *        i.e., each Kronecker products will contain a subset of these
 * \param num_y is the total number of tensors, each Kronecker product
 *        will act on one tensor but multiple products can act on one output
 *
 * \returns data ready for input to kronmult, the data is wrapped inside
 *          a pointer as to avoid relocation which would invalidate
 *          the pA, pX and pY pointer arrays
 *
 */
template<typename T, bool compute_reference = true>
std::unique_ptr<kronmult_intputs<T>>
make_kronmult_data(int dimensions, int n, int num_rows, int num_terms,
                   int num_matrices)
{
  int num_batch = num_rows * num_rows * num_terms;
  std::minstd_rand park_miller(42);
  std::uniform_real_distribution<T> unif(-1.0, 1.0);
  std::uniform_real_distribution<T> uniy(0, num_rows - 1);
  std::uniform_real_distribution<T> unim(0, num_matrices - 1);

  int num_data = 1;
  for (int i = 0; i < dimensions; i++)
    num_data *= n;

  auto result         = std::make_unique<kronmult_intputs<T>>();
  result->pointer_map = std::vector<int>((dimensions + 2) * num_batch);
  result->matrices    = std::vector<T>(n * n * num_matrices);
  result->input_x     = std::vector<T>(num_data * num_rows);
  result->output_y    = std::vector<T>(num_data * num_rows);
  result->reference_y = std::vector<T>(num_data * num_rows);
  result->pA          = std::vector<T *>(dimensions * num_batch);
  result->pX          = std::vector<T *>(num_batch);
  result->pY          = std::vector<T *>(num_batch);

  // pointer_map has 2D structure with num_batch strips of size (d+2)
  // the first entry of each strip is the input x
  // the next d entries are the matrices
  // the final entry is the output y
  auto ip = result->pointer_map.begin();
  int iy  = -1;
  for (int i = 0; i < num_rows * num_rows; i++)
  {
    if (i % num_rows == 0)
      iy++;
    int const ix = i % num_rows;
    for (int t = 0; t < num_terms; t++)
    {
      *ip++ = ix;
      for (int j = 0; j < dimensions; j++)
        *ip++ = unim(park_miller);
      *ip++ = iy;
    }
  }

#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(result->matrices.size());
       i++)
    result->matrices[i] = unif(park_miller);

#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(result->input_x.size()); i++)
  {
    result->input_x[i]  = unif(park_miller);
    result->output_y[i] = unif(park_miller);
  }

  result->reference_y = result->output_y;

  ip = result->pointer_map.begin();
  for (int i = 0; i < num_batch; i++)
  {
    result->pX[i] = &(result->input_x[*ip++ * num_data]);
    for (int j = 0; j < dimensions; j++)
      result->pA[i * dimensions + j] = &(result->matrices[*ip++ * n * n]);
    result->pY[i] = &(result->reference_y[*ip++ * num_data]);
  }

  if (compute_reference)
    reference_kronmult(dimensions, n, result->pA.data(), result->pX.data(),
                       result->pY.data(), num_batch);

  ip = result->pointer_map.begin();
  for (int i = 0; i < num_batch; i++)
  {
    ip += dimensions + 1;
    result->pY[i] = &(result->output_y[*ip++ * num_data]);
  }

  return result;
}

/*!
 * \brief Contains random inputs formatted for a call to kronmult.
 *
 * Contains mock up matrices and tensors that can directly feed into kronmult.
 */
template<typename P>
struct kronmult_intputs_welem
{
  //! \brief Creates the element table and
  kronmult_intputs_welem(int dims, int ksize, int range) :
      num_dimensions(dims), kron_size(ksize), index_range(range),
      coefficients(index_range * index_range * kron_size * kron_size)
  {
    int total = index_range;
    for(int d=1; d<num_dimensions; d++)
      total *= index_range;

    // create a tensor list of elements
    elem.reserve(total * num_dimensions);
    std::vector<int> index(num_dimensions);
    for(int i=0; i<total; i++)
    {
      int t = i;
      for(int d=0; d<num_dimensions; d++)
      {
        index[d] = t % num_dimensions;
        t /= num_dimensions;
      }

      elem.insert(index.begin(), index.end(), elem.end());
    }

    std::minstd_rand park_miller(42);
    std::uniform_real_distribution<T> unif(-1.0, 1.0);

    for(auto &c : coefficients)
      c = unif(park_miller);
  }

  int num_dimensions, kron_size, index_range;
  std::vector<int> elem;

  std::vector<P> coefficients;
  std::vector<P> input_x;
  std::vector<P> output_y;
  std::vector<P> reference_y;

};

/*!
 * \brief Generates data for a kronmult call using random inputs.
 *
 * \tparam P is double or float
 * \tparam compute_reference can be false to skip computing the reference solution,
 *         this is useful for benchmarking.
 *
 * \param dimensions is the number of dimensions of the tensors
 * \param n is the size of the matrix, e.g., 2 for linear and 3 for cubic basis
 * \param num_rows is the number of rows in the matrix
 * \param num_terms is the number of terms of the PDE,
 *        i.e., the number of Kronecker products is num_rows^2 * num_terms
 * \param index_range is the maximum index that will be used in the element table
 *
 * \returns data ready for input to kronmult, the data is wrapped inside
 *          a pointer as to avoid relocation which would invalidate
 *          some of the pointers
 *
 */
template<typename P, bool compute_reference = true>
std::unique_ptr<kronmult_intputs_welem<T>>
make_kronmult_welem(int dimensions, int n, int num_rows, int num_terms,
                    int index_range)
{
  kronmult_intputs_welem result(dimensions, n, index_range);
}
