
#pragma once

#include "./device/asgard_kronmult.hpp"
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
  int num_batch;

  std::vector<T> matrices;
  std::vector<T> input_x;
  std::vector<T> output_y;
  std::vector<T> reference_y;

  // vectors of pointers on the CPU
  std::vector<T *> pA;
  std::vector<T *> pX;
  std::vector<T *> pY;

#ifdef ASGARD_USE_CUDA
  // copy of the data on the GPU
  asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> gpux;
  asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> gpuy;
  asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> gpum;

  asgard::fk::vector<T *, asgard::mem_type::owner, asgard::resource::device>
      gpupX;
  asgard::fk::vector<T *, asgard::mem_type::owner, asgard::resource::device>
      gpupY;
  asgard::fk::vector<T *, asgard::mem_type::owner, asgard::resource::device>
      gpupA;
#endif
};

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
make_kronmult_data(int dimensions, int n, int num_batch, int num_matrices,
                   int num_y)
{
  std::minstd_rand park_miller(42);
  std::uniform_real_distribution<T> unif(-1.0, 1.0);
  std::uniform_real_distribution<T> uniy(0, num_y - 1);
  std::uniform_real_distribution<T> unim(0, num_matrices - 1);

  int num_data = 1;
  for (int i = 0; i < dimensions; i++)
    num_data *= n;

  std::vector<int> pointer_map((dimensions + 2) * num_batch);

  auto result         = std::make_unique<kronmult_intputs<T>>();
  result->num_batch   = num_batch;
  result->matrices    = std::vector<T>(n * n * num_matrices);
  result->input_x     = std::vector<T>(num_data * num_y);
  result->output_y    = std::vector<T>(num_data * num_y);
  result->reference_y = std::vector<T>(num_data * num_y);
  result->pA          = std::vector<T *>(dimensions * num_batch);
  result->pX          = std::vector<T *>(num_batch);
  result->pY          = std::vector<T *>(num_batch);

  // pointer_map has 2D structure with num_batch strips of size (d+2)
  // the first entry of each strip is the input x
  // the next d entries are the matrices
  // the final entry is the output y
  auto ip = pointer_map.begin();
  for (int i = 0; i < num_batch; i++)
  {
    *ip++ = uniy(park_miller);
    for (int j = 0; j < dimensions; j++)
      *ip++ = unim(park_miller);
    *ip++ = uniy(park_miller);
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

  ip = pointer_map.begin();
  for (int i = 0; i < num_batch; i++)
  {
    result->pX[i] = &(result->input_x[*ip++ * num_data]);
    for (int j = 0; j < dimensions; j++)
      result->pA[i * dimensions + j] = &(result->matrices[*ip++ * n * n]);
    result->pY[i] = &(result->reference_y[*ip++ * num_data]);
  }

  reference_kronmult(dimensions, n, result->pA.data(), result->pX.data(),
                     result->pY.data(), result->num_batch);

  ip = pointer_map.begin();
  for (int i = 0; i < num_batch; i++)
  {
    ip += dimensions + 1;
    result->pY[i] = &(result->output_y[*ip++ * num_data]);
  }

#ifdef ASGARD_USE_CUDA

  result->gpuy = asgard::fk::vector<T>(result->output_y).clone_onto_device();
  result->gpux = asgard::fk::vector<T>(result->input_x).clone_onto_device();
  result->gpum = asgard::fk::vector<T>(result->matrices).clone_onto_device();

  std::vector<T *> pX(result->pX.size());
  std::vector<T *> pY(result->pY.size());
  std::vector<T *> pA(result->pA.size());

  ip = pointer_map.begin();
  for (int i = 0; i < num_batch; i++)
  {
    pX[i] = result->gpux.begin() + *ip++ * num_data;
    for (int j = 0; j < dimensions; j++)
      pA[i * dimensions + j] = result->gpum.begin() + *ip++ * n * n;
    pY[i] = result->gpuy.begin() + *ip++ * num_data;
  }

  result->gpupX = asgard::fk::vector<T *>(pX).clone_onto_device();
  result->gpupY = asgard::fk::vector<T *>(pY).clone_onto_device();
  result->gpupA = asgard::fk::vector<T *>(pA).clone_onto_device();

#endif

  return result;
}
