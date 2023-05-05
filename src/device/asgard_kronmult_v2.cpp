#include <iostream>
#include <set>

#include "build_info.hpp"

#include "asgard_kronmult_v2.hpp"
#include "asgard_kronmult_cpu_kernels.hpp"

namespace asgard::kronmult
{

/*!
 * \brief Helper method that instantiates correct kernel based on alpha and beta.
 */
template<typename T>
void cpu_n0(int const d, int const rows, int const terms,
            int const iA[], T const vA[], T const alpha, T const x[],
            T const beta, T y[]){
  if (beta == 0){
    if (alpha == 1)
      cpu_n0<T, scalar_case::one, scalar_case::zero>(d, rows, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<T, scalar_case::neg_one, scalar_case::zero>(d, rows, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_n0<T, scalar_case::other, scalar_case::zero>(d, rows, terms, iA, vA, alpha, x, beta, y);
  }else if (beta == 1){
    if (alpha == 1)
      cpu_n0<T, scalar_case::one, scalar_case::one>(d, rows, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<T, scalar_case::neg_one, scalar_case::one>(d, rows, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_n0<T, scalar_case::other, scalar_case::one>(d, rows, terms, iA, vA, alpha, x, beta, y);
  }else if (beta == -1){
    if (alpha == 1)
      cpu_n0<T, scalar_case::one, scalar_case::neg_one>(d, rows, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<T, scalar_case::neg_one, scalar_case::neg_one>(d, rows, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_n0<T, scalar_case::other, scalar_case::neg_one>(d, rows, terms, iA, vA, alpha, x, beta, y);
  }else{
    if (alpha == 1)
      cpu_n0<T, scalar_case::one, scalar_case::other>(d, rows, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<T, scalar_case::neg_one, scalar_case::other>(d, rows, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_n0<T, scalar_case::other, scalar_case::other>(d, rows, terms, iA, vA, alpha, x, beta, y);
  }
}

/*!
 * \brief Helper method that instantiates correct kernel based on alpha and beta.
 */
template<typename T, int d, int n>
void cpu_dense(int const rows, int const terms, int const iA[],
               T const vA[], T const alpha, T const x[], T const beta,
               T y[])
{
  if (beta == 0){
    if (alpha == 1)
      cpu_dense<T, d, n, scalar_case::one, scalar_case::zero>(rows, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<T, d, n, scalar_case::neg_one, scalar_case::zero>(rows, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_dense<T, d, n, scalar_case::other, scalar_case::zero>(rows, terms, iA, vA, alpha, x, beta, y);
  }else if (beta == 1){
    if (alpha == 1)
      cpu_dense<T, d, n, scalar_case::one, scalar_case::one>(rows, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<T, d, n, scalar_case::neg_one, scalar_case::one>(rows, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_dense<T, d, n, scalar_case::other, scalar_case::one>(rows, terms, iA, vA, alpha, x, beta, y);
  }else if (beta == -1){
    if (alpha == 1)
      cpu_dense<T, d, n, scalar_case::one, scalar_case::neg_one>(rows, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<T, d, n, scalar_case::neg_one, scalar_case::neg_one>(rows, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_dense<T, d, n, scalar_case::other, scalar_case::neg_one>(rows, terms, iA, vA, alpha, x, beta, y);
  }else{
    if (alpha == 1)
      cpu_dense<T, d, n, scalar_case::one, scalar_case::other>(rows, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<T, d, n, scalar_case::neg_one, scalar_case::other>(rows, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_dense<T, d, n, scalar_case::other, scalar_case::other>(rows, terms, iA, vA, alpha, x, beta, y);
  }
}

template<typename T>
void cpu_dense(int const dimensions, int const n, int const num_rows,
               int const num_terms, int const iA[], T const vA[], T const alpha,
               T const x[], T const beta, T y[])
{
  switch (dimensions)
  {
  case 1:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, num_terms, iA, vA, alpha, x, beta, y);
      break;
    case 2:
      cpu_dense<T, 1, 2>(num_rows, num_terms, iA, vA, alpha, x, beta, y);
      break;
    //case 3:
    //  run_cpu_variant<T, 1, 3>(pA, lda, pX, pY, num_batch, output_stride);
    //  break;
    //case 4:
    //  run_cpu_variant<T, 1, 4>(pA, lda, pX, pY, num_batch, output_stride);
    //  break;
    default:
      throw std::runtime_error("unimplemented");
    }
    break;
  case 2:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, num_terms, iA, vA, alpha, x, beta, y);
      break;
    //case 2:
    //  run_cpu_variant<T, 2, 2>(pA, lda, pX, pY, num_batch, output_stride);
    //  break;
    //case 3:
    //  run_cpu_variant<T, 2, 3>(pA, lda, pX, pY, num_batch, output_stride);
    //  break;
    //case 4:
    //  run_cpu_variant<T, 2, 4>(pA, lda, pX, pY, num_batch, output_stride);
    //  break;
    default:
      //run_cpu_variant<T, 2>(n, pA, lda, pX, pY, num_batch, output_stride);
      throw std::runtime_error("unimplemented");
    }
    break;

  default:
    throw std::runtime_error(
        "kronmult unimplemented number of dimensions for the cpu");
  }
}

template void cpu_dense<float>(int const, int const, int const, int const,
                               int const[], float const[], float const,
                               float const[], float const, float[]);
template void cpu_dense<double>(int const, int const, int const, int const,
                                int const[], double const[], double const,
                                double const[], double const, double[]);



} // namespace asgard::kronmult
