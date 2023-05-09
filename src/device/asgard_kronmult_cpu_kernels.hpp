#pragma once

#include "asgard_kronmult_v2.hpp"

namespace asgard::kronmult
{

template<typename T, scalar_case alpha_case, scalar_case beta_case>
void cpu_n0(int const dimensions, int const num_rows, int const num_terms,
            int const iA[], T const vA[], T const alpha, T const x[],
            T const beta, T y[])
{
#pragma omp parallel for
  for (int iy = 0; iy < num_rows; iy++)
  {
    if constexpr(beta_case == scalar_case::zero)
      y[iy] = 0;
    else if constexpr(beta_case == scalar_case::neg_one)
      y[iy] = -y[iy];
    else if constexpr(beta_case == scalar_case::other)
      y[iy] *= beta;

    // ma is the starting index of the operators for this y
    int ma = iy * num_rows * num_terms * dimensions;

    for (int jx = 0; jx < num_rows; jx++)
    {
      for (int t = 0; t < num_terms; t++)
      {
        T totalA = 1;
        for (int d = 0; d < dimensions; d++)
          totalA *= vA[iA[ma++]];

        if constexpr(alpha_case == scalar_case::one)
          y[iy] += totalA * x[jx];
        else if constexpr(alpha_case == scalar_case::neg_one)
          y[iy] -= totalA * x[jx];
        else
          y[iy] += alpha * totalA * x[jx];
      }
    }
  }
}


template<typename T, int dimensions, int n,
         scalar_case alpha_case, scalar_case beta_case>
void cpu_dense(int const num_rows, int const num_terms, int const iA[],
               T const vA[], T const alpha, T const x[], T const beta,
               T y[])
{
  static_assert(1 <= dimensions and dimensions <= 6);
  static_assert(n > 1, "n must be positive and n==1 is a special case handled "
                       "by another method");

// always use one thread per kron-product
#pragma omp parallel for
  for (int iy = 0; iy < num_rows; iy++)
  {
    // tensor i (ti) is the first index of this tensor in y
    int const ti = iy * ipow<n, dimensions>();
    if constexpr(beta_case == scalar_case::zero)
      for (int j=0; j<ipow<n, dimensions>(); j++)
        y[ti + j] = 0;
    else if constexpr(beta_case == scalar_case::neg_one)
      for (int j=0; j<ipow<n, dimensions>(); j++)
        y[ti + j] = -y[ti + j];
    else if constexpr(beta_case == scalar_case::other)
      for (int j=0; j<ipow<n, dimensions>(); j++)
        y[ti + j] *= beta;

    // ma is the starting index of the operators for this y
    int ma = iy * num_rows * num_terms * dimensions;

    for (int jx = 0; jx < num_rows; jx++)
    {
      // tensor i (ti) is the first index of this tensor in x
      int const tj = jx * ipow<n, dimensions>();
      for (int t = 0; t < num_terms; t++)
      {
        if constexpr (dimensions == 1)
        {
          T const *A = &(vA[iA[ma++]]);
          T Y[n] = {{0}};
#pragma omp simd collapse(2)
          for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
              Y[j] += A[k * n + j] * x[tj + k];

#pragma omp simd
          for (int j = 0; j < n; j++)
            if constexpr(alpha_case == scalar_case::one)
              y[ti + j] += Y[j];
            else if constexpr(alpha_case == scalar_case::neg_one)
              y[ti + j] -= Y[j];
            else
              y[ti + j] += alpha * Y[j];
        }
        else if constexpr (dimensions == 2)
        {
          T const *A1 = &(vA[iA[ma++]]);
          T W[n][n] = {{{0}}}, Y[n][n] = {{{0}}};
#pragma omp simd collapse(3)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int k = 0; k < n; k++)
                W[s][k] += x[tj + n * j + k] * A1[j * n + s];
          T const *A0 = &(vA[iA[ma++]]);
#pragma omp simd collapse(3)
          for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
              for (int s = 0; s < n; s++)
                Y[k][s] += A0[s + j * n] * W[k][j];
#pragma omp simd collapse(2)
          for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
              y[ti + n * j + k] += Y[j][k];
        }
        else if constexpr (dimensions == 3)
        {
          T const *A2 = &(vA[iA[ma++]]);
          T W[n][n][n] = {{{{0}}}}, Y[n][n][n] = {{{{0}}}};
#pragma omp simd collapse(4)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  Y[s][l][k] +=
                      x[tj + n * n * j + n * l + k] * A2[j * n + s];
          T const *A1 = &(vA[iA[ma++]]);
#pragma omp simd collapse(4)
          for (int j = 0; j < n; j++)
            for (int l = 0; l < n; l++)
              for (int s = 0; s < n; s++)
                for (int k = 0; k < n; k++)
                  W[l][s][k] += Y[l][j][k] * A1[j * n + s];
          std::fill(&Y[0][0][0], &Y[0][0][0] + sizeof(W) / sizeof(T), T{0.});
          T const *A0 = &(vA[iA[ma++]]);
#pragma omp simd collapse(4)
          for (int j = 0; j < n; j++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                for (int s = 0; s < n; s++)
                  Y[l][k][s] += A0[j * n + s] * W[l][k][j];
#pragma omp simd collapse(3)
          for (int j = 0; j < n; j++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                if constexpr(alpha_case == scalar_case::one)
                  y[ti + n * n * j + n * l + k] += Y[j][l][k];
                else if constexpr(alpha_case == scalar_case::neg_one)
                  y[ti + n * n * j + n * l + k] -= Y[j][l][k];
                else
                  y[ti + n * n * j + n * l + k] += alpha * Y[j][l][k];
        }
        else if constexpr (dimensions == 4)
        {
          T W[n][n][n][n] = {{{{{0}}}}}, Y[n][n][n][n] = {{{{{0}}}}};
          T const *A3 = &(vA[iA[ma++]]);
#pragma omp simd collapse(4)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    W[s][p][l][k] +=
                        x[tj + n * n * n * j + n * n * p + n * l + k] *
                        A3[j * n + s];
          T const *A2 = &(vA[iA[ma++]]);
#pragma omp simd collapse(4)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    Y[p][s][l][k] += W[p][j][l][k] * A2[j * n + s];
          std::fill(&W[0][0][0][0], &W[0][0][0][0] + sizeof(W) / sizeof(T),
                    T{0.});
          T const *A1 = &(vA[iA[ma++]]);
#pragma omp simd collapse(4)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    W[p][l][s][k] += Y[p][l][j][k] * A1[j * n + s];
          std::fill(&Y[0][0][0][0], &Y[0][0][0][0] + sizeof(W) / sizeof(T),
                    T{0.});
          T const *A0 = &(vA[iA[ma++]]);
#pragma omp simd collapse(4)
          for (int j = 0; j < n; j++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  for (int s = 0; s < n; s++)
                    Y[p][l][k][s] += A0[j * n + s] * W[p][l][k][j];
#pragma omp simd collapse(3)
          for (int j = 0; j < n; j++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  if constexpr(alpha_case == scalar_case::one)
                    y[ti + n * n * n * j + n * n * p + n * l + k] += Y[j][p][l][k];
                  else if constexpr(alpha_case == scalar_case::neg_one)
                    y[ti + n * n * n * j + n * n * p + n * l + k] -= Y[j][p][l][k];
                  else
                    y[ti + n * n * n * j + n * n * p + n * l + k] += alpha * Y[j][p][l][k];
        }
        else if constexpr (dimensions == 5)
        {
          T W[n][n][n][n][n] = {{{{{{0}}}}}}, Y[n][n][n][n][n] = {{{{{{0}}}}}};
          T const *A4 = &(vA[iA[ma++]]);
#pragma omp simd collapse(5)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      Y[s][v][p][l][k] +=
                          x[tj + n * n * n * n * j + n * n * n * v + n * n * p +
                                n * l + k] *
                          A4[j * n + s];
          T const *A3 = &(vA[iA[ma++]]);
#pragma omp simd collapse(5)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      W[v][s][p][l][k] +=
                          Y[v][j][p][l][k] * A3[j * n + s];
          std::fill(&Y[0][0][0][0][0], &Y[0][0][0][0][0] + sizeof(W) / sizeof(T),
                    T{0.});
          T const *A2 = &(vA[iA[ma++]]);
#pragma omp simd collapse(5)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      Y[v][p][s][l][k] +=
                          W[v][p][j][l][k] * A2[j * n + s];
          std::fill(&W[0][0][0][0][0], &W[0][0][0][0][0] + sizeof(W) / sizeof(T),
                    T{0.});
          T const *A1 = &(vA[iA[ma++]]);
#pragma omp simd collapse(5)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      W[v][p][l][s][k] +=
                          Y[v][p][l][j][k] * A1[j * n + s];
          std::fill(&Y[0][0][0][0][0], &Y[0][0][0][0][0] + sizeof(W) / sizeof(T),
                    T{0.});
          T const *A0 = &(vA[iA[ma++]]);
#pragma omp simd collapse(5)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      Y[v][p][l][k][s] +=
                          A0[j * n + s] * W[v][p][l][k][j];
#pragma omp simd collapse(4)
          for (int j = 0; j < n; j++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    if constexpr(alpha_case == scalar_case::one)
                      y[ti + n * n * n * n * j + n * n * n * v + n * n * p + n * l +
                            k] += Y[j][v][p][l][k];
                    else if constexpr(alpha_case == scalar_case::neg_one)
                      y[ti + n * n * n * n * j + n * n * n * v + n * n * p + n * l +
                            k] -= Y[j][v][p][l][k];
                    else
                      y[ti + n * n * n * n * j + n * n * n * v + n * n * p + n * l +
                            k] += alpha * Y[j][v][p][l][k];
        }
        else if constexpr (dimensions == 6)
        {
          T W[n][n][n][n][n][n] = {{{{{{{0}}}}}}},
            Y[n][n][n][n][n][n] = {{{{{{{0}}}}}}};
          T const *A5 = &(vA[iA[ma++]]);
#pragma omp simd collapse(6)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int w = 0; w < n; w++)
                for (int v = 0; v < n; v++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        W[s][w][v][p][l][k] +=
                            x[tj + n * n * n * n * n * j + n * n * n * n * w +
                                  n * n * n * v + n * n * p + n * l + k] *
                            A5[j * n + s];
          T const *A4 = &(vA[iA[ma++]]);
#pragma omp simd collapse(6)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int w = 0; w < n; w++)
                for (int v = 0; v < n; v++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        Y[w][s][v][p][l][k] +=
                            W[w][j][v][p][l][k] * A4[j * n + s];
          std::fill(&W[0][0][0][0][0][0],
                    &W[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          T const *A3 = &(vA[iA[ma++]]);
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int w = 0; w < n; w++)
                for (int v = 0; v < n; v++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        W[w][v][s][p][l][k] +=
                            Y[w][v][j][p][l][k] * A3[j * n + s];
          std::fill(&Y[0][0][0][0][0][0],
                    &Y[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          T const *A2 = &(vA[iA[ma++]]);
#pragma omp simd collapse(6)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int w = 0; w < n; w++)
                for (int v = 0; v < n; v++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        Y[w][v][p][s][l][k] +=
                            W[w][v][p][j][l][k] * A2[j * n + s];
          std::fill(&W[0][0][0][0][0][0],
                    &W[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          T const *A1 = &(vA[iA[ma++]]);
#pragma omp simd collapse(6)
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int w = 0; w < n; w++)
                for (int v = 0; v < n; v++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        W[w][v][p][l][s][k] +=
                            Y[w][v][p][l][j][k] * A1[j * n + s];
          std::fill(&Y[0][0][0][0][0][0],
                    &Y[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          T const *A0 = &(vA[iA[ma++]]);
#pragma omp simd collapse(6)
          for (int j = 0; j < n; j++)
            for (int w = 0; w < n; w++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      for (int s = 0; s < n; s++)
                        Y[w][v][p][l][k][s] +=
                            A0[j * n + s] * W[w][v][p][l][k][j];
#pragma omp simd collapse(5)
          for (int j = 0; j < n; j++)
            for (int w = 0; w < n; w++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      if constexpr(alpha_case == scalar_case::one)
                        y[ti + n * n * n * n * n * j + n * n * n * n * w +
                              n * n * n * v + n * n * p + n * l + k] +=
                            Y[j][w][v][p][l][k];
                      else if constexpr(alpha_case == scalar_case::neg_one)
                        y[ti + n * n * n * n * n * j + n * n * n * n * w +
                              n * n * n * v + n * n * p + n * l + k] -=
                            Y[j][w][v][p][l][k];
                      else
                        y[ti + n * n * n * n * n * j + n * n * n * n * w +
                              n * n * n * v + n * n * p + n * l + k] +=
                            alpha * Y[j][w][v][p][l][k];
        }
      }
    }
  } // for iy loop
}

} // namespace asgard::kronmult
