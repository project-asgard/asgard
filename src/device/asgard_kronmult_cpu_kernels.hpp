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
          for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
              Y[k] += A[j * n + k] * x[tj + k];

          for (int j = 0; j < n; j++)
            if constexpr(alpha_case == scalar_case::one)
              y[ti + j] += Y[j];
            else if constexpr(alpha_case == scalar_case::neg_one)
              y[ti + j] -= Y[j];
            else
              y[ti + j] += alpha * Y[j];
        }
      }
    }
  }


//    for (int stride = 0; stride < output_stride; stride++)
//    {
//      int const i = iy * output_stride + stride;
//      if constexpr (dimensions == 1)
//      {
//        T Y[n] = {{0}};
//        for (int j = 0; j < n; j++)
//        {
//          for (int k = 0; k < n; k++)
//          {
//            Y[k] += pA[i][j * lda + k] * pX[i][j];
//          }
//        }
//        for (int j = 0; j < n; j++)
//        {
//          pY[i][j] += Y[j];
//        }
//      }
//      else if constexpr (dimensions == 2)
//      {
//        if constexpr (n == 2)
//        {
//          T w0 = pX[i][0] * pA[2 * i][0] + pX[i][2] * pA[2 * i][lda];
//          T w1 = pX[i][1] * pA[2 * i][0] + pX[i][3] * pA[2 * i][lda];
//          T w2 = pX[i][0] * pA[2 * i][1] + pX[i][2] * pA[2 * i][lda + 1];
//          T w3 = pX[i][1] * pA[2 * i][1] + pX[i][3] * pA[2 * i][lda + 1];
//          T y0 = pA[2 * i + 1][0] * w0 + pA[2 * i + 1][lda] * w1;
//          T y1 = pA[2 * i + 1][1] * w0 + pA[2 * i + 1][lda + 1] * w1;
//          T y2 = pA[2 * i + 1][0] * w2 + pA[2 * i + 1][lda] * w3;
//          T y3 = pA[2 * i + 1][1] * w2 + pA[2 * i + 1][lda + 1] * w3;
//          pY[i][0] += y0;
//          pY[i][1] += y1;
//          pY[i][2] += y2;
//          pY[i][3] += y3;
//        }
//        else if constexpr (n >= 3)
//        {
//          T W[n][n] = {{{0}}}, Y[n][n] = {{{0}}};
//          for (int j = 0; j < n; j++)
//            for (int k = 0; k < n; k++)
//              for (int s = 0; s < n; s++)
//                W[s][k] += pX[i][n * j + k] * pA[2 * i][j * lda + s];
//          for (int j = 0; j < n; j++)
//            for (int k = 0; k < n; k++)
//              for (int s = 0; s < n; s++)
//                Y[k][s] += pA[2 * i + 1][j * lda + s] * W[k][j];
//          for (int j = 0; j < n; j++)
//          {
//            for (int k = 0; k < n; k++)
//            {
//              pY[i][n * j + k] += Y[j][k];
//            }
//          }
//        }
//      }
//      else if constexpr (dimensions == 3)
//      {
//        T W[n][n][n] = {{{{0}}}}, Y[n][n][n] = {{{{0}}}};
//        for (int j = 0; j < n; j++)
//          for (int l = 0; l < n; l++)
//            for (int k = 0; k < n; k++)
//              for (int s = 0; s < n; s++)
//                Y[s][l][k] +=
//                    pX[i][n * n * j + n * l + k] * pA[3 * i][j * lda + s];
//        for (int j = 0; j < n; j++)
//          for (int l = 0; l < n; l++)
//            for (int k = 0; k < n; k++)
//              for (int s = 0; s < n; s++)
//                W[l][s][k] += Y[l][j][k] * pA[3 * i + 1][j * lda + s];
//        std::fill(&Y[0][0][0], &Y[0][0][0] + sizeof(W) / sizeof(T), T{0.});
//        for (int j = 0; j < n; j++)
//          for (int l = 0; l < n; l++)
//            for (int k = 0; k < n; k++)
//              for (int s = 0; s < n; s++)
//                Y[l][k][s] += pA[3 * i + 2][j * lda + s] * W[l][k][j];
//        for (int j = 0; j < n; j++)
//        {
//          for (int l = 0; l < n; l++)
//          {
//            for (int k = 0; k < n; k++)
//            {
//              pY[i][n * n * j + n * l + k] += Y[j][l][k];
//            }
//          }
//        }
//      }
//      else if constexpr (dimensions == 4)
//      {
//        T W[n][n][n][n] = {{{{{0}}}}}, Y[n][n][n][n] = {{{{{0}}}}};
//        for (int j = 0; j < n; j++)
//          for (int p = 0; p < n; p++)
//            for (int l = 0; l < n; l++)
//              for (int k = 0; k < n; k++)
//                for (int s = 0; s < n; s++)
//                  W[s][p][l][k] +=
//                      pX[i][n * n * n * j + n * n * p + n * l + k] *
//                      pA[4 * i][j * lda + s];
//        for (int j = 0; j < n; j++)
//          for (int p = 0; p < n; p++)
//            for (int l = 0; l < n; l++)
//              for (int k = 0; k < n; k++)
//                for (int s = 0; s < n; s++)
//                  Y[p][s][l][k] += W[p][j][l][k] * pA[4 * i + 1][j * lda + s];
//        std::fill(&W[0][0][0][0], &W[0][0][0][0] + sizeof(W) / sizeof(T),
//                  T{0.});
//        for (int j = 0; j < n; j++)
//          for (int p = 0; p < n; p++)
//            for (int l = 0; l < n; l++)
//              for (int k = 0; k < n; k++)
//                for (int s = 0; s < n; s++)
//                  W[p][l][s][k] += Y[p][l][j][k] * pA[4 * i + 2][j * lda + s];
//        std::fill(&Y[0][0][0][0], &Y[0][0][0][0] + sizeof(W) / sizeof(T),
//                  T{0.});
//        for (int j = 0; j < n; j++)
//          for (int p = 0; p < n; p++)
//            for (int l = 0; l < n; l++)
//              for (int k = 0; k < n; k++)
//                for (int s = 0; s < n; s++)
//                  Y[p][l][k][s] += pA[4 * i + 3][j * lda + s] * W[p][l][k][j];
//        for (int j = 0; j < n; j++)
//        {
//          for (int p = 0; p < n; p++)
//          {
//            for (int l = 0; l < n; l++)
//            {
//              for (int k = 0; k < n; k++)
//              {
//                pY[i][n * n * n * j + n * n * p + n * l + k] += Y[j][p][l][k];
//              }
//            }
//          }
//        }
//      }
//      else if constexpr (dimensions == 5)
//      {
//        T W[n][n][n][n][n] = {{{{{{0}}}}}}, Y[n][n][n][n][n] = {{{{{{0}}}}}};
//        for (int j = 0; j < n; j++)
//          for (int v = 0; v < n; v++)
//            for (int p = 0; p < n; p++)
//              for (int l = 0; l < n; l++)
//                for (int k = 0; k < n; k++)
//                  for (int s = 0; s < n; s++)
//                    Y[s][v][p][l][k] +=
//                        pX[i][n * n * n * n * j + n * n * n * v + n * n * p +
//                              n * l + k] *
//                        pA[5 * i][j * lda + s];
//        for (int j = 0; j < n; j++)
//          for (int v = 0; v < n; v++)
//            for (int p = 0; p < n; p++)
//              for (int l = 0; l < n; l++)
//                for (int k = 0; k < n; k++)
//                  for (int s = 0; s < n; s++)
//                    W[v][s][p][l][k] +=
//                        Y[v][j][p][l][k] * pA[5 * i + 1][j * lda + s];
//        std::fill(&Y[0][0][0][0][0], &Y[0][0][0][0][0] + sizeof(W) / sizeof(T),
//                  T{0.});
//        for (int j = 0; j < n; j++)
//          for (int v = 0; v < n; v++)
//            for (int p = 0; p < n; p++)
//              for (int l = 0; l < n; l++)
//                for (int k = 0; k < n; k++)
//                  for (int s = 0; s < n; s++)
//                    Y[v][p][s][l][k] +=
//                        W[v][p][j][l][k] * pA[5 * i + 2][j * lda + s];
//        std::fill(&W[0][0][0][0][0], &W[0][0][0][0][0] + sizeof(W) / sizeof(T),
//                  T{0.});
//        for (int j = 0; j < n; j++)
//          for (int v = 0; v < n; v++)
//            for (int p = 0; p < n; p++)
//              for (int l = 0; l < n; l++)
//                for (int k = 0; k < n; k++)
//                  for (int s = 0; s < n; s++)
//                    W[v][p][l][s][k] +=
//                        Y[v][p][l][j][k] * pA[5 * i + 3][j * lda + s];
//        std::fill(&Y[0][0][0][0][0], &Y[0][0][0][0][0] + sizeof(W) / sizeof(T),
//                  T{0.});
//        for (int j = 0; j < n; j++)
//          for (int v = 0; v < n; v++)
//            for (int p = 0; p < n; p++)
//              for (int l = 0; l < n; l++)
//                for (int k = 0; k < n; k++)
//                  for (int s = 0; s < n; s++)
//                    Y[v][p][l][k][s] +=
//                        pA[5 * i + 4][j * lda + s] * W[v][p][l][k][j];
//        for (int j = 0; j < n; j++)
//        {
//          for (int v = 0; v < n; v++)
//          {
//            for (int p = 0; p < n; p++)
//            {
//              for (int l = 0; l < n; l++)
//              {
//                for (int k = 0; k < n; k++)
//                {
//                  pY[i][n * n * n * n * j + n * n * n * v + n * n * p + n * l +
//                        k] += Y[j][v][p][l][k];
//                }
//              }
//            }
//          }
//        }
//      }
//      else if constexpr (dimensions == 6)
//      {
//        T W[n][n][n][n][n][n] = {{{{{{{0}}}}}}},
//          Y[n][n][n][n][n][n] = {{{{{{{0}}}}}}};
//        for (int j = 0; j < n; j++)
//          for (int w = 0; w < n; w++)
//            for (int v = 0; v < n; v++)
//              for (int p = 0; p < n; p++)
//                for (int l = 0; l < n; l++)
//                  for (int k = 0; k < n; k++)
//                    for (int s = 0; s < n; s++)
//                      W[s][w][v][p][l][k] +=
//                          pX[i][n * n * n * n * n * j + n * n * n * n * w +
//                                n * n * n * v + n * n * p + n * l + k] *
//                          pA[6 * i][j * lda + s];
//        for (int j = 0; j < n; j++)
//          for (int w = 0; w < n; w++)
//            for (int v = 0; v < n; v++)
//              for (int p = 0; p < n; p++)
//                for (int l = 0; l < n; l++)
//                  for (int k = 0; k < n; k++)
//                    for (int s = 0; s < n; s++)
//                      Y[w][s][v][p][l][k] +=
//                          W[w][j][v][p][l][k] * pA[6 * i + 1][j * lda + s];
//        std::fill(&W[0][0][0][0][0][0],
//                  &W[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
//        for (int j = 0; j < n; j++)
//          for (int w = 0; w < n; w++)
//            for (int v = 0; v < n; v++)
//              for (int p = 0; p < n; p++)
//                for (int l = 0; l < n; l++)
//                  for (int k = 0; k < n; k++)
//                    for (int s = 0; s < n; s++)
//                      W[w][v][s][p][l][k] +=
//                          Y[w][v][j][p][l][k] * pA[6 * i + 2][j * lda + s];
//        std::fill(&Y[0][0][0][0][0][0],
//                  &Y[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
//        for (int j = 0; j < n; j++)
//          for (int w = 0; w < n; w++)
//            for (int v = 0; v < n; v++)
//              for (int p = 0; p < n; p++)
//                for (int l = 0; l < n; l++)
//                  for (int k = 0; k < n; k++)
//                    for (int s = 0; s < n; s++)
//                      Y[w][v][p][s][l][k] +=
//                          W[w][v][p][j][l][k] * pA[6 * i + 3][j * lda + s];
//        std::fill(&W[0][0][0][0][0][0],
//                  &W[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
//        for (int j = 0; j < n; j++)
//          for (int w = 0; w < n; w++)
//            for (int v = 0; v < n; v++)
//              for (int p = 0; p < n; p++)
//                for (int l = 0; l < n; l++)
//                  for (int k = 0; k < n; k++)
//                    for (int s = 0; s < n; s++)
//                      W[w][v][p][l][s][k] +=
//                          Y[w][v][p][l][j][k] * pA[6 * i + 4][j * lda + s];
//        std::fill(&Y[0][0][0][0][0][0],
//                  &Y[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
//        for (int j = 0; j < n; j++)
//          for (int w = 0; w < n; w++)
//            for (int v = 0; v < n; v++)
//              for (int p = 0; p < n; p++)
//                for (int l = 0; l < n; l++)
//                  for (int k = 0; k < n; k++)
//                    for (int s = 0; s < n; s++)
//                      Y[w][v][p][l][k][s] +=
//                          pA[6 * i + 5][j * lda + s] * W[w][v][p][l][k][j];
//        for (int j = 0; j < n; j++)
//        {
//          for (int w = 0; w < n; w++)
//          {
//            for (int v = 0; v < n; v++)
//            {
//              for (int p = 0; p < n; p++)
//              {
//                for (int l = 0; l < n; l++)
//                {
//                  for (int k = 0; k < n; k++)
//                  {
//                    pY[i][n * n * n * n * n * j + n * n * n * n * w +
//                          n * n * n * v + n * n * p + n * l + k] +=
//                        Y[j][w][v][p][l][k];
//                  }
//                }
//              }
//            }
//          }
//        }
//      }
//    } // for output-length
//  }   // for loop
}

} // namespace asgard::kronmult
