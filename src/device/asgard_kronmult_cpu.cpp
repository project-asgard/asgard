#include "asgard_kronmult.hpp"

#ifndef KRON_MODE_GLOBAL

#include "asgard_kronmult_cpu_general.hpp"

namespace asgard::kronmult
{
/*!
 * \brief Handles the special case of n=1
 *
 * When n=1 the Kronecker product is just a product of constants,
 * we would be much better off expressing the problem as a sparse matrix
 * and using optimized sparse linear algebra libraries.
 * Nevertheless, this implementation is provided for completeness,
 * where the case n=1 is handled in the same data structures as n>1.
 *
 * TODO: can fix that in the kronmult_matrix factory and switch to dense
 *       matrix-matrix implementation.
 */
template<typename P, scalar_case alpha_case, scalar_case beta_case>
void cpu_n0(int const dimensions, int const num_rows, int const num_cols,
            int const num_terms, int const elem[], int const row_offset,
            int const col_offset, P const *const vA[], int const num_1d_blocks,
            P const alpha, P const x[], P const beta, P y[])
{
  int const vstride = num_1d_blocks * num_1d_blocks;
  (void)alpha;
  (void)beta;
#pragma omp parallel for
  for (int rowy = 0; rowy < num_rows; rowy++)
  {
    if constexpr (beta_case == scalar_case::zero)
      y[rowy] = 0;
    else if constexpr (beta_case == scalar_case::neg_one)
      y[rowy] = -y[rowy];
    else if constexpr (beta_case == scalar_case::other)
      y[rowy] *= beta;

    int const *iy = elem + (rowy + row_offset) * dimensions;

    for (int colx = 0; colx < num_cols; colx++)
    {
      int const *ix = elem + (colx + col_offset) * dimensions;

      for (int t = 0; t < num_terms; t++)
      {
        P totalA = 1;
        for (int d = 0; d < dimensions; d++)
          totalA *= vA[t][d * vstride + ix[d] * num_1d_blocks + iy[d]];

        if constexpr (alpha_case == scalar_case::one)
          y[rowy] += totalA * x[colx];
        else if constexpr (alpha_case == scalar_case::neg_one)
          y[rowy] -= totalA * x[colx];
        else
          y[rowy] += alpha * totalA * x[colx];
      }
    }
  }
}

/*!
 * \brief Helper method that instantiates correct kernel based on alpha and beta.
 */
template<typename P>
void cpu_n0(int const d, int const rows, int cols, int const terms,
            int const elem[], int const row_offset, int const col_offset,
            P const *const vA[], int const num_1d_blocks, P const alpha,
            P const x[], P const beta, P y[])
{
  if (beta == 0)
  {
    if (alpha == 1)
      cpu_n0<P, scalar_case::one, scalar_case::zero>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<P, scalar_case::neg_one, scalar_case::zero>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_n0<P, scalar_case::other, scalar_case::zero>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else if (beta == 1)
  {
    if (alpha == 1)
      cpu_n0<P, scalar_case::one, scalar_case::one>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<P, scalar_case::neg_one, scalar_case::one>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_n0<P, scalar_case::other, scalar_case::one>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else if (beta == -1)
  {
    if (alpha == 1)
      cpu_n0<P, scalar_case::one, scalar_case::neg_one>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<P, scalar_case::neg_one, scalar_case::neg_one>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_n0<P, scalar_case::other, scalar_case::neg_one>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else
  {
    if (alpha == 1)
      cpu_n0<P, scalar_case::one, scalar_case::other>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<P, scalar_case::neg_one, scalar_case::other>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_n0<P, scalar_case::other, scalar_case::other>(
          d, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
}

/*!
 * \brief Baseline kronmult algorithm on the CPU (dense case).
 *
 * The template handles up to 6D problems and could handle arbitrary n,
 * however, only n = 2, 3, 4 is instantiated in asgard_kronmult_cpu.cpp.
 * Fixing n as a template parameter allows for automatic unroll of
 * the for-loop of the general algorithm, but also significantly
 * increases the compile times.
 *
 * \tparam T is either float or double
 * \tparam dimensions is between 1 and 6
 * \tparam n is the size of the small matrices
 * \tparam alpha_case must be one if alpha is 1 and neg_one if alpha is -1,
 *         otherwise it must be scalar_case::other
 *         alpha_case cannot be scalar_case::zero since that means
 *         no multiplication
 * \tparam beta_case must match beta, one for beta = 1, neg_one for beta = -1,
 *         zero for beta = 0 and other in all other cases
 */
template<typename P, int dimensions, int n, scalar_case alpha_case,
         scalar_case beta_case>
void cpu_dense(int const num_rows, int num_cols, int const num_terms,
               int const elem[], int const row_offset, int const col_offset,
               P const *const vA[], int const num_1d_blocks, P const alpha,
               P const x[], P const beta, P y[])
{
  static_assert(1 <= dimensions and dimensions <= 6);
  static_assert(n > 1, "n must be positive and n==1 is a special case handled "
                       "by another method");

  int const vstride = num_1d_blocks * num_1d_blocks * n * n;

  (void)vstride;
  (void)alpha;
  (void)beta;
// always use one thread per kron-product
#pragma omp parallel for
  for (int rowy = 0; rowy < num_rows; rowy++)
  {
    // tensor i (ti) is the first index of this tensor in y
    int const ti = rowy * ipow<n, dimensions>();
    if constexpr (beta_case == scalar_case::zero)
      for (int j = 0; j < ipow<n, dimensions>(); j++)
        y[ti + j] = 0;
    else if constexpr (beta_case == scalar_case::neg_one)
      for (int j = 0; j < ipow<n, dimensions>(); j++)
        y[ti + j] = -y[ti + j];
    else if constexpr (beta_case == scalar_case::other)
      for (int j = 0; j < ipow<n, dimensions>(); j++)
        y[ti + j] *= beta;

    int const *iy = elem + (rowy + row_offset) * dimensions;

    for (int colx = 0; colx < num_cols; colx++)
    {
      int const *ix = elem + (colx + col_offset) * dimensions;

      // tensor i (ti) is the first index of this tensor in x
      int const tj = colx * ipow<n, dimensions>();
      for (int t = 0; t < num_terms; t++)
      {
        if constexpr (dimensions == 1)
        {
          P const *const A = &vA[t][n * n * (ix[0] * num_1d_blocks + iy[0])];
          P Y[n]           = {{0}};
          ASGARD_PRAGMA_OMP_SIMD(collapse(2))
          for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
              Y[k] += A[j * n + k] * x[tj + j];

          ASGARD_PRAGMA_OMP_SIMD()
          for (int j = 0; j < n; j++)
            if constexpr (alpha_case == scalar_case::one)
              y[ti + j] += Y[j];
            else if constexpr (alpha_case == scalar_case::neg_one)
              y[ti + j] -= Y[j];
            else
              y[ti + j] += alpha * Y[j];
        }
        else if constexpr (dimensions == 2)
        {
          P W[n][n] = {{{0}}}, Y[n][n] = {{{0}}};
          P const *A = &vA[t][n * n * (ix[0] * num_1d_blocks + iy[0])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(3))
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int k = 0; k < n; k++)
                W[s][k] += x[tj + n * j + k] * A[j * n + s];

          A = &vA[t][vstride + n * n * (ix[1] * num_1d_blocks + iy[1])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(3))
          for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                Y[k][s] += A[j * n + s] * W[k][j];
          ASGARD_PRAGMA_OMP_SIMD(collapse(2))
          for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
              if constexpr (alpha_case == scalar_case::one)
                y[ti + n * j + k] += Y[j][k];
              else if constexpr (alpha_case == scalar_case::neg_one)
                y[ti + n * j + k] -= Y[j][k];
              else
                y[ti + n * j + k] += alpha * Y[j][k];
        }
        else if constexpr (dimensions == 3)
        {
          P W[n][n][n] = {{{{0}}}}, Y[n][n][n] = {{{{0}}}};
          P const *A = &vA[t][n * n * (ix[0] * num_1d_blocks + iy[0])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(4))
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  Y[s][l][k] += x[tj + n * n * j + n * l + k] * A[j * n + s];
          A = &vA[t][vstride + n * n * (ix[1] * num_1d_blocks + iy[1])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(4))
          for (int l = 0; l < n; l++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int k = 0; k < n; k++)
                  W[l][s][k] += Y[l][j][k] * A[j * n + s];
          std::fill(&Y[0][0][0], &Y[0][0][0] + sizeof(W) / sizeof(P), P{0.});
          A = &vA[t][2 * vstride + n * n * (ix[2] * num_1d_blocks + iy[2])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(4))
          for (int l = 0; l < n; l++)
            for (int k = 0; k < n; k++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  Y[l][k][s] += A[j * n + s] * W[l][k][j];
          ASGARD_PRAGMA_OMP_SIMD(collapse(3))
          for (int j = 0; j < n; j++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                if constexpr (alpha_case == scalar_case::one)
                  y[ti + n * n * j + n * l + k] += Y[j][l][k];
                else if constexpr (alpha_case == scalar_case::neg_one)
                  y[ti + n * n * j + n * l + k] -= Y[j][l][k];
                else
                  y[ti + n * n * j + n * l + k] += alpha * Y[j][l][k];
        }
        else if constexpr (dimensions == 4)
        {
          P W[n][n][n][n] = {{{{{0}}}}}, Y[n][n][n][n] = {{{{{0}}}}};
          P const *A = &vA[t][n * n * (ix[0] * num_1d_blocks + iy[0])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(5))
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    W[s][p][l][k] +=
                        x[tj + n * n * n * j + n * n * p + n * l + k] *
                        A[j * n + s];
          A = &vA[t][vstride + n * n * (ix[1] * num_1d_blocks + iy[1])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(5))
          for (int p = 0; p < n; p++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    Y[p][s][l][k] += W[p][j][l][k] * A[j * n + s];
          std::fill(&W[0][0][0][0], &W[0][0][0][0] + sizeof(W) / sizeof(P),
                    P{0.});
          A = &vA[t][2 * vstride + n * n * (ix[2] * num_1d_blocks + iy[2])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(5))
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int k = 0; k < n; k++)
                    W[p][l][s][k] += Y[p][l][j][k] * A[j * n + s];
          std::fill(&Y[0][0][0][0], &Y[0][0][0][0] + sizeof(W) / sizeof(P),
                    P{0.});
          A = &vA[t][3 * vstride + n * n * (ix[3] * num_1d_blocks + iy[3])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(5))
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    Y[p][l][k][s] += A[j * n + s] * W[p][l][k][j];
          ASGARD_PRAGMA_OMP_SIMD(collapse(4))
          for (int j = 0; j < n; j++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  if constexpr (alpha_case == scalar_case::one)
                    y[ti + n * n * n * j + n * n * p + n * l + k] +=
                        Y[j][p][l][k];
                  else if constexpr (alpha_case == scalar_case::neg_one)
                    y[ti + n * n * n * j + n * n * p + n * l + k] -=
                        Y[j][p][l][k];
                  else
                    y[ti + n * n * n * j + n * n * p + n * l + k] +=
                        alpha * Y[j][p][l][k];
        }
        else if constexpr (dimensions == 5)
        {
          P const *A         = &vA[t][n * n * (ix[0] * num_1d_blocks + iy[0])];
          P W[n][n][n][n][n] = {{{{{{0}}}}}}, Y[n][n][n][n][n] = {{{{{{0}}}}}};
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      Y[s][v][p][l][k] +=
                          x[tj + n * n * n * n * j + n * n * n * v + n * n * p +
                            n * l + k] *
                          A[j * n + s];
          A = &vA[t][vstride + n * n * (ix[1] * num_1d_blocks + iy[1])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int v = 0; v < n; v++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      W[v][s][p][l][k] += Y[v][j][p][l][k] * A[j * n + s];
          std::fill(&Y[0][0][0][0][0],
                    &Y[0][0][0][0][0] + sizeof(W) / sizeof(P), P{0.});
          A = &vA[t][2 * vstride + n * n * (ix[2] * num_1d_blocks + iy[2])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      Y[v][p][s][l][k] += W[v][p][j][l][k] * A[j * n + s];
          std::fill(&W[0][0][0][0][0],
                    &W[0][0][0][0][0] + sizeof(W) / sizeof(P), P{0.});
          A = &vA[t][3 * vstride + n * n * (ix[3] * num_1d_blocks + iy[3])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    for (int k = 0; k < n; k++)
                      W[v][p][l][s][k] += Y[v][p][l][j][k] * A[j * n + s];
          std::fill(&Y[0][0][0][0][0],
                    &Y[0][0][0][0][0] + sizeof(W) / sizeof(P), P{0.});
          A = &vA[t][4 * vstride + n * n * (ix[4] * num_1d_blocks + iy[4])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  for (int j = 0; j < n; j++)
                    for (int s = 0; s < n; s++)
                      Y[v][p][l][k][s] += A[j * n + s] * W[v][p][l][k][j];
          ASGARD_PRAGMA_OMP_SIMD(collapse(5))
          for (int j = 0; j < n; j++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    if constexpr (alpha_case == scalar_case::one)
                      y[ti + n * n * n * n * j + n * n * n * v + n * n * p +
                        n * l + k] += Y[j][v][p][l][k];
                    else if constexpr (alpha_case == scalar_case::neg_one)
                      y[ti + n * n * n * n * j + n * n * n * v + n * n * p +
                        n * l + k] -= Y[j][v][p][l][k];
                    else
                      y[ti + n * n * n * n * j + n * n * n * v + n * n * p +
                        n * l + k] += alpha * Y[j][v][p][l][k];
        }
        else if constexpr (dimensions == 6)
        {
          P const *A = &vA[t][n * n * (ix[0] * num_1d_blocks + iy[0])];

          P W[n][n][n][n][n][n] = {{{{{{{0}}}}}}},
            Y[n][n][n][n][n][n] = {{{{{{{0}}}}}}};
          ASGARD_PRAGMA_OMP_SIMD(collapse(7))
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
                            A[j * n + s];
          A = &vA[t][vstride + n * n * (ix[1] * num_1d_blocks + iy[1])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(7))
          for (int w = 0; w < n; w++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int v = 0; v < n; v++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        Y[w][s][v][p][l][k] +=
                            W[w][j][v][p][l][k] * A[j * n + s];
          std::fill(&W[0][0][0][0][0][0],
                    &W[0][0][0][0][0][0] + sizeof(W) / sizeof(P), P{0.});
          A = &vA[t][2 * vstride + n * n * (ix[2] * num_1d_blocks + iy[2])];
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        W[w][v][s][p][l][k] +=
                            Y[w][v][j][p][l][k] * A[j * n + s];
          std::fill(&Y[0][0][0][0][0][0],
                    &Y[0][0][0][0][0][0] + sizeof(W) / sizeof(P), P{0.});
          A = &vA[t][3 * vstride + n * n * (ix[3] * num_1d_blocks + iy[3])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(7))
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        Y[w][v][p][s][l][k] +=
                            W[w][v][p][j][l][k] * A[j * n + s];
          std::fill(&W[0][0][0][0][0][0],
                    &W[0][0][0][0][0][0] + sizeof(W) / sizeof(P), P{0.});
          A = &vA[t][4 * vstride + n * n * (ix[4] * num_1d_blocks + iy[4])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(7))
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int j = 0; j < n; j++)
                    for (int s = 0; s < n; s++)
                      for (int k = 0; k < n; k++)
                        W[w][v][p][l][s][k] +=
                            Y[w][v][p][l][j][k] * A[j * n + s];
          std::fill(&Y[0][0][0][0][0][0],
                    &Y[0][0][0][0][0][0] + sizeof(W) / sizeof(P), P{0.});
          A = &vA[t][5 * vstride + n * n * (ix[5] * num_1d_blocks + iy[5])];
          ASGARD_PRAGMA_OMP_SIMD(collapse(7))
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    for (int j = 0; j < n; j++)
                      for (int s = 0; s < n; s++)
                        Y[w][v][p][l][k][s] +=
                            A[j * n + s] * W[w][v][p][l][k][j];
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int j = 0; j < n; j++)
            for (int w = 0; w < n; w++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      if constexpr (alpha_case == scalar_case::one)
                        y[ti + n * n * n * n * n * j + n * n * n * n * w +
                          n * n * n * v + n * n * p + n * l + k] +=
                            Y[j][w][v][p][l][k];
                      else if constexpr (alpha_case == scalar_case::neg_one)
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

/*!
 * \brief Helper method that instantiates correct kernel based on alpha and beta.
 */
template<typename P, int d, int n>
void cpu_dense(int const rows, int cols, int const terms, int const elem[],
               int const row_offset, int const col_offset, P const *const vA[],
               int const num_1d_blocks, P const alpha, P const x[],
               P const beta, P y[])
{
  if (beta == 0)
  {
    if (alpha == 1)
      cpu_dense<P, d, n, scalar_case::one, scalar_case::zero>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<P, d, n, scalar_case::neg_one, scalar_case::zero>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_dense<P, d, n, scalar_case::other, scalar_case::zero>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else if (beta == 1)
  {
    if (alpha == 1)
      cpu_dense<P, d, n, scalar_case::one, scalar_case::one>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<P, d, n, scalar_case::neg_one, scalar_case::one>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_dense<P, d, n, scalar_case::other, scalar_case::one>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else if (beta == -1)
  {
    if (alpha == 1)
      cpu_dense<P, d, n, scalar_case::one, scalar_case::neg_one>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<P, d, n, scalar_case::neg_one, scalar_case::neg_one>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_dense<P, d, n, scalar_case::other, scalar_case::neg_one>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else
  {
    if (alpha == 1)
      cpu_dense<P, d, n, scalar_case::one, scalar_case::other>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<P, d, n, scalar_case::neg_one, scalar_case::other>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_dense<P, d, n, scalar_case::other, scalar_case::other>(
          rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
}

/*!
 * \brief Helper method that instantiates correct kernel based on alpha and beta.
 */
template<typename P, int d>
void cpu_dense(int const n, int const rows, int cols, int const terms,
               int const elem[], int const row_offset, int const col_offset,
               P const *const vA[], int const num_1d_blocks, P const alpha,
               P const x[], P const beta, P y[])
{
  if (beta == 0)
  {
    if (alpha == 1)
      cpu_dense<P, d, scalar_case::one, scalar_case::zero>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<P, d, scalar_case::neg_one, scalar_case::zero>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_dense<P, d, scalar_case::other, scalar_case::zero>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else if (beta == 1)
  {
    if (alpha == 1)
      cpu_dense<P, d, scalar_case::one, scalar_case::one>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<P, d, scalar_case::neg_one, scalar_case::one>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_dense<P, d, scalar_case::other, scalar_case::one>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else if (beta == -1)
  {
    if (alpha == 1)
      cpu_dense<P, d, scalar_case::one, scalar_case::neg_one>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<P, d, scalar_case::neg_one, scalar_case::neg_one>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_dense<P, d, scalar_case::other, scalar_case::neg_one>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
  else
  {
    if (alpha == 1)
      cpu_dense<P, d, scalar_case::one, scalar_case::other>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else if (alpha == -1)
      cpu_dense<P, d, scalar_case::neg_one, scalar_case::other>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
    else
      cpu_dense<P, d, scalar_case::other, scalar_case::other>(
          n, rows, cols, terms, elem, row_offset, col_offset, vA, num_1d_blocks,
          alpha, x, beta, y);
  }
}

template<typename P>
void cpu_dense(int const dimensions, int const n, int const num_rows,
               int const num_cols, int const num_terms, int const elem[],
               int const row_offset, int const col_offset, P const *const vA[],
               int const num_1d_blocks, P const alpha, P const x[],
               P const beta, P y[])
{
  switch (dimensions)
  {
  case 1:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, num_cols, num_terms, elem, row_offset,
             col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 2:
      cpu_dense<P, 1, 2>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 3:
      cpu_dense<P, 1, 3>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 4:
      cpu_dense<P, 1, 4>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    default:
      cpu_dense<P, 1>(n, num_rows, num_cols, num_terms, elem, row_offset,
                      col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    }
    break;
  case 2:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, num_cols, num_terms, elem, row_offset,
             col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 2:
      cpu_dense<P, 2, 2>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 3:
      cpu_dense<P, 2, 3>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 4:
      cpu_dense<P, 2, 4>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    default:
      cpu_dense<P, 2>(n, num_rows, num_cols, num_terms, elem, row_offset,
                      col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    }
    break;
  case 3:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, num_cols, num_terms, elem, row_offset,
             col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 2:
      cpu_dense<P, 3, 2>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 3:
      cpu_dense<P, 3, 3>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 4:
      cpu_dense<P, 3, 4>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    default:
      cpu_dense<P, 3>(n, num_rows, num_cols, num_terms, elem, row_offset,
                      col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    }
    break;
  case 4:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, num_cols, num_terms, elem, row_offset,
             col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 2:
      cpu_dense<P, 4, 2>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 3:
      cpu_dense<P, 4, 3>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 4:
      cpu_dense<P, 4, 4>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    default:
      cpu_dense<P, 4>(n, num_rows, num_cols, num_terms, elem, row_offset,
                      col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    }
    break;
  case 5:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, num_cols, num_terms, elem, row_offset,
             col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 2:
      cpu_dense<P, 5, 2>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 3:
      cpu_dense<P, 5, 3>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 4:
      cpu_dense<P, 5, 4>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    default:
      cpu_dense<P, 5>(n, num_rows, num_cols, num_terms, elem, row_offset,
                      col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    }
    break;
  case 6:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, num_cols, num_terms, elem, row_offset,
             col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 2:
      cpu_dense<P, 6, 2>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 3:
      cpu_dense<P, 6, 3>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    case 4:
      cpu_dense<P, 6, 4>(num_rows, num_cols, num_terms, elem, row_offset,
                         col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    default:
      cpu_dense<P, 6>(n, num_rows, num_cols, num_terms, elem, row_offset,
                      col_offset, vA, num_1d_blocks, alpha, x, beta, y);
      break;
    }
    break;
  default:
    throw std::runtime_error(
        "kronmult unimplemented number of dimensions for the cpu");
  }
}

#ifndef ASGARD_USE_CUDA // no need to compile for the CPU if CUDA is on
#ifdef ASGARD_ENABLE_DOUBLE
template void cpu_dense<double>(int const, int const, int const, int const,
                                int const, int const[], int const, int const,
                                double const *const[], int const, double const,
                                double const[], double const, double y[]);
#endif
#ifdef ASGARD_ENABLE_FLOAT
template void cpu_dense<float>(int const, int const, int const, int const,
                               int const, int const[], int const, int const,
                               float const *const[], int const, float const,
                               float const[], float const, float y[]);
#endif
#endif

} // namespace asgard::kronmult

#endif // KRON_MODE_GLOBAL
