#include "asgard_kronmult.hpp"

#ifndef KRON_MODE_GLOBAL

#include "asgard_spkronmult_cpu_general.hpp"

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
template<typename T, scalar_case alpha_case, scalar_case beta_case>
void cpu_n0(int const dimensions, int const num_rows, int const pntr[],
            int const indx[], int const num_terms, int const iA[], T const vA[],
            T const alpha, T const x[], T const beta, T y[])
{
  (void)alpha;
  (void)beta;
#pragma omp parallel for
  for (int iy = 0; iy < num_rows; iy++)
  {
    if constexpr (beta_case == scalar_case::zero)
      y[iy] = 0;
    else if constexpr (beta_case == scalar_case::neg_one)
      y[iy] = -y[iy];
    else if constexpr (beta_case == scalar_case::other)
      y[iy] *= beta;

    // ma is the starting index of the operators for this y
    int ma = pntr[iy] * num_terms * dimensions;

    for (int jx = pntr[iy]; jx < pntr[iy + 1]; jx++)
    {
      for (int t = 0; t < num_terms; t++)
      {
        T totalA = 1;
        for (int d = 0; d < dimensions; d++)
          totalA *= vA[iA[ma++]];

        if constexpr (alpha_case == scalar_case::one)
          y[iy] += totalA * x[indx[jx]];
        else if constexpr (alpha_case == scalar_case::neg_one)
          y[iy] -= totalA * x[indx[jx]];
        else
          y[iy] += alpha * totalA * x[indx[jx]];
      }
    }
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
template<typename T, int dimensions, int n, scalar_case alpha_case,
         scalar_case beta_case>
void cpu_sparse(int const num_rows, int const pntr[], int const indx[],
                int const num_terms, int const iA[], T const vA[],
                T const alpha, T const x[], T const beta, T y[])
{
  static_assert(1 <= dimensions and dimensions <= 6);
  static_assert(n > 1, "n must be positive and n==1 is a special case handled "
                       "by another method");
  (void)alpha;
  (void)beta;
// always use one thread per kron-product
#pragma omp parallel for
  for (int iy = 0; iy < num_rows; iy++)
  {
    // tensor i (ti) is the first index of this tensor in y
    int const ti = iy * ipow<n, dimensions>();
    if constexpr (beta_case == scalar_case::zero)
      for (int j = 0; j < ipow<n, dimensions>(); j++)
        y[ti + j] = 0;
    else if constexpr (beta_case == scalar_case::neg_one)
      for (int j = 0; j < ipow<n, dimensions>(); j++)
        y[ti + j] = -y[ti + j];
    else if constexpr (beta_case == scalar_case::other)
      for (int j = 0; j < ipow<n, dimensions>(); j++)
        y[ti + j] *= beta;

    // ma is the starting index of the operators for this y
    int ma = pntr[iy] * num_terms * dimensions;

    for (int jx = pntr[iy]; jx < pntr[iy + 1]; jx++)
    {
      // tensor i (ti) is the first index of this tensor in x
      int const tj = indx[jx] * ipow<n, dimensions>();
      for (int t = 0; t < num_terms; t++)
      {
        if constexpr (dimensions == 1)
        {
          T const *const A = &(vA[iA[ma++]]);
          T Y[n]           = {{0}};
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
          T const *A = &(vA[iA[ma++]]); // A1
          T W[n][n] = {{{0}}}, Y[n][n] = {{{0}}};
          ASGARD_PRAGMA_OMP_SIMD(collapse(3))
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int k = 0; k < n; k++)
                W[s][k] += x[tj + n * j + k] * A[j * n + s];
          A = &(vA[iA[ma++]]); // A0
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
          T const *A   = &(vA[iA[ma++]]); // A2
          T W[n][n][n] = {{{{0}}}}, Y[n][n][n] = {{{{0}}}};
          ASGARD_PRAGMA_OMP_SIMD(collapse(4))
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  Y[s][l][k] += x[tj + n * n * j + n * l + k] * A[j * n + s];
          A = &(vA[iA[ma++]]); // A1
          ASGARD_PRAGMA_OMP_SIMD(collapse(4))
          for (int l = 0; l < n; l++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int k = 0; k < n; k++)
                  W[l][s][k] += Y[l][j][k] * A[j * n + s];
          std::fill(&Y[0][0][0], &Y[0][0][0] + sizeof(W) / sizeof(T), T{0.});
          A = &(vA[iA[ma++]]); // A0
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
          T W[n][n][n][n] = {{{{{0}}}}}, Y[n][n][n][n] = {{{{{0}}}}};
          T const *A = &(vA[iA[ma++]]); // A3
          ASGARD_PRAGMA_OMP_SIMD(collapse(5))
          for (int j = 0; j < n; j++)
            for (int s = 0; s < n; s++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    W[s][p][l][k] +=
                        x[tj + n * n * n * j + n * n * p + n * l + k] *
                        A[j * n + s];
          A = &(vA[iA[ma++]]); // A2
          ASGARD_PRAGMA_OMP_SIMD(collapse(5))
          for (int p = 0; p < n; p++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    Y[p][s][l][k] += W[p][j][l][k] * A[j * n + s];
          std::fill(&W[0][0][0][0], &W[0][0][0][0] + sizeof(W) / sizeof(T),
                    T{0.});
          A = &(vA[iA[ma++]]); // A1
          ASGARD_PRAGMA_OMP_SIMD(collapse(5))
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int k = 0; k < n; k++)
                    W[p][l][s][k] += Y[p][l][j][k] * A[j * n + s];
          std::fill(&Y[0][0][0][0], &Y[0][0][0][0] + sizeof(W) / sizeof(T),
                    T{0.});
          A = &(vA[iA[ma++]]); // A0
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
          T W[n][n][n][n][n] = {{{{{{0}}}}}}, Y[n][n][n][n][n] = {{{{{{0}}}}}};
          T const *A = &(vA[iA[ma++]]); // A4
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
          A = &(vA[iA[ma++]]); // A3
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int v = 0; v < n; v++)
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      W[v][s][p][l][k] += Y[v][j][p][l][k] * A[j * n + s];
          std::fill(&Y[0][0][0][0][0],
                    &Y[0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          A = &(vA[iA[ma++]]); // A2
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      Y[v][p][s][l][k] += W[v][p][j][l][k] * A[j * n + s];
          std::fill(&W[0][0][0][0][0],
                    &W[0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          A = &(vA[iA[ma++]]); // A1
          ASGARD_PRAGMA_OMP_SIMD(collapse(6))
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    for (int k = 0; k < n; k++)
                      W[v][p][l][s][k] += Y[v][p][l][j][k] * A[j * n + s];
          std::fill(&Y[0][0][0][0][0],
                    &Y[0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          A = &(vA[iA[ma++]]); // A0
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
          T W[n][n][n][n][n][n] = {{{{{{{0}}}}}}},
            Y[n][n][n][n][n][n] = {{{{{{{0}}}}}}};
          T const *A            = &(vA[iA[ma++]]); // A5
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
          A = &(vA[iA[ma++]]); // A4
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
                    &W[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          A = &(vA[iA[ma++]]); // A3
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
                    &Y[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          A = &(vA[iA[ma++]]); // A2
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
                    &W[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          A = &(vA[iA[ma++]]); // A1
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
                    &Y[0][0][0][0][0][0] + sizeof(W) / sizeof(T), T{0.});
          A = &(vA[iA[ma++]]); // A0
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
template<typename T>
void cpu_n0(int const d, int const rows, int const pntr[], int const indx[],
            int const terms, int const iA[], T const vA[], T const alpha,
            T const x[], T const beta, T y[])
{
  if (beta == 0)
  {
    if (alpha == 1)
      cpu_n0<T, scalar_case::one, scalar_case::zero>(d, rows, pntr, indx, terms,
                                                     iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<T, scalar_case::neg_one, scalar_case::zero>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_n0<T, scalar_case::other, scalar_case::zero>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else if (beta == 1)
  {
    if (alpha == 1)
      cpu_n0<T, scalar_case::one, scalar_case::one>(d, rows, pntr, indx, terms,
                                                    iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<T, scalar_case::neg_one, scalar_case::one>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_n0<T, scalar_case::other, scalar_case::one>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else if (beta == -1)
  {
    if (alpha == 1)
      cpu_n0<T, scalar_case::one, scalar_case::neg_one>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<T, scalar_case::neg_one, scalar_case::neg_one>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_n0<T, scalar_case::other, scalar_case::neg_one>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else
  {
    if (alpha == 1)
      cpu_n0<T, scalar_case::one, scalar_case::other>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_n0<T, scalar_case::neg_one, scalar_case::other>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_n0<T, scalar_case::other, scalar_case::other>(
          d, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
}

/*!
 * \brief Helper method that instantiates correct kernel based on alpha and beta.
 */
template<typename T, int d, int n>
void cpu_sparse(int const rows, int const pntr[], int const indx[],
                int const terms, int const iA[], T const vA[], T const alpha,
                T const x[], T const beta, T y[])
{
  if (beta == 0)
  {
    if (alpha == 1)
      cpu_sparse<T, d, n, scalar_case::one, scalar_case::zero>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_sparse<T, d, n, scalar_case::neg_one, scalar_case::zero>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_sparse<T, d, n, scalar_case::other, scalar_case::zero>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else if (beta == 1)
  {
    if (alpha == 1)
      cpu_sparse<T, d, n, scalar_case::one, scalar_case::one>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_sparse<T, d, n, scalar_case::neg_one, scalar_case::one>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_sparse<T, d, n, scalar_case::other, scalar_case::one>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else if (beta == -1)
  {
    if (alpha == 1)
      cpu_sparse<T, d, n, scalar_case::one, scalar_case::neg_one>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_sparse<T, d, n, scalar_case::neg_one, scalar_case::neg_one>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_sparse<T, d, n, scalar_case::other, scalar_case::neg_one>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else
  {
    if (alpha == 1)
      cpu_sparse<T, d, n, scalar_case::one, scalar_case::other>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_sparse<T, d, n, scalar_case::neg_one, scalar_case::other>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_sparse<T, d, n, scalar_case::other, scalar_case::other>(
          rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
}

/*!
 * \brief Helper method that instantiates correct kernel based on alpha and beta.
 */
template<typename T, int d>
void cpu_sparse(int const n, int const rows, int const pntr[], int const indx[],
                int const terms, int const iA[], T const vA[], T const alpha,
                T const x[], T const beta, T y[])
{
  if (beta == 0)
  {
    if (alpha == 1)
      cpu_sparse<T, d, scalar_case::one, scalar_case::zero>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_sparse<T, d, scalar_case::neg_one, scalar_case::zero>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_sparse<T, d, scalar_case::other, scalar_case::zero>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else if (beta == 1)
  {
    if (alpha == 1)
      cpu_sparse<T, d, scalar_case::one, scalar_case::one>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_sparse<T, d, scalar_case::neg_one, scalar_case::one>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_sparse<T, d, scalar_case::other, scalar_case::one>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else if (beta == -1)
  {
    if (alpha == 1)
      cpu_sparse<T, d, scalar_case::one, scalar_case::neg_one>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_sparse<T, d, scalar_case::neg_one, scalar_case::neg_one>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_sparse<T, d, scalar_case::other, scalar_case::neg_one>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
  else
  {
    if (alpha == 1)
      cpu_sparse<T, d, scalar_case::one, scalar_case::other>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else if (alpha == -1)
      cpu_sparse<T, d, scalar_case::neg_one, scalar_case::other>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
    else
      cpu_sparse<T, d, scalar_case::other, scalar_case::other>(
          n, rows, pntr, indx, terms, iA, vA, alpha, x, beta, y);
  }
}

template<typename T>
void cpu_sparse(int const dimensions, int const n, int const num_rows,
                int const pntr[], int const indx[], int const num_terms,
                int const iA[], T const vA[], T const alpha, T const x[],
                T const beta, T y[])
{
  switch (dimensions)
  {
  case 1:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
             beta, y);
      break;
    case 2:
      cpu_sparse<T, 1, 2>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 3:
      cpu_sparse<T, 1, 3>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 4:
      cpu_sparse<T, 1, 4>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    default:
      cpu_sparse<T, 1>(n, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                       beta, y);
      break;
    }
    break;
  case 2:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
             beta, y);
      break;
    case 2:
      cpu_sparse<T, 2, 2>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 3:
      cpu_sparse<T, 2, 3>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 4:
      cpu_sparse<T, 2, 4>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    default:
      cpu_sparse<T, 2>(n, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                       beta, y);
      break;
    }
    break;
  case 3:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
             beta, y);
      break;
    case 2:
      cpu_sparse<T, 3, 2>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 3:
      cpu_sparse<T, 3, 3>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 4:
      cpu_sparse<T, 3, 4>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    default:
      cpu_sparse<T, 3>(n, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                       beta, y);
      break;
    }
    break;
  case 4:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
             beta, y);
      break;
    case 2:
      cpu_sparse<T, 4, 2>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 3:
      cpu_sparse<T, 4, 3>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 4:
      cpu_sparse<T, 4, 4>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    default:
      cpu_sparse<T, 4>(n, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                       beta, y);
      break;
    }
    break;
  case 5:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
             beta, y);
      break;
    case 2:
      cpu_sparse<T, 5, 2>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 3:
      cpu_sparse<T, 5, 3>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 4:
      cpu_sparse<T, 5, 4>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    default:
      cpu_sparse<T, 5>(n, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                       beta, y);
      break;
    }
    break;
  case 6:
    switch (n)
    {
    case 1:
      cpu_n0(dimensions, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
             beta, y);
      break;
    case 2:
      cpu_sparse<T, 6, 2>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 3:
      cpu_sparse<T, 6, 3>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    case 4:
      cpu_sparse<T, 6, 4>(num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                          beta, y);
      break;
    default:
      cpu_sparse<T, 6>(n, num_rows, pntr, indx, num_terms, iA, vA, alpha, x,
                       beta, y);
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
template void cpu_sparse<double>(int const, int const, int const, int const[],
                                 int const[], int const, int const[],
                                 double const[], double const, double const[],
                                 double const, double[]);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void cpu_sparse<float>(int const, int const, int const, int const[],
                                int const[], int const, int const[],
                                float const[], float const, float const[],
                                float const, float[]);
#endif
#endif

} // namespace asgard::kronmult

#endif // KRON_MODE_GLOBAL
