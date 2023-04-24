#include <iostream>
#include <set>

#include "build_info.hpp"

#include "asgard_kronmult.hpp"

namespace asgard::kronmult
{
template<typename T>
inline void omp_atomic_add(T *p, T inc_value)
{
#pragma omp atomic
  (*p) += inc_value;
}

template<typename T, int dimensions>
class tensor
{
public:
  tensor(int n_)
      : n5(n_ * n_ * n_ * n_ * n_), n4(n_ * n_ * n_ * n_), n3(n_ * n_ * n_),
        n2(n_ * n_), n(n_),
        serialized(n *
                   ((dimensions == 6)
                        ? n5
                        : ((dimensions == 5)
                               ? n4
                               : ((dimensions == 4)
                                      ? n3
                                      : ((dimensions == 3)
                                             ? n2
                                             : ((dimensions == 2) ? n : 1))))))
  {
    static_assert(1 <= dimensions and dimensions <= 6);
  }
  T &operator()(int i, int j, int k, int l, int m, int p)
  {
    static_assert(dimensions == 6);
    return serialized[i * n5 + j * n4 + k * n3 + l * n2 + m * n + p];
  }
  T &operator()(int j, int k, int l, int m, int p)
  {
    static_assert(dimensions == 5);
    return serialized[j * n4 + k * n3 + l * n2 + m * n + p];
  }
  T &operator()(int k, int l, int m, int p)
  {
    static_assert(dimensions == 4);
    return serialized[k * n3 + l * n2 + m * n + p];
  }
  T &operator()(int l, int m, int p) {
    static_assert(dimensions == 3);
    return serialized[l * n2 + m * n + p];
  }
  T &operator()(int m, int p) {
    static_assert(dimensions == 2);
    return serialized[m * n + p];
  }
  T &operator()(int p) {
    static_assert(dimensions == 1);
    return serialized[p];
  }
  void zero()
  {
    for (auto &s : serialized)
      s = 0;
  }

private:
  int const n5, n4, n3, n2, n;
  std::vector<T> serialized;
};

template<typename T, int dimensions>
void run_cpu_variant(int const n, T const *const pA[], int const lda,
                     T const *const pX[], T *pY[], int const num_batch)
{
#pragma omp parallel
  {
    tensor<T, dimensions> Y(n), W(n);

#pragma omp for
    for (int i = 0; i < num_batch; i++)
    {
      if constexpr (dimensions == 1)
      {
        Y.zero();
        for (int j = 0; j < n; j++)
        {
          for (int k = 0; k < n; k++)
          {
            Y(k) += pA[i][j * lda + k] * pX[i][j];
          }
        }
        for (int j = 0; j < n; j++)
        {
#pragma omp atomic
          pY[i][j] += Y(j);
        }
      }
      else if constexpr (dimensions == 2)
      {
        Y.zero();
        W.zero();
        for (int j = 0; j < n; j++)
          for (int k = 0; k < n; k++)
            for (int s = 0; s < n; s++)
              W(s, k) += pX[i][n * j + k] * pA[2 * i][j * lda + s];
        Y.zero();
        for (int j = 0; j < n; j++)
          for (int k = 0; k < n; k++)
            for (int s = 0; s < n; s++)
              Y(k, s) += pA[2 * i + 1][j * lda + s] * W(k, j);
        for (int j = 0; j < n; j++)
        {
          for (int k = 0; k < n; k++)
          {
#pragma omp atomic
            pY[i][n * j + k] += Y(j, k);
          }
        }
      }
      else if constexpr (dimensions == 3)
      {
        Y.zero();
        W.zero();
        for (int j = 0; j < n; j++)
          for (int l = 0; l < n; l++)
            for (int k = 0; k < n; k++)
              for (int s = 0; s < n; s++)
                Y(s, l, k) +=
                    pX[i][n * n * j + n * l + k] * pA[3 * i][j * lda + s];
        for (int j = 0; j < n; j++)
          for (int l = 0; l < n; l++)
            for (int k = 0; k < n; k++)
              for (int s = 0; s < n; s++)
                W(l, s, k) += Y(l, j, k) * pA[3 * i + 1][j * lda + s];
        Y.zero();
        for (int j = 0; j < n; j++)
          for (int l = 0; l < n; l++)
            for (int k = 0; k < n; k++)
              for (int s = 0; s < n; s++)
                Y(l, k, s) += pA[3 * i + 2][j * lda + s] * W(l, k, j);
        for (int j = 0; j < n; j++)
        {
          for (int l = 0; l < n; l++)
          {
            for (int k = 0; k < n; k++)
            {
#pragma omp atomic
              pY[i][n * n * j + n * l + k] += Y(j, l, k);
            }
          }
        }
      }
      else if constexpr (dimensions == 4)
      {
        Y.zero();
        W.zero();
        for (int j = 0; j < n; j++)
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                for (int s = 0; s < n; s++)
                  W(s, p, l, k) +=
                      pX[i][n * n * n * j + n * n * p + n * l + k] *
                      pA[4 * i][j * lda + s];
        for (int j = 0; j < n; j++)
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                for (int s = 0; s < n; s++)
                  Y(p, s, l, k) += W(p, j, l, k) * pA[4 * i + 1][j * lda + s];
        W.zero();
        for (int j = 0; j < n; j++)
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                for (int s = 0; s < n; s++)
                  W(p, l, s, k) += Y(p, l, j, k) * pA[4 * i + 2][j * lda + s];
        Y.zero();
        for (int j = 0; j < n; j++)
          for (int p = 0; p < n; p++)
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                for (int s = 0; s < n; s++)
                  Y(p, l, k, s) += pA[4 * i + 3][j * lda + s] * W(p, l, k, j);
        for (int j = 0; j < n; j++)
        {
          for (int p = 0; p < n; p++)
          {
            for (int l = 0; l < n; l++)
            {
              for (int k = 0; k < n; k++)
              {
#pragma omp atomic
                pY[i][n * n * n * j + n * n * p + n * l + k] += Y(j, p, l, k);
              }
            }
          }
        }
      }
      else if constexpr (dimensions == 5)
      {
        Y.zero();
        W.zero();
        for (int j = 0; j < n; j++)
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  for (int s = 0; s < n; s++)
                    Y(s, v, p, l, k) +=
                        pX[i][n * n * n * n * j + n * n * n * v + n * n * p +
                              n * l + k] *
                        pA[5 * i][j * lda + s];
        for (int j = 0; j < n; j++)
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  for (int s = 0; s < n; s++)
                    W(v, s, p, l, k) +=
                        Y(v, j, p, l, k) * pA[5 * i + 1][j * lda + s];
        Y.zero();
        for (int j = 0; j < n; j++)
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  for (int s = 0; s < n; s++)
                    Y(v, p, s, l, k) +=
                        W(v, p, j, l, k) * pA[5 * i + 2][j * lda + s];
        W.zero();
        for (int j = 0; j < n; j++)
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  for (int s = 0; s < n; s++)
                    W(v, p, l, s, k) +=
                        Y(v, p, l, j, k) * pA[5 * i + 3][j * lda + s];
        Y.zero();
        for (int j = 0; j < n; j++)
          for (int v = 0; v < n; v++)
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  for (int s = 0; s < n; s++)
                    Y(v, p, l, k, s) +=
                        pA[5 * i + 4][j * lda + s] * W(v, p, l, k, j);
        for (int j = 0; j < n; j++)
        {
          for (int v = 0; v < n; v++)
          {
            for (int p = 0; p < n; p++)
            {
              for (int l = 0; l < n; l++)
              {
                for (int k = 0; k < n; k++)
                {
#pragma omp atomic
                  pY[i][n * n * n * n * j + n * n * n * v + n * n * p + n * l +
                        k] += Y(j, v, p, l, k);
                }
              }
            }
          }
        }
      }
      else if constexpr (dimensions == 6)
      {
        W.zero();
        for (int j = 0; j < n; j++)
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    for (int s = 0; s < n; s++)
                      W(s, w, v, p, l, k) +=
                          pX[i][n * n * n * n * n * j + n * n * n * n * w +
                                n * n * n * v + n * n * p + n * l + k] *
                          pA[6 * i][j * lda + s];
        Y.zero();
        for (int j = 0; j < n; j++)
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    for (int s = 0; s < n; s++)
                      Y(w, s, v, p, l, k) +=
                          W(w, j, v, p, l, k) * pA[6 * i + 1][j * lda + s];
        W.zero();
        for (int j = 0; j < n; j++)
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    for (int s = 0; s < n; s++)
                      W(w, v, s, p, l, k) +=
                          Y(w, v, j, p, l, k) * pA[6 * i + 2][j * lda + s];
        Y.zero();
        for (int j = 0; j < n; j++)
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    for (int s = 0; s < n; s++)
                      Y(w, v, p, s, l, k) +=
                          W(w, v, p, j, l, k) * pA[6 * i + 3][j * lda + s];
        W.zero();
        for (int j = 0; j < n; j++)
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    for (int s = 0; s < n; s++)
                      W(w, v, p, l, s, k) +=
                          Y(w, v, p, l, j, k) * pA[6 * i + 4][j * lda + s];
        Y.zero();
        for (int j = 0; j < n; j++)
          for (int w = 0; w < n; w++)
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    for (int s = 0; s < n; s++)
                      Y(w, v, p, l, k, s) +=
                          pA[6 * i + 5][j * lda + s] * W(w, v, p, l, k, j);
        for (int j = 0; j < n; j++)
        {
          for (int w = 0; w < n; w++)
          {
            for (int v = 0; v < n; v++)
            {
              for (int p = 0; p < n; p++)
              {
                for (int l = 0; l < n; l++)
                {
                  for (int k = 0; k < n; k++)
                  {
#pragma omp atomic
                    pY[i][n * n * n * n * n * j + n * n * n * n * w +
                          n * n * n * v + n * n * p + n * l + k] +=
                        Y(j, w, v, p, l, k);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

#define asgard_kronmult_cpu_instantiate(d)                                    \
  template void run_cpu_variant<float, (d)>(                                  \
      int n, float const *const pA[], int const lda, float const *const pX[], \
      float *pY[], int const num_batch);                                      \
  template void run_cpu_variant<double, (d)>(                                 \
      int n, double const *const pA[], int const lda,                         \
      double const *const pX[], double *pY[], int const num_batch);

asgard_kronmult_cpu_instantiate(1);
asgard_kronmult_cpu_instantiate(2);
asgard_kronmult_cpu_instantiate(3);
asgard_kronmult_cpu_instantiate(4);
asgard_kronmult_cpu_instantiate(5);
asgard_kronmult_cpu_instantiate(6);

} // namespace asgard::kronmult
