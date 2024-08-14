#include "asgard_kronmult.hpp"

namespace asgard::kronmult
{
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
  T &operator()(int l, int m, int p)
  {
    static_assert(dimensions == 3);
    return serialized[l * n2 + m * n + p];
  }
  T &operator()(int m, int p)
  {
    static_assert(dimensions == 2);
    return serialized[m * n + p];
  }
  T &operator()(int p)
  {
    static_assert(dimensions == 1);
    return serialized[p];
  }
  void zero()
  {
    for (auto &s : serialized)
      s = 0;
  }
  size_t size() const { return serialized.size(); }

private:
  int const n5, n4, n3, n2, n;
  std::vector<T> serialized;
};

template<typename T, int dimensions, scalar_case alpha_case,
         scalar_case beta_case>
void cpu_sparse(int const n, int const num_rows, int const pntr[],
                int const indx[], int const num_terms, int const iA[],
                T const vA[], T const alpha, T const x[], T const beta, T y[])
{
  (void)alpha;
  (void)beta;

#pragma omp parallel
  {
    tensor<T, dimensions> Y(n), W(n);

// always use one thread per kron-product
#pragma omp for
    for (int iy = 0; iy < num_rows; iy++)
    {
      // tensor i (ti) is the first index of this tensor in y
      int const ti = iy * Y.size();
      if constexpr (beta_case == scalar_case::zero)
        for (size_t j = 0; j < Y.size(); j++)
          y[ti + j] = 0;
      else if constexpr (beta_case == scalar_case::neg_one)
        for (size_t j = 0; j < Y.size(); j++)
          y[ti + j] = -y[ti + j];
      else if constexpr (beta_case == scalar_case::other)
        for (size_t j = 0; j < Y.size(); j++)
          y[ti + j] *= beta;

      // ma is the starting index of the operators for this y
      int ma = pntr[iy] * num_terms * dimensions;

      for (int jx = pntr[iy]; jx < pntr[iy + 1]; jx++)
      {
        // tensor i (ti) is the first index of this tensor in x
        int const tj = indx[jx] * Y.size();
        for (int t = 0; t < num_terms; t++)
        {
          if constexpr (dimensions == 1)
          {
            Y.zero();
            T const *const A = &(vA[iA[ma++]]);
            ASGARD_PRAGMA_OMP_SIMD(collapse(2))
            for (int j = 0; j < n; j++)
              for (int k = 0; k < n; k++)
                Y(k) += A[j * n + k] * x[tj + j];

            ASGARD_PRAGMA_OMP_SIMD()
            for (int j = 0; j < n; j++)
              if constexpr (alpha_case == scalar_case::one)
                y[ti + j] += Y(j);
              else if constexpr (alpha_case == scalar_case::neg_one)
                y[ti + j] -= Y(j);
              else
                y[ti + j] += alpha * Y(j);
          }
          else if constexpr (dimensions == 2)
          {
            Y.zero();
            W.zero();
            T const *A = &(vA[iA[ma++]]); // A1
            ASGARD_PRAGMA_OMP_SIMD(collapse(3))
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int k = 0; k < n; k++)
                  W(s, k) += x[tj + n * j + k] * A[j * n + s];
            A = &(vA[iA[ma++]]); // A0
            ASGARD_PRAGMA_OMP_SIMD(collapse(3))
            for (int j = 0; j < n; j++)
              for (int k = 0; k < n; k++)
                for (int s = 0; s < n; s++)
                  Y(k, s) += A[j * n + s] * W(k, j);
            ASGARD_PRAGMA_OMP_SIMD(collapse(2))
            for (int j = 0; j < n; j++)
              for (int k = 0; k < n; k++)
                if constexpr (alpha_case == scalar_case::one)
                  y[ti + n * j + k] += Y(j, k);
                else if constexpr (alpha_case == scalar_case::neg_one)
                  y[ti + n * j + k] -= Y(j, k);
                else
                  y[ti + n * j + k] += alpha * Y(j, k);
          }
          else if constexpr (dimensions == 3)
          {
            Y.zero();
            W.zero();
            T const *A = &(vA[iA[ma++]]); // A2
            ASGARD_PRAGMA_OMP_SIMD(collapse(4))
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    Y(s, l, k) += x[tj + n * n * j + n * l + k] * A[j * n + s];
            A = &(vA[iA[ma++]]); // A1
            ASGARD_PRAGMA_OMP_SIMD(collapse(4))
            for (int l = 0; l < n; l++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int k = 0; k < n; k++)
                    W(l, s, k) += Y(l, j, k) * A[j * n + s];
            Y.zero();
            A = &(vA[iA[ma++]]); // A0
            ASGARD_PRAGMA_OMP_SIMD(collapse(4))
            for (int l = 0; l < n; l++)
              for (int k = 0; k < n; k++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    Y(l, k, s) += A[j * n + s] * W(l, k, j);
            ASGARD_PRAGMA_OMP_SIMD(collapse(3))
            for (int j = 0; j < n; j++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  if constexpr (alpha_case == scalar_case::one)
                    y[ti + n * n * j + n * l + k] += Y(j, l, k);
                  else if constexpr (alpha_case == scalar_case::neg_one)
                    y[ti + n * n * j + n * l + k] -= Y(j, l, k);
                  else
                    y[ti + n * n * j + n * l + k] += alpha * Y(j, l, k);
          }
          else if constexpr (dimensions == 4)
          {
            Y.zero();
            W.zero();
            T const *A = &(vA[iA[ma++]]); // A3
            ASGARD_PRAGMA_OMP_SIMD(collapse(5))
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      W(s, p, l, k) +=
                          x[tj + n * n * n * j + n * n * p + n * l + k] *
                          A[j * n + s];
            A = &(vA[iA[ma++]]); // A2
            ASGARD_PRAGMA_OMP_SIMD(collapse(5))
            for (int p = 0; p < n; p++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      Y(p, s, l, k) += W(p, j, l, k) * A[j * n + s];
            W.zero();
            A = &(vA[iA[ma++]]); // A1
            ASGARD_PRAGMA_OMP_SIMD(collapse(5))
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    for (int k = 0; k < n; k++)
                      W(p, l, s, k) += Y(p, l, j, k) * A[j * n + s];
            Y.zero();
            A = &(vA[iA[ma++]]); // A0
            ASGARD_PRAGMA_OMP_SIMD(collapse(5))
            for (int p = 0; p < n; p++)
              for (int l = 0; l < n; l++)
                for (int k = 0; k < n; k++)
                  for (int j = 0; j < n; j++)
                    for (int s = 0; s < n; s++)
                      Y(p, l, k, s) += A[j * n + s] * W(p, l, k, j);
            ASGARD_PRAGMA_OMP_SIMD(collapse(4))
            for (int j = 0; j < n; j++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    if constexpr (alpha_case == scalar_case::one)
                      y[ti + n * n * n * j + n * n * p + n * l + k] +=
                          Y(j, p, l, k);
                    else if constexpr (alpha_case == scalar_case::neg_one)
                      y[ti + n * n * n * j + n * n * p + n * l + k] -=
                          Y(j, p, l, k);
                    else
                      y[ti + n * n * n * j + n * n * p + n * l + k] +=
                          alpha * Y(j, p, l, k);
          }
          else if constexpr (dimensions == 5)
          {
            Y.zero();
            W.zero();
            T const *A = &(vA[iA[ma++]]); // A4
            ASGARD_PRAGMA_OMP_SIMD(collapse(6))
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int v = 0; v < n; v++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        Y(s, v, p, l, k) +=
                            x[tj + n * n * n * n * j + n * n * n * v +
                              n * n * p + n * l + k] *
                            A[j * n + s];
            A = &(vA[iA[ma++]]); // A3
            ASGARD_PRAGMA_OMP_SIMD(collapse(6))
            for (int v = 0; v < n; v++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int p = 0; p < n; p++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        W(v, s, p, l, k) += Y(v, j, p, l, k) * A[j * n + s];
            Y.zero();
            A = &(vA[iA[ma++]]); // A2
            ASGARD_PRAGMA_OMP_SIMD(collapse(6))
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    for (int l = 0; l < n; l++)
                      for (int k = 0; k < n; k++)
                        Y(v, p, s, l, k) += W(v, p, j, l, k) * A[j * n + s];
            W.zero();
            A = &(vA[iA[ma++]]); // A1
            ASGARD_PRAGMA_OMP_SIMD(collapse(6))
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int j = 0; j < n; j++)
                    for (int s = 0; s < n; s++)
                      for (int k = 0; k < n; k++)
                        W(v, p, l, s, k) += Y(v, p, l, j, k) * A[j * n + s];
            Y.zero();
            A = &(vA[iA[ma++]]); // A0
            ASGARD_PRAGMA_OMP_SIMD(collapse(6))
            for (int v = 0; v < n; v++)
              for (int p = 0; p < n; p++)
                for (int l = 0; l < n; l++)
                  for (int k = 0; k < n; k++)
                    for (int j = 0; j < n; j++)
                      for (int s = 0; s < n; s++)
                        Y(v, p, l, k, s) += A[j * n + s] * W(v, p, l, k, j);
            ASGARD_PRAGMA_OMP_SIMD(collapse(5))
            for (int j = 0; j < n; j++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      if constexpr (alpha_case == scalar_case::one)
                        y[ti + n * n * n * n * j + n * n * n * v + n * n * p +
                          n * l + k] += Y(j, v, p, l, k);
                      else if constexpr (alpha_case == scalar_case::neg_one)
                        y[ti + n * n * n * n * j + n * n * n * v + n * n * p +
                          n * l + k] -= Y(j, v, p, l, k);
                      else
                        y[ti + n * n * n * n * j + n * n * n * v + n * n * p +
                          n * l + k] += alpha * Y(j, v, p, l, k);
          }
          else if constexpr (dimensions == 6)
          {
            W.zero();
            T const *A = &(vA[iA[ma++]]); // A5
            ASGARD_PRAGMA_OMP_SIMD(collapse(7))
            for (int j = 0; j < n; j++)
              for (int s = 0; s < n; s++)
                for (int w = 0; w < n; w++)
                  for (int v = 0; v < n; v++)
                    for (int p = 0; p < n; p++)
                      for (int l = 0; l < n; l++)
                        for (int k = 0; k < n; k++)
                          W(s, w, v, p, l, k) +=
                              x[tj + n * n * n * n * n * j + n * n * n * n * w +
                                n * n * n * v + n * n * p + n * l + k] *
                              A[j * n + s];
            Y.zero();
            A = &(vA[iA[ma++]]); // A4
            ASGARD_PRAGMA_OMP_SIMD(collapse(7))
            for (int w = 0; w < n; w++)
              for (int j = 0; j < n; j++)
                for (int s = 0; s < n; s++)
                  for (int v = 0; v < n; v++)
                    for (int p = 0; p < n; p++)
                      for (int l = 0; l < n; l++)
                        for (int k = 0; k < n; k++)
                          Y(w, s, v, p, l, k) +=
                              W(w, j, v, p, l, k) * A[j * n + s];
            W.zero();
            A = &(vA[iA[ma++]]); // A3
            ASGARD_PRAGMA_OMP_SIMD(collapse(7))
            for (int w = 0; w < n; w++)
              for (int v = 0; v < n; v++)
                for (int j = 0; j < n; j++)
                  for (int s = 0; s < n; s++)
                    for (int p = 0; p < n; p++)
                      for (int l = 0; l < n; l++)
                        for (int k = 0; k < n; k++)
                          W(w, v, s, p, l, k) +=
                              Y(w, v, j, p, l, k) * A[j * n + s];
            Y.zero();
            A = &(vA[iA[ma++]]); // A2
            ASGARD_PRAGMA_OMP_SIMD(collapse(7))
            for (int w = 0; w < n; w++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int j = 0; j < n; j++)
                    for (int s = 0; s < n; s++)
                      for (int l = 0; l < n; l++)
                        for (int k = 0; k < n; k++)
                          Y(w, v, p, s, l, k) +=
                              W(w, v, p, j, l, k) * A[j * n + s];
            W.zero();
            A = &(vA[iA[ma++]]); // A1
            ASGARD_PRAGMA_OMP_SIMD(collapse(7))
            for (int w = 0; w < n; w++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int j = 0; j < n; j++)
                      for (int s = 0; s < n; s++)
                        for (int k = 0; k < n; k++)
                          W(w, v, p, l, s, k) +=
                              Y(w, v, p, l, j, k) * A[j * n + s];
            Y.zero();
            A = &(vA[iA[ma++]]); // A0
            ASGARD_PRAGMA_OMP_SIMD(collapse(7))
            for (int w = 0; w < n; w++)
              for (int v = 0; v < n; v++)
                for (int p = 0; p < n; p++)
                  for (int l = 0; l < n; l++)
                    for (int k = 0; k < n; k++)
                      for (int j = 0; j < n; j++)
                        for (int s = 0; s < n; s++)
                          Y(w, v, p, l, k, s) +=
                              A[j * n + s] * W(w, v, p, l, k, j);
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
                              Y(j, w, v, p, l, k);
                        else if constexpr (alpha_case == scalar_case::neg_one)
                          y[ti + n * n * n * n * n * j + n * n * n * n * w +
                            n * n * n * v + n * n * p + n * l + k] -=
                              Y(j, w, v, p, l, k);
                        else
                          y[ti + n * n * n * n * n * j + n * n * n * n * w +
                            n * n * n * v + n * n * p + n * l + k] +=
                              alpha * Y(j, w, v, p, l, k);
          }
        }
      }
    }
  }
}

} // namespace asgard::kronmult
