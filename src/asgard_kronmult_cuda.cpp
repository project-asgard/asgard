
#include "asgard_kronmult.hpp"

namespace asgard::gpu
{
template<int n, int power>
__device__ constexpr int ipow()
{
  static_assert(power >= 0 and power <= 6,
                "gpu::ipow() does not works with specified power");
  if constexpr (power == 0)
  {
    return 1;
  }
  else if constexpr (power == 1)
  {
    return n;
  }
  else if constexpr (power == 2)
  {
    return n * n;
  }
  else if constexpr (power == 3)
  {
    return n * n * n;
  }
  else if constexpr (power == 4)
  {
    return n * n * n * n;
  }
  else if constexpr (power == 5)
  {
    return n * n * n * n * n;
  }
  else if constexpr (power == 6)
  {
    return n * n * n * n * n * n;
  }
  return 0;
}

}

namespace asgard::kronmult
{

__device__ inline
int binary_search(int first, int last, int const val, int const list[]) {
  while (first <= last) {
    int c = (first + last) / 2;
    if (list[c] < val) {
      first = c + 1;
    } else if (list[c] > val) {
      last = c - 1;
    } else {
      return c;
    }
  }
  return -1;
}

template<typename precision, int num_dims, int dim, int n, int team_size, int num_teams>
__device__ inline void vec_mult_add_cycle4(
        precision const A[], precision const x[], precision y[]) {
  static_assert(n >= 2, "degree 0 basis cannot be proceeded in 2 cycles");
  static_assert(num_dims >= 2, "1D is a special case, use 1 cycle only");

  int constexpr block_size = gpu::ipow<n, num_dims>();
  int constexpr num_last = block_size - 3 * team_size;

  int ia1 = 0;
  int ix1 = 0;
  int ia2 = 0;
  int ix2 = 0;
  int ia3 = 0;
  int ix3 = 0;
  int ia4 = 0;
  int ix4 = 0;

  int const TID2 = threadIdx.x + team_size;
  int const TID3 = threadIdx.x + 2 * team_size;
  int const TID4 = threadIdx.x + 3 * team_size;

  if constexpr (num_dims - dim == 1)
  {
    ix1 = n * (threadIdx.x / n);
    ia1 = threadIdx.x % n;
    ix2 = n * (TID2 / n);
    ia2 = TID2 % n;
    ix3 = n * (TID3 / n);
    ia3 = TID3 % n;
    ix4 = n * (TID4 / n);
    ia4 = TID4 % n;
  }
  else if constexpr (num_dims - dim == 2)
  {
    ix1 = threadIdx.x % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (threadIdx.x / gpu::ipow<n, 2>()));
    ia1 = threadIdx.x / n - ((num_dims == 2) ? 0 : n * (threadIdx.x / gpu::ipow<n, 2>()));
    ix2 = TID2 % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (TID2 / gpu::ipow<n, 2>()));
    ia2 = TID2 / n - ((num_dims == 2) ? 0 : n * (TID2 / gpu::ipow<n, 2>()));
    ix3 = TID3 % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (TID3 / gpu::ipow<n, 2>()));
    ia3 = TID3 / n - ((num_dims == 2) ? 0 : n * (TID3 / gpu::ipow<n, 2>()));
    ix4 = TID4 % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (TID4 / gpu::ipow<n, 2>()));
    ia4 = TID4 / n - ((num_dims == 2) ? 0 : n * (TID4 / gpu::ipow<n, 2>()));
  }
  else if constexpr (num_dims - dim == 3)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (threadIdx.x / gpu::ipow<n, 3>()));
    ia1 = threadIdx.x / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (threadIdx.x / gpu::ipow<n, 3>()));
    ix2 = TID2 % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (TID2 / gpu::ipow<n, 3>()));
    ia2 = TID2 / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (TID2 / gpu::ipow<n, 3>()));
    ix3 = TID3 % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (TID3 / gpu::ipow<n, 3>()));
    ia3 = TID3 / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (TID3 / gpu::ipow<n, 3>()));
    ix4 = TID4 % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (TID4 / gpu::ipow<n, 3>()));
    ia4 = TID4 / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (TID4 / gpu::ipow<n, 3>()));
  }
  else if constexpr (num_dims - dim == 4)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (threadIdx.x / gpu::ipow<n, 4>()));
    ia1 = threadIdx.x / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (threadIdx.x / gpu::ipow<n, 4>()));
    ix2 = TID2 % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (TID2 / gpu::ipow<n, 4>()));
    ia2 = TID2 / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (TID2 / gpu::ipow<n, 4>()));
    ix3 = TID3 % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (TID3 / gpu::ipow<n, 4>()));
    ia3 = TID3 / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (TID3 / gpu::ipow<n, 4>()));
    ix4 = TID4 % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (TID4 / gpu::ipow<n, 4>()));
    ia4 = TID4 / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (TID4 / gpu::ipow<n, 4>()));
  }
  else if constexpr (num_dims - dim == 5)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (threadIdx.x / gpu::ipow<n, 5>()));
    ia1 = threadIdx.x / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (threadIdx.x / gpu::ipow<n, 5>()));
    ix2 = TID2 % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (TID2 / gpu::ipow<n, 5>()));
    ia2 = TID2 / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (TID2 / gpu::ipow<n, 5>()));
    ix3 = TID3 % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (TID3 / gpu::ipow<n, 5>()));
    ia3 = TID3 / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (TID3 / gpu::ipow<n, 5>()));
    ix4 = TID4 % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (TID4 / gpu::ipow<n, 5>()));
    ia4 = TID4 / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (TID4 / gpu::ipow<n, 5>()));
  }
  else if constexpr (num_dims - dim == 6)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (threadIdx.x / gpu::ipow<n, 6>()));
    ia1 = threadIdx.x / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (threadIdx.x / gpu::ipow<n, 6>()));
    ix2 = TID2 % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (TID2 / gpu::ipow<n, 6>()));
    ia2 = TID2 / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (TID2 / gpu::ipow<n, 6>()));
    ix3 = TID3 % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (TID3 / gpu::ipow<n, 6>()));
    ia3 = TID3 / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (TID3 / gpu::ipow<n, 6>()));
    ix4 = TID4 % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (TID4 / gpu::ipow<n, 6>()));
    ia4 = TID4 / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (TID4 / gpu::ipow<n, 6>()));
  }

  bool constexpr use_shared_memory = false;
  if constexpr (use_shared_memory) {

    __shared__ precision XX[num_teams][2 * team_size];
    __shared__ precision AA[num_teams][n * n];

    XX[threadIdx.y][threadIdx.x] = x[threadIdx.x];
    XX[threadIdx.y][TID2] = x[TID2];
    if (threadIdx.x < n * n)
      AA[threadIdx.y][threadIdx.x] = A[threadIdx.x];
    __syncthreads();

    precision yinc1 = 0;
    for (int i = 0; i < n; i++)
      yinc1 += AA[threadIdx.y][ia1 + i * n] * XX[threadIdx.y][ix1 + i * gpu::ipow<n, num_dims - dim - 1>()];

    atomicAdd(&y[threadIdx.x], yinc1);

    if (threadIdx.x < num_last) {
      precision yinc2 = 0;
      for (int i = 0; i < n; i++)
        yinc2 += AA[threadIdx.y][ia2 + i * n] * XX[threadIdx.y][ix2 + i * gpu::ipow<n, num_dims - dim - 1>()];

      atomicAdd(&y[TID2], yinc2);
    }

  } else {

    precision yinc1 = 0;
    for (int i = 0; i < n; i++)
      yinc1 += A[ia1 + i * n] * x[ix1 + i * gpu::ipow<n, num_dims - dim - 1>()];
    atomicAdd(&y[threadIdx.x], yinc1);

    precision yinc2 = 0;
    for (int i = 0; i < n; i++)
      yinc2 += A[ia2 + i * n] * x[ix2 + i * gpu::ipow<n, num_dims - dim - 1>()];
    atomicAdd(&y[TID2], yinc2);

    precision yinc3 = 0;
    for (int i = 0; i < n; i++)
      yinc3 += A[ia3 + i * n] * x[ix3 + i * gpu::ipow<n, num_dims - dim - 1>()];
    atomicAdd(&y[TID3], yinc3);

    if (threadIdx.x < num_last) {
      precision yinc4 = 0;
      for (int i = 0; i < n; i++)
      yinc4 += A[ia4 + i * n] * x[ix4 + i * gpu::ipow<n, num_dims - dim - 1>()];

      atomicAdd(&y[TID4], yinc4);
    }
  }
}

template<typename precision, int num_dims, int dim, int n, int team_size, int num_teams>
__device__ inline void vec_mult_add_cycle3(
        precision const A[], precision const x[], precision y[]) {
  static_assert(n >= 2, "degree 0 basis cannot be proceeded in 2 cycles");
  static_assert(num_dims >= 2, "1D is a special case, use 1 cycle only");

  int constexpr block_size = gpu::ipow<n, num_dims>();
  int constexpr num_last = block_size - 2 * team_size;

  int ia1 = 0;
  int ix1 = 0;
  int ia2 = 0;
  int ix2 = 0;
  int ia3 = 0;
  int ix3 = 0;

  int const TID2 = threadIdx.x + team_size;
  int const TID3 = threadIdx.x + 2 * team_size;

  if constexpr (num_dims - dim == 1)
  {
    ix1 = n * (threadIdx.x / n);
    ia1 = threadIdx.x % n;
    ix2 = n * (TID2 / n);
    ia2 = TID2 % n;
    ix3 = n * (TID3 / n);
    ia3 = TID3 % n;
  }
  else if constexpr (num_dims - dim == 2)
  {
    ix1 = threadIdx.x % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (threadIdx.x / gpu::ipow<n, 2>()));
    ia1 = threadIdx.x / n - ((num_dims == 2) ? 0 : n * (threadIdx.x / gpu::ipow<n, 2>()));
    ix2 = TID2 % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (TID2 / gpu::ipow<n, 2>()));
    ia2 = TID2 / n - ((num_dims == 2) ? 0 : n * (TID2 / gpu::ipow<n, 2>()));
    ix3 = TID3 % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (TID3 / gpu::ipow<n, 2>()));
    ia3 = TID3 / n - ((num_dims == 2) ? 0 : n * (TID3 / gpu::ipow<n, 2>()));
  }
  else if constexpr (num_dims - dim == 3)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (threadIdx.x / gpu::ipow<n, 3>()));
    ia1 = threadIdx.x / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (threadIdx.x / gpu::ipow<n, 3>()));
    ix2 = TID2 % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (TID2 / gpu::ipow<n, 3>()));
    ia2 = TID2 / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (TID2 / gpu::ipow<n, 3>()));
    ix3 = TID3 % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (TID3 / gpu::ipow<n, 3>()));
    ia3 = TID3 / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (TID3 / gpu::ipow<n, 3>()));
  }
  else if constexpr (num_dims - dim == 4)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (threadIdx.x / gpu::ipow<n, 4>()));
    ia1 = threadIdx.x / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (threadIdx.x / gpu::ipow<n, 4>()));
    ix2 = TID2 % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (TID2 / gpu::ipow<n, 4>()));
    ia2 = TID2 / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (TID2 / gpu::ipow<n, 4>()));
    ix3 = TID3 % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (TID3 / gpu::ipow<n, 4>()));
    ia3 = TID3 / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (TID3 / gpu::ipow<n, 4>()));
  }
  else if constexpr (num_dims - dim == 5)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (threadIdx.x / gpu::ipow<n, 5>()));
    ia1 = threadIdx.x / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (threadIdx.x / gpu::ipow<n, 5>()));
    ix2 = TID2 % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (TID2 / gpu::ipow<n, 5>()));
    ia2 = TID2 / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (TID2 / gpu::ipow<n, 5>()));
    ix3 = TID3 % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (TID3 / gpu::ipow<n, 5>()));
    ia3 = TID3 / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (TID3 / gpu::ipow<n, 5>()));
  }
  else if constexpr (num_dims - dim == 6)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (threadIdx.x / gpu::ipow<n, 6>()));
    ia1 = threadIdx.x / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (threadIdx.x / gpu::ipow<n, 6>()));
    ix2 = TID2 % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (TID2 / gpu::ipow<n, 6>()));
    ia2 = TID2 / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (TID2 / gpu::ipow<n, 6>()));
    ix3 = TID3 % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (TID3 / gpu::ipow<n, 6>()));
    ia3 = TID3 / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (TID3 / gpu::ipow<n, 6>()));
  }

  bool constexpr use_shared_memory = false;
  if constexpr (use_shared_memory) {

    __shared__ precision XX[num_teams][3 * team_size];
    __shared__ precision AA[num_teams][n * n];

    XX[threadIdx.y][threadIdx.x] = x[threadIdx.x];
    XX[threadIdx.y][TID2] = x[TID2];
    XX[threadIdx.y][TID3] = x[TID3];
    if (threadIdx.x < n * n)
      AA[threadIdx.y][threadIdx.x] = A[threadIdx.x];
    __syncthreads();

    precision yinc1 = 0;
    for (int i = 0; i < n; i++)
      yinc1 += AA[threadIdx.y][ia1 + i * n] * XX[threadIdx.y][ix1 + i * gpu::ipow<n, num_dims - dim - 1>()];

    atomicAdd(&y[threadIdx.x], yinc1);

    precision yinc2 = 0;
    for (int i = 0; i < n; i++)
      yinc2 += AA[threadIdx.y][ia2 + i * n] * XX[threadIdx.y][ix2 + i * gpu::ipow<n, num_dims - dim - 1>()];

    atomicAdd(&y[TID2], yinc2);

    if (threadIdx.x < num_last) {
      precision yinc3 = 0;
      for (int i = 0; i < n; i++)
        yinc3 += AA[threadIdx.y][ia3 + i * n] * XX[threadIdx.y][ix3 + i * gpu::ipow<n, num_dims - dim - 1>()];

      atomicAdd(&y[TID3], yinc3);
    }

  } else {

    precision yinc1 = 0;
    for (int i = 0; i < n; i++)
      yinc1 += A[ia1 + i * n] * x[ix1 + i * gpu::ipow<n, num_dims - dim - 1>()];
    atomicAdd(&y[threadIdx.x], yinc1);

    precision yinc2 = 0;
    for (int i = 0; i < n; i++)
      yinc2 += A[ia2 + i * n] * x[ix2 + i * gpu::ipow<n, num_dims - dim - 1>()];
    atomicAdd(&y[TID2], yinc2);

    if (threadIdx.x < num_last) {
      precision yinc3 = 0;
      for (int i = 0; i < n; i++)
        yinc3 += A[ia3 + i * n] * x[ix3 + i * gpu::ipow<n, num_dims - dim - 1>()];

      atomicAdd(&y[TID3], yinc3);
    }
  }
}

template<typename precision, int num_dims, int dim, int n, int team_size, int num_teams>
__device__ inline void vec_mult_add_cycle2(
        precision const A[], precision const x[], precision y[]) {
  static_assert(n >= 2, "degree 0 basis cannot be proceeded in 2 cycles");
  static_assert(num_dims >= 2, "1D is a special case, use 1 cycle only");

  int constexpr block_size = gpu::ipow<n, num_dims>();
  int constexpr num_seconds = block_size - team_size;

  int ia1 = 0;
  int ix1 = 0;
  int ia2 = 0;
  int ix2 = 0;

  int const TID2 = threadIdx.x + team_size;

  if constexpr (num_dims - dim == 1)
  {
    ix1 = n * (threadIdx.x / n);
    ia1 = threadIdx.x % n;
    ix2 = n * (TID2 / n);
    ia2 = TID2 % n;
  }
  else if constexpr (num_dims - dim == 2)
  {
    ix1 = threadIdx.x % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (threadIdx.x / gpu::ipow<n, 2>()));
    ia1 = threadIdx.x / n - ((num_dims == 2) ? 0 : n * (threadIdx.x / gpu::ipow<n, 2>()));
    ix2 = TID2 % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (TID2 / gpu::ipow<n, 2>()));
    ia2 = TID2 / n - ((num_dims == 2) ? 0 : n * (TID2 / gpu::ipow<n, 2>()));
  }
  else if constexpr (num_dims - dim == 3)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (threadIdx.x / gpu::ipow<n, 3>()));
    ia1 = threadIdx.x / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (threadIdx.x / gpu::ipow<n, 3>()));
    ix2 = TID2 % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (TID2 / gpu::ipow<n, 3>()));
    ia2 = TID2 / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (TID2 / gpu::ipow<n, 3>()));
  }
  else if constexpr (num_dims - dim == 4)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (threadIdx.x / gpu::ipow<n, 4>()));
    ia1 = threadIdx.x / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (threadIdx.x / gpu::ipow<n, 4>()));
    ix2 = TID2 % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (TID2 / gpu::ipow<n, 4>()));
    ia2 = TID2 / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (TID2 / gpu::ipow<n, 4>()));
  }
  else if constexpr (num_dims - dim == 5)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (threadIdx.x / gpu::ipow<n, 5>()));
    ia1 = threadIdx.x / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (threadIdx.x / gpu::ipow<n, 5>()));
    ix2 = TID2 % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (TID2 / gpu::ipow<n, 5>()));
    ia2 = TID2 / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (TID2 / gpu::ipow<n, 5>()));
  }
  else if constexpr (num_dims - dim == 6)
  {
    ix1 = threadIdx.x % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (threadIdx.x / gpu::ipow<n, 6>()));
    ia1 = threadIdx.x / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (threadIdx.x / gpu::ipow<n, 6>()));
    ix2 = TID2 % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (TID2 / gpu::ipow<n, 6>()));
    ia2 = TID2 / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (TID2 / gpu::ipow<n, 6>()));
  }

  bool constexpr use_shared_memory = false;
  if constexpr (use_shared_memory) {

    __shared__ precision XX[num_teams][2 * team_size];
    __shared__ precision AA[num_teams][n * n];

    XX[threadIdx.y][threadIdx.x] = x[threadIdx.x];
    XX[threadIdx.y][TID2] = x[TID2];
    if (threadIdx.x < n * n)
      AA[threadIdx.y][threadIdx.x] = A[threadIdx.x];
    __syncthreads();

    precision yinc1 = 0;
    for (int i = 0; i < n; i++)
      yinc1 += AA[threadIdx.y][ia1 + i * n] * XX[threadIdx.y][ix1 + i * gpu::ipow<n, num_dims - dim - 1>()];

    atomicAdd(&y[threadIdx.x], yinc1);

    if (threadIdx.x < num_seconds) {
      precision yinc2 = 0;
      for (int i = 0; i < n; i++)
        yinc2 += AA[threadIdx.y][ia2 + i * n] * XX[threadIdx.y][ix2 + i * gpu::ipow<n, num_dims - dim - 1>()];

      atomicAdd(&y[TID2], yinc2);
    }

  } else {

    precision yinc1 = 0;
    for (int i = 0; i < n; i++)
      yinc1 += A[ia1 + i * n] * x[ix1 + i * gpu::ipow<n, num_dims - dim - 1>()];
    atomicAdd(&y[threadIdx.x], yinc1);

    if (threadIdx.x < num_seconds) {
      precision yinc2 = 0;
      for (int i = 0; i < n; i++)
      yinc2 += A[ia2 + i * n] * x[ix2 + i * gpu::ipow<n, num_dims - dim - 1>()];

      atomicAdd(&y[TID2], yinc2);
    }
  }
}

template<typename precision, int num_dims, int dim, int n, int team_size, int num_teams>
__device__ inline void vec_mult_add_cycle1(
        precision const A[], precision const x[], precision y[]) {
  static_assert(n >= 0);
  if constexpr (n == 1) {
    atomicAdd(y, A[0] * x[0]);
    return;
  }

  if constexpr (num_dims == 1) {
    if constexpr (n == 2) {
      precision const a0 = A[threadIdx.x];
      precision const a1 = A[threadIdx.x + n];
      atomicAdd(&y[threadIdx.x], a0 * x[0] + a1 * x[1]);
    } else {
      precision yinc = 0;
      for (int i = 0; i < n; i++)
        yinc += A[threadIdx.x + i * n] * x[i];
      atomicAdd(&y[threadIdx.x], yinc);
    }
    return;
  } else {

    int ia = 0;
    int ix = 0;

    if constexpr (num_dims - dim == 1)
    {
      ix = n * (threadIdx.x / n);
      ia = threadIdx.x % n;
    }
    else if constexpr (num_dims - dim == 2)
    {
      ix = threadIdx.x % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (threadIdx.x / gpu::ipow<n, 2>()));
      ia = threadIdx.x / n - ((num_dims == 2) ? 0 : n * (threadIdx.x / gpu::ipow<n, 2>()));
    }
    else if constexpr (num_dims - dim == 3)
    {
      ix = threadIdx.x % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (threadIdx.x / gpu::ipow<n, 3>()));
      ia = threadIdx.x / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (threadIdx.x / gpu::ipow<n, 3>()));
    }
    else if constexpr (num_dims - dim == 4)
    {
      ix = threadIdx.x % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (threadIdx.x / gpu::ipow<n, 4>()));
      ia = threadIdx.x / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (threadIdx.x / gpu::ipow<n, 4>()));
    }
    else if constexpr (num_dims - dim == 5)
    {
      ix = threadIdx.x % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (threadIdx.x / gpu::ipow<n, 5>()));
      ia = threadIdx.x / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (threadIdx.x / gpu::ipow<n, 5>()));
    }
    else if constexpr (num_dims - dim == 6)
    {
      ix = threadIdx.x % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (threadIdx.x / gpu::ipow<n, 6>()));
      ia = threadIdx.x / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (threadIdx.x / gpu::ipow<n, 6>()));
    }

    precision yinc = 0;
    for (int i = 0; i < n; i++)
      yinc += A[ia + i * n] * x[ix + i * gpu::ipow<n, num_dims - dim - 1>()];
    atomicAdd(&y[threadIdx.x], yinc);
  }
}

template<typename precision, int num_dimensions, int dim, int n, int num_teams,
         int num_cycles = 1>
__global__ void kernel_block_gpu_driver(
    int const grid_vecs, int const grid_pntr[],
    int const grid_order[], int const grid_sorted[],
    int const grid_vec_levels[],
    int const *const *conn_rowcol, int const *conn_nnz,
    precision const *const *vals, precision const x[], precision y[])
{
  // cycle1 case, the team size is n^dim, i.e., one thread per tensor entry
  // ID of member in the team is threadIdx.x
  // ID of the team in the block is threadIdx.y
  // ID of the team in the global workforce is threadIdx.y + blockIdx.x * blockDim.y
  constexpr int n2 = n * n;

  constexpr int block_size = ::asgard::gpu::ipow<n, num_dimensions>();

  int teamID = threadIdx.y + blockIdx.x * blockDim.y;

  int vec_id = 0;
  int cnnz   = 0;

  int level = grid_vec_levels[vec_id];
  int nnz   = conn_nnz[level];

  // process all the vectors, i.e., 1D vector of multi-indexes that match in all but one index
  while (vec_id < grid_vecs) {
    // finding the vec_id for this team
    // each vec needs a number of teams equal to the number of non-zeros in the pattern
    //    that is conn_pntr[grid_vec_levels[vec_id]][num-rows-per-level]

    // find an entry to process
    // look for vec_id such that cumulative_nnz <= teamID < cumulative_nnz + nnz
    // at the start of the loop, we are assuming that cumulative_nnz <= teamID
    while (cnnz + nnz <= teamID) {
      vec_id++; // skip one vector
      cnnz += nnz; // update the running total

      if (vec_id < grid_vecs) {
        level = grid_vec_levels[vec_id];  // update the level and num-rows
        nnz   = conn_nnz[level];
      } else
        return;
    }

    // from this point, vec_id is a valid vector of 1D multi-indexes
    // now we have to find the x/y index of the specific entry in the product
    int const j = teamID - cnnz;

    // here the ir/ic are the row/column indexes of the 1D block
    int const ir = conn_rowcol[level][3 * j];
    int const ic = conn_rowcol[level][3 * j + 1];
    int const ij = conn_rowcol[level][3 * j + 2];

    // need to convert ir/ic to a global ix/iy
    // with the added challenge that the sparse grid may not hold one or both indexes
    int const vec_begin = grid_pntr[vec_id];
    int const vec_end   = grid_pntr[vec_id + 1];
    int ix = vec_begin + ic;
    if (ix >= vec_end or grid_sorted[ix] != ic) {
      // using an adapted grid and we have missing nodes
      int iend = (ix < vec_end) ? ix : vec_end - 1;
      ix = binary_search(vec_begin, iend, ic, grid_sorted);
    }
    int iy = vec_begin + ir;
    if (ix > -1 and (iy >= vec_end or grid_sorted[iy] != ir)) {
      // using an adapted grid and we have missing nodes
      int iend = (iy < vec_end) ? iy : vec_end - 1;
      iy = binary_search(vec_begin, iend, ir, grid_sorted);
    }

    if (ix > -1 and iy > -1) {
      // we found an x/y pair
      if constexpr (num_cycles == 1) {
        vec_mult_add_cycle1<precision, num_dimensions, dim, n, block_size, num_teams>(
              vals[level] + ij * n2,
              x + grid_order[ix] * block_size,
              y + grid_order[iy] * block_size);
      } else if constexpr (num_cycles == 2) {
        int constexpr team_size = block_size / 2 + block_size % 2;
        vec_mult_add_cycle2<precision, num_dimensions, dim, n, team_size, num_teams>(
              vals[level] + ij * n2,
              x + grid_order[ix] * block_size,
              y + grid_order[iy] * block_size);
      } else if constexpr (num_cycles == 3) {
        int constexpr team_size = block_size / 3 + (block_size % 3 == 0 ? 0 : 1);
        vec_mult_add_cycle3<precision, num_dimensions, dim, n, team_size, num_teams>(
              vals[level] + ij * n2,
              x + grid_order[ix] * block_size,
              y + grid_order[iy] * block_size);
      } else if constexpr (num_cycles == 4) {
        int constexpr team_size = block_size / 4 + (block_size % 4 == 0 ? 0 : 1);
        vec_mult_add_cycle4<precision, num_dimensions, dim, n, team_size, num_teams>(
              vals[level] + ij * n2,
              x + grid_order[ix] * block_size,
              y + grid_order[iy] * block_size);
      }
    }

    teamID += gridDim.x * blockDim.y;
  }
}

template<typename precision, int num_dims, int dim, int n>
void launch_block_gpu(
    gpu_grid_data const &grid, gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  // Not the cleanest logic here and some manual tuning was involved.
  // Blocks of data have size n^num_dims and we need to launch a kernel
  // with a specific number of cuda blocks and threads (yes, block has 2 meanings).
  // The cycles refer to the number of data entries manipulated by a single
  // cuda thread, e.g., 1 thread works on 1 entry -> 1 cycle (same for 2, 3, 4).
  // The team size is the number of threads that will work on a single data-block.
  // The number of teams refers to the teams in a single cuda block,
  // the teams and team members for a 2d grid, x -> #team member, y -> #team.

  constexpr int max_threads = 1024;

  constexpr int block_size = ipow<n, num_dims>();

  constexpr int num_cycles = [&]() -> int {
      if constexpr (n == 1)
        return 1; // constant basis can only use one cycle
      if constexpr (num_dims == 6 and n == 4)
        return 4; // needs minimum 4 cycles
      if constexpr (num_dims == 3 and n == 3)
        return 1; // this is an exception

      if constexpr (num_dims >= 4) {
        return 4;
      } else if constexpr (num_dims >= 3) {
        return 2;
      } else {
        return 1;
      }
    }();

  constexpr int team_size = block_size / num_cycles
                           + (block_size % num_cycles == 0 ? 0 : 1);
  constexpr int max_num_teams = max_threads / team_size;
  constexpr int opt_num_teams = std::max(32 / block_size, 1);

  int constexpr num_teams = std::clamp(max_num_teams, 1, opt_num_teams);
  dim3 const launch_grid(team_size, num_teams);

  constexpr int launch_blocks = 2048;

  kernel_block_gpu_driver<precision, num_dims, dim, n, num_teams, num_cycles>
      <<<launch_blocks, launch_grid>>>
      (grid.num_vecs[dim], grid.pntr[dim].data(), grid.order[dim].data(),
       grid.sorted[dim].data(), grid.vec_levels[dim].data(),
       conns.rowcol(), conns.nnz(), vals, x, y);
}

template<typename precision, int num_dims, int dim>
void launch_block_gpu(
    int n, gpu_grid_data const &grid, gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  static_assert(dim < num_dims);
  switch (n)
  {
  case 1:
    launch_block_gpu<precision, num_dims, dim, 1>(grid, conns, vals, x, y);
    break;
  case 2:
    launch_block_gpu<precision, num_dims, dim, 2>(grid, conns, vals, x, y);
    break;
  case 3:
    launch_block_gpu<precision, num_dims, dim, 3>(grid, conns, vals, x, y);
    break;
  case 4:
    launch_block_gpu<precision, num_dims, dim, 4>(grid, conns, vals, x, y);
    break;
  case 5:
    launch_block_gpu<precision, num_dims, dim, 5>(grid, conns, vals, x, y);
    break;
  default:
    throw std::runtime_error("(kronmult-gpu) unimplemented n for given -degree");
  };
}

template<typename precision, int num_dims>
void launch_block_gpu(
    int n, gpu_grid_data const &grid, int dim, gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  expect(dim < num_dims);
  switch (dim)
  {
  case 0:
    launch_block_gpu<precision, num_dims, 0>(n, grid, conns, vals, x, y);
    break;
  case 1:
    if constexpr (num_dims >= 2) {
      launch_block_gpu<precision, num_dims, 1>(n, grid, conns, vals, x, y);
      break;
    }
  case 2:
    if constexpr (num_dims >= 3) {
      launch_block_gpu<precision, num_dims, 2>(n, grid, conns, vals, x, y);
      break;
    }
  case 3:
    if constexpr (num_dims >= 4) {
      launch_block_gpu<precision, num_dims, 3>(n, grid, conns, vals, x, y);
      break;
    }
  case 4:
    if constexpr (num_dims >= 5) {
      launch_block_gpu<precision, num_dims, 4>(n, grid, conns, vals, x, y);
      break;
    }
  case 5:
    if constexpr (num_dims >= 6) {
      launch_block_gpu<precision, num_dims, 5>(n, grid, conns, vals, x, y);
      break;
    }
  default:
    throw std::runtime_error("incorrect dim, incompatible with num_dimensions");
  }
  static_assert(1 <= num_dims and num_dims <= max_num_dimensions);
}

template<typename precision>
void launch_block_gpu(
    int num_dims, int n, gpu_grid_data const &grid, int dim,
    gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  switch (num_dims)
  {
  case 1:
    launch_block_gpu<precision, 1>(n, grid, dim, conns, vals, x, y);
    break;
  case 2:
    launch_block_gpu<precision, 2>(n, grid, dim, conns, vals, x, y);
    break;
  case 3:
    launch_block_gpu<precision, 3>(n, grid, dim, conns, vals, x, y);
    break;
  case 4:
    launch_block_gpu<precision, 4>(n, grid, dim, conns, vals, x, y);
    break;
  case 5:
    launch_block_gpu<precision, 5>(n, grid, dim, conns, vals, x, y);
    break;
  case 6:
    launch_block_gpu<precision, 6>(n, grid, dim, conns, vals, x, y);
    break;
  default:
    throw std::runtime_error("(kronmult-gpu) works with only up to 6 dimensions");
  }
}

template<typename precision>
void block_gpu(gpu::device dev, int n, sparse_grid const &grid,
               connection_patterns const &conns, permutes const &perm,
               std::array<gpu::vector<precision *>, max_num_dimensions> const &coeffs,
               precision alpha, precision const x[], precision beta, precision y[],
               workspace<precision> &work,
               std::array<block_sparse_matrix<precision>, max_num_dimensions> const &cmats)
{
  tools::time_event performance_("block_gpu");

  //{
  //  int64_t const num_entries = work.gpu_w1[dev.id].size();
  //  static std::vector<precision> cpu_x, cpu_y;
  //  gpu::copy_to_host(num_entries, x, cpu_x);
  //  gpu::copy_to_host(num_entries, y, cpu_y);
  //  block_cpu(n, grid, conns, perm, cmats,
  //            alpha, cpu_x.data(), beta, cpu_y.data(), work);
  //  gpu::copy_to_device(cpu_y, y);
  //  return;
  //}

  int64_t const num_entries = work.gpu_w1[dev.id].size();

  precision *w1 = work.gpu_w1[dev.id].data();
  precision *w2 = work.gpu_w2[dev.id].data();

  gpu_connect const &gpu_conn = conns.gpu_conns[dev.id];

  auto get_connect_1d = [&](conn_fill const fill)
      -> gpu_connect_1d const & {
    if (perm.flux_dir != -1 and fill == conn_fill::both)
      return gpu_conn.full();
    else
      return gpu_conn.patts[static_cast<int>(fill)];
  };

  int const num_dims    = grid.num_dims();
  int const active_dims = perm.num_dimensions();
  expect(active_dims > 0);

  for (size_t i = 0; i < perm.fill.size(); i++)
  {
    int dir = perm.direction[i][0];

    // std::cout << " working on dir = " << dir << '\n';

    compute->fill_zeros(num_entries, w1);
    launch_block_gpu(num_dims, n, grid.gpu_grid(dev), dir,
                     get_connect_1d(perm.fill[i][0]),
                     coeffs[dir].data(), x, w1);

    // block_cpu(num_dims, n, grid, dir, perm.fill[i][0],
    //             get_connect_1d(perm.fill[i][0]),
    //             cmats[dir].data(), x, w1, work.row_map);

    for (int d = 1; d < active_dims; d++)
    {
      dir = perm.direction[i][d];
      // block_cpu(num_dims, n, grid, dir, perm.fill[i][d],
      //           get_connect_1d(perm.fill[i][d]),
      //           cmats[dir].data(), w1, w2, work.row_map);

      compute->fill_zeros(num_entries, w2);
      launch_block_gpu(num_dims, n, grid.gpu_grid(dev), dir,
                       get_connect_1d(perm.fill[i][d]),
                       coeffs[dir].data(), w1, w2);

      std::swap(w1, w2);
    }

    compute->device_synchronize();
    cuda_check_error( cudaGetLastError() );

    if (i == 0) { // on iteration zero, scale y
      if (beta == 0)
        compute->fill_zeros(num_entries, y);
      else
        compute->scal(num_entries, beta, y);
    }
    compute->axpy(num_entries, alpha, w1, y);
  }
}

template<typename precision>
void block_gpu(gpu::device dev, int n, sparse_grid const &grid,
               connection_patterns const &conns, permutes const &perm,
               gpu::vector<precision *> const &coeffs,
               precision alpha, precision const x[], precision beta, precision y[],
               workspace<precision> &work,
               // the parameters below are used only for fallback
               block_sparse_matrix<precision> const &cmat)
{
  {
    int64_t const num_entries = work.gpu_w1[dev.id].size();
    static std::vector<precision> cpu_x, cpu_y;
    gpu::copy_to_host(num_entries, x, cpu_x);
    gpu::copy_to_host(num_entries, y, cpu_y);
    block_cpu(n, grid, conns, perm, cmat,
              alpha, cpu_x.data(), beta, cpu_y.data(), work);
    gpu::copy_to_device(cpu_y, y);
    return;
  }
  ignore(coeffs);
}

template<typename precision>
void blocksv_gpu(gpu::device dev, int n, sparse_grid const &grid,
                 connection_patterns const &conns,
                 gpu::vector<precision *> const &gpu_vals,
                 precision y[], workspace<precision> &work,
                 // the parameters below are used only for fallback
                 block_sparse_matrix<precision> const &gvals)
{
  {
    int64_t const num_entries = work.gpu_w1[dev.id].size();
    static std::vector<precision> cpu_y;
    gpu::copy_to_host(num_entries, y, cpu_y);
    blocksv_cpu(n, grid, conns[connect_1d::hierarchy::volume], gvals,
                cpu_y.data(), work);
    gpu::copy_to_device(cpu_y, y);
    return;
  }
  ignore(gpu_vals);
}

#ifdef ASGARD_ENABLE_DOUBLE

template void block_gpu<double>(
    gpu::device, int, sparse_grid const &, connection_patterns const &, permutes const &,
    std::array<gpu::vector<double *>, max_num_dimensions> const &,
    double, double const[], double, double[], workspace<double> &,
    std::array<block_sparse_matrix<double>, max_num_dimensions> const &);

template void block_gpu<double>(
    gpu::device, int, sparse_grid const &, connection_patterns const &, permutes const &,
    gpu::vector<double *> const &,
    double, double const[], double, double[],
    workspace<double> &, block_sparse_matrix<double> const &);

template void blocksv_gpu(
    gpu::device, int, sparse_grid const &, connection_patterns const &,
    gpu::vector<double *> const &, double[], workspace<double> &,
    block_sparse_matrix<double> const &);

#endif

#ifdef ASGARD_ENABLE_FLOAT

template void block_gpu<float>(
    gpu::device, int, sparse_grid const &, connection_patterns const &, permutes const &,
    std::array<gpu::vector<float *>, max_num_dimensions> const &,
    float, float const[], float, float[], workspace<float> &,
    std::array<block_sparse_matrix<precision>, max_num_dimensions> const &);

template void block_gpu<float>(
    gpu::device, int, sparse_grid const &, connection_patterns const &, permutes const &,
    gpu::vector<float *> const &,
    float, float const[], float, float[],
    workspace<float> &, block_sparse_matrix<float> const &);

template void blocksv_gpu(
    gpu::device, int, sparse_grid const &, connection_patterns const &,
    gpu::vector<float *> const &, float[], workspace<float> &,
    block_sparse_matrix<float> const &);

#endif

} // namespace asgard::kronmult
