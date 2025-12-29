#include "asgard_gpu_algorithms.hpp"

namespace asgard::gpu
{

inline int round_up(int64_t num, int max_threads) {
  int r = num / max_threads;
  if (r * max_threads < num) ++r;
  return r;
}

template<typename P, int num_threads>
__global__ void kernel_jacobi_apply(int64_t num, P const jacobi[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] *= jacobi[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void jacobi_apply(gpu::vector<P> const &jacobi, P y[])
{
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(jacobi.size(), max_threads);

  kernel_jacobi_apply<P, max_threads><<<num_blocks, max_threads>>>
      (jacobi.size(), jacobi.data(), y);
}

template<typename P, int num_threads>
__global__ void kernel_bicgstab_last(int64_t num, P beta, P omega, P const r[], P const v[], P p[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    p[i] = r[i] + beta * (p[i] - omega * v[i]);
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void compute_last_bicgstab(P beta, P omega, gpu::vector<P> const &r,
                           gpu::vector<P> const &v, gpu::vector<P> &p)
{
  expect(r.size() == v.size() and r.size() == p.size());
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(r.size(), max_threads);

  kernel_bicgstab_last<P, max_threads><<<num_blocks, max_threads>>>
      (r.size(), beta, omega, r.data(), v.data(), p.data());
}

template<typename P, int num_threads>
__global__ void kernel_xpby(int64_t num, P beta, P const x[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = x[i] + beta * y[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void xpby(gpu::vector<P> const &x, P beta, P y[]) {
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(x.size(), max_threads);

  kernel_xpby<P, max_threads><<<num_blocks, max_threads>>>(x.size(), beta, x.data(), y);
}

template<typename P, int num_threads>
__global__ void kernel_axpby(int64_t num, P alpha, P const x[], P beta, P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = alpha * x[i] + beta * y[i];
    i += num_threads * gridDim.x;
  }
}
template<typename P, int num_threads>
__global__ void kernel_axpby0(int64_t num, P alpha, P const x[], P y[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    y[i] = alpha * x[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void axpby(int64_t num, P alpha, P const x[], P beta, P y[]) {
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(num, max_threads);

  if (beta == 0)
    kernel_axpby0<P, max_threads><<<num_blocks, max_threads>>>(num, alpha, x, y);
  else
    kernel_axpby<P, max_threads><<<num_blocks, max_threads>>>(num, alpha, x, beta, y);
}

template<typename P, int num_threads>
__global__ void kernel_setscal(int64_t num, P alpha, P x[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    x[i] = alpha * x[i];
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void set_scal(int64_t num, P alpha, P x[]) {
  constexpr int max_threads = 1024;
  int const num_blocks = round_up(num, max_threads);
  kernel_setscal<P, max_threads><<<num_blocks, max_threads>>>(num, alpha, x);
}

template<typename P>
__global__ void kernel_tensor1d(int n, int num_indexes, int const indexes[], P const c1[], P x[])
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    x[i * n + threadIdx.x] = c1[ indexes[i] * n + threadIdx.x ];
    i += gridDim.x * blockDim.y;
  }
}

template<typename P>
__global__ void kernel_tensor2d(int n, int num_indexes, int const indexes[],
                                P const c1[], P const c2[], P x[])
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    P v = c1[ indexes[2 * i] * n + threadIdx.x % n ];
    v *= c2[ indexes[2 * i + 1] * n + threadIdx.x / n ];

    x[i * n * n + threadIdx.x] = v;
    i += gridDim.x * blockDim.y;
  }
}

template<typename P>
__global__ void kernel_tensor3d(int n, int num_indexes, int const indexes[],
                                P const c1[], P const c2[], P const c3[], P x[])
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    int d = threadIdx.x;
    P v = c1[ indexes[3 * i] * n + d % n ];
    d /= n;
    v *= c2[ indexes[3 * i + 1] * n + d % n ];
    d /= n;
    v *= c3[ indexes[3 * i + 2] * n + d ];

    x[i * n * n * n + threadIdx.x] = v;
    i += gridDim.x * blockDim.y;
  }
}

template<typename P>
__global__ void kernel_tensor4d(int n, int num_indexes, int const indexes[],
                                P const c1[], P const c2[], P const c3[], P const c4[], P x[])
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    int d = threadIdx.x;
    P v = c1[ indexes[4 * i] * n + d % n ];
    d /= n;
    v *= c2[ indexes[4 * i + 1] * n + d % n ];
    d /= n;
    v *= c3[ indexes[4 * i + 2] * n + d % n ];
    d /= n;
    v *= c4[ indexes[4 * i + 3] * n + d ];

    x[i * n * n * n * n + threadIdx.x] = v;
    i += gridDim.x * blockDim.y;
  }
}

template<typename P>
__global__ void kernel_tensor5d(int n, int num_indexes, int const indexes[],
    P const c1[], P const c2[], P const c3[], P const c4[], P const c5[], P x[])
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    int d = threadIdx.x;
    P v = c1[ indexes[5 * i] * n + d % n ];
    d /= n;
    v *= c2[ indexes[5 * i + 1] * n + d % n ];
    d /= n;
    v *= c3[ indexes[5 * i + 2] * n + d % n ];
    d /= n;
    v *= c4[ indexes[5 * i + 3] * n + d % n ];
    d /= n;
    v *= c5[ indexes[5 * i + 4] * n + d ];

    x[i * n * n * n * n * n + threadIdx.x] = v;
    i += gridDim.x * blockDim.y;
  }
}

template<typename P>
__global__ void kernel_tensor6d_1(int n, int num_indexes, int const indexes[],
    P const c1[], P const c2[], P const c3[], P const c4[], P const c5[], P const c6[], P x[])
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    int d = threadIdx.x;
    P v = c1[ indexes[6 * i] * n + d % n ];
    d /= n;
    v *= c2[ indexes[6 * i + 1] * n + d % n ];
    d /= n;
    v *= c3[ indexes[6 * i + 2] * n + d % n ];
    d /= n;
    v *= c4[ indexes[6 * i + 3] * n + d % n ];
    d /= n;
    v *= c5[ indexes[6 * i + 4] * n + d % n ];
    d /= n;
    v *= c6[ indexes[6 * i + 5] * n + d ];

    x[i * n * n * n * n * n * n + threadIdx.x] = v;
    i += gridDim.x * blockDim.y;
  }
}

template<typename P> // cycle 4
__global__ void kernel_tensor6d_4(int n, int num_indexes, int const indexes[],
    P const c1[], P const c2[], P const c3[], P const c4[], P const c5[], P const c6[], P x[])
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    for (int j = 0; j < 4; j++) {
      int d = threadIdx.x + j * 1024;
      P v = c1[ indexes[6 * i] * n + d % n ];
      d /= n;
      v *= c2[ indexes[6 * i + 1] * n + d % n ];
      d /= n;
      v *= c3[ indexes[6 * i + 2] * n + d % n ];
      d /= n;
      v *= c4[ indexes[6 * i + 3] * n + d % n ];
      d /= n;
      v *= c5[ indexes[6 * i + 4] * n + d % n ];
      d /= n;
      v *= c6[ indexes[6 * i + 5] * n + d ];

      x[i * n * n * n * n * n * n + threadIdx.x + j * 1024] = v;
    }
    i += gridDim.x * blockDim.y;
  }
}

template<typename P>
void tensor_by_index(int n, int num_dims, int num_indexes, int const indexes[],
    P const c1[], P const c2[], P const c3[], P const c4[], P const c5[], P const c6[], P x[])
{
  constexpr int max_threads = 1024;
  const int team_size = (num_dims == 6 and n == 4) ? 1024 : fm::ipow(n, num_dims);
  const int num_teams = max_threads / team_size;
  dim3 const launch_grid(team_size, num_teams);
  constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;

  switch (num_dims) {
  case 1:
    kernel_tensor1d<P><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, x);
    break;
  case 2:
    kernel_tensor2d<P><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, c2, x);
    break;
  case 3:
    kernel_tensor3d<P><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, c2, c3, x);
    break;
  case 4:
    kernel_tensor4d<P><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, c2, c3, c4, x);
    break;
  case 5:
    kernel_tensor5d<P><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, c2, c3, c4, c5, x);
    break;
  case 6:
    if (num_dims == 6 and n == 4)
      kernel_tensor6d_4<P><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, c2, c3, c4, c5, c6, x);
    else
      kernel_tensor6d_1<P><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, c2, c3, c4, c5, c6, x);
    break;
  default:
    break; // unreachable
  };
}

#ifdef ASGARD_ENABLE_DOUBLE

template void jacobi_apply<double>(gpu::vector<double> const &, double[]);

template void compute_last_bicgstab<double>(double, double, gpu::vector<double> const &,
                                            gpu::vector<double> const &, gpu::vector<double> &);

template void xpby(gpu::vector<double> const &x, double beta, double y[]);
template void axpby(int64_t, double, double const[], double, double[]);
template void set_scal(int64_t, double, double[]);

template void tensor_by_index(int, int, int, int const[], double const[], double const[], double const[],
                              double const[], double const[], double const[], double[]);
#endif

#ifdef ASGARD_ENABLE_FLOAT

template void jacobi_apply<float>(gpu::vector<float> const &, float[]);

template void compute_last_bicgstab<float>(float, float, gpu::vector<float> const &,
                                           gpu::vector<float> const &, gpu::vector<float> &);

template void xpby(gpu::vector<float> const &x, float beta, float y[]);
template void axpby(int64_t, float, float const[], float, float[]);
template void set_scal(int64_t, float, float[]);

template void tensor_by_index(int, int, int, int const[], float const[], float const[], float const[],
                              float const[], float const[], float const[], float[]);
#endif
}
