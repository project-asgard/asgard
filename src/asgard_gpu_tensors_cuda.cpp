#include "asgard_gpu_tensors.hpp"

namespace asgard::gpu
{

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
    P v = c1[ indexes[2 * i] * n + threadIdx.x / n ];
    v *= c2[ indexes[2 * i + 1] * n + threadIdx.x % n ];

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
    P v = c3[ indexes[3 * i + 2] * n + d % n ];
    d /= n;
    v *= c2[ indexes[3 * i + 1] * n + d % n ];
    d /= n;
    v *= c1[ indexes[3 * i] * n + d ];

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
    P v = c4[ indexes[4 * i + 3] * n + d % n ];
    d /= n;
    v *= c3[ indexes[4 * i + 2] * n + d % n ];
    d /= n;
    v *= c2[ indexes[4 * i + 1] * n + d % n ];
    d /= n;
    v *= c1[ indexes[4 * i] * n + d ];

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
    P v = c5[ indexes[5 * i + 4] * n + d % n ];
    d /= n;
    v *= c4[ indexes[5 * i + 3] * n + d % n ];
    d /= n;
    v *= c3[ indexes[5 * i + 2] * n + d % n ];
    d /= n;
    v *= c2[ indexes[5 * i + 1] * n + d % n ];
    d /= n;
    v *= c1[ indexes[5 * i] * n + d ];

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
    P v = c6[ indexes[6 * i + 5] * n + d % n ];
    d /= n;
    v *= c5[ indexes[6 * i + 4] * n + d % n ];
    d /= n;
    v *= c4[ indexes[6 * i + 3] * n + d % n ];
    d /= n;
    v *= c3[ indexes[6 * i + 2] * n + d % n ];
    d /= n;
    v *= c2[ indexes[6 * i + 1] * n + d % n ];
    d /= n;
    v *= c1[ indexes[6 * i] * n + d ];

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
      P v = c6[ indexes[6 * i + 5] * n + d % n ];
      d /= n;
      v *= c5[ indexes[6 * i + 4] * n + d % n ];
      d /= n;
      v *= c4[ indexes[6 * i + 3] * n + d % n ];
      d /= n;
      v *= c3[ indexes[6 * i + 2] * n + d % n ];
      d /= n;
      v *= c2[ indexes[6 * i + 1] * n + d % n ];
      d /= n;
      v *= c1[ indexes[6 * i] * n + d ];

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
template void tensor_by_index(int, int, int, int const[], double const[], double const[], double const[],
                              double const[], double const[], double const[], double[]);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void tensor_by_index(int, int, int, int const[], float const[], float const[], float const[],
                              float const[], float const[], float const[], float[]);
#endif

}
