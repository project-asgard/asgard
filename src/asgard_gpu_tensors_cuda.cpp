#include "asgard_gpu_tensors.hpp"

namespace asgard::gpu
{

//// ========================== source kernels ============================ ////

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

template<typename P, int max_threads> // cycle 4
__global__ void kernel_tensor6d_4(int n, int num_indexes, int const indexes[],
    P const c1[], P const c2[], P const c3[], P const c4[], P const c5[], P const c6[], P x[])
{
  static_assert(max_threads == 1024, "this is an assumption related to the number of cycles, "
                                     "the last cycle must be changed if block-size isn''t 4096 "
                                     "or max-threads isn't 1024 or 4096 / 1024 != 4");
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    for (int j = 0; j < 4; j++) {
      int d = threadIdx.x + j * max_threads;
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

      x[i * n * n * n * n * n * n + threadIdx.x + j * max_threads] = v;
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
      kernel_tensor6d_4<P, max_threads><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, c2, c3, c4, c5, c6, x);
    else
      kernel_tensor6d_1<P><<<launch_blocks, launch_grid>>>(n, num_indexes, indexes, c1, c2, c3, c4, c5, c6, x);
    break;
  default:
    break; // unreachable
  };
}

//// ========================== moment kernels ============================ ////

template<typename P, int vdims>
__global__ void kernel_moment_l0(int pdof, int pos_block, int full_block,
                                 int const num_rij, int const rij[],
                                 P const integ0[], P const integ1[], P const integ2[],
                                 P const state[], P vals[])
{
  static_assert(1 <= vdims and vdims <= 3);
  int blk = threadIdx.y + blockIdx.x * blockDim.y;
  while (blk < num_rij)
  {
    if constexpr (vdims == 1)
    {
      P &out      = vals[rij[2 * blk] * pos_block + threadIdx.x];
      P const *in = state + rij[2 * blk + 1] * full_block + threadIdx.x * pdof;
      P sum = 0;
      for (int k = 0; k < pdof; k++)
        sum += integ0[k] * (*in++);
      out = sum;
    }
    else if constexpr (vdims == 2)
    {
      P &out      = vals[rij[2 * blk] * pos_block + threadIdx.x];
      P const *in = state + rij[2 * blk + 1] * full_block + threadIdx.x * pdof * pdof;
      P sum0 = 0;
      for (int k0 = 0; k0 < pdof; k0++) {
        P sum1 = 0;
        for (int k1 = 0; k1 < pdof; k1++) {
          sum1 += integ1[k1] * (*in++);
        }
        sum0 += integ0[k0] * sum1;
      }
      out = sum0;
    }
    else // if constexpr (vdims == 3)
    {
      P &out      = vals[rij[2 * blk] * pos_block + threadIdx.x];
      P const *in = state + rij[2 * blk + 1] * full_block + threadIdx.x * pdof * pdof * pdof;
      P sum0 = 0;
      for (int k0 = 0; k0 < pdof; k0++) {
        P sum1 = 0;
        for (int k1 = 0; k1 < pdof; k1++) {
          P sum2 = 0;
          for (int k2 = 0; k2 < pdof; k2++) {
            sum2 += integ2[k2] * (*in++);
          }
          sum1 += integ1[k1] * sum2;
        }
        sum0 += integ0[k0] * sum1;
      }
      out = sum0;
    }

    blk += gridDim.x * blockDim.y;
  }
}

template<typename P, int pdims, int vdims>
__global__ void kernel_moment(int pdof, int pos_block, int full_block,
                              unsigned int zeros, int const indexes[],
                              int const num_rij, int const rij[],
                              P const integ0[], P const integ1[], P const integ2[],
                              P const state[], P vals[])
{
  static_assert(1 <= vdims and vdims <= 3);

  bool const zero_l0 = (zeros & 1u != 0);
  bool const zero_l1 = (zeros & 2u != 0);
  bool const zero_l2 = (zeros & 4u != 0);

  int blk = threadIdx.y + blockIdx.x * blockDim.y;
  while (blk < num_rij)
  {
    int const i = rij[2 * blk]; // destination block
    int const j = rij[2 * blk + 1]; // source block

    if constexpr (vdims == 1)
    {
      P const *itg0 = integ0 + indexes[j * (pdims + vdims) + pdims] * pdof;
      P const *in   = state + j * full_block + threadIdx.x * pdof;

      P sum = 0;
      for (int k = 0; k < pdof; k++)
        sum += itg0[k] * (*in++);
      atomicAdd(&vals[i * pos_block + threadIdx.x], sum);
    }
    else if constexpr (vdims == 2)
    {
      int const idx0 = indexes[j * (pdims + vdims) + pdims];
      int const idx1 = indexes[j * (pdims + vdims) + pdims + 1];

      // if using only level 0 and the index is non-zero, skip
      if ((zero_l0 and idx0 != 0) or (zero_l1 and idx1 != 0))
        continue;

      P const *in = state + j * full_block + threadIdx.x * pdof * pdof;
      P const *itg0 = integ0 + idx0 * pdof;
      P const *itg1 = integ1 + idx1 * pdof;

      P sum0 = 0;
      for (int k0 = 0; k0 < pdof; k0++) {
        P sum1 = 0;
        for (int k1 = 0; k1 < pdof; k1++) {
          sum1 += itg1[k1] * (*in++);
        }
        sum0 += itg0[k0] * sum1;
      }
      atomicAdd(&vals[i * pos_block + threadIdx.x], sum0);
    }
    else // if constexpr (vdims == 3)
    {
      int const idx0 = indexes[j * (pdims + vdims) + pdims];
      int const idx1 = indexes[j * (pdims + vdims) + pdims + 1];
      int const idx2 = indexes[j * (pdims + vdims) + pdims + 2];

      // if using only level 0 and the index is non-zero, skip
      if ((zero_l0 and idx0 != 0) or (zero_l1 and idx1 != 0) or (zero_l2 and idx2 != 0))
        continue;

      P const *in = state + j * full_block + threadIdx.x * pdof * pdof * pdof;
      P const *itg0 = integ0 + idx0 * pdof;
      P const *itg1 = integ1 + idx1 * pdof;
      P const *itg2 = integ1 + idx2 * pdof;

      P sum0 = 0;
      for (int k0 = 0; k0 < pdof; k0++) {
        P sum1 = 0;
        for (int k1 = 0; k1 < pdof; k1++) {
          P sum2 = 0;
          for (int k2 = 0; k2 < pdof; k2++) {
            sum2 += itg2[k2] * (*in++);
          }
          sum1 += itg1[k1] * sum2;
        }
        sum0 += itg0[k0] * sum1;
      }
      atomicAdd(&vals[i * pos_block + threadIdx.x], sum0);
    }

    blk += gridDim.x * blockDim.y;
  }
}

template<typename P>
void moment_reduce_zero(int pdof, int pos_block, int full_block, int vdims,
                        gpu::vector<int> const &rij,
                        std::array<P const *, max_mom_dims> const &integ,
                        gpu::vector<P> const &state, gpu::vector<P> &vals)
{
  static_assert(max_mom_dims == 3, "making assumptions here");
  constexpr int max_threads = 1024;
  const int num_teams = max_threads / pos_block;
  dim3 const launch_grid(pos_block, num_teams);
  constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;

  switch (vdims) {
  case 1:
    kernel_moment_l0<P, 1><<<launch_blocks, launch_grid>>>(
        pdof, pos_block, full_block, static_cast<int>(rij.size()), rij.data(),
        integ[0], nullptr, nullptr, state.data(), vals.data());
    break;
  case 2:
    kernel_moment_l0<P, 1><<<launch_blocks, launch_grid>>>(
        pdof, pos_block, full_block, static_cast<int>(rij.size()), rij.data(),
        integ[0], integ[1], nullptr, state.data(), vals.data());
    break;
  case 3:
    kernel_moment_l0<P, 1><<<launch_blocks, launch_grid>>>(
        pdof, pos_block, full_block, static_cast<int>(rij.size()), rij.data(),
        integ[0], integ[1], integ[2], state.data(), vals.data());
    break;
  default:
    break; // unreachable
  }
}

template<typename P>
void moment_reduce(int pdof, int pos_block, int full_block, int pdims, int vdims,
                   std::array<bool, max_mom_dims> lzero, int const *indexes,
                   gpu::vector<int> const &rij,
                   std::array<P const *, max_mom_dims> const &integ,
                   gpu::vector<P> const &state, gpu::vector<P> &vals)
{
  unsigned int const zeros = [&]() -> unsigned int {
      unsigned int res = 0;
      if (lzero[0]) res |= 1u;
      if (lzero[1]) res |= 2u;
      if (lzero[2]) res |= 4u;
      return res;
    }();

  constexpr int max_threads = 1024;
  const int num_teams = max_threads / pos_block;
  dim3 const launch_grid(pos_block, num_teams);
  constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;

  switch (pdims) {
  case 1:
    switch (vdims) {
    case 1:
      kernel_moment<P, 1, 1><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], nullptr, nullptr, state.data(), vals.data());
      break;
    case 2:
      kernel_moment<P, 1, 2><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], integ[1], nullptr, state.data(), vals.data());
      break;
    case 3:
      kernel_moment<P, 1, 3><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], integ[1], integ[2], state.data(), vals.data());
      break;
    default:
      break; // unreachable
    }
    break;
  case 2:
    switch (vdims) {
    case 1:
      kernel_moment<P, 2, 1><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], nullptr, nullptr, state.data(), vals.data());
      break;
    case 2:
      kernel_moment<P, 2, 2><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], integ[1], nullptr, state.data(), vals.data());
      break;
    case 3:
      kernel_moment<P, 2, 3><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], integ[1], integ[2], state.data(), vals.data());
      break;
    default:
      break; // unreachable
    }
    break;
  case 3:
    switch (vdims) {
    case 1:
      kernel_moment<P, 3, 1><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], nullptr, nullptr, state.data(), vals.data());
      break;
    case 2:
      kernel_moment<P, 3, 2><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], integ[1], nullptr, state.data(), vals.data());
      break;
    case 3:
      kernel_moment<P, 3, 3><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, static_cast<int>(rij.size()), rij.data(),
          integ[0], integ[1], integ[2], state.data(), vals.data());
      break;
    default:
      break; // unreachable
    }
    break;
  default:
    break;
  }
}

template<typename P, int num_cycles = 1, int cycle_size = 1>
__global__ void kernel_moment_expand(int pos_block, int vel_block,
                                     int const num_rij, int const rij[],
                                     P const pos_data[], P vals[])
{
  int blk = threadIdx.y + blockIdx.x * blockDim.y;
  while (blk < num_rij)
  {
    int const i = rij[2 * blk]; // source block
    int const j = rij[2 * blk + 1]; // destination block

    if constexpr (num_cycles == 1) {
      P const src = pos_data[i * pos_block + threadIdx.x / vel_block];
      vals[j * pos_block * vel_block + threadIdx.x] = src;
    } else
      for (int c = 0; c < num_cycles; c++) {
        P const src = pos_data[i * pos_block + (threadIdx.x + c * cycle_size) / vel_block];
        vals[j * pos_block * vel_block + threadIdx.x + (threadIdx.x + c * cycle_size)] = src;
      }

    blk += gridDim.x * blockDim.y;
  }
}

template<typename P>
void moment_expand(int pdof, int num_pos, int num_vel, gpu::vector<int> const &rij,
                   gpu::vector<P> const &pos_data, gpu::vector<P> &vals)
{
  int const pos_block = fm::ipow(pdof, num_pos);
  int const vel_block = fm::ipow(pdof, num_vel);

  constexpr int max_threads = 1024;
  constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;

  int const team_size = pos_block * vel_block;
  if (team_size > max_threads) { // multiple cycles
    expect(pdof == 4 and num_pos == 3 and num_vel == 3);
    dim3 const launch_grid(max_threads, 1);
    kernel_moment_expand<P, 4, max_threads><<<launch_blocks, launch_grid>>>(
        pos_block, vel_block, static_cast<int>(rij.size()), rij.data(), pos_data.data(), vals.data());
  } else {
    const int num_teams = max_threads / team_size;
    dim3 const launch_grid(team_size, num_teams);
    kernel_moment_expand<P><<<launch_blocks, launch_grid>>>(
        pos_block, vel_block, static_cast<int>(rij.size()), rij.data(), pos_data.data(), vals.data());
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template void tensor_by_index(int, int, int, int const[], double const[], double const[], double const[],
                              double const[], double const[], double const[], double[]);

template void moment_reduce_zero(int, int, int, int, gpu::vector<int> const &,
                                 std::array<double const *, max_mom_dims> const &,
                                 gpu::vector<double> const &, gpu::vector<double> &);

template void moment_reduce(
    int, int, int, int, int, std::array<bool, max_mom_dims>, int const[],
    gpu::vector<int> const &, std::array<double const *, max_mom_dims> const &,
    gpu::vector<double> const &, gpu::vector<double> &);

template void moment_expand(
    int, int, int, gpu::vector<int> const &, gpu::vector<double> const &, gpu::vector<double> &);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void tensor_by_index(int, int, int, int const[], float const[], float const[], float const[],
                              float const[], float const[], float const[], float[]);

template void moment_reduce_zero(int, int, int, int, gpu::vector<int> const &,
                                 std::array<float const *, max_mom_dims> const &,
                                 gpu::vector<float> const &, gpu::vector<float> &);

template void moment_reduce(
    int, int, int, int, int, std::array<bool, max_mom_dims>, int const[],
    gpu::vector<int> const &, std::array<float const *, max_mom_dims> const &,
    gpu::vector<float> const &, gpu::vector<float> &);

template void moment_expand(
    int, int, int, gpu::vector<int> const &, gpu::vector<float> const &, gpu::vector<float> &);
#endif

}
