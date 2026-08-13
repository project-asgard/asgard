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

  bool const zero_l0 = ((zeros & 1u) != 0);
  bool const zero_l1 = ((zeros & 2u) != 0);
  bool const zero_l2 = ((zeros & 4u) != 0);

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

  int const num_rij = static_cast<int>(rij.size() / 2);

  switch (vdims) {
  case 1:
    kernel_moment_l0<P, 1><<<launch_blocks, launch_grid>>>(
        pdof, pos_block, full_block, num_rij, rij.data(),
        integ[0], nullptr, nullptr, state.data(), vals.data());
    break;
  case 2:
    kernel_moment_l0<P, 2><<<launch_blocks, launch_grid>>>(
        pdof, pos_block, full_block, num_rij, rij.data(),
        integ[0], integ[1], nullptr, state.data(), vals.data());
    break;
  case 3:
    kernel_moment_l0<P, 3><<<launch_blocks, launch_grid>>>(
        pdof, pos_block, full_block, num_rij, rij.data(),
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

  int const num_rij = static_cast<int>(rij.size() / 2);

  switch (pdims) {
  case 1:
    switch (vdims) {
    case 1:
      kernel_moment<P, 1, 1><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
          integ[0], nullptr, nullptr, state.data(), vals.data());
      break;
    case 2:
      kernel_moment<P, 1, 2><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
          integ[0], integ[1], nullptr, state.data(), vals.data());
      break;
    case 3:
      kernel_moment<P, 1, 3><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
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
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
          integ[0], nullptr, nullptr, state.data(), vals.data());
      break;
    case 2:
      kernel_moment<P, 2, 2><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
          integ[0], integ[1], nullptr, state.data(), vals.data());
      break;
    case 3:
      kernel_moment<P, 2, 3><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
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
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
          integ[0], nullptr, nullptr, state.data(), vals.data());
      break;
    case 2:
      kernel_moment<P, 3, 2><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
          integ[0], integ[1], nullptr, state.data(), vals.data());
      break;
    case 3:
      kernel_moment<P, 3, 3><<<launch_blocks, launch_grid>>>(
          pdof, pos_block, full_block, zeros, indexes, num_rij, rij.data(),
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
        vals[j * pos_block * vel_block + (threadIdx.x + c * cycle_size)] = src;
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

  int const num_rij = static_cast<int>(rij.size() / 2);

  int const team_size = pos_block * vel_block;
  if (team_size > max_threads) { // multiple cycles
    assert(pdof == 4 and num_pos == 3 and num_vel == 3);
    dim3 const launch_grid(max_threads, 1);
    kernel_moment_expand<P, 4, max_threads><<<launch_blocks, launch_grid>>>(
        pos_block, vel_block, num_rij, rij.data(), pos_data.data(), vals.data());
  } else {
    const int num_teams = max_threads / team_size;
    dim3 const launch_grid(team_size, num_teams);
    kernel_moment_expand<P><<<launch_blocks, launch_grid>>>(
        pos_block, vel_block, num_rij, rij.data(), pos_data.data(), vals.data());
  }
}

template<typename P, bool dol2, int num_cycles = 1, int num_threads = 1024>
__global__ void kernel_weights(int num_indexes, P const state[], P weights[])
{
  static_assert(num_cycles == 1 or num_cycles == 4);

  static_assert(sizeof(P) == sizeof(unsigned int) or sizeof(P) == sizeof(unsigned long long int),
                "CUDA does not provide 'atomicMax' operation and integer max is used instead, "
                "but this works only if the sizes of float/double match int32_t/int64_t");

  extern __shared__ double2 mem_[];
  P *data = reinterpret_cast<P *>(mem_);

  int const team_size = blockDim.x;

  int const nd = (team_size % 2 == 0) ? num_cycles * (team_size + 1) : num_cycles * team_size;

  P *const wtotal = weights + num_indexes;

  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    if constexpr (num_cycles == 1) {
      P x = state[i * team_size + threadIdx.x];
      if constexpr (dol2) {
        data[nd * threadIdx.y + threadIdx.x] = x * x;
      } else {
        data[nd * threadIdx.y + threadIdx.x] = abs(x);
      }
    } else {
      for (int k = 0; k < num_cycles; k++) {
        P x = state[i * team_size * num_cycles + k * team_size + threadIdx.x];
        if constexpr (dol2) {
          data[nd * threadIdx.y + k * team_size + threadIdx.x] = x * x;
        } else {
          data[nd * threadIdx.y + k * team_size + threadIdx.x] = abs(x);
        }
      }
    }

    __syncthreads();

    int num = team_size;

    while (num > 1) {
      int const r = (num + 1) / 2;
      if constexpr (dol2) {
        if (threadIdx.x + r < num)
          data[nd * threadIdx.y + threadIdx.x] += data[nd * threadIdx.y + threadIdx.x + r];
      } else {
        if (threadIdx.x + r < num) {
          P const v = data[nd * threadIdx.y + threadIdx.x + r];
          if (data[nd * threadIdx.y + threadIdx.x] < v) data[nd * threadIdx.y + threadIdx.x] = v;
        }
      }

      num = r;

      __syncthreads();
    }

    if (threadIdx.x == 0) {
      P const v = data[nd * threadIdx.y];
      weights[i] = v;
      if constexpr (dol2) {
        atomicAdd(wtotal, v);
      } else {
        // CUDA does not provide atomicMax method for floating point numbers; however,
        // according to IEEE 754 casting the bit-patterns of floating point numbers
        // into integers and comparing the integers is equivalent to comparing
        // the floating point numbers directly, so long as the numbers are positive
        if constexpr (sizeof(P) == sizeof(unsigned int))
          atomicMax(reinterpret_cast<unsigned int *>(wtotal),
                    *reinterpret_cast<unsigned int const *>(&v));
        else
          atomicMax(reinterpret_cast<unsigned long long int *>(wtotal),
                    *reinterpret_cast<unsigned long long int const *>(&v));
      }
    }

    i += gridDim.x * blockDim.y;
  }
}

template<typename P, bool use_l2>
void compute_nrm_tmpl(int block_size, int num_indexes, gpu::vector<P> const &state,
                      gpu::vector<P> &weights, P &l2)
{
  constexpr int max_threads = 1024;
  bool const one_cycle = (block_size <= max_threads);
  int const team_size = (one_cycle) ? block_size : max_threads;
  const int num_teams = max_threads / team_size;
  dim3 const launch_grid(team_size, num_teams);
  constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;

  weights.resize(num_indexes + 1);
  compute->fill_zeros(weights);

  if (one_cycle)
    kernel_weights<P, use_l2, 1, max_threads><<<launch_blocks, launch_grid, (max_threads + num_teams) * sizeof(P)>>>(
        num_indexes, state.data(), weights.data());
  else
    kernel_weights<P, use_l2, 4, max_threads><<<launch_blocks, launch_grid, 4 * (max_threads + num_teams) * sizeof(P)>>>(
        num_indexes, state.data(), weights.data());

  gpu::memcopy_dev2host(1, weights.data() + num_indexes, &l2);
}

template<typename P>
void compute_l2_weights(int block_size, int num_indexes, gpu::vector<P> const &state,
                        gpu::vector<P> &weights, P &l2)
{
  bool constexpr use_l2 = true;
  compute_nrm_tmpl<P, use_l2>(block_size, num_indexes, state, weights, l2);
}

template<typename P>
void compute_max_weights(int block_size, int num_indexes, gpu::vector<P> const &state,
                         gpu::vector<P> &weights, P &wmax)
{
  bool constexpr use_l2 = false; // when not l2 then using weights
  compute_nrm_tmpl<P, use_l2>(block_size, num_indexes, state, weights, wmax);
}

template<typename P, int num_threads, bool force_set>
__global__ void kernel_set_istatus(int64_t num, P tol, P const weights[], sparse_grid::istatus status[])
{
  int i = threadIdx.x + blockIdx.x * num_threads;
  while (i < num) {
    if constexpr (force_set)
      status[i] = (sqrt(weights[i]) >= tol) ? sparse_grid::istatus::refine : sparse_grid::istatus::clear;
    else {
      if (status[i] == sparse_grid::istatus::clear and weights[i] >= tol)
        status[i] = sparse_grid::istatus::refine;
    }
    i += num_threads * gridDim.x;
  }
}

template<typename P>
void set_istatus(int num_indexes, P tolerance, gpu::vector<P> const &weights,
                 gpu::vector<sparse_grid::istatus> &status)
{
  constexpr int max_threads = 1024;
  int const num_blocks = (num_indexes + max_threads - 1) / max_threads;

  constexpr bool force_set = true;

  kernel_set_istatus<P, max_threads, force_set><<<num_blocks, max_threads>>>
      (num_indexes, tolerance, weights.data(), status.data());
}

template<typename P>
void update_istatus(int num_indexes, P tolerance, gpu::vector<P> const &weights,
                    gpu::vector<sparse_grid::istatus> &status)
{
  constexpr int max_threads = 1024;
  int const num_blocks = (num_indexes + max_threads - 1) / max_threads;

  constexpr bool force_set = false; // update, assuming status is already set

  kernel_set_istatus<P, max_threads, force_set><<<num_blocks, max_threads>>>
      (num_indexes, tolerance, weights.data(), status.data());

}

template<typename P, int num_cycles = 1, int num_threads = 1024>
__global__ void kernel_remap(int num_indexes, int const map[], P const old_state[], P new_state[])
{
  static_assert(num_cycles == 1 or num_cycles == 4);

  int const team_size  = blockDim.x;
  int const block_size = blockDim.x * num_cycles;

  int i = threadIdx.y + blockIdx.x * blockDim.y;
  while (i < num_indexes)
  {
    int const old_i = map[i];
    if (map[i] > -1) {
      if constexpr (num_cycles == 1)
        new_state[i * block_size + threadIdx.x] = old_state[old_i * block_size + threadIdx.x];
      else
        for (int k = 0; k < num_cycles; k++)
          new_state[i * block_size + k * team_size + threadIdx.x] =
            old_state[old_i * block_size + k * team_size + threadIdx.x];
    } else {
      if constexpr (num_cycles == 1)
        new_state[i * block_size + threadIdx.x] = P{0};
      else
        for (int k = 0; k < num_cycles; k++)
          new_state[i * block_size + k * team_size + threadIdx.x] = P{0};
    }

    i += gridDim.x * blockDim.y;
  }
}

template<typename P>
void remap_state(int block_size, gpu::vector<int> const &map, gpu::vector<P> &state)
{
  int const num_indexes = static_cast<int>(map.size());

  constexpr int max_threads = 1024;
  bool const one_cycle = (block_size <= max_threads);
  int const team_size = (one_cycle) ? block_size : max_threads;
  assert(one_cycle or 4 * team_size == block_size);
  const int num_teams = max_threads / team_size;
  dim3 const launch_grid(team_size, num_teams);
  constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;

  gpu::vector<P> new_state(map.size() * block_size);

  if (one_cycle)
    kernel_remap<P, 1><<<launch_blocks, launch_grid>>>
        (num_indexes, map.data(), state.data(), new_state.data());
  else
    kernel_remap<P, 4><<<launch_blocks, launch_grid>>>
        (num_indexes, map.data(), state.data(), new_state.data());

  state = std::move(new_state);
}

template<typename P, data_mode dmode, int ib_dim>
__global__ void kernel_merge_boundary_grids(int num_dims, int num_indexes, int const indexes[],
                                            int const map[], P const con1d[], P const bnd[],
                                            int pdof, P alpha, int subgrid_block_size,
                                            int block_size, int ib_stride, P y[])
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;

  while (i < num_indexes)
  {
    int const *idx = indexes + i * num_dims;
    int const isub = map[i];

    P const *block1d = con1d + pdof * idx[ib_dim];
    P const *subblock = bnd + subgrid_block_size * isub;

    int const j = threadIdx.x;

    // Coordinate in the flux dimension.
    int const vib = (j / ib_stride) % pdof;

    // Remove the flux-dimension coordinate from the flattened index.
    int const block = j / (pdof * ib_stride);
    int const offset = j % ib_stride;
    int const ib = block * ib_stride + offset;

    P const b1 = block1d[vib];
    P const b2 = subblock[ib];

    if constexpr (dmode == data_mode::replace)
      y[i * block_size + j] = b1 * b2;
    else if constexpr (dmode == data_mode::scal_rep)
      y[i * block_size + j] = alpha * b1 * b2;
    else if constexpr (dmode == data_mode::increment)
      y[i * block_size + j] += b1 * b2;
    else if constexpr (dmode == data_mode::scal_inc)
      y[i * block_size + j] += alpha * b1 * b2;

    i += gridDim.x * blockDim.y;
  }
}

template<typename P, data_mode dmode, int ib_dim, int ib_stride>
__global__ void kernel_merge_boundary_grids_6d_4(int num_indexes, int const indexes[], int const map[],
                                                 P const con1d[], P const bnd[], P alpha,
                                                 int subgrid_block_size, P y[])
{
  static_assert(ib_dim < 6);

  constexpr int max_threads = 1024;
  constexpr int block_size = 4096;
  constexpr int pdof = 4;
  constexpr int num_dims = 6;

  int i = threadIdx.y + blockIdx.x * blockDim.y;

  while (i < num_indexes)
  {
    int const *idx = indexes + i * num_dims;
    int const isub = map[i];

    P const *block1d = con1d + pdof * idx[ib_dim];
    P const *subblock = bnd + subgrid_block_size * isub;

    // Each thread handles four coefficients.
    for (int c = 0; c < pdof; c++)
    {
      int const j = threadIdx.x + c * max_threads;

      // Coordinate in the flux dimension.
      int const vib = (j / ib_stride) % pdof;

      // Remove the flux-dimension coordinate from the flattened index.
      int const block = j / (pdof * ib_stride);
      int const offset = j % ib_stride;
      int const ib = block * ib_stride + offset;

      P const b1 = block1d[vib];
      P const b2 = subblock[ib];

      if constexpr (dmode == data_mode::replace)
        y[i * block_size + j] = b1 * b2;
      else if constexpr (dmode == data_mode::scal_rep)
        y[i * block_size + j] = alpha * b1 * b2;
      else if constexpr (dmode == data_mode::increment)
        y[i * block_size + j] += b1 * b2;
      else if constexpr (dmode == data_mode::scal_inc)
        y[i * block_size + j] += alpha * b1 * b2;
    }

    i += gridDim.x * blockDim.y;
  }
}

template<typename P, data_mode dmode>
void merge_boundary_grids(sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
                          gpu::vector<int> const &map, gpu::vector<P> const &con1d,
                          gpu::vector<P> const &bnd, int pdof, P alpha, P y[])
{
  int const num_dims = grid.num_dims();
  int const num_indexes = grid.num_indexes();

  int const block_size = fm::ipow(pdof, num_dims);
  int const subgrid_block_size = subgrid.block_size();

  constexpr int max_threads = 1024;
  constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;

  assert(0 <= flux_dim and flux_dim < num_dims);

  if (num_dims == 6 and pdof == 4) {
    constexpr int team_size = 1024;
    dim3 const launch_grid(team_size, 1);

    switch (flux_dim) {
      case 0:
      {
        constexpr int ib_stride = 1024;
        kernel_merge_boundary_grids_6d_4<P, dmode, 0, ib_stride><<<launch_blocks, launch_grid>>>(
            num_indexes, grid.gpu_indexes(), map.data(), con1d.data(), bnd.data(),
            alpha, subgrid_block_size, y);
        break;
      }
      case 1:
      {
        constexpr int ib_stride = 256;
        kernel_merge_boundary_grids_6d_4<P, dmode, 1, ib_stride><<<launch_blocks, launch_grid>>>(
            num_indexes, grid.gpu_indexes(), map.data(), con1d.data(), bnd.data(),
            alpha, subgrid_block_size, y);
        break;
      }
      case 2:
      {
        constexpr int ib_stride = 64;
        kernel_merge_boundary_grids_6d_4<P, dmode, 2, ib_stride><<<launch_blocks, launch_grid>>>(
            num_indexes, grid.gpu_indexes(), map.data(), con1d.data(), bnd.data(),
            alpha, subgrid_block_size, y);
        break;
      }
      case 3:
      {
        constexpr int ib_stride = 16;
        kernel_merge_boundary_grids_6d_4<P, dmode, 3, ib_stride><<<launch_blocks, launch_grid>>>(
            num_indexes, grid.gpu_indexes(), map.data(), con1d.data(), bnd.data(),
            alpha, subgrid_block_size, y);
        break;
      }
      case 4:
      {
        constexpr int ib_stride = 4;
        kernel_merge_boundary_grids_6d_4<P, dmode, 4, ib_stride><<<launch_blocks, launch_grid>>>(
            num_indexes, grid.gpu_indexes(), map.data(), con1d.data(), bnd.data(),
            alpha, subgrid_block_size, y);
        break;
      }
      case 5:
      {
        constexpr int ib_stride = 1;
        kernel_merge_boundary_grids_6d_4<P, dmode, 5, ib_stride><<<launch_blocks, launch_grid>>>(
            num_indexes, grid.gpu_indexes(), map.data(), con1d.data(), bnd.data(),
            alpha, subgrid_block_size, y);
        break;
      }
      default:
        break; // unreachable
    }
  } else {
    assert(block_size <= max_threads);

    int const num_teams = max_threads / block_size;
    dim3 const launch_grid(block_size, num_teams);
    int ib_stride = 1;
    for (int i = flux_dim; i < num_dims - 1; i++)
      ib_stride *= pdof;

    switch (flux_dim)
    {
      case 0:
        kernel_merge_boundary_grids<P, dmode, 0><<<launch_blocks, launch_grid>>>(
            num_dims, num_indexes, grid.gpu_indexes(), map.data(), con1d.data(),
            bnd.data(), pdof, alpha, subgrid_block_size, block_size, ib_stride, y);
        break;
      case 1:
        kernel_merge_boundary_grids<P, dmode, 1><<<launch_blocks, launch_grid>>>(
            num_dims, num_indexes, grid.gpu_indexes(), map.data(), con1d.data(),
            bnd.data(), pdof, alpha, subgrid_block_size, block_size, ib_stride, y);
        break;
      case 2:
        kernel_merge_boundary_grids<P, dmode, 2><<<launch_blocks, launch_grid>>>(
            num_dims, num_indexes, grid.gpu_indexes(), map.data(), con1d.data(),
            bnd.data(), pdof, alpha, subgrid_block_size, block_size, ib_stride, y);
        break;
      case 3:
        kernel_merge_boundary_grids<P, dmode, 3><<<launch_blocks, launch_grid>>>(
            num_dims, num_indexes, grid.gpu_indexes(), map.data(), con1d.data(),
            bnd.data(), pdof, alpha, subgrid_block_size, block_size, ib_stride, y);
        break;
      case 4:
        kernel_merge_boundary_grids<P, dmode, 4><<<launch_blocks, launch_grid>>>(
            num_dims, num_indexes, grid.gpu_indexes(), map.data(), con1d.data(),
            bnd.data(), pdof, alpha, subgrid_block_size, block_size, ib_stride, y);
        break;
      case 5:
        kernel_merge_boundary_grids<P, dmode, 5><<<launch_blocks, launch_grid>>>(
            num_dims, num_indexes, grid.gpu_indexes(), map.data(), con1d.data(),
            bnd.data(), pdof, alpha, subgrid_block_size, block_size, ib_stride, y);
        break;
      default:
        break; // unreachable
    }
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

template void compute_l2_weights(int, int, gpu::vector<double> const &, gpu::vector<double> &, double &);
template void compute_max_weights(int, int, gpu::vector<double> const &, gpu::vector<double> &, double &);

template void set_istatus(int, double, gpu::vector<double> const &, gpu::vector<sparse_grid::istatus> &);
template void update_istatus(int, double, gpu::vector<double> const &, gpu::vector<sparse_grid::istatus> &);

template void merge_boundary_grids<double, data_mode::replace>(
    sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
    gpu::vector<int> const &map, gpu::vector<double> const &con1d,
    gpu::vector<double> const &bnd, int pdof, double alpha, double y[]);
template void merge_boundary_grids<double, data_mode::scal_rep>(
    sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
    gpu::vector<int> const &map, gpu::vector<double> const &con1d,
    gpu::vector<double> const &bnd, int pdof, double alpha, double y[]);
template void merge_boundary_grids<double, data_mode::increment>(
    sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
    gpu::vector<int> const &map, gpu::vector<double> const &con1d,
    gpu::vector<double> const &bnd, int pdof, double alpha, double y[]);
template void merge_boundary_grids<double, data_mode::scal_inc>(
    sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
    gpu::vector<int> const &map, gpu::vector<double> const &con1d,
    gpu::vector<double> const &bnd, int pdof, double alpha, double y[]);
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

template void compute_l2_weights(int, int, gpu::vector<float> const &, gpu::vector<float> &, float &);
template void compute_max_weights(int, int, gpu::vector<float> const &, gpu::vector<float> &, float &);

template void set_istatus(int, float, gpu::vector<float> const &, gpu::vector<sparse_grid::istatus> &);
template void update_istatus(int, float, gpu::vector<float> const &, gpu::vector<sparse_grid::istatus> &);

template void merge_boundary_grids<float, data_mode::replace>(
    sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
    gpu::vector<int> const &map, gpu::vector<float> const &con1d,
    gpu::vector<float> const &bnd, int pdof, float alpha, float y[]);
template void merge_boundary_grids<float, data_mode::scal_rep>(
    sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
    gpu::vector<int> const &map, gpu::vector<float> const &con1d,
    gpu::vector<float> const &bnd, int pdof, float alpha, float y[]);
template void merge_boundary_grids<float, data_mode::increment>(
    sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
    gpu::vector<int> const &map, gpu::vector<float> const &con1d,
    gpu::vector<float> const &bnd, int pdof, float alpha, float y[]);
template void merge_boundary_grids<float, data_mode::scal_inc>(
    sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
    gpu::vector<int> const &map, gpu::vector<float> const &con1d,
    gpu::vector<float> const &bnd, int pdof, float alpha, float y[]);
#endif

// compiling both cases, doesn't take long and no need to #ifdef guard in the indexset
template void remap_state(int, gpu::vector<int> const &, gpu::vector<double> &);
template void remap_state(int, gpu::vector<int> const &, gpu::vector<float> &);

}
