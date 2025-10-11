
#include "asgard_kronmult.hpp"

namespace asgard::gpu
{
template<int n, int power>
__device__ constexpr int ipow()
{
  static_assert(power >= 0 and power <= 6,
                "gpu::ipow() does not works with specified power");
  if constexpr (power == 0)
    return 1;
  else if constexpr (power == 1)
    return n;
  else if constexpr (power == 2)
    return n * n;
  else if constexpr (power == 3)
    return n * n * n;
  else if constexpr (power == 4)
    return n * n * n * n;
  else if constexpr (power == 5)
    return n * n * n * n * n;
  else if constexpr (power == 6)
    return n * n * n * n * n * n;

  return 0;
}

}  // namespace asgard::gpu

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

template<int n, int num_dims, int relative_dir>
__device__ int offset_ix(int base) {
  if constexpr (relative_dir == 1) {
    return n * (base / n);
  } else if constexpr (relative_dir == 2) {
    return base % n + ((num_dims == 2) ? 0 : gpu::ipow<n, 2>() * (base / gpu::ipow<n, 2>()));
  } else if constexpr (relative_dir == 3) {
    return base % gpu::ipow<n, 2>() + ((num_dims == 3) ? 0 : gpu::ipow<n, 3>() * (base / gpu::ipow<n, 3>()));
  } else if constexpr (relative_dir == 4) {
    return base % gpu::ipow<n, 3>() + ((num_dims == 4) ? 0 : gpu::ipow<n, 4>() * (base / gpu::ipow<n, 4>()));
  } else if constexpr (relative_dir == 5) {
    return base % gpu::ipow<n, 4>() + ((num_dims == 5) ? 0 : gpu::ipow<n, 5>() * (base / gpu::ipow<n, 5>()));
  } else if constexpr (relative_dir == 6) {
    return base % gpu::ipow<n, 5>() + ((num_dims == 6) ? 0 : gpu::ipow<n, 6>() * (base / gpu::ipow<n, 6>()));
  }
}

template<int n, int num_dims, int relative_dir>
__device__ int offset_ia(int base) {
  if constexpr (relative_dir == 1) {
    return base % n;
  } else if constexpr (relative_dir == 2) {
    return base / n - ((num_dims == 2) ? 0 : n * (base / gpu::ipow<n, 2>()));
  } else if constexpr (relative_dir == 3) {
    return base / gpu::ipow<n, 2>() - ((num_dims == 3) ? 0 : n * (base / gpu::ipow<n, 3>()));
  } else if constexpr (relative_dir == 4) {
    return base / gpu::ipow<n, 3>() - ((num_dims == 4) ? 0 : n * (base / gpu::ipow<n, 4>()));
  } else if constexpr (relative_dir == 5) {
    return base / gpu::ipow<n, 4>() - ((num_dims == 5) ? 0 : n * (base / gpu::ipow<n, 5>()));
  } else if constexpr (relative_dir == 6) {
    return base / gpu::ipow<n, 5>() - ((num_dims == 6) ? 0 : n * (base / gpu::ipow<n, 6>()));
  }
}

template<typename precision, int num_dims, int dim, int n, int block_size, int num_teams, int num_cycles>
__device__ inline void vec_mult_add(
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
    // flip the dimension index, the fastest index is the last,
    // so we flip the index here
    constexpr int dflip = num_dims - dim;

    int ix = offset_ix<n, num_dims, dflip>(threadIdx.x);
    int ia = offset_ia<n, num_dims, dflip>(threadIdx.x);

    precision yinc = 0;
    for (int i = 0; i < n; i++)
      yinc += A[ia + i * n] * x[ix + i * gpu::ipow<n, dflip - 1>()];
    atomicAdd(&y[threadIdx.x], yinc);

    // constexpr int block_size = gpu::ipow<n, num_dims>();
    int constexpr team_size = block_size / num_cycles + (block_size % num_cycles == 0 ? 0 : 1);
    int constexpr last_size = block_size - (num_cycles - 1) * team_size;

    if constexpr (num_cycles >= 2) {
      ix = offset_ix<n, num_dims, dflip>(threadIdx.x + team_size);
      ia = offset_ia<n, num_dims, dflip>(threadIdx.x + team_size);

      if (num_cycles != 2 or threadIdx.x < last_size) {
        yinc = 0;
        for (int i = 0; i < n; i++)
          yinc += A[ia + i * n] * x[ix + i * gpu::ipow<n, dflip - 1>()];
        atomicAdd(&y[threadIdx.x + team_size], yinc);
      }
    }

    if constexpr (num_cycles >= 3) {
      ix = offset_ix<n, num_dims, dflip>(threadIdx.x + 2 * team_size);
      ia = offset_ia<n, num_dims, dflip>(threadIdx.x + 2 * team_size);

      if (num_cycles != 3 or threadIdx.x < last_size) {
        yinc = 0;
        for (int i = 0; i < n; i++)
          yinc += A[ia + i * n] * x[ix + i * gpu::ipow<n, dflip - 1>()];
        atomicAdd(&y[threadIdx.x + 2 * team_size], yinc);
      }
    }

    if constexpr (num_cycles >= 4) {
      ix = offset_ix<n, num_dims, dflip>(threadIdx.x + 3 * team_size);
      ia = offset_ia<n, num_dims, dflip>(threadIdx.x + 3 * team_size);

      if (num_cycles != 4 or threadIdx.x < last_size) {
        yinc = 0;
        for (int i = 0; i < n; i++)
          yinc += A[ia + i * n] * x[ix + i * gpu::ipow<n, dflip - 1>()];
        atomicAdd(&y[threadIdx.x + 3 * team_size], yinc);
      }
    }
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
      vec_mult_add<precision, num_dimensions, dim, n, block_size, num_teams, num_cycles>(
              vals[level] + ij * n2,
              x + grid_order[ix] * block_size,
              y + grid_order[iy] * block_size);
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

  constexpr int launch_blocks = ASGARD_NUM_GPU_BLOCKS;

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
               std::array<block_sparse_matrix<precision>, max_num_dimensions> const &)
{
  tools::time_event performance_("block_gpu");

  int64_t const num_entries = work.gpu_w1[dev.id].size();

  precision *w1 = work.gpu_w1[dev.id].data();
  precision *w2 = work.gpu_w2[dev.id].data();

  gpu_connect const &gpu_conn = conns.gpu_conns[dev.id];

  auto get_connect_1d = [&](conn_fill const fill)
      -> gpu_connect_1d const & {
    if (perm.flux_dir != -1 and fill == conn_fill::both)
      return gpu_conn.full();
    else if (fill == conn_fill::lower_udiag)
      return gpu_conn.patts[static_cast<int>(conn_fill::lower)];
    else
      return gpu_conn.patts[static_cast<int>(fill)];
  };

  int const num_dims    = grid.num_dims();
  int const active_dims = perm.num_dimensions();
  expect(active_dims > 0);

  for (size_t i = 0; i < perm.fill.size(); i++)
  {
    int dir = perm.direction[i][0];

    compute->fill_zeros(num_entries, w1);
    launch_block_gpu(num_dims, n, grid.gpu_grid(dev), dir,
                     get_connect_1d(perm.fill[i][0]),
                     coeffs[dir].data(), x, w1);

    if (perm.fill[i][0] == conn_fill::lower_udiag)
      compute->axpy(num_entries, x, w1);

    for (int d = 1; d < active_dims; d++)
    {
      dir = perm.direction[i][d];

      compute->fill_zeros(num_entries, w2);
      launch_block_gpu(num_dims, n, grid.gpu_grid(dev), dir,
                       get_connect_1d(perm.fill[i][d]),
                       coeffs[dir].data(), w1, w2);

      if (perm.fill[i][d] == conn_fill::lower_udiag)
        compute->axpy(num_entries, w1, w2);

      std::swap(w1, w2);
    }

    // compute->device_synchronize();
    // cuda_check_error( cudaGetLastError() );

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
               block_sparse_matrix<precision> const &)
{
  tools::time_event performance_("block_gpu");

  int64_t const num_entries = work.gpu_w1[dev.id].size();

  precision *w1 = work.gpu_w1[dev.id].data();
  precision *w2 = work.gpu_w2[dev.id].data();

  gpu_connect const &gpu_conn = conns.gpu_conns[dev.id];

  auto get_connect_1d = [&](conn_fill const fill)
      -> gpu_connect_1d const & {
    // cannot happen until we connect interpolation to fux
    // if (perm.flux_dir != -1 and fill == conn_fill::both) {
    //   return gpu_conn.full();
    // } else
    if (fill == conn_fill::lower_udiag)
      return gpu_conn.patts[static_cast<int>(conn_fill::lower)];
    return gpu_conn.patts[static_cast<int>(fill)];
  };

  int const num_dims    = grid.num_dims();
  int const active_dims = perm.num_dimensions();
  expect(active_dims > 0);

  for (size_t i = 0; i < perm.fill.size(); i++)
  {
    int dir = perm.direction[i][0];

    compute->fill_zeros(num_entries, w1);
    launch_block_gpu(num_dims, n, grid.gpu_grid(dev), dir,
                     get_connect_1d(perm.fill[i][0]),
                     coeffs.data(), x, w1);

    if (perm.fill[i][0] == conn_fill::lower_udiag)
      compute->axpy(num_entries, x, w1);

    for (int d = 1; d < active_dims; d++)
    {
      dir = perm.direction[i][d];

      compute->fill_zeros(num_entries, w2);
      launch_block_gpu(num_dims, n, grid.gpu_grid(dev), dir,
                       get_connect_1d(perm.fill[i][d]),
                       coeffs.data(), w1, w2);

      if (perm.fill[i][d] == conn_fill::lower_udiag)
        compute->axpy(num_entries, w1, w2);

      std::swap(w1, w2);
    }

    // compute->device_synchronize();
    // cuda_check_error( cudaGetLastError() );

    if (i == 0) { // on iteration zero, scale y
      if (beta == 0)
        compute->fill_zeros(num_entries, y);
      else
        compute->scal(num_entries, beta, y);
    }
    compute->axpy(num_entries, alpha, w1, y);
  }
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
    double, double const[], double, double[], workspace<double> &,
    block_sparse_matrix<double> const &);

#endif

#ifdef ASGARD_ENABLE_FLOAT

template void block_gpu<float>(
    gpu::device, int, sparse_grid const &, connection_patterns const &, permutes const &,
    std::array<gpu::vector<float *>, max_num_dimensions> const &,
    float, float const[], float, float[], workspace<float> &,
    std::array<block_sparse_matrix<float>, max_num_dimensions> const &);

template void block_gpu<float>(
    gpu::device, int, sparse_grid const &, connection_patterns const &, permutes const &,
    gpu::vector<float *> const &,
    float, float const[], float, float[], workspace<float> &,
    block_sparse_matrix<float> const &);

#endif

} // namespace asgard::kronmult
