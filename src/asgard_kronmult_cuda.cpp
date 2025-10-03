
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

inline constexpr int blocks(int64_t work_size, int work_per_block) {
  int constexpr max_blocks = 320;
  return std::min(max_blocks, static_cast<int>((work_size + work_per_block - 1) / work_per_block));
}

inline constexpr int num_teams(int team_size) {
  int constexpr max_threads = 1024;
  return max_threads / team_size;
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

template<typename precision, int num_dims, int dim, int n>
__device__ inline void vec_mult_add(precision const A[], precision const x[], precision y[]) {
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
  }

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

template<typename precision, int num_dimensions, int dim, int n>
__global__ void kernel_block_gpu_cycle1(
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
  // printf(" block_size = %d\n", block_size);
  // printf(" kernel enter\n");

  int teamID = threadIdx.y + blockIdx.x * blockDim.y;

  int vec_id = 0;
  int cnnz   = 0;

  int level = grid_vec_levels[vec_id];
  int nnz   = conn_nnz[level];
  // printf(" level = %d   \n", level);

  // printf(" kernel start: vec_id = %d  nnz = %d \n", vec_id, nnz);

  // process all the vectors, i.e., 1D vector of multi-indexes that match in all but one index
  while (vec_id < grid_vecs) {
    // finding the vec_id for this team
    // each vec needs a number of teams equal to the number of non-zeros in the pattern
    //    that is conn_pntr[grid_vec_levels[vec_id]][num-rows-per-level]

    // printf(" vec_id = %d   level = %d   nnz = %d \n", vec_id, level, nnz);

    // find an entry to process
    // look for vec_id such that cumulative_nnz <= teamID < cumulative_nnz + nnz
    // at the start of the loop, we are assuming that cumulative_nnz <= teamID
    while (cnnz + nnz <= teamID) {
      vec_id++; // skip one vector
      // printf(" new vec_id = %d \n", vec_id);
      cnnz += nnz; // update the running total

      if (vec_id < grid_vecs) {
        level = grid_vec_levels[vec_id];  // update the level and num-rows
        nnz   = conn_nnz[level];
      } else
        return;
      // printf(" level = %d    nnz = %d\n", level, nnz);
    }

    // printf(" vec_id = %d   level = %d   nnz = %d \n", vec_id, level, nnz);

    // if (vec_id >= grid_vecs) // we overran the number of 1D-vectors
    //   break;

    // from this point, vec_id is a valid vector of 1D multi-indexes
    // now we have to find the x/y index of the specific entry in the product

    // printf(" vec_id = %d   level = %d   cumulative_nnz = %d \n", vec_id, level, cumulative_nnz);

    int const j = teamID - cnnz;

    // here the ir/ic are the row/column indexes of the 1D block
    int const ir = conn_rowcol[level][3 * j];
    int const ic = conn_rowcol[level][3 * j + 1];
    int const ij = conn_rowcol[level][3 * j + 2];

    // printf("(ir, ic) = (%d, %d)\n", ir, ic);

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

    // printf("ix = %d  iy = %d \n", ix, iy);
    // printf("(iy, ix) = (%d, %d)  (ir, ic) = (%d, %d)   %e    %e    %e \n", grid_order[iy], grid_order[ix], ir, ic,
    //         (vals[level] + j * n2)[0], (x + grid_order[ix] * block_size)[0], (y + grid_order[iy] * block_size)[0]);
    // printf("(iy, ix) = (%d, %d)  (ir, ic) = (%d, %d)\n", grid_order[iy], grid_order[ix], ir, ic);
    if (ix > -1 and iy > -1) {
      // we found an x/y pair
      vec_mult_add<precision, num_dimensions, dim, n>(
            vals[level] + ij * n2, // <- if using lower/upper this may be a different j, j from the global pattern, needs a conn_map
            x + grid_order[ix] * block_size,
            y + grid_order[iy] * block_size);

      // printf("done mult\n");
    }

    teamID += gridDim.x * blockDim.y;
    // printf(" teamID = %d\n", teamID);
  }
  // printf("kernel end\n");
}

template<typename precision, int num_dims, int dim, int n>
void launch_block_gpu(
    gpu_grid_data const &grid, gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  constexpr int max_threads = 1024;

  constexpr int block_size = ipow<n, num_dims>();

  if constexpr (block_size <= max_threads) {
    // using cycle 1 kernel, i.e., one thread per tensor entry

    constexpr int team_size = block_size;
    constexpr int max_num_teams = max_threads / team_size;
    constexpr int opt_num_teams = std::max(32 / block_size, 1);

    // Given the maximum number of threads, using 1 thread per block, we have
    // the theoretical maximum number of teams due to hardware limitations.
    // This is not necessarily the optimum, the formula for opt num-teams
    // was derived empirically on a GV100 (need to recheck a newer device).
    int const num_teams = std::clamp(max_num_teams, 1, opt_num_teams);
    dim3 const launch_grid(team_size, num_teams);

    constexpr int launch_blocks = 2048;

    kernel_block_gpu_cycle1<precision, num_dims, dim, n>
        <<<launch_blocks, launch_grid>>>
        (grid.num_vecs[dim], grid.pntr[dim].data(), grid.order[dim].data(),
         grid.sorted[dim].data(), grid.vec_levels[dim].data(),
         conns.rowcol(), conns.nnz(), vals, x, y);
  }

  // constexpr int team_size = ipow<n, num_dims>();
  // int num_teams = ::asgard::gpu::num_teams(team_size) / 32;
  // if (num_teams == 0) num_teams = 1;
  // // constexpr int num_teams = 1;
  //
  // int const nvecs = grid.num_vecs[dim];
  //
  // dim3 const launch_grid(team_size, num_teams);
  // // int const launch_blocks = ::asgard::gpu::blocks(nvecs, num_teams);
  // int const launch_blocks = 1640; // Test to figure out this number
  // // int const launch_blocks = 1;
  //
  // auto nz = conns.nnz_.copy_to_host();
  // std::cout << " nnz entries = " << nz.size() << "  " << conns.nnz_.size() << "\n";
  // for (auto &x : nz)
  //   std::cout << " nz = " << x << '\n';

  // std::cout << " team_size = " << team_size
  //           << " num_teams = " << num_teams
  //           << " launch_grid.x = " << launch_grid.x
  //           << " launch_grid.y = " << launch_grid.y
  //           << " launch_grid.z = " << launch_grid.z
  //           << " launch_blocks = " << launch_blocks
  //           << '\n';

  // std::cout << " kernel launch\n";
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
