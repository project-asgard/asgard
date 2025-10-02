
#include "asgard_kronmult.hpp"

namespace asgard::gpu
{
template<int n, int power>
__device__ constexpr int ipow()
{
  static_assert(power >= 1 and power <= 6,
                "gpu::ipow() does not works with specified power");
  if constexpr (power == 1)
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
  int constexpr max_blocks = 300;
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

template<typename precision, int num_dimensions, int dim, int n>
__device__ inline void vec_mult_add(precision const A[], precision const x[], precision y[]) {
  if constexpr (n == 1) {
    atomicAdd(y, A[0] * x[0]);
  }

  // static_assert(num_dimensions >= 1 and num_dimensions <= 6);
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
  constexpr int n2 = ::asgard::gpu::ipow<n, 2>();

  constexpr int64_t block_size = ::asgard::gpu::ipow<n, num_dimensions>();

  int teamID = threadIdx.y + blockIdx.x * blockDim.y;

  int vec_id = 0;
  int cumulative_nnz = 0;

  // process all the vectors, i.e., 1D vector of multi-indexes that match in all but one index
  while (vec_id < grid_vecs) {
    // finding the vec_id for this team
    // each vec needs a number of teams equal to the number of non-zeros in the pattern
    //    that is conn_pntr[grid_vec_levels[vec_id]][num-rows-per-level]

    int level = grid_vec_levels[vec_id];
    int nnz   = conn_nnz[level];

    // find an entry to process
    // look for vec_id such that cumulative_nnz <= teamID < cumulative_nnz + nnz
    // at the start of the loop, we are assuming that cumulative_nnz <= teamID
    while (vec_id < grid_vecs and cumulative_nnz + nnz <= teamID) {
      vec_id++; // skip one vector
      cumulative_nnz += nnz; // update the running total

      level = grid_vec_levels[vec_id];  // update the level and num-rows
      nnz   = conn_nnz[level];
    }

    if (vec_id >= grid_vecs) // we overran the number of 1D-vectors
      break;

    // from this point, vec_id is a valid vector of 1D multi-indexes
    // now we have to find the x/y index of the specific entry in the product

    int const j = teamID - cumulative_nnz;

    // here the ir/ic are the row/column indexes of the 1D block
    int const ir = conn_rowcol[level][2 * j];
    int const ic = conn_rowcol[level][2 * j + 1];

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
      vec_mult_add<precision, num_dimensions, dim, n>(
            vals[level] + j * n2,
            x + grid_sorted[ix] * block_size,
            y + grid_sorted[iy] * block_size);
    }

    teamID += gridDim.x * blockDim.y;
  }
}

template<typename precision, int num_dimensions, int dim, int n>
void launch_block_gpu(
    gpu_grid_data const &grid, gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  constexpr int team_size = n;
  constexpr int num_teams = ::asgard::gpu::num_teams(team_size);

  int const nvecs = grid.num_vecs[dim];

  dim3 const launch_grid(team_size, num_teams);
  int const launch_blocks = ::asgard::gpu::blocks(nvecs, num_teams);

  kernel_block_gpu_cycle1<precision, num_dimensions, dim, n>
    <<<launch_blocks, launch_grid>>>
    (nvecs, grid.pntr[dim].data(), grid.order[dim].data(), grid.sorted[dim].data(),
     grid.vec_levels[dim].data(),
     conns.rowcol(), conns.nnz(), vals, x, y);
}

template<typename precision, int num_dimensions, int dim>
void launch_block_gpu(
    int n, gpu_grid_data const &grid, gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  static_assert(dim < num_dimensions);
  switch (n)
  {
  case 1:
    launch_block_gpu<precision, num_dimensions, dim, 1>(grid, conns, vals, x, y);
    break;
  default:
    throw std::runtime_error("(kronmult-gpu) unimplemented n for given -degree");
  };
}

template<typename precision, int num_dimensions>
void launch_block_gpu(
    int n, gpu_grid_data const &grid, int dim, gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  if constexpr (num_dimensions == 1)
  {
    launch_block_gpu<precision, num_dimensions, 0>(n, grid, conns, vals, x, y);
  }
  else if constexpr (num_dimensions == 2)
  {
    if (dim == 0)
      launch_block_gpu<precision, num_dimensions, 0>(n, grid, conns, vals, x, y);
    else
      launch_block_gpu<precision, num_dimensions, 1>(n, grid, conns, vals, x, y);
  }
  static_assert(1 <= num_dimensions and num_dimensions <= 2);
}

template<typename precision>
void launch_block_gpu(
    int num_dimensions, int n, gpu_grid_data const &grid, int dim,
    gpu_connect_1d const &conns,
    precision const *const *vals, precision const x[], precision y[])
{
  switch (num_dimensions)
  {
  case 1:
    launch_block_gpu<precision, 1>(n, grid, dim, conns, vals, x, y);
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

      launch_block_gpu(num_dims, n, grid.gpu_grid(dev), dir,
                       get_connect_1d(perm.fill[i][d]),
                       coeffs[dir].data(), w1, w2);

      std::swap(w1, w2);
    }

    int64_t num_entries = work.gpu_w1[dev.id].size();

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
