
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

template<typename precision, permutes::matrix_fill fill, int num_dimensions, int dim, int n>
__global__ void kernel_block_gpu_cycle1(
    int const grid_vecs, int const grid_pntr[], int const grid_order[], int const grid_sorted[],
    int const grid_vec_levels[],
    int const **conn_pntr, int const **conn_indx, int const **conn_diag,
    precision const **vals, precision const x[], precision y[])
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
    int num_rows = (1 << level);
    int nnz = conn_pntr[level][num_rows];

    // find an entry to process
    // assumption here is that teamID >= cumulative_nnz, so we are looking for vec_id
    // so that teamID < cumulative_nnz + nnz
    while (vec_id < grid_vecs and teamID > cumulative_nnz + nnz) {
      vec_id++;
      cumulative_nnz += nnz;

      level = grid_vec_levels[vec_id];
      num_rows = (1 << level);
      nnz = conn_pntr[level][num_rows];
    }

    if (vec_id >= grid_vecs) //
      break;

    // from this point, vec_id is a valid vector of 1D multi-indexes
    // now we have to find the x/y index of the specific entry in the product


  }


//   dimension_sort const &dsort = grid.dsort();
//
//   int const num_vecs = dsort.num_vecs(dim);
//
// #ifdef _OPENMP
//   int const max_threads = omp_get_max_threads();
// #else
//   int const max_threads = 1;
// #endif
//
//   if (static_cast<int>(row_wspace.size()) < max_threads)
//     row_wspace.resize(max_threads);
//
//   int threadid = 0;
// #pragma omp parallel
//   {
//     int64_t my_block_count = 0;
//
//     int tid;
// #pragma omp critical
//     tid = threadid++;
//
//     // xidx holds indexes for the entries of the current
//     // sparse row that are present in the current ilist
//     std::vector<int64_t> &xidx = row_wspace[tid];
//     if (static_cast<int>(xidx.size()) < conn.num_rows())
//       xidx.resize(conn.num_rows(), -1);
//
// #pragma omp for schedule(dynamic)
//     for (int vec_id = 0; vec_id < num_vecs; vec_id++)
//     {
//       int const vec_begin = dsort.vec_begin(dim, vec_id);
//       int const vec_end   = dsort.vec_end(dim, vec_id);
//       // map the indexes of present entries
//       for (int j = vec_begin; j < vec_end; j++)
//         xidx[grid.dsorted(dim, j)] = dsort.map(dim, j) * block_size;
//
//       // matrix-vector product using xidx as a row
//       for (int rj = vec_begin; rj < vec_end; rj++)
//       {
//         // row in the 1d pattern
//         int const row = grid.dsorted(dim, rj);
//
//         precision *const local_y = y + xidx[row];
//
//         // columns for the 1d pattern
//         int col_begin = (fill == permutes::matrix_fill::upper) ? conn.row_diag(row) : conn.row_begin(row);
//         int col_end   = (fill == permutes::matrix_fill::lower) ? conn.row_diag(row) : conn.row_end(row);
//
//         if constexpr (n != -1)
//           for (int j = 0; j < block_size; j++)
//             local_y[j] = precision{0};
//
//         for (int c = col_begin; c < col_end; c++)
//         {
//           int64_t const xj = xidx[conn[c]];
//           if (xj != -1)
//           {
//             if constexpr (n == -1)
//               my_block_count += 1;
//             else
//               gbkron_mult_add<precision, num_dimensions, dim, n>(vals + n2 * c, x + xj, local_y);
//           }
//         }
//       }
//
//       // restore the entries
//       for (int j = vec_begin; j < vec_end; j++)
//         xidx[grid.dsorted(dim, j)] = -1;
//     }
//
//     if constexpr (n == -1)
// #pragma omp atomic
//       asgard_kronmult_nblocks_ += my_block_count;
//   } // pragma parallel
}

// void block_cpu(int n, sparse_grid const &grid,
//                connect_1d const &conn, precision const vals[],
//                precision const x[], precision y[],
//                std::vector<std::vector<int64_t>> &row_wspace

template<typename precision>
void launch_block_gpu(
    int n, gpu_grid_data const &grid, gpu_connect_1d const &conns,
    std::array<gpu::vector<precision *>, max_num_dimensions> const &coeffs,
    precision const x[], precision y[])
{
  ignore(n);
  int constexpr nn   = 2;
  int constexpr dim  = 0;
  int constexpr dims = 2;

  permutes::matrix_fill constexpr fill = permutes::matrix_fill::lower;

  constexpr int team_size = nn;
  constexpr int num_teams = ::asgard::gpu::num_teams(team_size);

  int const nvecs = grid.num_vecs[dim];

  dim3 const launch_grid(team_size, num_teams);
  int const launch_blocks = ::asgard::gpu::blocks(nvecs, num_teams);

  kernel_block_gpu_cycle1<precision, fill, dims, dim, nn>
    <<<launch_blocks, launch_grid>>>
    (nvecs, grid.pntr[dim].data(), grid.order[dim].data(), grid.sorted[dim].data(),
     grid.vec_levels[dim].data(),
     conns.pntr.data(), conns.indx.data(), conns.diag.data(),
     coeffs[dim].data(), x, y);
}

template<typename precision>
void block_gpu(gpu::device dev, int n, sparse_grid const &grid,
               connection_patterns const &conns, permutes const &perm,
               std::array<gpu::vector<precision *>, max_num_dimensions> const &coeffs,
               precision alpha, precision const x[], precision beta, precision y[],
               workspace<precision> &work,
               std::array<block_sparse_matrix<precision>, max_num_dimensions> const &cmats)
{
  {
    int64_t const num_entries = work.gpu_w1[dev.id].size();
    static std::vector<precision> cpu_x, cpu_y;
    gpu::copy_to_host(num_entries, x, cpu_x);
    gpu::copy_to_host(num_entries, y, cpu_y);
    block_cpu(n, grid, conns, perm, cmats,
              alpha, cpu_x.data(), beta, cpu_y.data(), work);
    gpu::copy_to_device(cpu_y, y);
    return;
  }

  precision *w1 = work.gpu_w1[dev.id].data();
  precision *w2 = work.gpu_w2[dev.id].data();

  // auto get_connect_1d = [&](permutes::matrix_fill const fill)
  //     -> gpu_connect_1d const & {
  //   // if the term has flux, i.e., fdir != -1
  //   // then the direction using fill::both will use the flux+volume connectivity
  //   // otherwise we will use only the volume connectivity
  //   if (perm.flux_dir != -1 and fill == permutes::matrix_fill::both)
  //     return conns[connect_1d::hierarchy::full];
  //   else
  //     return conns[connect_1d::hierarchy::volume];
  // };

  int const num_dims    = grid.num_dims();
  int const active_dims = perm.num_dimensions();
  expect(active_dims > 0);

  for (size_t i = 0; i < perm.fill.size(); i++)
  {
    int dir = perm.direction[i][0];

    // block_cpu(num_dims, n, grid, dir, perm.fill[i][0],
    //             get_connect_1d(perm.fill[i][0]),
    //             cmats[dir].data(), x, w1, work.row_map);


    //launch_block_gpu(n, grid)

    for (int d = 1; d < active_dims; d++)
    {
      dir = perm.direction[i][d];
      // block_cpu(num_dims, n, grid, dir, perm.fill[i][d],
      //           get_connect_1d(perm.fill[i][d]),
      //           cmats[dir].data(), w1, w2, work.row_map);
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
