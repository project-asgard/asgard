
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
