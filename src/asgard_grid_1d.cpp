#include "asgard_grid_1d.hpp"

namespace asgard
{
#ifdef ASGARD_USE_GPU
void connection_patterns::load_to_gpu()
{
  int const num_gpus  = compute->num_gpus();
  int const max_level = conns[0].max_loaded_level();

  int const lend = max_level + 1;
  lconns[0].resize(max_level + 1);
  lconns[1].resize(max_level + 1);
  #pragma omp parallel for
  for (int l = 0; l < 2 * lend; l++) {
    lconns[l % 2][l / 2]
      = connect_1d(l/ 2, (l % 2 == 0) ? connect_1d::hierarchy::volume : connect_1d::hierarchy::full);
  }

  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});
    gpu_conns[g] = gpu_connect(max_level);
  }
}
#endif

} // namespace asgard
