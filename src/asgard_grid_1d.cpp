#include "asgard_grid_1d.hpp"

namespace asgard
{
#ifdef ASGARD_USE_GPU
void connection_patterns::load_to_gpu()
{
  int const num_gpus = compute->num_gpus();

  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});
    gpu_conns[g][0] = gpu_connect_1d(conns[0]);
    gpu_conns[g][1] = gpu_connect_1d(conns[1]);
  }
}
#endif

} // namespace asgard
