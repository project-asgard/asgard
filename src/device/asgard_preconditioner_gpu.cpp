#include "asgard_kronmult.hpp"

namespace asgard::kronmult
{
#ifdef ASGARD_USE_CUDA

namespace kernel
{
/*!
 * \brief Kernel to apply diagonal Jacobi preconditioner
 *
 * x[i] = 1.0 / (1.0 - dt * precon[i])
 *
 * \tparam T is float or double
 *
 * \param num is the size of prec and x
 * \param dt is the scaling factor, i.e., time-step
 * \param prec is the preconditiner, i.e., the diagonal entries
 * \param x is the right-hand-side
 */
template<typename T>
__global__ void gpu_precon_jacobi(int64_t size, T dt, T const prec[], T x[])
{
  int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < size)
  {
    x[i] /= (1.0 - dt * prec[i]);

    i += gridDim.x * blockDim.x;
  }
}
} // namespace kernel

template<typename T>
void gpu_precon_jacobi(int64_t size, T dt, T const prec[], T x[])
{
  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;

  int num_blocks = std::min(max_blocks, static_cast<int>((size + max_threads - 1) / max_threads));
  kernel::gpu_precon_jacobi<T><<<num_blocks, max_threads>>>(size, dt, prec, x);
}

#ifdef ASGARD_ENABLE_DOUBLE

template void gpu_precon_jacobi(int64_t size, double dt, double const prec[], double x[]);

#endif

#ifdef ASGARD_ENABLE_FLOAT

template void gpu_precon_jacobi(int64_t size, float dt, float const prec[], float x[]);

#endif

#endif

} // namespace asgard::kronmult
