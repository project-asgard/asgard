#pragma once

#include "asgard_transformations.hpp"

namespace asgard
{
/*!
 * \brief Manages the data-structures for the non-separable operations
 */
template<typename P>
class interpolation_manager {
public:
  //! default empty constructor, must reinitialize to use the class
  interpolation_manager() = default;
  //! initialize the manager
  interpolation_manager(pde_domain<P> const &domain, hierarchy_manipulator<P> const &hier,
                        connection_patterns const &conn);

  //! (mostly testing) returns the hierarchical form of the 1d nodes
  std::vector<P> const &nodes1d() const { return nodes1d_; }
  //! returns the nodes corresponding to the grid
  vector2d<P> const &nodes(sparse_grid const &grid) const;

  //! compute nodal values for the field
  void wav2nodal(sparse_grid const &grid, connection_patterns const &conn,
                 P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 0;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(
                  pdof, grid, conn, perm, P{wav_scale}, P{0}, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("wavelet-to-nodal", flops);
    #else
    tools::time_event performance_("wavelet-to-nodal");
    #endif
    block_cpu(pdof, grid, conn, perm, wav2nodal_,
              P{wav_scale}, f, P{0}, vals, work);
  }

  //! compute nodal values for the field
  void nodal2wav(sparse_grid const &grid, connection_patterns const &conn,
                 P alpha, P const f[], P beta, P vals[],
                 kronmult::workspace<P> &work, std::vector<P> &t1) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 1;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = 2 * kronmult::block_cpu(
                  pdof, grid, conn, perm, alpha * P{iwav_scale}, beta, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-wavelet", flops);
    #else
    tools::time_event performance_("nodal-to-wavelet");
    #endif
    block_cpu(pdof, grid, conn, perm, nodal2hier_,
              P{1}, f, P{0}, t1.data(), work);
    block_cpu(pdof, grid, conn, perm, hier2wav_,
              alpha * P{iwav_scale}, t1.data(), beta, vals, work);
  }

  /*!
   * \brief Performs the interpolation of the function func
   *
   * Given the grid, connection patterns, and current time:
   * 1. recomputes the nodes
   * 2. computes the values of the state at the nodes
   * 3. call func() with the time, nodes, state values as "f", and computes vals
   * 4. projects the result back in the basis and y = alpha * vals + beta * y
   *
   * The workspace is needed to call kronmult, the t1 and t2 are additional
   * workspace with size equal to the state.
   * The names t1/t2 come because this uses term_manager scratch space for working with chains
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, P time, P const state[],
       P alpha, md_func_f<P> const &func, P beta, P y[],
       kronmult::workspace<P> &work, std::vector<P> &t1, std::vector<P> &t2) const
  {
    tools::time_event performance_("interpolation operation");
    wav2nodal(grid, conn, state, t1.data(), work);
    {
      tools::time_event perf_("interpolation function");
      func(time, nodes(grid), t1, t2);
    }
    nodal2wav(grid, conn, alpha, t2.data(), beta, y, work, t1);
  }
  /*!
   * \brief Performs the interpolation of the function func
   *
   * Vector variant
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, P time,
       std::vector<P> const &state,
       P alpha, md_func_f<P> const &func, P beta, std::vector<P> &y,
       kronmult::workspace<P> &work, std::vector<P> &t1, std::vector<P> &t2) const
  {
    expect(state.size() == t1.size() and t1.size() == t2.size());
    if (beta == 0)
      y.resize(state.size());
    else
      expect(y.size() == state.size());
    (*this)(grid, conn, time, state.data(), alpha, func, beta, y.data(),
            work, t1, t2);
  }
  /*!
   * \brief Performs the interpolation of the function func
   *
   * Given the grid, connection patterns, and current time:
   * 1. recomputes the nodes
   * 2. call func() with the time, nodes, and computes vals
   * 3. projects the result back in the basis and y = alpha * vals + beta * y
   *
   * The workspace is needed to call kronmult, the t1 and t2 are additional
   * workspace with size equal to the state.
   * The names t1/t2 come because this sues term_manager scratch space for working with chains
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, P time,
       P alpha, md_func<P> const &func, P beta, P y[],
       kronmult::workspace<P> &work, std::vector<P> &t1, std::vector<P> &t2) const
  {
    tools::time_event performance_("interpolation project");
    {
      tools::time_event perf_("interpolation eval");
      func(time, nodes(grid), t1);
    }
    nodal2wav(grid, conn, alpha, t1.data(), beta, y, work, t2);
  }
  /*!
   * \brief Performs the interpolation of the function func
   *
   * Vector variant
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, P time,
       P alpha, md_func<P> const &func, P beta, std::vector<P> &y,
       kronmult::workspace<P> &work, std::vector<P> &t1, std::vector<P> &t2) const
  {
    if (beta == 0)
      y.resize(t1.size());
    else
      expect(y.size() == t1.size());
    (*this)(grid, conn, time, alpha, func, beta, y.data(), work, t1, t2);
  }

  //! indicates whether the manager has been initialized
  operator bool () const { return (num_dims > 0); }

  #ifdef ASGARD_USE_GPU
  //! compute nodal values for the field
  void wav2nodal(gpu::device dev, sparse_grid const &grid,
                 connection_patterns const &conn, P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 0;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(n, grid, conn, perm, P{wav_scale}, P{0}, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("wavelet-to-nodal", flops);
    #else
    tools::time_event performance_("wavelet-to-nodal");
    #endif
    block_gpu(dev, pdof, grid, conn, perm, gpu_wav2nodal_[dev.id], P{wav_scale}, f,
              P{0}, vals, work, wav2nodal_);
  }
  //! compute nodal values for the field
  void nodal2wav(gpu::device dev, sparse_grid const &grid,
                 connection_patterns const &conn,
                 P alpha, P const f[], P beta, P vals[],
                 kronmult::workspace<P> &work, gpu::vector<P> &t1) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 1;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = 2 * kronmult::block_cpu(
                  pdof, grid, conn, perm, alpha * P{iwav_scale}, beta, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-wavelet", flops);
    #else
    tools::time_event performance_("nodal-to-wavelet");
    #endif
    block_gpu(dev, pdof, grid, conn, perm, gpu_nodal2hier_[dev.id],
              P{1}, f, P{0}, t1.data(), work, nodal2hier_);
    block_gpu(dev, pdof, grid, conn, perm, gpu_hier2wav_[dev.id],
              alpha * P{iwav_scale}, t1.data(), beta, vals, work, hier2wav_);
  }

  /*!
   * \brief Performs the interpolation of the function func
   */
  void operator ()
      (gpu::device dev, sparse_grid const &grid,
       connection_patterns const &conn, P time, P const state[],
       P alpha, md_func_f<P> const &func, P beta, P y[],
       kronmult::workspace<P> &work,
       std::vector<P> &t1, std::vector<P> &t2,
       gpu::vector<P> &gpu_t1, gpu::vector<P> &gpu_t2) const
  {
    tools::time_event performance_("interpolation operation");
    wav2nodal(dev, grid, conn, state, gpu_t1.data(), work);
    gpu_t1.copy_to_host(t1);
    {
      tools::time_event perf_("interpolation function");
      func(time, nodes(grid), t1, t2);
    }
    gpu_t1 = t2;
    nodal2wav(dev, grid, conn, alpha, gpu_t1.data(), beta, y, work, gpu_t2);
  }
  /*!
   * \brief Computes the interpolation function on the CPU and moves the data to the GPU
   *
   * In this context, the kronmult work is done on the GPU
   * but the function evaluation is done on the CPU side.
   */
  void operator ()
      (gpu::device dev, sparse_grid const &grid,
       connection_patterns const &conn, P time,
       P alpha, md_func<P> const &func, P beta, P y[],
       kronmult::workspace<P> &work,
       std::vector<P> &t1,
       gpu::vector<P> &gpu_t1, gpu::vector<P> &gpu_t2) const
  {
    func(time, nodes(grid), t1);
    gpu_t1 = t1;
    nodal2wav(dev, grid, conn, alpha, gpu_t1.data(), beta, y, work, gpu_t2);
  }
  #endif

private:
  int num_dims = 0;
  int pdof = 0;
  int block_size = 0;
  std::array<P, max_num_dimensions> xmin, xscale;
  P wav_scale = 0, iwav_scale = 0;

  mutable int grid_gen = -1;

  std::vector<P> nodes1d_;
  mutable vector2d<P> nodes_;

  kronmult::permutes perm;

  block_sparse_matrix<P> wav2nodal_;
  block_sparse_matrix<P> nodal2hier_;
  block_sparse_matrix<P> hier2wav_;

  #ifdef ASGARD_USE_GPU
  //! gpu coefficient matrices for different levels wavelet to nodal
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lwav2nodal_;
  //! gpu coefficient matrices for different levels nodal to hierarchical
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lnodal2hier_;
  //! gpu coefficient matrices for different levels hierarchical to wavelet
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lhier2wav_;
  //! pointers to gpu matrices for different levels
  std::array<gpu::vector<P*>, max_num_gpus> gpu_wav2nodal_;
  //! pointers to gpu matrices for different levels
  std::array<gpu::vector<P*>, max_num_gpus> gpu_nodal2hier_;
  //! pointers to gpu matrices for different levels
  std::array<gpu::vector<P*>, max_num_gpus> gpu_hier2wav_;
  #endif

  #ifdef ASGARD_USE_FLOPCOUNTER
  struct flop_info_entry {
    int grid_gen = -1;
    int64_t flops = 0;
  };
  // indexes are wav2nodal (0), nodal2wav (1)
  mutable std::array<flop_info_entry, 2> flop_info;
  #endif
};

} // namespace asgard
