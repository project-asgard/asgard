#pragma once

#include "asgard_transformations.hpp"

namespace asgard
{
/*!
 * \brief Describes the stages of the interpolation operation.
 *
 * Uses bit operations to avoid storing multiple bools.
 */
struct interpolation_plan
{
  //! holds the information about the plan
  int plan_mode_ = 0;

  //! enable/disable the interpolation plan
  void enable(bool val = true) {
    if (val)
      plan_mode_ |= (1 << enabled_);
    else
      plan_mode_ &= ~(1 << enabled_);
  }
  //! use the existing interpolated field or start from wavelet coefficients
  void use_field(bool val = true) {
    if (val)
      plan_mode_ |= (1 << field_);
    else
      plan_mode_ &= ~(1 << field_);
  }
  //! does the current function use moments
  void use_moments(bool val = true) {
    if (val)
      plan_mode_ |= (1 << moments_);
    else
      plan_mode_ &= ~(1 << moments_);
  }
  //! do we stop at the hierarchical coefficients or go back to wavelet basis
  void stop_hier(bool val = true) {
    if (val)
      plan_mode_ |= (1 << hier_);
    else
      plan_mode_ &= ~(1 << hier_);
  }

  //! indicates whether the plan has been enabled
  bool is_enabled() const {
    // if any flags is set, this is an interpolatory term
    return (plan_mode_ != 0);
  }
  //! indicates whether the plan uses pre-interpolated field
  bool uses_field() const { return (plan_mode_ & (1 << field_)) != 0; }
  //! indicates whether the plan uses moments
  bool uses_moments() const { return (plan_mode_ & (1 << moments_)) != 0; }
  //! indicates whether the plan stops at the hierarchy
  bool uses_hier() const { return (plan_mode_ & (1 << hier_)) != 0; }

  //! tag for whether to use the enabled
  static int constexpr enabled_ = 0;
  //! tag for whether to use the field
  static int constexpr field_ = 1;
  //! tag for whether to use moments
  static int constexpr moments_ = 2;
  //! tag for whether to stop at the hierarchy
  static int constexpr hier_ = 3;
};

/*!
 * \brief Manages the data-structures for the non-separable operations
 */
template<typename P>
class interpolation_manager {
public:
  //! default empty constructor, must reinitialize to use the class
  interpolation_manager() = default;
  //! initialize the manager
  interpolation_manager(prog_opts const &opts,
                        pde_domain<P> const &domain,
                        hierarchy_manipulator<P> const &hier,
                        connection_patterns const &conn);

  //! the program options are needed only to potentially set new point
  interpolation_manager(pde_domain<P> const &domain,
                        hierarchy_manipulator<P> const &hier,
                        connection_patterns const &conn)
    : interpolation_manager(prog_opts{}, domain, hier, conn)
  {}

  //! (mostly testing) returns the hierarchical form of the 1d nodes
  std::vector<P> const &nodes1d() const { return nodes1d_; }
  //! returns the nodes corresponding to the grid
  vector2d<P> const &nodes(sparse_grid const &grid) const;

  //! compute nodal values for the field
  void wav2nodal(sparse_grid const &grid, P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 0;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn_reduced, perm, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("wavelet-to-nodal", flops);
    #else
    tools::time_event performance_("wavelet-to-nodal");
    #endif
    block_cpu(pdof, grid, conn_reduced, perm, wav2nodal_, P{wav_scale}, f, P{0}, vals, work);
  }
  //! compute values for the field, vector overload
  void wav2nodal(sparse_grid const &grid, P const f[], std::vector<P> &vals,
                 kronmult::workspace<P> &work) const
  {
    size_t const num_entries = static_cast<size_t>(grid.num_indexes() * block_size);
    vals.resize(num_entries);
    wav2nodal(grid, f, vals.data(), work);
  }

  //! compute nodal values for the moment position coefficients
  void pos2nodal(sparse_grid const &grid, connection_patterns const &conns,
                 P const f[], P scal, P vals[], kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 1;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn_reduced, perm, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("position-to-nodal", flops);
    #else
    tools::time_event performance_("position-to-nodal");
    #endif
    ignore(conns);
    block_cpu(pdof, grid, conn_reduced, perm_pos, wav2nodal_, scal, f, P{0}, vals, work);
  }
  //! compute values for the moment position coefficients, vector overload
  void pos2nodal(sparse_grid const &grid, connection_patterns const &conn,
                 P const f[], P scal, std::vector<P> &vals,
                 kronmult::workspace<P> &work) const
  {
    size_t num_entries = static_cast<size_t>(grid.num_indexes() * fm::ipow(pdof, grid.num_dims()));
    vals.resize(num_entries);
    pos2nodal(grid, conn, f, scal, vals.data(), work);
  }

  //! converts interpolated nodal values to hierarchical coefficients
  void nodal2hier(sparse_grid const &grid, connection_patterns const &conn,
                  P const f[], P hier[], kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 2;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn, perm_low, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-hier", flops);
    #else
    tools::time_event performance_("nodal-to-hier");
    #endif
    block_cpu(pdof, grid, conn, perm_low, nodal2hier_, P{1}, f, P{0}, hier, work);
  }

  //! compute nodal values for the field
  void nodal2wav(sparse_grid const &grid, connection_patterns const &conn,
                 P alpha, P const f[], P beta, P vals[],
                 kronmult::workspace<P> &work, std::vector<P> &t1) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 3;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = 2 * kronmult::block_cpu(pdof, grid, conn, perm_up, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-wavelet", flops);
    #else
    tools::time_event performance_("nodal-to-wavelet");
    #endif
    block_cpu(pdof, grid, conn, perm_low, nodal2hier_,
              P{1}, f, P{0}, t1.data(), work);
    block_cpu(pdof, grid, conn, perm_up, hier2wav_,
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
   * Depending on the plan options:
   * 1. instead of computing the values of the state at the nodes,
   *    the pre-computed ifield vector will be used
   * 2. the func() will be called with the given moment set
   * 3. the inversion may stop at the intermediate hierarchical interpolation
   *    surpluses, when the projection to the wavelet basis is merged with
   *    the follow-on separable term in a chain
   *
   * The workspace is needed to call kronmult, the t1 and t2 are additional
   * workspace with size equal to the state.
   * The names t1/t2 come because this uses term_manager scratch space for working with chains
   */
  void operator ()
      (interpolation_plan const &plan, sparse_grid const &grid,
       connection_patterns const &conn, momentset<P> const &moments,
       P time, P const state[], std::vector<P> const &ifield,
       P alpha, term_md<P> const &tmd, P beta, P y[],
       kronmult::workspace<P> &work, std::vector<P> &t1, std::vector<P> &t2) const
  {
    expect(plan.is_enabled());
    std::vector<P> const &nodal = [&]() -> std::vector<P> const &
      {
        if (plan.uses_field()) {
          wav2nodal(grid, state, t1.data(), work);
          return t1;
        } else {
          return ifield;
        }
      }();
    {
      tools::time_event perf_("interpolation function");
      if (plan.uses_moments()) {
        tmd.interp(time, nodes(grid), moments, nodal, t2);
      } else {
        tmd.interp(time, nodes(grid), nodal, t2);
      }
    }
    if (plan.uses_hier())
      nodal2hier(grid, conn, t2.data(), y, work);
    else
      nodal2wav(grid, conn, alpha, t2.data(), beta, y, work, t1);
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

  //! returns the diagonal form of the hier2wav matrix
  block_diag_matrix<P> const &get_raw_hier2wav() const { return diag_h2w; }
  //! returns the final form of the hier2wav matrix
  block_sparse_matrix<P> const &get_hier2wav() const { return hier2wav_; }

  //! multiplies the diagonal matrix by diagonal hier2wav and transforms to hierarchical form
  block_sparse_matrix<P> mult_transform_h2w(hierarchy_manipulator<P> const &hier,
                                            connection_patterns const &conns,
                                            block_diag_matrix<P> const &mat,
                                            block_diag_matrix<P> &work) const;
  //! multiplies the tri-diagonal matrix by diagonal hier2wav and transforms to hierarchical form
  block_sparse_matrix<P> mult_transform_h2w(hierarchy_manipulator<P> const &hier,
                                            connection_patterns const &conns,
                                            block_tri_matrix<P> const &mat,
                                            block_tri_matrix<P> &work) const;
  //! returns the wavelet scale factor for hier2wav
  P wav_scale_h2w() const { return iwav_scale; }


  #ifdef ASGARD_USE_GPU
  //! compute nodal values for the field
  void wav2nodal(gpu::device dev, sparse_grid const &grid, P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 0;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn_reduced, perm, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("wavelet-to-nodal-gpu", flops);
    #else
    tools::time_event performance_("wavelet-to-nodal-gpu");
    #endif
    grid.use_gpu_reduced_xy();
    block_gpu(dev, pdof, grid, conn_reduced, perm, gpu_wav2nodal_[dev.id], P{wav_scale}, f,
              P{0}, vals, work, wav2nodal_);
    grid.use_gpu_default_xy();
  }
  //! compute nodal values for the moment
  void pos2nodal(gpu::device dev, sparse_grid const &grid, P const f[], P scal, P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 1;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn_reduced, perm, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("position-to-nodal-gpu", flops);
    #else
    tools::time_event performance_("position-to-nodal-gpu");
    #endif
    grid.use_gpu_reduced_xy();
    block_gpu(dev, pdof, grid, conn_reduced, perm_pos, gpu_wav2nodal_[dev.id], scal, f,
              P{0}, vals, work, wav2nodal_);
    grid.use_gpu_default_xy();
  }
  //! compute hirarchical coefficients from nodal values
  void nodal2hier(gpu::device dev, sparse_grid const &grid,
                 connection_patterns const &conn,
                 P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 2;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn, perm, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-hier-gpu", flops);
    #else
    tools::time_event performance_("nodal-to-hier-gpu");
    #endif
    block_gpu(dev, pdof, grid, conn, perm_low, gpu_nodal2hier_[dev.id],
              P{1}, f, P{0}, vals, work, nodal2hier_);
  }
  //! compute wavelet coefficients from nodal values
  void nodal2wav(gpu::device dev, sparse_grid const &grid,
                 connection_patterns const &conn,
                 P alpha, P const f[], P beta, P vals[],
                 kronmult::workspace<P> &work, gpu::vector<P> &t1) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 3;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = 2 * kronmult::block_cpu(pdof, grid, conn, perm, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-wavelet-gpu", flops);
    #else
    tools::time_event performance_("nodal-to-wavelet-gpu");
    #endif
    block_gpu(dev, pdof, grid, conn, perm_low, gpu_nodal2hier_[dev.id],
              P{1}, f, P{0}, t1.data(), work, nodal2hier_);
    block_gpu(dev, pdof, grid, conn, perm_up, gpu_hier2wav_[dev.id],
              alpha * P{iwav_scale}, t1.data(), beta, vals, work, hier2wav_);
  }
  /*!
   * \brief Performs the interpolation of the function func
   */
  void operator ()
      (gpu::device dev, interpolation_plan const &plan, sparse_grid const &grid,
       connection_patterns const &conn, momentset<P> const &moments,
       P time, P const state[], std::vector<P> const &ifield,
       P alpha, term_md<P> const &tmd, P beta, P y[],
       kronmult::workspace<P> &work, std::vector<P> &t1, std::vector<P> &t2,
       gpu::vector<P> &gpu_t1, gpu::vector<P> &gpu_t2) const
  {
    expect(plan.is_enabled());
    std::vector<P> const &nodal = [&]() -> std::vector<P> const &
      {
        if (plan.uses_field()) {
          wav2nodal(dev, grid, state, gpu_t1.data(), work);
          gpu_t1.copy_to_host(t1);
          return t1;
        } else {
          return ifield;
        }
      }();
    {
      tools::time_event perf_("interpolation function");
      if (plan.uses_moments()) {
        tmd.interp(time, nodes(grid), moments, nodal, t2);
      } else {
        tmd.interp(time, nodes(grid), nodal, t2);
      }
    }
    gpu_t1 = t2;
    if (plan.uses_hier())
      nodal2hier(dev, grid, conn, gpu_t1.data(), y, work);
    else
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
    {
      tools::time_event perf_("source function");
      func(time, nodes(grid), t1);
    }
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

  std::vector<double> points;
  std::vector<int> horder;

  mutable int grid_gen = -1;

  std::vector<P> trans_mats_; // transform for the hierarchical basis
  block_diag_matrix<P> diag_h2w;

  std::vector<P> nodes1d_;
  mutable vector2d<P> nodes_;

  kronmult::permutes perm;
  kronmult::permutes perm_low; // only lower matrices
  kronmult::permutes perm_up; // only upper matrices
  kronmult::permutes perm_pos; // position only permutations

  block_sparse_matrix<P> wav2nodal_;
  block_sparse_matrix<P> nodal2hier_;
  block_sparse_matrix<P> hier2wav_;

  connection_patterns conn_reduced;

  #ifdef ASGARD_USE_GPU
  #ifdef ASGARD_GPU_MEMGREEDY
  //! the type of the matrix, either a single matrix or pointers to levels
  using mat_type = gpu::vector<P>;
  #else
  using mat_type = gpu::vector<P*>;
  //! gpu coefficient matrices for different levels wavelet to nodal
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lwav2nodal_;
  //! gpu coefficient matrices for different levels nodal to hierarchical
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lnodal2hier_;
  //! gpu coefficient matrices for different levels hierarchical to wavelet
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lhier2wav_;
  #endif
  //! gpu matrices for each device
  std::array<mat_type, max_num_gpus> gpu_wav2nodal_;
  //! gpu matrices for each device
  std::array<mat_type, max_num_gpus> gpu_nodal2hier_;
  //! gpu matrices for each device
  std::array<mat_type, max_num_gpus> gpu_hier2wav_;
  #endif

  #ifdef ASGARD_USE_FLOPCOUNTER
  struct flop_info_entry {
    int grid_gen = -1;
    int64_t flops = 0;
  };
  // indexes are wav2nodal (0), wav2nodal position-only (1)
  //             nodal2hier (2), nodal2wav (3)
  mutable std::array<flop_info_entry, 4> flop_info;
  #endif
};

} // namespace asgard
