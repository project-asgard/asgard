#pragma once

#include "asgard_transformations.hpp"

namespace asgard
{

/*!
 * \brief Describes the stages of the interpolation operation.
 *
 * An interpolation operation can have multiple stages and multiple shortcuts can be taken
 * to avoid repeated or redundant work.
 * These are expressed with multiple flags, which are stored here in a single int
 * using bit-wise operations.
 *
 * Interpolation is applied to either sources or terms but two have differences.
 * Both separable and interpolatory terms are stored together in a single data-structure
 * for purposes of chaining and several of the plan flags depend on the previous and following
 * terms of the chain, i.e., the plan cannot be figured out by just looking at a single term.
 * In contrast, sources are stored separately and interpolatory sources do not use
 * the majority of the flags in the plan.
 * All relevant information for the source can be determined from looking just at a single source.
 * Therefore, the plan is used only for terms and not sources.
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
  //! does the current function use GPU arrays
  void use_gpu_func(bool val = true) {
    if (val)
      plan_mode_ |= (1 << gpu_func_);
    else
      plan_mode_ &= ~(1 << gpu_func_);
  }
  //! does the interpolation project only in position directions (incomplete feature)
  void use_hybrid(bool val = true) {
    if (val)
      plan_mode_ |= (1 << hybrid_);
    else
      plan_mode_ &= ~(1 << hybrid_);
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
  //! indicates whether to use GPU arrays
  bool uses_gpu_func() const { return (plan_mode_ & (1 << gpu_func_)) != 0; }
  //! indicates whether to project only in position directions
  bool uses_hybrid() const { return (plan_mode_ & (1 << hybrid_)) != 0; }

  //! tag for whether to use the enabled
  static int constexpr enabled_ = 0;
  //! tag for whether to use the field
  static int constexpr field_ = 1;
  //! tag for whether to use moments
  static int constexpr moments_ = 2;
  //! tag for whether to stop at the hierarchy
  static int constexpr hier_ = 3;
  //! tag for whether to call a function on the GPU
  static int constexpr gpu_func_ = 4;
  //! tag for whether to use hybrid position-only interpolation
  static int constexpr hybrid_ = 5;
};

/*!
 * \brief Manages the data-structures for the non-separable operations
 */
template<typename P>
class interpolation_manager {
public:
  #ifdef ASGARD_USE_GPU
  #ifdef ASGARD_GPU_MEMGREEDY
  //! the type of the GPU matrix, either a single matrix or pointers to levels
  using mat_type = gpu::vector<P>;
  #else
  using mat_type = gpu::vector<P*>;
  #endif
  #endif

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
  vector2d<P> const &nodes(sparse_grid const &grid) const {
    if (grid.generation() == grid_gen)
      return nodes_;
    grid_gen = grid.generation();
    return nodes(grid, nodes_);
  }
  //! constructs the nodes corresponding to the grid
  template<int missing_dim = -1>
  vector2d<P> const &nodes(sparse_grid const &grid, vector2d<P> &vnodes) const;

  //! interpolation mode, which dimension gets interpolation applied to
  enum class interp_dims {
    //! using all dimensions
    full,
    //! using hybrid interpolation, pos-dimensions
    hybrid_pos
  };

  //! compute nodal values for the field
  template<interp_dims imode = interp_dims::full>
  void wav2nodal(sparse_grid const &grid, P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    kronmult::permutes const &ip = (imode == interp_dims::full) ? perm : perm_pos;
    P const scale = (imode == interp_dims::full) ? wav_scale : pos_wav_scale;

    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = flop_id<method::wav2nodal, imode>();
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn_reduced, ip, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("wavelet-to-nodal", flops);
    #else
    // tools::time_event performance_("wavelet-to-nodal");
    #endif
    block_cpu(pdof, grid, conn_reduced, ip, wav2nodal_, scale, f, P{0}, vals, work);
  }

  void wav2nodal(interp_dims imode, sparse_grid const &grid, P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    if (imode == interp_dims::full)
      wav2nodal<interp_dims::full>(grid, f, vals, work);
    else
      wav2nodal<interp_dims::hybrid_pos>(grid, f, vals, work);
  }

  //! compute nodal values for the moment position coefficients
  void pos2nodal(sparse_grid const &grid, P const f[], P vals[], kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = flop_id<method::pos2nodal, interp_dims::full>();
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn_reduced, perm_pos, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("position-to-nodal", flops);
    #else
    // tools::time_event performance_("position-to-nodal");
    #endif
    block_cpu(pdof, grid, conn_reduced, perm_pos, wav2nodal_, pos_wav_scale, f, P{0}, vals, work);
  }
  //! converts interpolated nodal values to hierarchical coefficients
  template<interp_dims imode = interp_dims::full>
  void nodal2hier(sparse_grid const &grid, connection_patterns const &conn,
                  P const f[], P hier[], kronmult::workspace<P> &work) const
  {
    kronmult::permutes const &ip = (imode == interp_dims::full) ? perm_low : perm_low_pos;

    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = flop_id<method::nodal2hier, imode>();
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn, ip, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-hier", flops);
    #else
    // tools::time_event performance_("nodal-to-hier");
    #endif
    block_cpu(pdof, grid, conn, ip, nodal2hier_, P{1}, f, P{0}, hier, work);
  }

  //! converts interpolated nodal values to hierarchical coefficients
  void nodal2hier(interp_dims imode, sparse_grid const &grid, connection_patterns const &conn,
                  P const f[], P hier[], kronmult::workspace<P> &work) const
  {
    if (imode == interp_dims::full)
      nodal2hier<interp_dims::full>(grid, conn, f, hier, work);
    else
      nodal2hier<interp_dims::hybrid_pos>(grid, conn, f, hier, work);
  }
  //! compute nodal values for the field
  template<interp_dims imode = interp_dims::full>
  void nodal2wav(sparse_grid const &grid, connection_patterns const &conn,
                 P alpha, P const f[], P beta, P vals[],
                 kronmult::workspace<P> &work, std::vector<P> &t1) const
  {
    kronmult::permutes const &ip_low = (imode == interp_dims::full) ? perm_low : perm_low_pos;
    kronmult::permutes const &ip_up  = (imode == interp_dims::full) ? perm_up : perm_up_pos;

    P const scale = (imode == interp_dims::full) ? iwav_scale : pos_iwav_scale;

    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = flop_id<method::nodal2wav, imode>();
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = 2 * kronmult::block_cpu(pdof, grid, conn, ip_up, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-wavelet", flops);
    #else
    // tools::time_event performance_("nodal-to-wavelet");
    #endif
    block_cpu(pdof, grid, conn, ip_low, nodal2hier_, P{1}, f, P{0}, t1.data(), work);
    block_cpu(pdof, grid, conn, ip_up, hier2wav_, alpha * scale, t1.data(), beta, vals, work);
  }
  //! compute nodal values for the field
  void nodal2wav(interp_dims imode, sparse_grid const &grid, connection_patterns const &conn,
                 P alpha, P const f[], P beta, P vals[],
                 kronmult::workspace<P> &work, std::vector<P> &t1) const
  {
    if (imode == interp_dims::full)
      nodal2wav<interp_dims::full>(grid, conn, alpha, f, beta, vals, work, t1);
    else
      nodal2wav<interp_dims::hybrid_pos>(grid, conn, alpha, f, beta, vals, work, t1);
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
  template<typename tmd_type>
  void operator ()
      (interpolation_plan const &plan, sparse_grid const &grid,
       connection_patterns const &conn, momentset<P> const &moments,
       P time, P const state[], P alpha, tmd_type const &tmd, P beta, P y[],
       kronmult::workspace<P> &work) const
  {
    tools::time_event perf_("interpolation term");
    assert(plan.is_enabled());

    // interp_dims const idim = (plan.uses_hybrid()) ? interp_dims::hybrid_pos : interp_dims::full;
    interp_dims constexpr idim = interp_dims::full;

    std::vector<P> const &nodal = [&]() -> std::vector<P> const &
      {
        if (plan.uses_field()) {
          // return (plan.uses_hybrid()) ? hybrid_ifield : ifield;
          return ifield;
        } else {
          wav2nodal(idim, grid, state, it1.data(), work);
          return it1;
        }
      }();
    {
      // tools::time_event perf_("interpolation func");
      if (plan.uses_moments()) {
        tmd.interp(time, nodes(grid), moments, nodal, it2);
      } else {
        tmd.interp(time, nodes(grid), nodal, it2);
      }
    }
    if (plan.uses_hier()) {
      nodal2hier(idim, grid, conn, it2.data(), y, work);
    } else {
      nodal2wav(idim, grid, conn, alpha, it2.data(), beta, y, work, it1);
    }
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
  template<typename tmd_type>
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, momentset<P> const &moments,
       P time, P alpha, tmd_type const &func, P beta, P y[], kronmult::workspace<P> &work) const
  {
    {
      tools::time_event perf_("interpolation source");
      func(time, nodes(grid), moments, grid.indexes(), it1);
    }

    interp_dims const idim = (func.is_hybrid()) ? interp_dims::hybrid_pos : interp_dims::full;

    nodal2wav(idim, grid, conn, alpha, it1.data(), beta, y, work, it2);
  }
  /*!
   * \brief Perform interpolation on a basic md_func
   *
   * This signature is used to set the initial conditions for the PDE,
   * it does not use moments or any kind of hybrid interpolation.
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn,
       P time, P alpha, md_func<P> const &func, P beta, P y[], kronmult::workspace<P> &work) const
  {
    {
      tools::time_event perf_("interpolation source");
      func(time, nodes(grid), it1);
    }
    nodal2wav(grid, conn, alpha, it1.data(), beta, y, work, it2);
  }

  //! indicates whether the manager has been initialized
  operator bool () const { return (pdof > 0); }

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
  P hybrid_wav_scale_h2w() const { return pos_iwav_scale; }


  #ifdef ASGARD_USE_GPU
  //! returns the nodes corresponding to the grid
  P const *gpu_nodes(gpu::device dev, sparse_grid const &grid) const;
  //! compute nodal values for the field
  void wav2nodal(gpu::device dev, sparse_grid const &grid, P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = flop_id<method::wav2nodal, interp_dims::full>();;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn_reduced, perm, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("wavelet-to-nodal-gpu", flops);
    #else
    // tools::time_event performance_("wavelet-to-nodal-gpu");
    #endif
    grid.use_gpu_reduced_xy();
    block_gpu(dev, pdof, grid, conn_reduced, perm, gpu_wav2nodal_[dev.id], P{wav_scale}, f,
              P{0}, vals, work, wav2nodal_);
    grid.use_gpu_default_xy();
  }
  //! compute nodal values for the moment
  void pos2nodal(gpu::device dev, sparse_grid const &grid, P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = flop_id<method::pos2nodal, interp_dims::full>();
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn_reduced, perm_pos, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("position-to-nodal-gpu", flops);
    #else
    // tools::time_event performance_("position-to-nodal-gpu");
    #endif
    grid.use_gpu_reduced_xy();
    block_gpu(dev, pdof, grid, conn_reduced, perm_pos, gpu_wav2nodal_[dev.id], pos_wav_scale, f,
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
    int constexpr id = flop_id<method::nodal2hier, interp_dims::full>();
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(pdof, grid, conn, perm, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-hier-gpu", flops);
    #else
    // tools::time_event performance_("nodal-to-hier-gpu");
    #endif
    block_gpu(dev, pdof, grid, conn, perm_low, gpu_nodal2hier_[dev.id],
              P{1}, f, P{0}, vals, work, nodal2hier_);
  }
  //! compute wavelet coefficients from nodal values
  template<interp_dims imode = interp_dims::full>
  void nodal2wav(gpu::device dev, sparse_grid const &grid,
                 connection_patterns const &conn,
                 P alpha, P const f[], P beta, P vals[],
                 kronmult::workspace<P> &work, gpu::vector<P> &t1) const
  {
    kronmult::permutes const &ip_low = (imode == interp_dims::full) ? perm_low : perm_low_pos;
    kronmult::permutes const &ip_up  = (imode == interp_dims::full) ? perm_up : perm_up_pos;

    P const scale = (imode == interp_dims::full) ? iwav_scale : pos_iwav_scale;

    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = flop_id<method::nodal2wav, imode>();
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = 2 * kronmult::block_cpu(pdof, grid, conn, ip_low, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-wavelet-gpu", flops);
    #else
    // tools::time_event performance_("nodal-to-wavelet-gpu");
    #endif
    block_gpu(dev, pdof, grid, conn, ip_low, gpu_nodal2hier_[dev.id],
              P{1}, f, P{0}, t1.data(), work, nodal2hier_);
    block_gpu(dev, pdof, grid, conn, ip_up, gpu_hier2wav_[dev.id],
              alpha * scale, t1.data(), beta, vals, work, hier2wav_);
  }

 //! compute nodal values for the field on the gpu
  void nodal2wav(gpu::device dev, interp_dims imode, sparse_grid const &grid,
                 connection_patterns const &conn,
                 P alpha, P const f[], P beta, P vals[],
                 kronmult::workspace<P> &work, gpu::vector<P> &t1) const
  {
    if (imode == interp_dims::full)
      nodal2wav<interp_dims::full>(dev, grid, conn, alpha, f, beta, vals, work, t1);
    else
      nodal2wav<interp_dims::hybrid_pos>(dev, grid, conn, alpha, f, beta, vals, work, t1);
  }

  //! compute hirarchical coefficients from hybrid nodal values
  void pos2hier(gpu::device dev, sparse_grid const &grid, connection_patterns const &conn,
                P const f[], P vals[], kronmult::workspace<P> &work) const
  {
    block_gpu(dev, pdof, grid, conn, perm_low_pos, gpu_nodal2hier_[dev.id],
              P{1}, f, P{0}, vals, work, nodal2hier_);
  }

  //! compute wavelet coefficients from hybrid nodal values
  void pos2wav(gpu::device dev, sparse_grid const &grid, connection_patterns const &conn,
               P alpha, P const f[], P beta, P vals[],
               kronmult::workspace<P> &work, gpu::vector<P> &t1) const
  {
    block_gpu(dev, pdof, grid, conn, perm_low_pos, gpu_nodal2hier_[dev.id],
              P{1}, f, P{0}, t1.data(), work, nodal2hier_);
    block_gpu(dev, pdof, grid, conn, perm_up_pos, gpu_hier2wav_[dev.id],
              alpha * P{pos_iwav_scale}, t1.data(), beta, vals, work, hier2wav_);
  }

  /*!
   * \brief Performs the interpolation of the function func
   */
  template<typename tmd_type, typename mom_type>
  void operator ()
      (gpu::device dev, interpolation_plan const &plan, sparse_grid const &grid,
       connection_patterns const &conn, mom_type const &moms, P time, P const state[],
       P alpha, tmd_type const &tmd, P beta, P y[],
       kronmult::workspace<P> &work) const
  {
    tools::time_event perf_("interpolation term-gpu");
    std::vector<P> &t1 = cpu_it1[dev()];
    std::vector<P> &t2 = cpu_it2[dev()];
    gpu::vector<P> &gpu_t1 = gpu_it1[dev()];
    gpu::vector<P> &gpu_t2 = gpu_it2[dev()];

    assert(plan.is_enabled());
    if (plan.uses_gpu_func()) {
      gpu::vector<P> const &nodal = [&]() -> gpu::vector<P> const &
        {
          if (plan.uses_field()) {
            return gpu_ifield;
          } else {
            // if (plan.uses_hybrid())
            //   pos2nodal(dev, grid, state, gpu_t1.data(), work);
            // else
              wav2nodal(dev, grid, state, gpu_t1.data(), work);
            return gpu_t1;
          }
        }();
      {
        // tools::time_event perf_("interpolation func-gpu");
        if (plan.uses_moments()) {
          tmd.interp(nodal.size(), time, gpu_nodes(dev, grid), moms.get_cached_interps(dev),
                     nodal.data(), gpu_t2.data());
        } else {
          tmd.interp(nodal.size(), time, gpu_nodes(dev, grid), nodal.data(), gpu_t2.data());
        }
      }
      if (plan.uses_hier()) {
        // if (plan.uses_hybrid())
        //   pos2hier(dev, grid, conn, gpu_t2.data(), y, work);
        // else
          nodal2hier(dev, grid, conn, gpu_t2.data(), y, work);
      } else {
        // if (plan.uses_hybrid())
        //   pos2wav(dev, grid, conn, alpha, gpu_t2.data(), beta, y, work, gpu_t1);
        // else
          nodal2wav(dev, grid, conn, alpha, gpu_t2.data(), beta, y, work, gpu_t1);
      }
    } else {
      std::vector<P> const &nodal = [&]() -> std::vector<P> const &
        {
          if (plan.uses_field()) {
            // return (plan.uses_hybrid()) ? hybrid_ifield : ifield;
            return ifield;
          } else {
            // if (plan.uses_hybrid())
            //   pos2nodal(dev, grid, state, gpu_t1.data(), work);
            // else
              wav2nodal(dev, grid, state, gpu_t1.data(), work);
            gpu_t1.copy_to_host(t1);
            return t1;
          }
        }();
      {
        // tools::time_event perf_("interpolation func");
        if (plan.uses_moments()) {
          tmd.interp(time, nodes(grid), moms.get_cached_interps(), nodal, t2);
        } else {
          tmd.interp(time, nodes(grid), nodal, t2);
        }
      }
      gpu_t1 = t2;
      if (plan.uses_hier()) {
        // if (plan.uses_hybrid())
        //   pos2hier(dev, grid, conn, gpu_t1.data(), y, work);
        // else
          nodal2hier(dev, grid, conn, gpu_t1.data(), y, work);
      } else {
        // if (plan.uses_hybrid())
        //   pos2wav(dev, grid, conn, alpha, gpu_t1.data(), beta, y, work, gpu_t2);
        // else
          nodal2wav(dev, grid, conn, alpha, gpu_t1.data(), beta, y, work, gpu_t2);
      }
    }
  }
  /*!
   * \brief Computes the interpolation function on the CPU and moves the data to the GPU
   *
   * In this context, the kronmult work is done on the GPU
   * but the function evaluation is done on the CPU side.
   */
  template<typename tmd_type>
  void operator ()
      (gpu::device dev, sparse_grid const &grid,
       connection_patterns const &conn, momentset<P> const &moments, P time,
       P alpha, tmd_type const &func, P beta, P y[],
       kronmult::workspace<P> &work) const
  {
    tools::time_event perf_("interpolation source-gpu");
    {
      // tools::time_event perf_("source func");
      func(time, nodes(grid), moments, grid.indexes(), cpu_it1[dev()]);
    }
    gpu_it1[dev()] = cpu_it1[dev()];

    interp_dims const idim = (func.is_hybrid()) ? interp_dims::hybrid_pos : interp_dims::full;

    nodal2wav(dev, idim, grid, conn, alpha, gpu_it1[dev()].data(), beta, y, work, gpu_it2[dev()]);
  }
  /*!
   * \brief Computes the interpolation function on the GPU
   *
   * In this context, all work is done on the GPU.
   */
  template<typename tmd_type>
  void operator ()
      (gpu::device dev, sparse_grid const &grid,
       connection_patterns const &conn, momentset_gpu<P> const &moments, P time,
       P alpha, tmd_type const &func, P beta, P y[],
       kronmult::workspace<P> &work) const
  {
    tools::time_event perf_("interpolation source-gpu");
    {
      // tools::time_event perf_("source func (gpu)");
      func(gpu_it1[dev()].size(), time, gpu_nodes(dev, grid), moments, grid.gpu_indexes(), gpu_it1[dev()].data());
    }

    interp_dims const idim = (func.is_hybrid()) ? interp_dims::hybrid_pos : interp_dims::full;

    nodal2wav(dev, idim, grid, conn, alpha, gpu_it1[dev()].data(), beta, y, work, gpu_it2[dev()]);
  }

  //! field value sitting on the GPU
  mutable gpu::vector<P> gpu_ifield;
  mutable std::array<std::vector<P>, max_num_gpus> cpu_it1, cpu_it2;
  mutable std::array<gpu::vector<P>, max_num_gpus> gpu_it1, gpu_it2;

  mat_type const &matrix_gpu_nodal2hier(gpu::device gpu) const { return gpu_nodal2hier_[gpu()]; }
  mat_type const &matrix_gpu_hier2wav(gpu::device gpu) const { return gpu_hier2wav_[gpu()]; }
  #endif

  //! computes approximate memory usage by the object
  size_t used_bytes() const {
    size_t t = diag_h2w.used_bytes() + nodes1d_.size() * sizeof(P)
              + nodes1d_.size() * sizeof(P)
              + (ifield.size() + it1.size() + it2.size()) * sizeof(P);
    t += wav2nodal_.used_bytes() + nodal2hier_.used_bytes() + hier2wav_.used_bytes();
    return t;
  }
  //! values for the interpolation field, allows reuse for several interp ops
  mutable std::vector<P> ifield;
  // //! hybrid interpolation field, nodal in position and wavelet in velocity
  // mutable std::vector<P> hybrid_ifield;
  //! temporary workspace vector
  mutable std::vector<P> it1;
  //! temporary workspace vector
  mutable std::vector<P> it2;
  //! provides access to the wav2nodal matrix
  block_sparse_matrix<P> const &matrix_wav2nodal() const { return wav2nodal_; }
  //! provides access to the nodal2hier matrix
  block_sparse_matrix<P> const &matrix_nodal2hier() const { return nodal2hier_; }
  //! provides access to the hier2wav matrix
  block_sparse_matrix<P> const &matrix_hier2wav() const { return hier2wav_; }

  //! provides access to the permutations
  kronmult::permutes const &pos_permute() const { return perm_pos; }
  //! provides access to the permutations
  kronmult::permutes const &pos_permute_low() const { return perm_low_pos; }
  //! provides access to the permutations
  kronmult::permutes const &pos_permute_up() const { return perm_up_pos; }

  //! reduced connection pattern
  connection_patterns const &reduced_connection() const { return conn_reduced; }

private:
  int pdof = 0;
  std::array<P, max_num_dimensions> xmin, xscale;
  P wav_scale = 0, iwav_scale = 0;
  P pos_wav_scale = 0, pos_iwav_scale = 0;

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
  kronmult::permutes perm_low_pos; // position only lower matrices
  kronmult::permutes perm_up_pos; // position only upper matrices

  block_sparse_matrix<P> wav2nodal_;
  block_sparse_matrix<P> nodal2hier_;
  block_sparse_matrix<P> hier2wav_;

  connection_patterns conn_reduced;

  #ifdef ASGARD_USE_GPU
  #ifndef ASGARD_GPU_MEMGREEDY
  //! gpu coefficient matrices for different levels wavelet to nodal
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lwav2nodal_;
  //! gpu coefficient matrices for different levels nodal to hierarchical
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lnodal2hier_;
  //! gpu coefficient matrices for different levels hierarchical to wavelet
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lhier2wav_;
  #endif
  //! grid gen for the nodes loaded on the GPU
  mutable std::array<int, max_num_gpus> gpu_nodes_grid_gen_;
  //! nodal matrices, TODO for multi-GPU
  mutable std::array<gpu::vector<P>, max_num_gpus> gpu_nodes_;
  //! gpu matrices for each device
  std::array<mat_type, max_num_gpus> gpu_wav2nodal_;
  //! gpu matrices for each device
  std::array<mat_type, max_num_gpus> gpu_nodal2hier_;
  //! gpu matrices for each device
  std::array<mat_type, max_num_gpus> gpu_hier2wav_;
  #endif

  #ifdef ASGARD_USE_FLOPCOUNTER
  static int constexpr max_flop_counted_operations = 7;

  enum class method {
    wav2nodal, // full grid, either all dimensions or pos-dimensions only
    pos2nodal, // position grid only, no dimension variant
    nodal2hier, // full grid, with dimension variants
    nodal2wav  // full grid, with dimension variants
  };

  template<method operation, interp_dims imode>
  static constexpr int flop_id() {
    if constexpr (operation == method::wav2nodal) {
      if constexpr (imode == interp_dims::full)
        return 0;
      else
        return 1;
    } else if constexpr (operation == method::pos2nodal) {
      return 2;
    } else if constexpr (operation == method::nodal2hier) {
      if constexpr (imode == interp_dims::full)
        return 3;
      else
        return 4;
    } else if constexpr (operation == method::nodal2wav) {
      if constexpr (imode == interp_dims::full)
        return 5;
      else
        return 6;
    }
    static_assert(max_flop_counted_operations == 7);
  }
  struct flop_info_entry {
    int grid_gen = -1;
    int64_t flops = 0;
  };
  mutable std::array<flop_info_entry, max_flop_counted_operations> flop_info;
  #endif
};

} // namespace asgard
