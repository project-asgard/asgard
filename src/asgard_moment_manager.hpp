#pragma once

#include "asgard_poisson.hpp"

namespace asgard
{
/*!
 * \brief Manages the moment computations
 *
 * Handles groups, domain, etc.
 */
template<typename P>
class moment_manager {
public:
  //! creates a default empty manager, no moments
  moment_manager() = default;
  //! create the manager with the new groups and potentially lower degree
  moment_manager(pde_domain<P> const &domain, int max_level, legendre_basis<P> const &basis,
                 hierarchy_manipulator<P> const &hier,
                 moments_list &&mlist_in,
                 std::vector<moments_list> const &mom_groups = std::vector<moments_list>{});

  /*!
   * \brief set mass term in the given dimension
   *
   * This is useful to reuse the data for the points computed from
   * the construction of the mass term.
   * However, if coeff is empty, it will be resized and filled with 1 for the values.
   */
  void set_mass(int dim, P xleft, P xright, int max_level, legendre_basis<P> const &basis,
                hierarchy_manipulator<P> const &hier, P scale, rhs_raw_data<P> &coeff);

  //! returns the number of velocity dimensions
  int num_vel() const { return num_vel_; }
  //! returns the number of position dimensions
  int num_pos() const { return num_pos_; }
  //! returns the total number of moments
  int num_moments() const { return mlist.size(); }
  //! returns true if the manager has been initialized
  operator bool () const { return (not mlist.empty()); }

  //! return the specified moment
  moment const &get_by_id(moment_id id) const { return mlist[id]; }
  //! returns the ID of an existing moment
  moment_id find_id(moment const &m) const { return mlist.get_id(m); }
  //! returns a read-only reference to the list of moments
  moments_list const &moments() const { return mlist; }
  //! returns true if there is a poisson solver
  bool has_poisson() const { return (not std::holds_alternative<std::monostate>(poisson_solver)); }

  //! returns a grid indexes, used for I/O
  std::vector<int> const &get_grid_indexes() const { return pos_grid.iset_.indexes_; }
  //! computes the specified moment
  void mcompute(sparse_grid const &grid, moment_id id,
                std::vector<P> const &state, std::vector<P> &vals) const;

  //! load all moments into the data-structures
  void cache_moments(group_id group, sparse_grid const &grid, std::vector<P> const &state,
                     connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                     interpolation_manager<P> const &interp, kronmult::workspace<P> &work) const;
  //! load all moments into the data-structures
  void cache_moments(sparse_grid const &grid, std::vector<P> const &state,
                     connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                     interpolation_manager<P> const &interp, kronmult::workspace<P> &work) const {
    cache_moments(group_id::all(), grid, state, conn, hier, interp, work);
  }
  //! computes and caches a specific moment
  void cache_moment(moment_id id, sparse_grid const &grid, std::vector<P> const &state) const;

  //! returns the  moment vector after expanding to full level and reconstructing
  std::vector<P> const &get_cached_level(moment_id id, hierarchy_manipulator<P> const &hier) const {
    assert(pos_grid.num_dims() == 1); // levels work only for position 1d
    if (full_level[id].empty())
      complete_level(hier, raw_vals[id], full_level.get(id));
    return full_level[id];
  }
  //! returns the  moment vector, assumes it has already been reconstructed
  std::vector<P> const &get_cached_level(moment_id id) const {
    assert(pos_grid.num_dims() == 1); // levels work only for position 1d
    assert(not full_level[id].empty());
    return full_level[id];
  }
  //! returns true if the moment requires a poisson solve
  bool needs_poisson(moment_id id) const {
    return mlist[id].is_electric();
  }

  //! fill the vector to a full 1d level, only for position 1d
  void complete_level(hierarchy_manipulator<P> const &hier, std::vector<P> const &raw,
                      std::vector<P> &vals) const;
  //! cache the raw moment from the level, only for position 1d and electric field moments
  void cache_raw_from_level(moment_id id, hierarchy_manipulator<P> const &hier) const {
    assert(num_pos_ == 1 and needs_poisson(id)); // this should only be called for 1D electric field moments
    int const level = pos_grid.current_level(0);
    constexpr int level_degree = 0;
    std::vector<P> padded_level = pad_degree(level_degree, pdof - 1, full_level[id]);
    hier.transform(level, padded_level, raw_vals[id]);
  }
  //! cache a number of ids listed as the first n entries of a container ids, where ids[i] is moment_id
  template<typename vec_type>
  void cache_levels(int num, hierarchy_manipulator<P> const &hier, vec_type const &ids) const {
    assert(num <= static_cast<int>(ids.size()));
    assert(pos_grid.num_dims() == 1); // levels work only for position 1d
    static_assert(std::is_same_v<decltype(ids[0]), moment_id const> or std::is_same_v<decltype(ids[0]), moment_id const &>);
    for (int i = 0; i < num; i++) {
      if (full_level[ids[i]].empty())
        complete_level(hier, raw_vals[ids[i]], full_level.get(ids[i]));
    }
  }
  //! return the set of cached raw moments defined on the position grid
  momentset<P> const &get_cached_raws() const { return raw_vals; }
  //! return the set of cached levels, all relevant moments must be cached already
  momentset<P> const &get_cached_levels() const { return full_level; }
  //! return the set of cached interpolation values, all relevant moments must be cached already
  momentset<P> const &get_cached_interps() const { return interps; }
  //! return the current position grid where the raw moments are defined
  sparse_grid const &get_position_grid() const { return pos_grid; }
  //! return the cached raw moment defined on the position grid with moment id mid
  std::vector<P> const &get_cached_raw(moment_id mid) const {
    rassert(not (num_pos_ == 1 and needs_poisson(mid)), 
      "The electric field moment is only computed for the full level in 1D, use get_cached_raw(mid, hier) to convert from the level to the raw moment instead");
    return raw_vals.get(mid);
  }
  //! return the cached raw moment defined on the position grid with moment id mid, overload for 1d electric moments
  std::vector<P> const &get_cached_raw(moment_id mid, hierarchy_manipulator<P> const &hier) const {
    if (num_pos_ == 1 and needs_poisson(mid))
      cache_raw_from_level(mid, hier);
    return raw_vals.get(mid);
  }

  //! solves the poisson equation and caches them as electric moments
  void solve_poisson(sparse_grid const &grid, connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                     interpolation_manager<P> const &interp, kronmult::workspace<P> &work) const {
    constexpr bool raw_on_cpu = false;
    solve_poisson(grid, conn, hier, interp, work, raw_on_cpu);
  }
  //! solves the poisson equation and caches them as electric moments
  void solve_poisson(sparse_grid const &grid, connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                     interpolation_manager<P> const &interp, kronmult::workspace<P> &work, bool raw_on_cpu) const;
  //! compute the specified interpolated moments
  void compute_interps(std::vector<moment_id> const &ids, sparse_grid const &grid,
                       std::vector<P> const &state, interpolation_manager<P> const &interp,
                       connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                       kronmult::workspace<P> &work) const;

  //! computes approximate memory usage by the object
  size_t used_bytes() const;

  #ifdef ASGARD_USE_GPU
  /*!
   * \brief Set the distribution of inteprolatory moments across GPU devices
   *
   * Interpolation moments require most expensive kronmult operations and can be fed into
   * user provided functions on the GPU device. But this must respect the distribution
   * of moments across GPU devices, and this applies only to interpolatory moments.
   * The format is device dev, group grp has a vector of moments mom[dev][grp].
   *
   * CPU moments can be interpolated, regular or inactive. The interpolated moments
   * will be computed on device 0 and moved back to the CPU.
   * Regular moments will be computed on the CPU.
   */
  void set_moment_distribution(std::array<std::vector<std::vector<moment_id>>, max_num_gpus> const &gpu_mom,
                               std::vector<std::vector<moment_id>> const &cpu_raw,
                               std::vector<std::vector<moment_id>> const &cpu_interp,
                               std::vector<std::vector<moment_id>> const &skip_interp);
  //! return the set of cached interpolation values, all relevant moments must be cached already
  momentset_gpu<P> const &get_cached_interps(gpu::device dev) const { return gpu_interps[dev.id]; }
  //! load all moments into the data-structures
  void compute_moments(group_id group, sparse_grid const &grid, interpolation_manager<P> const &interp,
                       connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                       kronmult::workspace<P> &kwork, gpu::vector<P> const &state) const;
  //! load all moments into the data-structures
  void compute_moments(sparse_grid const &grid, interpolation_manager<P> const &interp,
                       connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                       kronmult::workspace<P> &kwork, gpu::vector<P> const &state) const
  {
    compute_moments(group_id::all(), grid, interp, conn, hier, kwork, state);
  }
  //! load the given moments into the data-structures at device 0
  void compute_moments(std::vector<moment_id> const &mids, sparse_grid const &grid,
                       interpolation_manager<P> const &interp, connection_patterns const &conn,
                       hierarchy_manipulator<P> const &hier, kronmult::workspace<P> &kwork,
                       gpu::vector<P> const &state) const {
    constexpr bool result_to_cpu = false;
    compute_moments(mids, grid, interp, conn, hier, kwork, state, result_to_cpu);
  }
  //! load the given moments into the data-structures at device 0
  void compute_moments(std::vector<moment_id> const &mids, sparse_grid const &grid,
                       interpolation_manager<P> const &interp, connection_patterns const &conn,
                       hierarchy_manipulator<P> const &hier, kronmult::workspace<P> &kwork,
                       gpu::vector<P> const &state, bool result_to_cpu) const;
  //! solves the poisson equation and caches them as electric moments, on gpu
  void solve_poisson_gpu(connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                         interpolation_manager<P> const &interp, kronmult::workspace<P> &work) const {
    constexpr bool raw_on_cpu = false;
    solve_poisson_gpu(conn, hier, interp, work, raw_on_cpu);
  }
  //! solves the poisson equation and caches them as electric moments, on gpu
  void solve_poisson_gpu(connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                         interpolation_manager<P> const &interp, kronmult::workspace<P> &work, bool raw_on_cpu) const;
  #endif
  /*!
   * \brief Defines moments that should be used as raw or interpolation
   *
   * The raw moments are the ones computed directly from the state and those are defined on
   * the position grid. The level and interp moments are computed from the raw moments,
   * where the level moments are generated on call.
   * Thus, the raw moments include both the level moments and interp moments.
   * The interp moments are those that require interpolation.
   *
   * Using access: raws[group][moment]
   */
  void set_moment_types(std::vector<std::vector<moment_id>> const &raws,
                        std::vector<std::vector<moment_id>> const &intps);
  //! sets up the poisson solver if one is needed
  void set_poisson(int const max_level, sparse_grid const &grid,
                   std::array<P, max_num_dimensions> const &xleft,
                   std::array<P, max_num_dimensions> const &xright,
                   connection_patterns const &conn,
                   hierarchy_manipulator<P> const &hier,
                   build_term_func<P> build_func, prog_opts const &opts);

  //! Updates the position grid if needed
  void update_position_grid(sparse_grid const &grid) const;
  //! Updates the dsort_ field of the position grid
  void update_position_grid_dsort() const;

  //! Poisson solver data
  mutable std::variant<std::monostate, poisson_1d<P>, poisson_md<P>> poisson_solver = std::monostate{};

protected:
  //! set the new groups
  moment_manager(moments_list &&mlist_in,
                 std::vector<moments_list> const &mom_groups = std::vector<moments_list>{});

  //! set a dimension where only level 0 will contain moment data
  void set_level_zero(pde_domain<P> const &domain, legendre_basis<P> const &basis,
                      moment const &max_moms, int dim);
  /*!
   * \brief computes the specified moment
   *
   * This assumes that the position grid (pos_grid) has been set together with the
   * offsets of the nodes within the global grid.
   * The method templates on the number of velocity dimensions and polynomial
   * degrees of freedom (pdof) to speed up work.
   */
  template<int nvel, int tpdof>
  void mcompute(sparse_grid const &grid, moment_id id,
                std::vector<P> const &state, std::vector<P> &vals) const;
  //! mid-step, realizes the template from above using the pdof
  template<int nvel>
  void mcompute(sparse_grid const &grid, moment_id id,
                std::vector<P> const &state, std::vector<P> &vals) const;
  /*!
   * \brief computes the position grid from the given global grid
   *
   * Computes both the position indexes and the pntr array linking the position
   * multi-indexes to the corresponding zero-th index in the grid.
   */
  template<int npos>
  void reduce_grid(sparse_grid const &grid) const;

  /*!
   * \brief computes the nodal values of the moment
   */
  void make_nodal(moment_id id, interpolation_manager<P> const &interp,
                  hierarchy_manipulator<P> const &hier,
                  kronmult::workspace<P> &work, std::vector<P> &workspace) const;

  #ifdef ASGARD_USE_GPU
  //! prepare the position grid and optionally set the interpolation dsort
  void prepare_pos_grid_gpu(group_id group, sparse_grid const &grid) const;
  #endif

  //! returns an iterator to the first entry of the given group using serialized vector
  template<typename vector_like>
  static auto first_in(group_id group, vector_like const &vec) {
    int gid = 0;
    auto mid = vec.begin();
    while (group != group_id{gid}) {
      if (*mid == moment_id::unset()) gid++;
      mid++;
    }
    return mid;
  }

private:
  //! indicates whether level 0 contains all the needed moment data
  enum class moment_level {
    //! all moments are at level 0
    zero,
    //! need to consider all levels
    all,
  };

  int num_dims_ = 0;
  int num_vel_ = 0;
  int num_pos_ = 0;
  int pdof = 0;

  int vel_block = 0;
  int full_block = 0;

  P wav_scale = 0;

  mutable int dsort_generation = -1; // keeps track of when dsort is set in the grid
  mutable sparse_grid pos_grid; // holds the reduced grid (could be 1 cell)
  // location of the zero-th entry of pos_grid in the global grid
  mutable std::vector<int> pntr;

  moments_list mlist;
  std::vector<std::vector<moment_id>> groups_;

  std::vector<moment_id> raw_moments_;
  std::vector<moment_id> interp_moments_;

  bool all_levels_zero = true;
  std::array<moment_level, max_mom_dims> dim_level;

  std::array<vector2d<P>, max_mom_dims> integ;

  mutable momentset<P> raw_vals; // computed on pos-grid
  mutable momentset<P> full_level; // operator matrices need full level moments
  mutable momentset<P> interps; // moment values for interpolation

  #ifdef ASGARD_USE_GPU
  //! combines the moment_id with a flag if the raw-data is needed on the CPU
  struct mom_on_gpu {
    //! create and set the moment id
    mom_on_gpu(moment_id id = moment_id::unset()) : mid(id) {}
    //! the moment id
    moment_id mid;
    //! flag whether to keep on the cpu or gpu
    unsigned int flags = 0;
    //! indicates if the moment is unset
    operator bool () const { return (mid != moment_id::unset()); }
    //! indicates if the moment is unset
    bool is_unset() const { return (mid == moment_id::unset()); }
    //! indicates whether to use raw-value on the cpu
    bool raw_on_cpu() const { return ((flags & 1u) != 0); }
    //! indicates whether to use interp value on the cpu
    bool interp_on_cpu() const { return ((flags & 2u) != 0); }
    //! indicated whether this is regular or interp moment
    bool skip_interp() const { return  ((flags & 4u) != 0); }
    //! sets the moment as needing raw on the cpu
    void set_raw_on_cpu() { flags |= 1u; }
    //! sets the moment as needing interp on the cpu
    void set_interp_on_cpu() { flags |= 2u; }
    //! sets the moment as not needing interp on the gpu
    void set_skip_interp() { flags |= 4u; }
  };

  std::array<std::array<gpu::vector<P>, max_mom_dims>, max_num_gpus> gpu_integ;
  mutable std::array<gpu::vector<int>, max_num_gpus> reduce_ij; // pairs of ij corresponding to pos-grid to global-grid
  mutable std::array<gpu::vector<int>, max_num_gpus> reduce_ij_allzero; // special case, only using level zero
  std::vector<bool> has_interp; // if moments have to computed on the CPU too
  std::array<std::vector<mom_on_gpu>, max_num_gpus> gpu_moments; // distribution of moments across GPU devices

  mutable std::array<momentset_gpu<P>, max_num_gpus> gpu_interps; // moment values for interpolation on the GPU
  #endif
};

} // namespace asgard
