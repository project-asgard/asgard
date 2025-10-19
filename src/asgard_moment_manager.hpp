#pragma once

#include "asgard_interp.hpp"

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
  //! creates a default empty maanger, no moments
  moment_manager() = default;
  //! create the manager with the new groups
  moment_manager(pde_domain<P> const &domain, int degree,
                 moments_list &&mlist_in,
                 std::vector<moments_list> const &mom_groups = std::vector<moments_list>{});
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

  //! returns the loaded dimensions
  int num_dims() const { return num_dims_; }
  //! returns the number of velocity dimensions
  int num_vel() const { return num_vel_; }
  //! returns the total number of moments
  int num_moments() const { return mlist.size(); }
  //! returns true if the manager has been initialized
  operator bool () const { return (not mlist.empty()); }

  //! return the specified moment
  moment const &get_by_id(moment_id id) const { return mlist[id]; }
  //! returns the ID of an existing moment
  moment_id find_id(moment const &m) const { return mlist.get_id(m); }
  //! update the action for the given moment
  void set_action(moment_id id, moment::moment_type action) {
    mlist.set_action(id, action);
  }

  //! returns a grid defined over the position dimensions ready for kronmult
  sparse_grid const &get_kronmult_grid() const {
    if (dsort_generation != pos_grid.generation_) {
      pos_grid.dsort_ = dimension_sort(pos_grid.iset_);
      dsort_generation = pos_grid.generation_;
    }
    return pos_grid;
  }
  //! returns a grid indexes, used for I/O
  std::vector<int> const &get_grid_indexes() const { return pos_grid.iset_.indexes_; }
  //! computes the specified moment
  void compute(sparse_grid const &grid, moment_id id,
               std::vector<P> const &state, std::vector<P> &vals) const;

  //! load all moments into the data-structures
  void cache_moments(sparse_grid const &grid, std::vector<P> const &state, int group = -1) const;
  //! computes and caches a specific moment
  void cache_moment(moment_id id, sparse_grid const &grid, std::vector<P> const &state);
  //! get the cached moment
  std::vector<P> const &get_cached(moment_id id) const {
    return raw_vals[id];
  }
  //! returns the  moment vector after expanding to full level and reconstructing
  std::vector<P> const &get_cached_level(moment_id id, hierarchy_manipulator<P> const &hier) const {
    expect(pos_grid.num_dims() == 1); // levels work only for position 1d
    if (full_level[id].empty())
      complete_level(hier, raw_vals[id], full_level.get(id));
    return full_level[id];
  }
  //! returns the  moment vector, assumes it has already been reconstructed
  std::vector<P> const &get_cached_level(moment_id id) const {
    expect(pos_grid.num_dims() == 1); // levels work only for position 1d
    expect(not full_level[id].empty());
    return full_level[id];
  }
  //! returns the Poisson solution on the position grid, 1D position uses poisson_level() only
  std::vector<P> const &poisson_raw() const { return poisson_raw_; }
  //! returns the Poisson solution on the full 1D level (position 1D case)
  std::vector<P> const &poisson_level() const { return poisson_level_; }
  //! returns the Poisson solution expanded to the interpolation nodes
  std::vector<P> const &poisson_interp() const { return poisson_interp_; }

  //! returns the Poisson solution on the position grid, 1D position uses poisson_level() only
  std::vector<P> &edit_poisson_raw() const { return poisson_raw_; }
  //! returns the Poisson solution on the full 1D level (position 1D case)
  std::vector<P> &edit_poisson_level() const { return poisson_level_; }
  //! returns the Poisson solution expanded to the interpolation nodes
  std::vector<P> &edit_poisson_interp() const { return poisson_interp_; }

  //! fill the vector to a full 1d level, only for position 1d
  void complete_level(hierarchy_manipulator<P> const &hier, std::vector<P> const &raw,
                      std::vector<P> &vals) const;
  //! cache a number of ids listed as the first n entries of a container ids, were ids[i] is moment_id
  template<typename vec_type>
  void cache_levels(int num, hierarchy_manipulator<P> const &hier, vec_type const &ids) const {
    expect(num <= static_cast<int>(ids.size()));
    expect(pos_grid.num_dims() == 1); // levels work only for position 1d
    static_assert(std::is_same_v<decltype(ids[0]), moment_id const> or std::is_same_v<decltype(ids[0]), moment_id const &>);
    for (int i = 0; i < num; i++) {
      if (full_level[ids[i]].empty())
        complete_level(hier, raw_vals[ids[i]], full_level.get(ids[i]));
    }
  }
  //! return the set of cached levels, all relevant moments must be cached already
  momentset<P> const &get_cached_levels() const { return full_level; }

  //! load the inteprolatory moments, all groups
  void load_interp(interpolation_manager<P> const &interp,
                   connection_patterns const &conn, kronmult::workspace<P> &work,
                   std::vector<P> &workspace) const;
  //! load the inteprolatory moments, specified group
  void load_interp(int groupid, interpolation_manager<P> const &interp,
                   connection_patterns const &conn, kronmult::workspace<P> &work,
                   std::vector<P> &workspace) const;

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
  void compute(sparse_grid const &grid, moment_id id,
               std::vector<P> const &state, std::vector<P> &vals) const;
  //! mid-step, realizes the template from above using the pdof
  template<int nvel>
  void compute(sparse_grid const &grid, moment_id id,
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
                  connection_patterns const &conn, kronmult::workspace<P> &work,
                  std::vector<P> &workspace) const;

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
  int pdof = 0;

  int pos_block = 0;
  int vel_block = 0;
  int full_block = 0;

  P wav_scale = 0;

  mutable int dsort_generation = -1; // keeps track of when dsort is set in the grid
  mutable sparse_grid pos_grid; // holds the reduced grid (could be 1 cell)
  // location of the zero-th entry of pos_grid in the global grid
  mutable std::vector<int> pntr;

  moments_list mlist;
  std::vector<std::vector<moment_id>> groups_;

  bool all_levels_zero = true;
  std::array<moment_level, max_mom_dims> dim_level;

  std::array<vector2d<P>, max_mom_dims> integ;

  mutable momentset<P> raw_vals; // computed on pos-grid
  mutable momentset<P> full_level; // operator matrices need full level moments
  mutable momentset<P> interps; // moment values for interpolation

  mutable std::vector<P> poisson_raw_; // computed on pos-grid (or full grid for 1D)
  mutable std::vector<P> poisson_level_; // Poisson extended to full level
  mutable std::vector<P> poisson_interp_; // Poisson extended to the interp nodes
};

} // namespace asgard
