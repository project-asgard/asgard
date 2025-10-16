#pragma once
#include "asgard_pde.hpp"
#include "asgard_pde_functions.hpp"
#include "asgard_wavelet_basis.hpp"
#include "asgard_transformations.hpp"

namespace asgard
{
/*!
 * \brief Holds information about the moments
 *
 * Initializes with a given number of moments over a specified domain,
 * this class can compute the moments and represent them via the non-hierarchical
 * Legendre basis functions.
 * The moments can then be used to construct operators.
 *
 * The assumption here is that the first dimension corresponds to "position"
 * while the remaining dimensions holds the "velocity".
 */
template<typename P>
class moments1d {
public:
  //! empty constructor, no moments
  moments1d() {}
  //! constructor, prepares the given number of moments, for degree and up to the max_level
  moments1d(int num_mom, int degree, int max_level, pde_domain<P> const &domain);

  /*!
   * \brief Given the grid and solution state, compute the moments
   *
   * If no position dimension is present, the moments will collapse to a single value.
   * Otherwise, the moments will be populated.
   */
  void project_moments(sparse_grid const &grid, std::vector<P> const &state,
                       std::vector<P> &moments) const;

  /*!
   * \brief Given the solution state and table, compute only one moment
   *
   * Simpler version of project_moments() that avoids recomputing everything.
   * Works up to moments with second power.
   */
  void project_moment(int const mom, sparse_grid const &grid, std::vector<P> const &state,
                      std::vector<P> &moment) const;

  /*!
   * \brief Given the current grid, just resize the moment vector
   */
  void resize_moments(sparse_grid const &grid, std::vector<P> &moments) const {
    int const mom_outs = 1 + (num_dims_ - 1) * (num_mom_ - 1);
    int const nout = fm::ipow2(grid.current_level(0));
    moments.resize(nout * mom_outs * (degree_ + 1));
  }

  //! \brief Returns the number of loaded moments, based on the power of v
  int num_mom() const { return num_mom_; }

  //! \brief Returns the number of loaded moments, based on the dimension and power
  int num_comp_mom() const { return 1 + (num_mom_ - 1) * (num_dims_ - 1); }

protected:
  /*!
   * \brief Computes the moment integrals over a sub-range of the domain
   *
   * The canonical interval (-1, 1) corresponds to the physical interval (a, b).
   * The output is the integral of the basis functions (b0 ... b_degree)
   * in blocks for each moment.
   *
   * No side-effects here, only reading from num_mom_ and degree_, thread-safe.
   *
   * The input work vector should be equal to 4 * quad.left_nodes().size()
   */
  void integrate(basis::canonical_integrator const &quad, P a, P b, scalar_func<P> const &dv,
                 vector2d<P> const &basis, std::vector<P> &work, span2d<P> integ) const;

  //! compute the projection of a 1d cell
  template<int ndims>
  void project_cell(P const x[], int const idx[], span2d<P> moments, std::vector<P> &work) const;

  //! compute the projection of a 1d cell
  template<int ndims>
  void project_cell(int const mom, P const x[], int const idx[], P moment[],
                    std::vector<P> &work) const;

private:
  //! number of moments
  int num_mom_ = 0;
  //! number of dimensions
  int num_dims_ = 0;
  //! number of position dimensions
  int num_pos_ = 0;
  //! the degree of the basis
  int degree_ = 0;
  //! integral of the canonical basis, each index holds num_mom_ * (degree_ + 1) entries
  std::array<vector2d<P>, max_num_dimensions> integ;
};

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
                 std::vector<moments_list> &&mom_groups = std::vector<moments_list>{});
  //! create the manager with the new groups and potentially lower degree
  moment_manager(pde_domain<P> const &domain, int max_level,
                 hierarchy_manipulator<P> const &hier,
                 moments_list &&mlist_in,
                 std::vector<moments_list> &&mom_groups = std::vector<moments_list>{});

  /*!
   * \brief set mass term in the given dimension
   *
   * This is useful to reuse the data for the points computed from
   * the construction of the mass term.
   * However, if coeff is empty, it will be resized and filled with 1 for the values.
   */
  void set_mass(int dim, P xleft, P xright, int max_level,
                hierarchy_manipulator<P> const &hier, rhs_raw_data<P> &coeff);

  //! returns the loaded dimensions
  int num_dims() const { return num_dims_; }
  //! returns the total number of moments
  int num_moments() const { return mlist.size(); }
  //! returns true if the manager has been initialized
  operator bool () const { return (num_dims_ > 0); }

  //! return the specified moment
  moment const &get_by_id(moment_id id) const { return mlist[id]; }
  //! returns the ID of an existing moment
  moment_id find_id(moment const &m) const { return mlist.get_id(m); }

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
  //! returns the  moment vector after expanding to full level and reconstructing
  std::vector<P> const &get_cached_level(moment_id id, hierarchy_manipulator<P> const &hier) const {
    expect(pos_grid.num_dims() == 1); // this must be changed for higher dims
    if (full_level[id].empty()) {
      std::vector<P> const &raw = raw_vals[id];
      std::vector<P> &vals = full_level.get(id);
      vals.resize(pdof * fm::ipow2(pos_grid.level_[0]));
      for (int i = 0; i < pos_grid.num_indexes(); i++)
        vals[pos_grid[i][0]] = raw[i];
    }
    return full_level[id];
  }
  //! returns the  moment vector, assumes it has already been reconstructed
  std::vector<P> const &get_cached_level(moment_id id) const {
    expect(pos_grid.num_dims() == 1); // this must be changed for higher dims
    expect(not full_level[id].empty());
    return full_level[id];
  }
  //! returns the Poisson solution on the position grid, 1D position uses poisson_level() only
  std::vector<P> &poisson_raw() const { return poisson_raw_; }
  //! returns the Poisson solution on the full 1D level (position 1D case)
  std::vector<P> &poisson_level() const { return poisson_level_; }
  //! returns the Poisson solution expanded to the interpolation nodes
  std::vector<P> &poisson_interp() const { return poisson_interp_; }
protected:
  //! set the new groups
  moment_manager(moments_list &&mlist_in,
                 std::vector<moments_list> &&mom_groups = std::vector<moments_list>{});

  //! set a dimension where only level 0 will contain moment data
  void set_level_zero(pde_domain<P> const &domain, moment const &max_moms, int dim);
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
