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

} // namespace asgard
