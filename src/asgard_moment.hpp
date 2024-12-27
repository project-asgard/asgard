#pragma once
#include "asgard_basis.hpp"
#include "asgard_elements.hpp"
#include "asgard_transformations.hpp"

namespace asgard
{
#ifdef ASGARD_USE_CUDA
static constexpr resource sparse_resrc = resource::device;
#else
static constexpr resource sparse_resrc = resource::host;
#endif

/*!
 * \brief Holds information about the moments
 *
 * Initializes with a given number of moments over a specified domain,
 * this class can compute the moments and represent them via the non-hierachical
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
  //! constructor, prepares the given number of momemnts, for dgree and up to the max_level
  moments1d(int num_mom, int degree, int max_level, std::vector<dimension<P>> const &dims);

  /*!
   * \brief Given the solution state and table, compute the moments
   *
   * The dim0_level is the current level of dimenison zero and will determine
   * the size of moments, but if any indexes are not present in the etable,
   * those will be filled with zeros.
   */
  void project_moments(int const dim0_level, std::vector<P> const &state,
                       elements::table const &etable, std::vector<P> &moments) const;

  /*!
   * \brief Given the solution state and table, compute only one moment
   *
   * Simpler version of project_moments() that avoids recomputing everything.
   */
  void project_moment(int const mom, int const dim0_level, std::vector<P> const &state,
                      elements::table const &etable, std::vector<P> &moment) const;

  //! \brief Returns the number of loaded moments
  int num_mom() const { return num_mom_; }

protected:
  /*!
   * \brief Computes the moment integrals over a sub-range of the domain
   *
   * The canonical interval (-1, 1) corresponds to the physical interval (a, b).
   * The outpout is the integral of the basis functions (b0 ... b_degree)
   * in blocks for each moment.
   *
   * No side-effects here, only reading from num_mom_ and degree_, thread-safe.
   *
   * The input work vector should be equal to 4 * quad.left_nodes().size()
   */
  void integrate(basis::canonical_integrator const &quad, P a, P b, g_func_type<P> const &dv,
                 vector2d<P> const &basis, std::vector<P> &work, span2d<P> integ) const;

  //! compute the projection of a 1d cell
  template<int ndims>
  void project_cell(P const x[], int const idx[], span2d<P> moments, std::vector<P> &work) const;

  //! compute the projection of a 1d cell
  template<int ndims>
  void project_cell(int const mom, P const x[], int const idx[], P moment[],
                    std::vector<P> &work) const;

  //! construct global indexe list from the etable
  static vector2d<int> get_cells(int num_dimensions, elements::table const &etable);

private:
  //! number of momemnts
  int num_mom_ = 0;
  //! number of dimensions
  int num_dims_ = 0;
  //! the degree of the basis
  int degree_ = 0;
  //! ingeral of the canonical basis, each index holds num_mom_ * (degree_ + 1) entries
  std::array<vector2d<P>, max_num_dimensions> integ;
};


template<typename P>
class moment
{
public:
  moment(std::vector<md_func_type<P>> md_funcs_);
  void createFlist(PDE<P> const &pde);
  void createMomentVector(PDE<P> const &pde,
                          elements::table const &hash_table);

  std::vector<md_func_type<P>> const &get_md_funcs() const { return md_funcs; }
  fk::vector<P> const &get_vector() const { return vector; }
  std::vector<std::vector<fk::vector<P>>> const &get_fList() const
  {
    return fList;
  }
  fk::sparse<P, sparse_resrc> const &get_moment_matrix_dev() const
  {
    return sparse_mat;
  }

  void createMomentReducedMatrix(PDE<P> const &pde,
                                 elements::table const &hash_table);

  fk::vector<P> const &get_realspace_moment() const { return realspace; }
  void set_realspace_moment(fk::vector<P> &&realspace_in)
  {
    realspace = std::move(realspace_in);
  }

  fk::vector<P> &create_realspace_moment(
      PDE<P> const &pde_1d, fk::vector<P> &wave, elements::table const &table,
      asgard::basis::wavelet_transform<P, resource::host> const &transformer,
      std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace);

  fk::vector<P> &create_realspace_moment(
      PDE<P> const &pde_1d,
      fk::vector<P, mem_type::owner, resource::device> &wave,
      elements::table const &table,
      basis::wavelet_transform<P, resource::host> const &transformer,
      std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace);

private:
  template<int nvdim>
  void createMomentReducedMatrix_nd(PDE<P> const &pde,
                                    elements::table const &hash_table);

  std::vector<md_func_type<P>> md_funcs;
  std::vector<std::vector<fk::vector<P>>> fList;
  fk::vector<P> vector;
  fk::vector<P> realspace;
  fk::sparse<P, sparse_resrc> sparse_mat;
};

} // namespace asgard
