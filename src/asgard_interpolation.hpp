#pragma once

#include "asgard_interpolation1d.hpp"
#include "asgard_kronmult_matrix.hpp"

namespace asgard
{
#ifdef KRON_MODE_GLOBAL

/*!
 * \brief Handles the interpolation operations in multidimensional context
 *
 * The interpolation takes the number of dimensions, a volume connectivity
 * object and kronmult workspace. Aliases are kept for the connect_1d
 * and workspace objects, those will be held in the kron_operators.
 */
template<typename precision>
class interpolation
{
public:
  //! constructs and empty interpolation class
  interpolation()
  {}
  //! convert to true if the class has been initialized
  operator bool() const { return (num_dimenisons_ > 0); }

  //! new interpolation method for the given dimensions and connect_1d
  interpolation(int num_dimenisons, connect_1d const *conn_in,
                kronmult::block_global_workspace<precision> *workspace)
      : num_dimenisons_(num_dimenisons), block_size(fm::ipow(pterms, num_dimenisons_)),
        perms(num_dimenisons_), wav1d(conn_in), workspace_(workspace)
  {}

  //! get the interpolation nodes for the cells
  vector2d<precision> get_nodes(vector2d<int> const &cells) const
  {
    int const num_incell = fm::ipow(pterms, num_dimenisons_);

    vector2d<precision> nodes(num_dimenisons_, num_incell * cells.num_strips());

    std::array<int, max_num_dimensions> offsets;

    for (int c = 0; c < cells.num_strips(); c++)
    {
      for (int d = 0; d < num_dimenisons_; d++)
        offsets[d] = pterms * cells[c][d];

      for (int i = 0; i < num_incell; i++)
      {
        int const inode = c * num_incell + i;

        int t = i;
        for (int d = num_dimenisons_ - 1; d >= 0; d--)
        {
          nodes[inode][d] = wav1d.node(offsets[d] + t % pterms);
          t /= pterms;
        }
      }
    }

    return nodes;
  }

  /*!
   * \brief convert projected coefficients to nodal values
   *
   * nodal = scale * proj2node * proj
   *
   * scale should be 1 / sqrt(size/area/volume of the domain)
   */
  void get_nodal_values(vector2d<int> const &cells, dimension_sort const &dsort,
                        precision scale,
                        precision const proj[], precision nodal[]) const
  {
    kronmult::global_cpu(num_dimenisons_, pterms, block_size, cells, dsort, perms,
                         wav1d.get_conn(), wav1d.proj2node(), scale, proj,
                         nodal, *workspace_);
  }

  /*!
   * \brief convert nodal values to hierarchical interpolation coefficients
   *
   * nodal = node2hier^{-1} * nodal (inversion/solve)
   */
  void compute_hierarchical_coeffs(vector2d<int> const &cells,
                                   dimension_sort const &dsort,
                                   precision nodal[]) const
  {
    kronmult::globalsv_cpu(num_dimenisons_, pterms, cells, dsort, wav1d.get_conn(),
                           wav1d.node2hier(), nodal, *workspace_);
  }

  /*!
   * \brief converts hierarchical interpolation coefficients to projection coefficients
   *
   * proj = scale * hier2proj * interp
   *
   * The scale should be sqrt(size/area/volume of the domain)
   */
  void get_projection_coeffs(vector2d<int> const &cells,
                             dimension_sort const &dsort,
                             precision scale,
                             precision const hier[], precision proj[]) const
  {
    kronmult::global_cpu(num_dimenisons_, pterms, block_size, cells, dsort, perms,
                         wav1d.get_conn(), wav1d.hier2proj(), scale, hier, proj,
                         *workspace_);
  }
  /*!
   * \brief converts hierarchical interpolation coefficients to projection coefficients
   *
   * proj = hier2proj * interp
   *
   * Uses scale = 1 for situations when the scaling will be applied externally,
   * e.g., in conjuction with another operation.
   */
  void get_projection_coeffs(vector2d<int> const &cells,
                             dimension_sort const &dsort,
                             precision const hier[], precision proj[]) const
  {
    kronmult::global_cpu(num_dimenisons_, pterms, block_size, cells, dsort, perms,
                         wav1d.get_conn(), wav1d.hier2proj(), hier, proj,
                         *workspace_);
  }

  //! overload getting data from the global matrix
  void get_nodal_values(block_global_kron_matrix<precision> const &mat,
                        precision scale,
                        std::vector<precision> const &proj,
                        std::vector<precision> &nodal) const
  {
    get_nodal_values(mat.get_cells(), mat.get_dsort(), scale, proj.data(),
                     nodal.data());
  }

  //! overload getting data from the global matrix
  void compute_hierarchical_coeffs(block_global_kron_matrix<precision> const &mat,
                                   std::vector<precision> &nodal) const
  {
    compute_hierarchical_coeffs(mat.get_cells(), mat.get_dsort(), nodal.data());
  }
  //! overload getting data from the global matrix
  void get_projection_coeffs(block_global_kron_matrix<precision> const &mat,
                             std::vector<precision> const &interp,
                             std::vector<precision> &proj) const
  {
    get_projection_coeffs(
        mat.get_cells(), mat.get_dsort(), interp.data(), proj.data());
  }
  //! overload getting data from the global matrix
  void get_projection_coeffs(block_global_kron_matrix<precision> const &mat,
                             precision scale,
                             std::vector<precision> const &interp,
                             std::vector<precision> &proj) const
  {
    get_projection_coeffs(
        mat.get_cells(), mat.get_dsort(), scale, interp.data(), proj.data());
  }

private:
  int num_dimenisons_         = 0;
  static constexpr int pterms = 2; // remove static later
  int64_t block_size          = 0;
  kronmult::permutes perms;

  wavelet_interp1d<1, precision> wav1d;

  mutable kronmult::block_global_workspace<precision> *workspace_ = nullptr;
};

#endif

} // namespace asgard
