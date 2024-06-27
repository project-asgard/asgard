#pragma once

#include "asgard_interpolation1d.hpp"
#include "asgard_kronmult_matrix.hpp"

namespace asgard
{
#ifdef KRON_MODE_GLOBAL_BLOCK

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
  interpolation() : num_dimenisons_(0), block_size(0), workspace_(nullptr)
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

  //! convert projected coefficients to nodal values
  void get_nodal_values(vector2d<int> const &cells,
                        dimension_sort const &dsort,
                        precision const proj[], precision nodal[]) const
  {
    kronmult::global_cpu(num_dimenisons_, pterms, block_size, cells, dsort, perms,
                         wav1d.get_conn(), wav1d.get_eval_matrix(), proj, nodal,
                         *workspace_);
  }

  //! compute hierarchical coefficients
  void compute_hierarchical_coeffs(vector2d<int> const &cells,
                                   dimension_sort const &dsort,
                                   precision nodal[]) const
  {
    kronmult::globalsv_cpu(num_dimenisons_, pterms, cells, dsort, wav1d.get_conn(),
                           wav1d.get_interp_matrix(), nodal, *workspace_);
  }

  //! convert interpolation coefficients to projection coefficients
  void get_projection_coeffs(vector2d<int> const &cells,
                             dimension_sort const &dsort,
                             precision const interp[], precision proj[]) const
  {
    kronmult::global_cpu(num_dimenisons_, pterms, block_size, cells, dsort, perms,
                         wav1d.get_conn(), wav1d.get_integ_matrix(), interp, proj,
                         *workspace_);
  }

  //! overload accepting a vector
  void compute_hierarchical_coeffs(vector2d<int> const &cells,
                                   dimension_sort const &dsort,
                                   std::vector<precision> &nodal) const
  {
    expect(static_cast<int64_t>(nodal.size()) == cells.num_strips() * block_size);
    compute_hierarchical_coeffs(cells, dsort, nodal.data());
  }

  //! overload getting data from the global matrix
  void get_nodal_values(block_global_kron_matrix<precision> const &mat,
                        precision const proj[], precision nodal[]) const
  {
    get_nodal_values(mat.get_cells(), mat.get_dsort(), proj, nodal);
  }
  //! overload getting data from the global matrix
  void compute_hierarchical_coeffs(block_global_kron_matrix<precision> const &mat,
                                   precision nodal[]) const
  {
    compute_hierarchical_coeffs(mat.get_cells(), mat.get_dsort(), nodal);
  }
  //! overload getting data from the global matrix
  void get_projection_coeffs(block_global_kron_matrix<precision> const &mat,
                             precision const interp[], precision proj[]) const
  {
    get_nodal_values(mat.get_cells(), mat.get_dsort(), interp, proj);
  }

private:
  int num_dimenisons_;
  static constexpr int pterms = 2; // remove static later
  int64_t block_size;
  kronmult::permutes perms;

  wavelet_interp1d<1, precision> wav1d;

  mutable kronmult::block_global_workspace<precision> *workspace_;
};

#endif

} // namespace asgard
