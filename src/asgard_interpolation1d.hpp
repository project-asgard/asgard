#pragma once

#include "asgard_indexset.hpp"

namespace asgard
{
#ifdef KRON_MODE_GLOBAL

/*!
 * \brief Describes the 1D operations associated with wavelet interpolation
 *
 * The class generates and stores for quick access
 * the interpolation nodes on a canonical interval (0, 1).
 *
 * The interpolation matrices maps the coefficients within a cell
 * to the values of the wavelet-basis at the nodes.
 * The matrix at level 0 correspond to the Legendre polynomials,
 * at level 1 gives the wavelet polynomials (at level 1).
 * The matrices at higher levels should be scaled by sqrt(2.0) * level
 */
template<int order, typename precision>
class wavelet_interp1d
{
public:
  //! number of polynomial terms, e.g., 2 for linear basis
  static constexpr int pterms = order + 1;
  //! size of the matrix block
  static constexpr int matsize = pterms * pterms;

  //! null constructor
  wavelet_interp1d()
  {}
  //! cache interpolation points and transformation matrices
  wavelet_interp1d(connect_1d const *conn_in) : conn(conn_in)
  {
    cache_nodes();
    prepare_proj2node();
    prepare_node2hier();
    prepare_hier2proj();
  }

  //! location of each node on canonical interval (0, 1)
  precision node(int i) const { return nodes_[i]; }
  //! returns the entire nodes vector
  std::vector<precision> const &nodes() const { return nodes_; }
  //! get the wavelet values matrix
  precision const *proj2node() const
  {
    return proj2node_.data();
  }
  //! get the interpolation values matrix
  precision const *node2hier() const
  {
    return node2hier_.data();
  }
  //! get the integration matrix (mass)
  precision const *hier2proj() const
  {
    return hier2proj_.data();
  }
  //! get the connectivity patter
  connect_1d const &get_conn() const { return *conn; }

  //! number of nodes per cell
  static constexpr int num_cell_nodes() { return pterms; }
  //! the loaded max number of levels
  int max_level() const { return conn->max_loaded_level(); }

protected:
  //! pre-computed constats, std::sqrt(2.0)
  static precision constexpr s2 = 1.41421356237309505; // sqrt(2.0)
  //! pre-computed constats, std::sqrt(3.0)
  static precision constexpr s3 = 1.73205080756887729; // sqrt(3.0)

  //! creates the interpolation nodes
  void cache_nodes();
  //! creates the matrix for projection-to-nodal values
  void prepare_proj2node();
  //! creates the matrix for nodal-to-(interp)hierarchical values/coeffs
  void prepare_node2hier();
  //! creates the matrix for (interp)hierarchical-to-projection coeffs
  void prepare_hier2proj();

  // helper methods that find the values of interp. and proj. wavelet
  // at normalized nodes. There is a second layer of scaling and
  // corrections based on the level and support in the prepare functions.
  void make_wavelet_wmat0(std::array<precision, pterms> const &nodes,
                          precision mat[]);
  void make_wavelet_wmat(std::array<precision, pterms> const &nodes,
                         precision mat[]);
  void make_wavelet_imat0(std::array<precision, pterms> const &nodes,
                          precision mat[]);
  void make_wavelet_imat(std::array<precision, pterms> const &nodes,
                         precision mat[]);

private:
  connect_1d const *conn = nullptr;
  std::vector<precision> nodes_;
  std::vector<precision> proj2node_;
  std::vector<precision> node2hier_;
  std::vector<precision> hier2proj_;
};

#endif // KRON_MODE_GLOBAL
} // namespace asgard
