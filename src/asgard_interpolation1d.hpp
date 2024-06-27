#pragma once

#include "asgard_indexset.hpp"

namespace asgard
{
#ifdef KRON_MODE_GLOBAL_BLOCK

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
  wavelet_interp1d() : conn(nullptr)
  {}
  //! cache interpolation points and transformation matrices
  wavelet_interp1d(connect_1d const *conn_in) : conn(conn_in)
  {
    cache_nodes();
    prepare_wmatrix();
    prepare_imatrix();
    prepare_iematrix();
  }

  //! location of each node on canonical interval (0, 1)
  precision node(int i) const { return nodes[i]; }
  //! get the wavelet values matrix
  precision const *get_eval_matrix() const
  {
    return eval_matrix.data();
  }
  //! get the interpolation values matrix
  precision const *get_interp_matrix() const
  {
    return interp_matrix.data();
  }
  //! get the integration matrix (mass)
  precision const *get_integ_matrix() const
  {
    return ie_matrix.data();
  }
  //! get the connectivity patter
  connect_1d const &get_conn() const { return *conn; }

  //! number of nodes per cell
  static constexpr int num_cell_nodes() { return pterms; }
  //! the loaded max number of levels
  int max_level() const { return conn->max_loaded_level(); }
  //! the number of loaded nodes
  size_t num_loaded_nodes() const { return nodes.size(); }

protected:
  //! pre-computed constats, std::sqrt(2.0)
  static precision constexpr s2 = 1.41421356237309505; // sqrt(2.0)
  //! pre-computed constats, std::sqrt(3.0)
  static precision constexpr s3 = 1.73205080756887729; // sqrt(3.0)

  void cache_nodes();
  void prepare_wmatrix();
  void prepare_imatrix();
  void prepare_iematrix();

  void make_wavelet_wmat0(std::array<precision, pterms> const &nodes,
                          std::array<precision, matsize> &mat);
  //! evaluates the wavelets on a canonical nodes0/1
  void make_wavelet_wmat(std::array<precision, pterms> const &nodes,
                         std::array<precision, matsize> &mat);
  //! evaluates the wavelets on a canonical nodes0/1
  void make_wavelet_imat0(std::array<precision, pterms> const &nodes,
                          std::array<precision, matsize> &mat);
  //! evaluates the wavelets on a canonical nodes0/1
  void make_wavelet_imat(std::array<precision, pterms> const &nodes,
                         std::array<precision, matsize> &mat);

private:
  connect_1d const *conn;
  std::vector<precision> nodes;
  std::vector<precision> eval_matrix;
  std::vector<precision> interp_matrix;
  std::vector<precision> ie_matrix;
};

#endif // KRON_MODE_GLOBAL_BLOCK
} // namespace asgard
