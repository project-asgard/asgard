#pragma once

#include "asgard_transformations.hpp"

namespace asgard
{

template<typename P, int degree>
class interpolation_manager1d {
public:
  //! create an empty interpolation manager
  interpolation_manager1d() = default;
  //! initialize the manager using the connection pattern
  interpolation_manager1d(connect_1d const &conn) {
    static_assert(1 <= degree and degree <= 3);
    initialize_nodes(conn.max_loaded_level());

    vector2d<P> const w0 = basis::legendre_poly<P>(degree);
    basis::canonical_integrator quad(degree);
    vector2d<P> const w1 = basis::wavelet_poly<P>(w0, quad);

    make_wav2nodal(w0, w1, conn);

  }
  //! converts to true if the manager has been initialized
  operator bool () const { return (nodes_.num_strips() > 0); }

  //! return the canonical nodes, nodes()[i][j] where i is the cell id
  vector2d<P> const &nodes() const { return nodes_; }
  //! return the wavelet to nodal matrix
  block_sparse_matrix<P> const& wav2nodal() const { return wav2nodal_; }

protected:
  //! pre-computed constant, std::sqrt(2.0)
  static P constexpr s2 = 1.41421356237309505; // sqrt(2.0)
  //! number of polynomial degrees of freedom
  static constexpr int n = degree + 1;

  void initialize_nodes(int const max_level);
  void make_wav2nodal(vector2d<P> const &w0, vector2d<P> const &w1,
                      connect_1d const &conn);

private:
  vector2d<P> nodes_;
  block_sparse_matrix<P> wav2nodal_;
};

} // namespace asgard
