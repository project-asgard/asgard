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
    static_assert(0 <= degree and degree <= 3);
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
  block_sparse_matrix<P> const &wav2nodal() const { return wav2nodal_; }

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

/*!
 * \brief Handles the data-structures for interpolation
 */
template<typename P>
class interpolation_manager {
public:
  //! default constructor, no interpolation
  interpolation_manager() = default;
  //! initialize new interpolation manager over the domain
  interpolation_manager(pde_domain<P> const &domain, connect_1d const &conn,
                        int degree)
      : num_dims(domain.num_dims()), n(degree + 1), perm(num_dims)
  {
    block_size = 1;
    for (int d : iindexof(num_dims)) {
      block_size *= (degree + 1);
      xmin[d]   = domain.xleft(d);
      xscale[d] = (domain.xright(d) - domain.xleft(d));
    }

    switch (degree) {
      case 0:
        interp = interpolation_manager1d<P, 0>(conn);
        break;
      case 1:
        interp = interpolation_manager1d<P, 1>(conn);
        break;
      case 2:
        interp = interpolation_manager1d<P, 2>(conn);
        break;
      case 3:
        interp = interpolation_manager1d<P, 3>(conn);
        break;
      default:
        throw std::runtime_error("invalid degree used for interpolaton_manager");
    };
  }
  //! returns the stored degree
  int degree() const { return interp.index(); }
  //! returns the nodes corresponding to the grid
  vector2d<P> const &nodes(sparse_grid const &grid) const;

  //! compute nodal values for the field
  void wav2nodal(sparse_grid const &grid, connection_patterns const &conn,
                 P const f[], P vals[],
                 kronmult::block_global_workspace<P> &workspace)
  {
    block_cpu(n, grid, conn, perm, wav2nodal1d(), P{1}, f, P{0}, vals, workspace);
  }
  //! compute nodal values for the field
  void wav2nodal(sparse_grid const &grid, connection_patterns const &conn,
                 std::vector<P> const &f, std::vector<P> &vals,
                 kronmult::block_global_workspace<P> &workspace)
  {
    expect(static_cast<int64_t>(f.size()) == block_size * grid.num_indexes());
    vals.resize(f.size());
    wav2nodal(grid, conn, f.data(), vals.data(), workspace);
  }

private:
  //! returns the 1d nodes
  vector2d<P> const &nodes1d() const {
    switch(interp.index()) {
      case 0: return std::get<0>(interp).nodes();
      case 1: return std::get<1>(interp).nodes();
      case 2: return std::get<2>(interp).nodes();
      default: // case 3
        return std::get<3>(interp).nodes();
    }
  }
  //! return the 1d wav2noal matrix
  block_sparse_matrix<P> const &wav2nodal1d() const {
    switch(interp.index()) {
      case 0: return std::get<0>(interp).wav2nodal();
      case 1: return std::get<1>(interp).wav2nodal();
      case 2: return std::get<2>(interp).wav2nodal();
      default: // case 3
        return std::get<3>(interp).wav2nodal();
    }
  }

private:
  int num_dims = 0;
  int n = 0;
  std::array<P, max_num_dimensions> xmin, xscale;

  std::variant<
    interpolation_manager1d<P, 0>,
    interpolation_manager1d<P, 1>,
    interpolation_manager1d<P, 2>,
    interpolation_manager1d<P, 3>
    > interp;

  int grid_gen = -1;
  int block_size = 0;
  mutable vector2d<P> nodes_;

  kronmult::permutes perm;
};

} // namespace asgard
