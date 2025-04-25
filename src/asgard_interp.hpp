#pragma once

#include "asgard_transformations.hpp"

namespace asgard
{
/*!
 * \brief Defines an interpolation basis for the given degree
 */
template<typename P, int degree>
class interp_basis {
public:
  interp_basis(vector2d<P> const &nodes);

  //! number of basis functions per level
  static constexpr int n = degree + 1;
  //! number of level 0 nodes in the left half-cell
  static constexpr int nL = n / 2 + n % 2;
  //! number of level 1 nodes in the left half-cell
  static constexpr int nR = n - nL;

  void eval0(std::array<P, n> const &x, P vals[]) const {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        P v = 1;
        for (int k = 0; k < j; k++)
          v *= (x[i] - x0[k]);
        for (int k = j + 1; k < n; k++)
          v *= (x[i] - x0[k]);
        vals[j * n + i] = -v * w0[j];
      }
    }
  }
  void eval1(std::array<P, n> const &x, P vals[]) const {
    for (int i = 0; i < n; i++) {
      if (x[i] < 0 or x[i] > 1) {
        for (int j = 0; j < n; j++)
          vals[j * n + i] = 0;
      } else if (x[i] < 0.5) {
        for (int j = 0; j < nL; j++) {
          P v = 1;
          for (int k = 0; k < j; k++)
            v *= (x[i] - xL[k]);
          for (int k = j + 1; k < n; k++)
            v *= (x[i] - xL[k]);
          vals[j * n + i] = -v * wL[j];
        }
        for (int j = nL; j < n; j++)
          vals[j * n + i] = 0;
      } else {
        for (int j = 0; j < nL; j++)
          vals[j * n + i] = 0;
        for (int j = nL; j < n; j++) {
          P v = 1;
          for (int k = 0; k < j; k++)
            v *= (x[i] - xR[k]);
          for (int k = j + 1; k < n; k++)
            v *= (x[i] - xR[k]);
          vals[j * n + i] = -v * wR[j - nL];
        }
      }
    }
  }

private:
  // nodes and Lagrane weights at level 0
  std::array<P, n> x0, w0;
  // nodes and additional zeros at level 1, left-right
  std::array<P, n> xL, xR;
  // Lagrange weights at level 1, left
  std::array<P, nL> wL;
  // Lagrange weights at level 1, right
  std::array<P, nR> wR;
};

template<typename P, int degree>
class interpolation_manager1d {
public:
  //! create an empty interpolation manager
  interpolation_manager1d() = default;
  //! initialize the manager using the connection pattern
  interpolation_manager1d(connect_1d const &conn) {
    static_assert(0 <= degree and degree <= 3);
    initialize_nodes(std::max(1, conn.max_loaded_level()));

    vector2d<P> const w0 = basis::legendre_poly<P>(degree);
    basis::canonical_integrator quad(degree);
    vector2d<P> const w1 = basis::wavelet_poly<P>(w0, quad);

    make_wav2nodal(w0, w1, conn);

    interp_basis<P, degree> basis(nodes_);
    make_nodal2hier(conn, basis);
  }
  //! converts to true if the manager has been initialized
  operator bool () const { return (nodes_.num_strips() > 0); }

  //! return the canonical nodes, nodes()[i][j] where i is the cell id
  vector2d<P> const &nodes() const { return nodes_; }
  //! return the wavelet to nodal matrix
  block_sparse_matrix<P> const &wav2nodal() const { return wav2nodal_; }
  //! return the nodal to hierarchical matrix
  block_sparse_matrix<P> const &nodal2hier() const { return nodal2hier_; }

protected:
  //! pre-computed constant, std::sqrt(2.0)
  static P constexpr s2 = 1.41421356237309505; // sqrt(2.0)
  //! number of polynomial degrees of freedom
  static constexpr int n = degree + 1;

  void initialize_nodes(int const max_level);
  void make_wav2nodal(vector2d<P> const &w0, vector2d<P> const &w1,
                      connect_1d const &conn);
  void make_nodal2hier(connect_1d const &conn, interp_basis<P, degree> const &basis);

private:
  vector2d<P> nodes_;
  block_sparse_matrix<P> wav2nodal_;
  block_sparse_matrix<P> nodal2hier_;
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

  //! compute hierarchical representation from the nodal values
  void nodal2hier(sparse_grid const &grid, connection_patterns const &conn,
                  P vals[], kronmult::block_global_workspace<P> &workspace)
  {
    globalsv_cpu(num_dims, n, grid, conn[connect_1d::hierarchy::volume],
                 nodal2hier1d(), vals, workspace);
  }
  //! compute hierarchical representation from the nodal values
  void nodal2hier(sparse_grid const &grid, connection_patterns const &conn,
                  std::vector<P> &vals, kronmult::block_global_workspace<P> &workspace)
  {
    expect(static_cast<int64_t>(vals.size()) == block_size * grid.num_indexes());
    nodal2hier(grid, conn, vals.data(), workspace);
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
  //! return the 1d nodal2hier matrix
  block_sparse_matrix<P> const &nodal2hier1d() const {
    switch(interp.index()) {
      case 0: return std::get<0>(interp).nodal2hier();
      case 1: return std::get<1>(interp).nodal2hier();
      case 2: return std::get<2>(interp).nodal2hier();
      default: // case 3
        return std::get<3>(interp).nodal2hier();
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
