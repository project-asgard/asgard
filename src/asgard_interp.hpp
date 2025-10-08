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
  //! construct the basis, first two level of nodes are always used
  interp_basis(vector2d<P> const &nodes);

  //! number of basis functions per level
  static constexpr int n = degree + 1;
  //! number of level 0 nodes in the left half-cell
  static constexpr int nL = n / 2;
  //! number of level 1 nodes in the left half-cell
  static constexpr int nR = n - nL;

  //! computes the (negative) values of the interpolation basis at level 0
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
  //! computes the (negative) values of the interpolation basis at level 1
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
  //! get the values of the j-th interpolation function at level 0
  P ival0(int j, P x) const
  {
    P v = 1, w = 0.5 * x + 0.5;
    for (int k = 0; k < j; k++)
      v *= (w - x0[k]);
    for (int k = j + 1; k < n; k++)
      v *= (w - x0[k]);
    return v * w0[j];
  }
  //! get the values of the j-th interpolation function at level 1, left
  P ival1L(int j, P x) const
  {
    P v = 1, w = 0.5 * x + 0.5;
    for (int k = 0; k < j; k++)
      v *= (w - xL[k]);
    for (int k = j + 1; k < n; k++)
      v *= (w - xL[k]);
    return v * wL[j];
  }
  //! get the values of the j-th interpolation function at level 1, right
  P ival1R(int j, P x) const
  {
    P v = 1, w = 0.5 * x + 0.5;
    for (int k = 0; k < j; k++)
      v *= (w - xR[k]);
    for (int k = j + 1; k < n; k++)
      v *= (w - xR[k]);
    return v * wR[j - nL];
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

/*!
 * \brief Integrator of wavelets and interpolation basis functions
 *
 * This is a wrapper around the data-structures and allows computing values and integrals
 * by combining data from all components, e.g., left-right quadrature points,
 * wavelet values and values for the interpolation basis.
 * This also handles issues of the support, i.e., wavelets have support over the entire
 * domain, but are split into two components (left and right), the interpolation basis
 * has only left or only right component past level 0.
 */
template<typename P, int degree>
class interp_wavelet_integrator {
public:
  //! wrap wavelet and interpolation basis, and the integrator
  interp_wavelet_integrator(vector2d<P> const &w0_in, vector2d<P> const &w1_in,
                            interp_basis<P, degree> const &ibasis_in,
                            basis::canonical_integrator const &quad_in)
      : w0(w0_in), w1(w1_in), ibasis(ibasis_in), quad(quad_in)
  {}
  //! number of basis functions per level
  static constexpr int n = degree + 1;
  //! pre-computed constant, std::sqrt(2.0)
  static P constexpr s2 = 1.41421356237309505; // sqrt(2.0)
  //! integrate wavelets to i-basis at level 0
  void mat00(P block[]) const {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.left_nodes().size(); k++) {
          P const x = quad.left_nodes()[k];
          q += quad.left_weights()[k] * wval0(i, x) * ibasis.ival0(j, x);
        }
        for (size_t k = 0; k < quad.right_nodes().size(); k++) {
          P const x = quad.right_nodes()[k];
          q += quad.right_weights()[k] * wval0(i, x) * ibasis.ival0(j, x);
        }
        block[j * n + i] = q;
      }
    }
  }
  //! integrate wavelets level 0 to i-basis at level 1
  void mat01(P block[]) const {
    for (int j = 0; j < ibasis.nL; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.left_nodes().size(); k++) {
          P const x = quad.left_nodes()[k];
          q += quad.left_weights()[k] * wval0(i, x) * ibasis.ival1L(j, x);
        }
        block[j * n + i] = q;
      }
    }
    for (int j = ibasis.nL; j < n; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.right_nodes().size(); k++) {
          P const x = quad.right_nodes()[k];
          q += quad.right_weights()[k] * wval0(i, x) * ibasis.ival1R(j, x);
        }
        block[j * n + i] = q;
      }
    }
  }
  //! integrate wavelets level 1 to i-basis at level 0
  void mat10(P block[]) const {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.left_nodes().size(); k++) {
          P const x = quad.left_nodes()[k];
          q += quad.left_weights()[k] * wval1L(i, x) * ibasis.ival0(j, x);
        }
        for (size_t k = 0; k < quad.right_nodes().size(); k++) {
          P const x = quad.right_nodes()[k];
          q += quad.right_weights()[k] * wval1R(i, x) * ibasis.ival0(j, x);
        }
        block[j * n + i] = q;
      }
    }
  }
  //! integrate wavelets level 0 to i-basis at higher level, i-basis is on a sub-domain
  void mat01i(P const xs, P const xi, P block[]) const {
    for (int j = 0; j < ibasis.nL; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.left_nodes().size(); k++) {
          P const x = quad.left_nodes()[k];
          q += quad.left_weights()[k] * wval0(i, xs * x + xi) * ibasis.ival1L(j, x);
        }
        block[j * n + i] = q * xs;
      }
    }
    for (int j = ibasis.nL; j < n; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.right_nodes().size(); k++) {
          P const x = quad.right_nodes()[k];
          q += quad.right_weights()[k] * wval0(i, xs * x + xi) * ibasis.ival1R(j, x);
        }
        block[j * n + i] = q * xs;
      }
    }
  }
  //! integrate wavelets at higher level to i-basis at level 0, wavelet is on a sub-domain
  void mat10w(P const xs, P const xi, P const scale, P block[]) const {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.left_nodes().size(); k++) {
          P const x = quad.left_nodes()[k];
          q += quad.left_weights()[k] * wval1L(i, x) * ibasis.ival0(j, xs * x + xi);
        }
        for (size_t k = 0; k < quad.right_nodes().size(); k++) {
          P const x = quad.right_nodes()[k];
          q += quad.right_weights()[k] * wval1R(i, x) * ibasis.ival0(j, xs * x + xi);
        }
        block[j * n + i] = scale * q;
      }
    }
  }
  //! integrate wavelets i-basis at level > 0 and matching support
  void mat11(P const scale, P block[]) const {
    for (int j = 0; j < ibasis.nL; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.left_nodes().size(); k++) {
          P const x = quad.left_nodes()[k];
          q += quad.left_weights()[k] * wval1L(i, x) * ibasis.ival1L(j, x);
        }
        block[j * n + i] = scale * q;
      }
    }
    for (int j = ibasis.nL; j < n; j++) {
      for (int i = 0; i < n; i++) {
        P q = 0;
        for (size_t k = 0; k < quad.right_nodes().size(); k++) {
          P const x = quad.right_nodes()[k];
          q += quad.right_weights()[k] * wval1R(i, x) * ibasis.ival1R(j, x);
        }
        block[j * n + i] = scale * q;
      }
    }
  }
  //! integrate wavelets and i-basis at higher level, i-basis is on a sub-domain
  void mat11i(P const xs, P const xi, P const scale, P block[]) const {
    // check if the sub-domain overlaps with the left or right wavelet
    if (xs * quad.left_nodes()[0] + xi < 0) {
      for (int j = 0; j < ibasis.nL; j++) {
        for (int i = 0; i < n; i++) {
          P q = 0;
          for (size_t k = 0; k < quad.left_nodes().size(); k++) {
            P const x = quad.left_nodes()[k];
            q += quad.left_weights()[k] * wval1L(i, xs * x + xi) * ibasis.ival1L(j, x);
          }
          block[j * n + i] = scale * q * xs;
        }
      }
      for (int j = ibasis.nL; j < n; j++) {
        for (int i = 0; i < n; i++) {
          P q = 0;
          for (size_t k = 0; k < quad.right_nodes().size(); k++) {
            P const x = quad.right_nodes()[k];
            q += quad.right_weights()[k] * wval1L(i, xs * x + xi) * ibasis.ival1R(j, x);
          }
          block[j * n + i] = scale * q * xs;
        }
      }
    } else {
      for (int j = 0; j < ibasis.nL; j++) {
        for (int i = 0; i < n; i++) {
          P q = 0;
          for (size_t k = 0; k < quad.left_nodes().size(); k++) {
            P const x = quad.left_nodes()[k];
            q += quad.left_weights()[k] * wval1R(i, xs * x + xi) * ibasis.ival1L(j, x);
          }
          block[j * n + i] = scale * q * xs;
        }
      }
      for (int j = ibasis.nL; j < n; j++) {
        for (int i = 0; i < n; i++) {
          P q = 0;
          for (size_t k = 0; k < quad.right_nodes().size(); k++) {
            P const x = quad.right_nodes()[k];
            q += quad.right_weights()[k] * wval1R(i, xs * x + xi) * ibasis.ival1R(j, x);
          }
          block[j * n + i] = scale * q * xs;
        }
      }
    }
  }
  //! integrate wavelets and i-basis at higher level, wavelet is on a sub-domain
  void mat11w(P const xs, P const xi, P const scale, P block[]) const {
    if (xs * quad.left_nodes()[0] + xi < 0) { // the sub-domain is on the left, right i-basis is 0
      for (int j = 0; j < ibasis.nL; j++) {
        for (int i = 0; i < n; i++) {
          P q = 0;
          for (size_t k = 0; k < quad.left_nodes().size(); k++) {
            P const x = quad.left_nodes()[k];
            q += quad.left_weights()[k] * wval1L(i, x) * ibasis.ival1L(j, xs * x + xi);
          }
          for (size_t k = 0; k < quad.right_nodes().size(); k++) {
            P const x = quad.right_nodes()[k];
            q += quad.right_weights()[k] * wval1R(i, x) * ibasis.ival1L(j, xs * x + xi);
          }
          block[j * n + i] = scale * q * xs;
        }
      }
      for (int j = ibasis.nL; j < n; j++) {
        for (int i = 0; i < n; i++) {
          block[j * n + i] = 0;
        }
      }
    } else { // the sub-domain is on the right, left i-basis is zero
      for (int j = 0; j < ibasis.nL; j++) {
        for (int i = 0; i < n; i++) {
          block[j * n + i] = 0;
        }
      }
      for (int j = ibasis.nL; j < n; j++) {
        for (int i = 0; i < n; i++) {
          P q = 0;
          for (size_t k = 0; k < quad.left_nodes().size(); k++) {
            P const x = quad.left_nodes()[k];
            q += quad.left_weights()[k] * wval1L(i, x) * ibasis.ival1R(j, xs * x + xi);
          }
          for (size_t k = 0; k < quad.right_nodes().size(); k++) {
            P const x = quad.right_nodes()[k];
            q += quad.right_weights()[k] * wval1R(i, x) * ibasis.ival1R(j, xs * x + xi);
          }
          block[j * n + i] = scale * q * xs;
        }
      }
    }
  }

  //! get the values of the j-th wavelet function at level 0
  P wval0(int j, P x) const
  {
    P b = 0, m = 1;
    for (int k = 0; k <= j; k++) {
      b += m * w0[j][k];
      m *= x;
    }
    return b / s2;
  }
  //! get the values of the j-th wavelet function at level 1, left
  P wval1L(int j, P x) const
  {
    P b = 0, m = 1;
    for (int k = 0; k < n; k++) {
      b += m * w1[j][k];
      m *= x;
    }
    return b;
  }
  //! get the values of the j-th wavelet function at level 1, right
  P wval1R(int j, P x) const
  {
    P b = 0, m = 1;
    for (int k = 0; k < n; k++) {
      b += m * w1[j][k + n];
      m *= x;
    }
    return b;
  }

private:
  // wavelet basis, levels 0 and 1
  vector2d<P> const &w0, &w1;
  // interpolation basis
  interp_basis<P, degree> const &ibasis;
  // integrator
  basis::canonical_integrator const &quad;
};

/*!
 * \brief Handles the 1d data-structure for the interpolation
 */
template<typename P, int degree>
class interpolation_manager1d {
public:
  //! create an empty interpolation manager
  interpolation_manager1d() = default;
  //! initialize the manager using the connection pattern
  interpolation_manager1d(connection_patterns const &conns) {
    connect_1d const &conn = conns[connect_1d::hierarchy::volume];
    static_assert(0 <= degree and degree <= 3);
    initialize_nodes(std::max(1, conn.max_loaded_level()));

    vector2d<P> const w0 = basis::legendre_poly<P>(degree);
    basis::canonical_integrator quad(degree);
    vector2d<P> const w1 = basis::wavelet_poly<P>(w0, quad);

    make_wav2nodal(w0, w1, conn);

    interp_basis<P, degree> basis(nodes_);
    make_nodal2hier(conn, basis);

    interp_wavelet_integrator<P, degree> integ(w0, w1, basis, quad);
    make_hier2wav(conn, integ);

    #ifdef ASGARD_USE_GPU
    load_to_gpu(conns);
    #endif
  }
  //! converts to true if the manager has been initialized
  operator bool () const { return (nodes_.num_strips() > 0); }

  //! return the canonical nodes, nodes()[i][j] where i is the cell id
  vector2d<P> const &nodes() const { return nodes_; }
  //! return the wavelet to nodal matrix
  block_sparse_matrix<P> const &wav2nodal() const { return wav2nodal_; }
  //! return the nodal to hierarchical matrix
  block_sparse_matrix<P> const &nodal2hier() const { return nodal2hier_; }
  //! return the hierarchical to wavelet matrix
  block_sparse_matrix<P> const &hier2wav() const { return hier2wav_; }

  #ifdef ASGARD_USE_GPU
  //! return the wavelet to nodal matrix
  gpu::vector<P*> const &wav2nodal(gpu::device dev) const {
    return gpu_wav2nodal_[dev.id];
  }
  //! return the nodal to hierarchical matrix
  gpu::vector<P*> const &nodal2hier(gpu::device dev) const {
    return gpu_nodal2hier_[dev.id];
  }
  //! return the hierarchical to wavelet matrix
  gpu::vector<P*> const &hier2wav(gpu::device dev) const {
    return gpu_hier2wav_[dev.id];
  }
  #endif

protected:
  //! pre-computed constant, std::sqrt(2.0)
  static P constexpr s2 = 1.41421356237309505; // sqrt(2.0)
  //! number of polynomial degrees of freedom
  static constexpr int n = degree + 1;

  //! initialize the interpolation nodes
  void initialize_nodes(int const max_level);
  //! make the matrix for wavelet-to-nodal transformation
  void make_wav2nodal(vector2d<P> const &w0, vector2d<P> const &w1,
                      connect_1d const &conn);
  //! make the matrix for nodal to hierarchical transformation
  void make_nodal2hier(connect_1d const &conn, interp_basis<P, degree> const &basis);
  //! make the matrix hierarchical basis to wavelet transformation
  void make_hier2wav(connect_1d const &conn, interp_wavelet_integrator<P, degree> const &integ);

  #ifdef ASGARD_USE_GPU
  //! load the matrices to the GPU devices
  void load_to_gpu(connection_patterns const &conns);
  #endif

private:
  vector2d<P> nodes_;
  block_sparse_matrix<P> wav2nodal_;
  block_sparse_matrix<P> nodal2hier_;
  block_sparse_matrix<P> hier2wav_;

  #ifdef ASGARD_USE_GPU
  //! gpu coefficient matrices for different levels wavelet to nodal
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lwav2nodal_;
  //! gpu coefficient matrices for different levels nodal to hierarchical
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lnodal2hier_;
  //! gpu coefficient matrices for different levels hierarchical to wavelet
  std::array<std::vector<gpu::vector<P>>, max_num_gpus> gpu_lhier2wav_;
  //! pointers to gpu matrices for different levels
  std::array<gpu::vector<P*>, max_num_gpus> gpu_wav2nodal_;
  //! pointers to gpu matrices for different levels
  std::array<gpu::vector<P*>, max_num_gpus> gpu_nodal2hier_;
  //! pointers to gpu matrices for different levels
  std::array<gpu::vector<P*>, max_num_gpus> gpu_hier2wav_;
  #endif
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
  interpolation_manager(pde_domain<P> const &domain, connection_patterns const &conns,
                        int degree)
      : num_dims(domain.num_dims()), n(degree + 1), perm(num_dims)
  {
    block_size = 1;
    wav_scale  = 1;
    for (int d : iindexof(num_dims)) {
      block_size *= (degree + 1);
      xmin[d]   = domain.xleft(d);
      xscale[d] = (domain.xright(d) - domain.xleft(d));
      wav_scale *= xscale[d];
    }
    iwav_scale = std::sqrt(wav_scale);
    wav_scale = P{1} / iwav_scale;

    switch (degree) {
      case 0:
        interp = interpolation_manager1d<P, 0>(conns);
        break;
      case 1:
        interp = interpolation_manager1d<P, 1>(conns);
        break;
      case 2:
        interp = interpolation_manager1d<P, 2>(conns);
        break;
      case 3:
        interp = interpolation_manager1d<P, 3>(conns);
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
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 0;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(n, grid, conn, perm, P{wav_scale}, P{0}, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("wavelet-to-nodal", flops);
    #else
    tools::time_event performance_("wavelet-to-nodal");
    #endif
    block_cpu(n, grid, conn, perm, wav2nodal1d(), P{wav_scale}, f, P{0}, vals, work);
  }
  //! compute nodal values for the field
  void wav2nodal(sparse_grid const &grid, connection_patterns const &conn,
                 std::vector<P> const &f, std::vector<P> &vals,
                 kronmult::workspace<P> &work) const
  {
    expect(static_cast<int64_t>(f.size()) == block_size * grid.num_indexes());
    vals.resize(f.size());
    wav2nodal(grid, conn, f.data(), vals.data(), work);
  }

  //! compute hierarchical representation from the nodal values
  void nodal2hier(sparse_grid const &grid, connection_patterns const &conn,
                  P vals[], kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 1;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::blocksv_cpu(n, grid, conn[connect_1d::hierarchy::volume], work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-hier", flops);
    #else
    tools::time_event performance_("nodal-to-hier");
    #endif
    blocksv_cpu(n, grid, conn[connect_1d::hierarchy::volume],
                nodal2hier1d(), vals, work);
  }
  //! compute hierarchical representation from the nodal values
  void nodal2hier(sparse_grid const &grid, connection_patterns const &conn,
                  std::vector<P> &vals, kronmult::workspace<P> &work) const
  {
    expect(static_cast<int64_t>(vals.size()) == block_size * grid.num_indexes());
    nodal2hier(grid, conn, vals.data(), work);
  }

  //! compute nodal values for the field
  void hier2wav(sparse_grid const &grid, connection_patterns const &conn,
                P const f[], P vals[],
                kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 2;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(n, grid, conn, perm, P{iwav_scale}, P{0}, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("hier-to-wavelet", flops);
    #else
    tools::time_event performance_("hier-to-wavelet");
    #endif
    block_cpu(n, grid, conn, perm, hier2wav1d(), P{iwav_scale}, f, P{0}, vals, work);
  }
  //! compute nodal values for the field
  void hier2wav(sparse_grid const &grid, connection_patterns const &conn,
                std::vector<P> const &f, std::vector<P> &vals,
                kronmult::workspace<P> &work) const
  {
    expect(static_cast<int64_t>(f.size()) == block_size * grid.num_indexes());
    vals.resize(f.size());
    hier2wav(grid, conn, f.data(), vals.data(), work);
  }
  //! compute nodal values for the field
  void hier2wav(sparse_grid const &grid, connection_patterns const &conn,
                P alpha, P const f[], P beta, P vals[],
                kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 2;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(n, grid, conn, perm, alpha * iwav_scale, beta, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("hier-to-wavelet", flops);
    #else
    tools::time_event performance_("hier-to-wavelet");
    #endif
    block_cpu(n, grid, conn, perm, hier2wav1d(), alpha * iwav_scale, f, beta, vals, work);
  }
  //! compute nodal values for the field
  void hier2wav(sparse_grid const &grid, connection_patterns const &conn,
                P alpha, std::vector<P> const &f, P beta, std::vector<P> &vals,
                kronmult::workspace<P> &work) const
  {
    expect(static_cast<int64_t>(f.size()) == block_size * grid.num_indexes());
    if (beta == 0)
      vals.resize(f.size());
    else
      expect(f.size() == vals.size());
    hier2wav(grid, conn, alpha, f.data(), beta, vals.data(), work);
  }
  //! returns true if the manager has been initialized
  operator bool () const { return (num_dims > 0); }

  /*!
   * \brief Performs the interpolation of the function func
   *
   * Given the grid, connection patterns, and current time:
   * 1. recomputes the nodes
   * 2. computes the values of the state at the nodes
   * 3. call func() with the time, nodes, state values as "f", and computes vals
   * 4. projects the result back in the basis and y = alpha * vals + beta * y
   *
   * The workspace is needed to call kronmult, the t1 and t2 are additional
   * workspace with size equal to the state.
   * The names t1/t2 come because this sues term_manager scratch space for working with chains
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, P time, P const state[],
       P alpha, md_func_f<P> const &func, P beta, P y[],
       kronmult::workspace<P> &work,
       std::vector<P> &t1, std::vector<P> &t2) const
  {
    tools::time_event performance_("interpolation operation");
    wav2nodal(grid, conn, state, t1.data(), work);
    {
      tools::time_event perf_("interpolation function");
      func(time, nodes(grid), t1, t2);
    }
    nodal2hier(grid, conn, t2, work);
    hier2wav(grid, conn, alpha, t2.data(), beta, y, work);
  }
  /*!
   * \brief Performs the interpolation of the function func
   *
   * Vector variant
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, P time,
       std::vector<P> const &state,
       P alpha, md_func_f<P> const &func, P beta, std::vector<P> &y,
       kronmult::workspace<P> &work,
       std::vector<P> &t1, std::vector<P> &t2) const
  {
    expect(state.size() == t1.size() and t1.size() == t2.size());
    if (beta == 0)
      y.resize(state.size());
    else
      expect(y.size() == state.size());
    (*this)(grid, conn, time, state.data(), alpha, func, beta, y.data(),
            work, t1, t2);
  }
  /*!
   * \brief Performs the interpolation of the function func
   *
   * Given the grid, connection patterns, and current time:
   * 1. recomputes the nodes
   * 2. call func() with the time, nodes, and computes vals
   * 3. projects the result back in the basis and y = alpha * vals + beta * y
   *
   * The workspace is needed to call kronmult, the t1 and t2 are additional
   * workspace with size equal to the state.
   * The names t1/t2 come because this sues term_manager scratch space for working with chains
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, P time,
       P alpha, md_func<P> const &func, P beta, P y[],
       kronmult::workspace<P> &work,
       std::vector<P> &t1) const
  {
    func(time, nodes(grid), t1);
    nodal2hier(grid, conn, t1, work);
    hier2wav(grid, conn, alpha, t1.data(), beta, y, work);
  }
  /*!
   * \brief Performs the interpolation of the function func
   *
   * Vector variant
   */
  void operator ()
      (sparse_grid const &grid, connection_patterns const &conn, P time,
       P alpha, md_func<P> const &func, P beta, std::vector<P> &y,
       kronmult::workspace<P> &work,
       std::vector<P> &t1) const
  {
    if (beta == 0)
      y.resize(t1.size());
    else
      expect(y.size() == t1.size());
    (*this)(grid, conn, time, alpha, func, beta, y.data(), work, t1);
  }

  #ifdef ASGARD_USE_GPU
  //! compute nodal values for the field
  void wav2nodal(gpu::device dev, sparse_grid const &grid,
                 connection_patterns const &conn, P const f[], P vals[],
                 kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 0;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(n, grid, conn, perm, P{wav_scale}, P{0}, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("wavelet-to-nodal", flops);
    #else
    tools::time_event performance_("wavelet-to-nodal");
    #endif
    block_gpu(dev, n, grid, conn, perm, wav2nodal1d(dev), P{wav_scale}, f,
              P{0}, vals, work, wav2nodal1d());
  }
  //! compute hierarchical representation from the nodal values
  void nodal2hier(gpu::device dev, sparse_grid const &grid,
                  connection_patterns const &conn, P vals[],
                  kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 1;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::blocksv_cpu(n, grid, conn[connect_1d::hierarchy::volume], work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("nodal-to-hier", flops);
    #else
    tools::time_event performance_("nodal-to-hier");
    #endif
    blocksv_gpu(dev, n, grid, conn, nodal2hier1d(dev), vals, work, nodal2hier1d());
  }
  //! compute nodal values for the field
  void hier2wav(gpu::device dev, sparse_grid const &grid,
                connection_patterns const &conn,
                P alpha, P const f[], P beta, P vals[],
                kronmult::workspace<P> &work) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int constexpr id = 2;
    int64_t const flops = [&, this]()-> int64_t {
        if (flop_info[id].grid_gen != grid.generation()) {
          flop_info[id].flops = kronmult::block_cpu(n, grid, conn, perm, alpha * iwav_scale, beta, work);
          flop_info[id].grid_gen = grid.generation();
        }
        return flop_info[id].flops;
      }();
    tools::time_event performance_("hier-to-wavelet", flops);
    #else
    tools::time_event performance_("hier-to-wavelet");
    #endif
    block_gpu(dev, n, grid, conn, perm, hier2wav1d(dev), alpha * iwav_scale, f, beta, vals, work, hier2wav1d());
  }
  /*!
   * \brief Performs the interpolation of the function func
   */
  void operator ()
      (gpu::device dev, sparse_grid const &grid,
       connection_patterns const &conn, P time, P const state[],
       P alpha, md_func_f<P> const &func, P beta, P y[],
       kronmult::workspace<P> &work,
       std::vector<P> &t1, std::vector<P> &t2, gpu::vector<P> &gpu_t1) const
  {
    tools::time_event performance_("interpolation operation");
    wav2nodal(dev, grid, conn, state, gpu_t1.data(), work);
    gpu_t1.copy_to_host(t1);
    {
      tools::time_event perf_("interpolation function");
      func(time, nodes(grid), t1, t2);
    }
    gpu_t1 = t2;
    nodal2hier(dev, grid, conn, gpu_t1.data(), work);
    hier2wav(dev, grid, conn, alpha, gpu_t1.data(), beta, y, work);
  }
  /*!
   * \brief Computes the interpolation function on the CPU and moves the data to the GPU
   *
   * In this context, the kronmult work is done on the GPU
   * but the function evaluation is done on the CPU side.
   */
  void operator ()
      (gpu::device dev, sparse_grid const &grid,
       connection_patterns const &conn, P time,
       P alpha, md_func<P> const &func, P beta, P y[],
       kronmult::workspace<P> &work,
       std::vector<P> &t1, gpu::vector<P> &gpu_t1) const
  {
    func(time, nodes(grid), t1);
    gpu_t1 = t1;
    nodal2hier(dev, grid, conn, gpu_t1.data(), work);
    hier2wav(dev, grid, conn, alpha, gpu_t1.data(), beta, y, work);
  }
  #endif

protected:
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
  //! return the 1d hier2wav matrix
  block_sparse_matrix<P> const &hier2wav1d() const {
    switch(interp.index()) {
      case 0: return std::get<0>(interp).hier2wav();
      case 1: return std::get<1>(interp).hier2wav();
      case 2: return std::get<2>(interp).hier2wav();
      default: // case 3
        return std::get<3>(interp).hier2wav();
    }
  }

  #ifdef ASGARD_USE_GPU
  //! return the 1d wav2noal matrix
  gpu::vector<P *> const &wav2nodal1d(gpu::device dev) const {
    switch(interp.index()) {
      case 0: return std::get<0>(interp).wav2nodal(dev);
      case 1: return std::get<1>(interp).wav2nodal(dev);
      case 2: return std::get<2>(interp).wav2nodal(dev);
      default: // case 3
        return std::get<3>(interp).wav2nodal(dev);
    }
  }
  //! return the 1d nodal2hier matrix
  gpu::vector<P *> const &nodal2hier1d(gpu::device dev) const {
    switch(interp.index()) {
      case 0: return std::get<0>(interp).nodal2hier(dev);
      case 1: return std::get<1>(interp).nodal2hier(dev);
      case 2: return std::get<2>(interp).nodal2hier(dev);
      default: // case 3
        return std::get<3>(interp).nodal2hier(dev);
    }
  }
  //! return the 1d hier2wav matrix
  gpu::vector<P *> const &hier2wav1d(gpu::device dev) const {
    switch(interp.index()) {
      case 0: return std::get<0>(interp).hier2wav(dev);
      case 1: return std::get<1>(interp).hier2wav(dev);
      case 2: return std::get<2>(interp).hier2wav(dev);
      default: // case 3
        return std::get<3>(interp).hier2wav(dev);
    }
  }
  #endif

private:
  int num_dims = 0;
  int n = 0;
  std::array<P, max_num_dimensions> xmin, xscale;
  P wav_scale = 0, iwav_scale = 0;

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

  #ifdef ASGARD_USE_FLOPCOUNTER
  struct flop_info_entry {
    int grid_gen = -1;
    int64_t flops = 0;
  };
  // indexes are wav2nodal (0), nodal2hier (1), hier2wav(2)
  mutable std::array<flop_info_entry, 3> flop_info;
  #endif
};

/*!
 * \brief Manages the data-structures for the non-separable operations
 */
template<typename P>
class quadmd_manager {
public:
  //! default empty constructor, must reinitialize to use the class
  quadmd_manager() = default;
  //! initialize the manager
  quadmd_manager(pde_domain<P> const &domain, hierarchy_manipulator<P> const &hier,
                 connection_patterns const &conn);

  //! (mostly testing) returns the hierarchical form of the 1d nodes
  std::vector<P> const &nodes1d() const { return nodes1d_; }
  //! returns the nodes corresponding to the grid
  vector2d<P> const &nodes(sparse_grid const &grid) const;

private:
  int num_dims = 0;
  int pdof = 0;
  int block_size = 0;
  std::array<P, max_num_dimensions> xmin, xscale;

  int grid_gen = -1;

  std::vector<P> nodes1d_;
  mutable vector2d<P> nodes_;

  kronmult::permutes perm;

};

} // namespace asgard
