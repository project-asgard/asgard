#include "asgard_interp.hpp"

namespace asgard
{

template<typename P, int degree>
interp_basis<P, degree>::interp_basis(vector2d<P> const &nodes) {
  expect(nodes.num_strips() >= 2);
  expect(nodes.stride() == n);

  std::copy_n(nodes[0], n, x0.begin());

  std::copy_n(nodes[1], nL, xL.begin());
  std::copy_n(nodes[0], nR, xL.begin() + nL);

  std::copy_n(nodes[0] + nR, nL, xR.begin());
  std::copy_n(nodes[1] + nL, nR, xR.begin() + nL);
  for (int i = 0; i < n; i++) {
    w0[i] = 1;
    for (int j = 0; j < i; j++)
      w0[i] *= (x0[i] - x0[j]);
    for (int j = i + 1; j < n; j++)
      w0[i] *= (x0[i] - x0[j]);
    w0[i] = P{1} / w0[i];
  }
  for (int i = 0; i < nL; i++) {
    wL[i] = 1;
    for (int j = 0; j < i; j++)
      wL[i] *= (xL[i] - xL[j]);
    for (int j = i + 1; j < n; j++)
      wL[i] *= (xL[i] - xL[j]);
    wL[i] = P{1} / wL[i];
  }
  for (int i = 0; i < nR; i++) {
    wR[i] = 1;
    for (int j = 0; j < nL + i; j++)
      wR[i] *= (xR[i + nL] - xR[j]);
    for (int j = nL + i + 1; j < n; j++)
      wR[i] *= (xR[i + nL] - xR[j]);
    wR[i] = P{1} / wR[i];
  }
}

template<typename P, int degree>
void interpolation_manager1d<P, degree>::initialize_nodes(int const max_level)
{
  int const num_cells = fm::ipow2(max_level);

  nodes_ = vector2d<P>(n, num_cells);

  // for degree 0, 1, 2, 3 ..., start start_den is 3, 3, 5, 5 ...
  P constexpr start_den = (degree == 0) ? 1 : 2 * (degree / 2) + 3;

  std::array<P, n> const num0 = []() -> std::array<P, n> {
      if constexpr (degree == 0)
        return {0, };
      else if constexpr (degree == 1)
        return {1, 2};
      else if constexpr (degree == 2)
        return {1, 2, 4};
      else // if constexpr (degree == 3) {
        return {1, 2, 3, 4};
    }();

  std::array<P, n> const num1 = []() -> std::array<P, n> {
      if constexpr (degree == 0)
        return {1, };
      else if constexpr (degree == 1)
        return {1, 5};
      else if constexpr (degree == 2)
        return {1, 6, 9};
      else // if constexpr (degree == 3) {
        return {1, 3, 7, 9, };
    }();

  P den = start_den;
  for (int j = 0; j < n; j++)
    nodes_[0][j] = num0[j] / den;

  int ncells = 1;
  for (int l = 1; l <= max_level; l++) {
    den *= 2;

    P offset = 0;
    for (int c = ncells; c < 2 * ncells; c++) {
      for (int j = 0; j < n; j++)
        nodes_[c][j] = (num1[j] + offset) / den;
      offset += 2 * start_den;
    }

    ncells *= 2;
  }
}

template<typename P, int degree>
void interpolation_manager1d<P, degree>::make_wav2nodal(
    vector2d<P> const &w0, vector2d<P> const &w1, connect_1d const &conn)
{
  wav2nodal_ = block_sparse_matrix<P>(n * n, conn.num_connections(), connect_1d::hierarchy::volume);

  std::array<P, n> x;

  // fill the block with (i, j) values of \phi_j(x_i)
  // uses wavelet basis at level 0, x is in (0, 1)
  auto apply_w0 = [&](P block[]) -> void {
    for (int i = 0; i < n; i++) { // point i
      P const w = 2 * x[i] - 1;
      for (int j = 0; j < n; j++) { // basis j
        P b = 0, m = 1;
        for (int k = 0; k <= j; k++) {
          b += m * w0[j][k];
          m *= w;
        }
        block[j * n + i] = s2 * b;
      }
    }
  };
  // uses wavelet basis at level > 0, x is in (-1, 1)
  auto apply_w1 = [&](P block[], P const scale) -> void {
    for (int i = 0; i < n; i++) { // point i
      P const w = x[i];
      if (w < -1 or w > 1) { // out of bounds
        for (int j = 0; j < n; j++)
          block[j * n + i] = 0;
      } else { // point is in the cell
        int const offset = (w < 0) ? 0 : n; // left/right wavelet
        for (int j = 0; j < n; j++) { // basis j
          P b = 0, m = 1;
          for (int k = 0; k < n; k++) {
            b += m * w1[j][k + offset];
            m *= w;
          }
          block[j * n + i] = scale * b;
        }
      }
    }
  };

  if (conn.max_loaded_level() == 0) {
    std::copy_n(nodes_[0], n, x.begin());
    apply_w0(wav2nodal_[0]);
    return;
  }

  for (int row : iindexof(conn.num_rows()))
  {
    P const *const raw_x = nodes_[row];

    std::copy_n(raw_x, n, x.begin());

    int const row_end   = conn.row_end(row);

    int c = conn.row_begin(row);

    // first two cells always have global support
    apply_w0(wav2nodal_[c++]);

    for (int i = 0; i < n; i++)
        x[i] = -1 + 2 * x[i];
    apply_w1(wav2nodal_[c++], 1);

    // the above gets us to level 2
    int level_begin = 2; // first cell on each level
    P scale = s2;
    P dx    = 0.5; // cell size

    // loop over the rest of the row
    for (; c < row_end; c++)
    {
      int const col = conn[c]; // connected cell

      // move to the next level
      while (col >= 2 * level_begin)
      {
        level_begin *= 2;
        scale *= s2;
        dx    *= 0.5;
      }

      P xl = dx * (col - level_begin); // left-most node

      #pragma omp simd
      for (int i = 0; i < n; i++)
        x[i] = -1 + 2 * (raw_x[i] - xl) / dx;

      apply_w1(wav2nodal_[c], scale); // uses captured x
    }
  }
}

template<typename P, int degree>
void interpolation_manager1d<P, degree>::make_hier2wav(connect_1d const &conn,
    interp_wavelet_integrator<P, degree> const &integ)
{
  hier2wav_ = block_sparse_matrix<P>(n * n, conn.num_connections(), connect_1d::hierarchy::volume);

  int const num_rows = conn.num_rows();
  if (num_rows == 1) {
    integ.mat00(hier2wav_[0]);
    return;
  }

  // first two entries on row 0
  integ.mat00(hier2wav_[0]);
  integ.mat01(hier2wav_[1]);

  // the above gets us to level 2
  int level_begin = 2; // first cell on each level
  P xs            = 0.5; // slope is the ratio of full canonical domain to sub-domain

  for (int c = conn.row_begin(0) + 2; c < conn.row_end(0); c++)
  {
    int const col = conn[c]; // connected cell

    // move to the next level
    while (col >= 2 * level_begin)
    {
      level_begin *= 2;
      xs          *= 0.5;
    }

    // transformation intercept for this cell
    P const xi = 2 * xs * (col - level_begin) + xs - 1;

    integ.mat01i(xs, xi, hier2wav_[c]);
  }

  // do row 2
  integ.mat10(hier2wav_[conn.row_begin(1)]);

  integ.mat11(P{0.5}, hier2wav_[conn.row_begin(1) + 1]);

  // finish row 2
  level_begin = 2;
  xs          = 0.5;

  for (int c = conn.row_begin(1) + 2; c < conn.row_end(1); c++)
  {
    int const col = conn[c]; // connected cell

    // move to the next level
    while (col >= 2 * level_begin)
    {
      level_begin *= 2;
      xs          *= 0.5;
    }

    // transformation intercept for this cell
    P const xi = 2 * xs * (col - level_begin) + xs - 1;

    integ.mat11i(xs, xi, P{0.5}, hier2wav_[c]);
  }

  int wlbegin = 2; // wavelet functions, level begin
  P wscale    = 0.5 * s2; // wavelet function scale factor
  P wxs       = 0.5; // wavelet functions, ratio of support over canonical domain

  for (int row = 2; row < num_rows; row++)
  {
    if (row >= 2 * wlbegin)
    {
      wlbegin *= 2;
      wscale *= s2;
      wxs *= 0.5;
    }

    int c = conn.row_begin(row);

    P const wxi = 2 * wxs * (row - wlbegin) + wxs - 1;

    // 1. handle the functions at the higher level, the wavelet is on a sub-domain
    integ.mat10w(wxi, wxs, wscale, hier2wav_[c++]); // TODO: remove, this is always 0
    integ.mat11w(wxi, wxs, wscale, hier2wav_[c++]);

    level_begin = 2;
    xs          = 0.5;

    for (; c < conn.row_diag(row); c++) {
      int const col = conn[c]; // connected cell
      if (col >= 2 * level_begin)
      {
        level_begin *= 2;
        xs          *= 0.5;
      }

      P const xi = 2 * xs * (conn[c] - level_begin) + xs - 1;

      integ.mat11w(wxs / xs, (wxi - xi) / xs, wscale * xs, hier2wav_[c]);
    }

    // cell to self
    integ.mat11(wscale * wxs, hier2wav_[c++]);

    level_begin *= 2;
    xs          *= 0.5;

    for (; c < conn.row_end(row); c++) {
      int const col = conn[c]; // connected cell
      if (col >= 2 * level_begin)
      {
        level_begin *= 2;
        xs          *= 0.5;
      }

      P const xi = 2 * xs * (conn[c] - level_begin) + xs - 1;

      integ.mat11i(xs / wxs, (xi - wxi) / wxs, wscale * wxs, hier2wav_[c]);
    }
  }
}

template<typename P, int degree>
void interpolation_manager1d<P, degree>::make_nodal2hier(
    connect_1d const &conn, interp_basis<P, degree> const &basis)
{
  nodal2hier_ = block_sparse_matrix<P>(n * n, conn.num_connections(), connect_1d::hierarchy::volume);

  std::array<P, n> x;

  if (conn.max_loaded_level() == 0) {
    std::copy_n(nodes_[0], n, x.begin());
    basis.eval0(x, nodal2hier_[0]);
    return;
  }

  std::copy_n(nodes_[1], n, x.begin());
  basis.eval0(x, nodal2hier_[conn.row_begin(1)]);

  for (int row = 2; row < conn.num_rows(); row++)
  {
    P const *const raw_x = nodes_[row];

    std::copy_n(raw_x, n, x.begin());

    int const row_end = conn.row_diag(row);

    int c = conn.row_begin(row);

    // first two cells always have global support
    basis.eval0(x, nodal2hier_[c++]);
    basis.eval1(x, nodal2hier_[c++]);

    // the above gets us to level 2
    int level_begin = 2; // first cell on each level

    P dx = 0.5; // cell size

    // loop over the rest of the row
    for (; c < row_end; c++)
    {
      int const col = conn[c]; // connected cell

      // move to the next level
      while (col >= 2 * level_begin)
      {
        level_begin *= 2;
        dx    *= 0.5;
      }

      P xl = dx * (col - level_begin); // left-most node

      #pragma omp simd
      for (int i = 0; i < n; i++)
        x[i] = (raw_x[i] - xl) / dx;

      basis.eval1(x, nodal2hier_[c]); // uses captured x
    }
  }
}

template<typename P>
vector2d<P> const &interpolation_manager<P>::nodes(
    sparse_grid const &grid) const
{
  if (grid.generation() == grid_gen)
    return nodes_;

  int64_t const num_points = grid.num_indexes() * block_size;

  nodes_.resize(num_dims, num_points);

  vector2d<P> const &nd1d = nodes1d();

  #pragma omp parallel
  {
    std::array<P const *, max_num_dimensions> offs;

    #pragma omp for
    for (int64_t i = 0; i < grid.num_indexes(); i++)
    {
      for (int d = 0; d < num_dims; d++)
        offs[d] = nd1d[grid[i][d]];

      for (int j : iindexof(block_size))
      {
        int64_t t = j;
        for (int d = num_dims - 1; d >= 0; d--) {
          nodes_[i * block_size + j][d] = offs[d][t % n];
          t /= n;
        }
      }

      ASGARD_PRAGMA_OMP_SIMD(collapse(2))
      for (int j = 0; j < block_size; j++)
        for (int d = 0; d < num_dims; d++)
          nodes_[i * block_size + j][d] = xmin[d] + nodes_[i * block_size + j][d] * xscale[d];
    }
  }

  return nodes_;
}

#ifdef ASGARD_ENABLE_DOUBLE
template class interp_basis<double, 0>;
template class interp_basis<double, 1>;
template class interp_basis<double, 2>;
template class interp_basis<double, 3>;

template class interpolation_manager1d<double, 0>;
template class interpolation_manager1d<double, 1>;
template class interpolation_manager1d<double, 2>;
template class interpolation_manager1d<double, 3>;

template class interpolation_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class interp_basis<float, 0>;
template class interp_basis<float, 1>;
template class interp_basis<float, 2>;
template class interp_basis<float, 3>;

template class interpolation_manager1d<float, 0>;
template class interpolation_manager1d<float, 1>;
template class interpolation_manager1d<float, 2>;
template class interpolation_manager1d<float, 3>;

template class interpolation_manager<float>;
#endif

} // namespace asgard
