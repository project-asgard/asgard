#include "asgard_interp.hpp"

#include "asgard_small_mats.hpp"

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

#ifdef ASGARD_USE_GPU
template<typename P, int degree>
void interpolation_manager1d<P, degree>::load_to_gpu(connection_patterns const &conns)
{
  int level = conns.max_loaded_level();

  int const num_gpus = compute->num_gpus();
  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});

    std::vector<P*> coeff_pntrs(level + 1, nullptr);

    // loading wav2nodal_
    gpu_lwav2nodal_[g].resize(level + 1);
    for (int l = 0; l < level; l++) {
      gpu_lwav2nodal_[g][l] = wav2nodal_.get_subpattern(l, conns).data_vector();
      coeff_pntrs[l]        = gpu_lwav2nodal_[g][l].data();
    }
    gpu_lwav2nodal_[g][level] = wav2nodal_.data_vector();
    coeff_pntrs[level]        = gpu_lwav2nodal_[g][level].data();

    gpu_wav2nodal_[g] = coeff_pntrs;

    // loading nodal2hier_
    gpu_lnodal2hier_[g].resize(level + 1);
    for (int l = 0; l < level; l++) {
      gpu_lnodal2hier_[g][l] = nodal2hier_.get_subpattern(l, conns).data_vector();
      coeff_pntrs[l]         = gpu_lnodal2hier_[g][l].data();
    }
    gpu_lnodal2hier_[g][level] = nodal2hier_.data_vector();
    coeff_pntrs[level]         = gpu_lnodal2hier_[g][level].data();

    gpu_nodal2hier_[g] = coeff_pntrs;

    // loading hier2wav_
    gpu_lhier2wav_[g].resize(level + 1);
    for (int l = 0; l < level; l++) {
      gpu_lhier2wav_[g][l] = hier2wav_.get_subpattern(l, conns).data_vector();
      coeff_pntrs[l]       = gpu_lhier2wav_[g][l].data();
    }
    gpu_lhier2wav_[g][level] = hier2wav_.data_vector();
    coeff_pntrs[level]       = gpu_lhier2wav_[g][level].data();

    gpu_hier2wav_[g] = coeff_pntrs;
  }
}
#endif

template<typename P>
vector2d<P> const &interpolation_manager<P>::nodes(sparse_grid const &grid) const
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

template<typename P>
quadmd_manager<P>::quadmd_manager(
    pde_domain<P> const &domain, hierarchy_manipulator<P> const &hier,
    connection_patterns const &conn)
    : num_dims(domain.num_dims()), pdof(hier.degree() + 1), block_size(hier.block_size()),
      perm(num_dims)
{
  wav_scale  = 1;
  for (int d : iindexof(num_dims)) {
    xmin[d]   = domain.xleft(d);
    xscale[d] = (domain.xright(d) - domain.xleft(d));
    wav_scale *= xscale[d];
  }
  iwav_scale = std::sqrt(wav_scale);
  wav_scale  = P{1} / iwav_scale;

  // points represents the point locations in the canonical element (-1, 1)
  // horder represents the hierarchical order
  //   e.g., two adjacent non-hierarchical cells with pdof points each
  //         (p_0, p_1, p_2) (p_3, p_4, p_5)
  //         will merge into two hierarchical cells
  //         (h_0, h_1, h_2)
  //         (h_3, h_4, h_5)
  //         h-order is the list of p indexes that will form (h_0, h_1, h_2)

  std::vector<double> points;
  std::vector<int> horder;
  switch (pdof) {
  case 1: // constant
    points = {-1.0, };
    horder = {0, }; // take the left cell
    break;
  case 2: // linear
    points = {-1.0 / 3.0, +1.0 / 3.0}; // interior nodes
    horder = {1, 2};
    // points = {-1.0, +1.0}; // edge interpolation
    // horder = {0, 3};
    break;
  case 3: // quadratic
    points = {-1.0, -0.0/3.0, 1.0};
    horder = {0, 2, 5};
    // points = {-0.6, -0.2, 0.6};
    // horder = {1, 3, };
    break;
  case 4: // cubic
    points = {-1.0, -1.0/3.0, 1.0/3.0, 1.0};
    horder = {0, 2, 5, 7};
    // points = {-0.6, -0.2, +0.2, +0.6};
    // horder = {1, 3, 4, 6};
    break;
  default:
    break;
  };

  expect(points.size() == static_cast<size_t>(pdof));
  expect(horder.size() == points.size());

  int const pdof2 = pdof * pdof;

  // transformation matrices for the permutation and hierarchical coefficients
  // 4 matrices of size 2 * pdof X 2 * pdof, plus the lower order nodes
  std::vector<P> trans_mats_(3 * 4 * pdof2 + pdof);
  smmat::matrix<P> permute(2 * pdof, trans_mats_.data());
  smmat::matrix<P> hier_coeff(2 * pdof, trans_mats_.data() + 4 * pdof * pdof);
  smmat::matrix<P> ihier_coeff(2 * pdof, trans_mats_.data() + 8 * pdof * pdof);
  P *lorder = trans_mats_.data() + 12 * pdof2;

  { // find the set of lower node indexes, i.e., not included in horder
    int idx = 0;
    for (int i = 0; i < 2 * pdof; i++)
      if (std::none_of(horder.begin(), horder.end(),
                       [&](int m) -> bool { return (m == i); }))
        lorder[idx++] = i;
    expect(idx == pdof);
  }

  { // construct the permutation transform
    for (int r = 0; r < pdof; r++) // upper level nodes
      permute(r, horder[r]) = 1;

    for (int r = 0; r < pdof; r++) // lower level nodes
      permute(r + pdof, lorder[r]) = 1;
  }
  // permute.print();

  // ------------------------------------------------------------
  // construct the points and remap to hierarchical order
  // ------------------------------------------------------------
  int const level     = conn.max_loaded_level();
  int const num_cells = conn.conns[0].num_rows();
  P const cell_size = P{1} / static_cast<P>(num_cells);
  P const sqrt_size = std::sqrt(static_cast<P>(num_cells));

  std::vector<P> cell_nodes(num_cells * pdof);
  #pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
    for (int j = 0; j < pdof; j++)
      cell_nodes[i * pdof + j] = cell_size * (i + P{0.5} + P{0.5} * points[j]);

  hier.transform(permute.data(), level, cell_nodes, nodes1d_);

  // ------------------------------------------------------------
  // transforming hierarchical Legendre coefficients to nodal values
  // ------------------------------------------------------------
  block_diag_matrix<P> mat(pdof * pdof, num_cells);

  {
    // values of Legendre polynomials at the interpolation points
    // functions are scaled by 1/sqrt(dx), i.e., sqrt(num-points)
    auto legendre = legendre_vals(points, pdof - 1);
    auto &leg_vals = legendre[0];

    for(auto &l : leg_vals) l *= sqrt_size;

    if constexpr (is_double<P>) {
      fill_pattern(leg_vals.data(), mat);
    } else {
      std::vector<P> fleg(leg_vals.size());
      std::copy(leg_vals.begin(), leg_vals.end(), fleg.begin());
      fill_pattern(fleg.data(), mat);
    }
  }

  wav2nodal_ = hier.diag2block(hierarchy_manipulator<P>::operation::custom_unitary,
                               permute.data(),
                               hierarchy_manipulator<P>::operation::transform,
                               nullptr, mat, level, conn);

  // ------------------------------------------------------------
  // transforming nodal coefficients to hierarchical coefficients
  // ------------------------------------------------------------

  // the nodal-values -> interpolatory hierarchical coefficients starts
  // with the permutation, then the values of the Lagrange polynomials on
  // the coarser level are subtracted
  std::copy_n(permute.data(), 4 * pdof2, hier_coeff.data());

  std::vector<P> canonical_hier;
  { // build the canonical hierarchical nodes and eval Legendre basis
    std::vector<P> two_cells(2 * pdof);

    for (int j = 0; j < pdof; j++) // (-1, 1) -> (-1, 0)
      two_cells[j] = P{0.5} * points[j] - P{0.5};
    for (int j = 0; j < pdof; j++) // (-1, 1) -> (0, 1)
      two_cells[pdof + j] = P{0.5} * points[j] + P{0.5};

    hier.transform(permute.data(), 1, two_cells, canonical_hier);
  }

  for (int r = 0; r < pdof; r++)
    for (int i = 0; i < pdof; i++)
      hier_coeff(r + pdof, horder[i]) = -fm::lagrange<double>(points, i, canonical_hier[r + pdof]);

  // hier_coeff.print();

  fill_pattern(smmat::make_identity<P>(pdof).data(), mat); // start with identity

  nodal2hier_ = hier.diag2block(
                    hierarchy_manipulator<P>::operation::custom_non_unitary,
                    hier_coeff.data(),
                    hierarchy_manipulator<P>::operation::custom_unitary,
                    permute.data(), mat, level, conn);

  // ------------------------------------------------------------
  // projecting hierarchical interpolation basis to hierarchical Legendre
  // ------------------------------------------------------------

  // form the inverse map for the hierarchical coefficients
  for (int i = 0; i < pdof; i++)
    ihier_coeff(horder[i], i) = 1; // high order nodes stay as is
  for (int i = 0; i < pdof; i++)
    ihier_coeff(lorder[i], pdof + i) = 1; // low order nodes to self

  for (int i = 0; i < pdof; i++) // each low order point
    for (int c = 0; c < pdof; c++) // each high order basis function
      ihier_coeff(lorder[i], c) = fm::lagrange<double>(points, c, canonical_hier[i + pdof]);

  // ihier_coeff.print();

  { // nodal cell-by-cell projection
    auto [pnts, wts]     = legendre_weights(pdof - 1, -1, 1);
    auto [lvals, lprime] = legendre_vals(pnts, pdof - 1);
    ignore(lprime);

    int const num_quad = static_cast<int>(pnts.size());

    std::vector<double> legw(lvals.size());
    smmat::col_scal(num_quad, pdof, wts.data(), lvals.data(), legw.data());

    std::vector<double> lag(legw.size());
    for (int c = 0; c < pdof; c++) // each Lagrange function
      for (int r = 0; r < num_quad; r++) // each quadrature point
        lag[c * num_quad + r] = fm::lagrange(points, c, pnts[r]);

    std::vector<double> base(pdof * pdof);
    smmat::gemm_tn<1>(pdof, num_quad, legw.data(), lag.data(), base.data());

    double const scale = P{0.5} / sqrt_size;
    for(auto &s : base) s *= scale;

    if constexpr (is_double<P>) {
      fill_pattern(base.data(), mat);
    } else {
      std::vector<P> fbase(base.size());
      std::copy(base.begin(), base.end(), fbase.begin());
      fill_pattern(fbase.data(), mat);
    }
  }

  hier2wav_ = hier.diag2block(
                  hierarchy_manipulator<P>::operation::transform, nullptr,
                  hierarchy_manipulator<P>::operation::custom_non_unitary,
                  ihier_coeff.data(), mat, level, conn);


  // wav2nodal_.to_full(conn).print();
  // nodal2hier_.to_full(conn).print();
  // hier2wav_.to_full(conn).print();

#ifdef ASGARD_USE_GPU
  int const num_gpus = compute->num_gpus();
  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});

    std::vector<P*> coeff_pntrs(level + 1, nullptr);

    // loading wav2nodal_
    gpu_lwav2nodal_[g].resize(level + 1);
    for (int l = 0; l < level; l++) {
      gpu_lwav2nodal_[g][l] = wav2nodal_.get_subpattern(l, conn).data_vector();
      coeff_pntrs[l]        = gpu_lwav2nodal_[g][l].data();
    }
    gpu_lwav2nodal_[g][level] = wav2nodal_.data_vector();
    coeff_pntrs[level]        = gpu_lwav2nodal_[g][level].data();

    gpu_wav2nodal_[g] = coeff_pntrs;

    // loading nodal2hier_
    gpu_lnodal2hier_[g].resize(level + 1);
    for (int l = 0; l < level; l++) {
      gpu_lnodal2hier_[g][l] = nodal2hier_.get_subpattern(l, conn).data_vector();
      coeff_pntrs[l]         = gpu_lnodal2hier_[g][l].data();
    }
    gpu_lnodal2hier_[g][level] = nodal2hier_.data_vector();
    coeff_pntrs[level]         = gpu_lnodal2hier_[g][level].data();

    gpu_nodal2hier_[g] = coeff_pntrs;

    // loading hier2wav_
    gpu_lhier2wav_[g].resize(level + 1);
    for (int l = 0; l < level; l++) {
      gpu_lhier2wav_[g][l] = hier2wav_.get_subpattern(l, conn).data_vector();
      coeff_pntrs[l]       = gpu_lhier2wav_[g][l].data();
    }
    gpu_lhier2wav_[g][level] = hier2wav_.data_vector();
    coeff_pntrs[level]       = gpu_lhier2wav_[g][level].data();

    gpu_hier2wav_[g] = coeff_pntrs;
  }
#endif
}

template<typename P>
vector2d<P> const &quadmd_manager<P>::nodes(sparse_grid const &grid) const
{
  if (grid.generation() == grid_gen)
    return nodes_;

  int64_t const num_points = grid.num_indexes() * block_size;

  nodes_.resize(num_dims, num_points);

  span2d<P const> const nd1d(pdof, -1, nodes1d_.data());

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
          nodes_[i * block_size + j][d] = offs[d][t % pdof];
          t /= pdof;
        }
      }

      ASGARD_PRAGMA_OMP_SIMD(collapse(2))
      for (int j = 0; j < block_size; j++)
        for (int d = 0; d < num_dims; d++)
          nodes_[i * block_size + j][d] = xmin[d] + nodes_[i * block_size + j][d] * xscale[d];
    }
  }

  grid_gen = grid.generation();

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

template class quadmd_manager<double>;
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

template class quadmd_manager<float>;
#endif

} // namespace asgard
