#include "asgard_interp.hpp"

#include "asgard_small_mats.hpp"

namespace asgard
{
template<typename P>
interpolation_manager<P>::interpolation_manager(
    prog_opts const &opts,
    pde_domain<P> const &domain, hierarchy_manipulator<P> const &hier,
    connection_patterns const &conn)
    : num_dims(domain.num_dims()), pdof(hier.degree() + 1),
      block_size(hier.block_size()),
      perm(num_dims),
      perm_low(num_dims, conn_fill::lower_udiag),
      perm_up(num_dims, conn_fill::upper)
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
  // testing purposes, allow setting different points in the options
  if (not opts.interp_points.empty()) {
    rassert(opts.interp_points.size() == static_cast<size_t>(pdof),
            "the size of interp_points must be degree + 1");
    rassert(opts.interp_horder.size() == opts.interp_points.size(),
            "the size of interp_horder must match interp_points");
    points = opts.interp_points;
    horder = opts.interp_horder;
  }

  expect(points.size() == static_cast<size_t>(pdof));
  expect(horder.size() == points.size());

  int const pdof2 = pdof * pdof;

  // transformation matrices for the permutation and hierarchical coefficients
  // 4 matrices of size 2 * pdof X 2 * pdof, plus the lower order nodes
  trans_mats_.resize(3 * 4 * pdof2 + pdof);
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
  diag_h2w = block_diag_matrix<P>(pdof * pdof, num_cells);

  {
    // values of Legendre polynomials at the interpolation points
    // functions are scaled by 1/sqrt(dx), i.e., sqrt(num-points)
    auto legendre = legendre_vals(points, pdof - 1);
    auto &leg_vals = legendre[0];

    for(auto &l : leg_vals) l *= sqrt_size;

    if constexpr (is_double<P>) {
      fill_pattern(leg_vals.data(), diag_h2w);
    } else {
      std::vector<P> fleg(leg_vals.size());
      std::copy(leg_vals.begin(), leg_vals.end(), fleg.begin());
      fill_pattern(fleg.data(), diag_h2w);
    }
  }

  wav2nodal_ = hier.diag2block(hierarchy_manipulator<P>::operation::custom_unitary,
                               permute.data(),
                               hierarchy_manipulator<P>::operation::transform,
                               nullptr, diag_h2w, level, conn);

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

  fill_pattern(smmat::make_identity<P>(pdof).data(), diag_h2w); // start with identity

  nodal2hier_ = hier.diag2block(
                    hierarchy_manipulator<P>::operation::custom_non_unitary,
                    hier_coeff.data(),
                    hierarchy_manipulator<P>::operation::custom_unitary,
                    permute.data(), diag_h2w, level, conn);

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
      fill_pattern(base.data(), diag_h2w);
    } else {
      std::vector<P> fbase(base.size());
      std::copy(base.begin(), base.end(), fbase.begin());
      fill_pattern(fbase.data(), diag_h2w);
    }
  }

  hier2wav_ = hier.diag2block(
                  hierarchy_manipulator<P>::operation::transform, nullptr,
                  hierarchy_manipulator<P>::operation::custom_non_unitary,
                  ihier_coeff.data(), diag_h2w, level, conn);


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
vector2d<P> const &interpolation_manager<P>::nodes(sparse_grid const &grid) const
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

template<typename P> block_sparse_matrix<P>
interpolation_manager<P>::mult_transform_h2w(hierarchy_manipulator<P> const &hier,
                                             connection_patterns const &conns,
                                             block_diag_matrix<P> const &mat,
                                             block_diag_matrix<P> &work) const
{
  expect(mat.nblock() == pdof * pdof);
  expect(mat.nrows() == diag_h2w.nrows());

  work.check_resize(mat);
  gemm_block_diag(pdof, mat, diag_h2w, work);

  return hier.diag2block(hierarchy_manipulator<P>::operation::transform, nullptr,
                         hierarchy_manipulator<P>::operation::custom_non_unitary,
                         trans_mats_.data() + 8 * pdof * pdof, work,
                         fm::intlog2(mat.nrows()), conns);
}

template<typename P> block_sparse_matrix<P>
interpolation_manager<P>::mult_transform_h2w(hierarchy_manipulator<P> const &hier,
                                             connection_patterns const &conns,
                                             block_tri_matrix<P> const &mat,
                                             block_tri_matrix<P> &work) const
{
  expect(mat.nblock() == pdof * pdof);
  expect(mat.nrows() == diag_h2w.nrows());

  work.check_resize(mat);
  gemm_tri_diag(pdof, mat, diag_h2w, work);

  return hier.tri2block(hierarchy_manipulator<P>::operation::transform, nullptr,
                        hierarchy_manipulator<P>::operation::custom_non_unitary,
                        trans_mats_.data() + 8 * pdof * pdof, work,
                        fm::intlog2(mat.nrows()), conns);
}

#ifdef ASGARD_ENABLE_DOUBLE
template class interpolation_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class interpolation_manager<float>;
#endif

} // namespace asgard
