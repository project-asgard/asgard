#include "asgard_interpolation1d.hpp"

#include "asgard_wavelet_basis.hpp"

namespace asgard
{
#ifdef KRON_MODE_GLOBAL

/*
 * The 0 level basis (Legendre polynomials or simple Lagrane ones)
 * has support over the whole domain and it just linear functions.
 * But all wavelets (projection or interpolation) have either a kinck
 * or a dicontinuity at the middle of the domain.
 * The inteprolation wavelets are also zero on half of the domain.
 * Therefore, we need to consider intergation in two sub-domains,
 * labeled L and R (left-right).
 *
 * Linear times linear functions give quadratic, 2 points
 * of Gauss-Legendre is enough to integrate up to cubic basis.
 *
 * The other trick is when the two basis have partial overlap,
 * then integration is done over the domain of the shorter support,
 * taking that to be canonical (0, 1), which is transformed
 * into the higher support with xl + scale * x_canonical,
 * where xl is the left-point and scale is the ratio of
 * the support of the two functions (scale > 1).
 *
 * At scale == 1, the domains overlap and no transformation is needed.
 *
 * The scaling is done outside of this class.
 * Naturally, all integrals and and basis functions have to be
 * scaled to the appropriate size. The integrals are scaled
 * by the support of the smaller functions and the projection
 * basis is scaled by sqrt(2.0)^level.
 *
 * Domain transformation and how to compute xl and scale:
 *
 * Assume that we have cells r and c (row/column but the argument
 * is symmetic and can be applied in reverse).
 * Also assume that size-r < size-c and r is inside c.
 * We start with a canonical variable over r, call that \xi \in (0, 1),
 * and we want to translate that to a canonical variable over cell c.
 * We call the second variable z \in (0, 1).
 * The reason for the translation is because we only have formulas
 * for the basis functions over the canonical cell [0, 1].
 *
 * Over cell r, we integrate using canonical points and basis,
 * but the canonical quadrature points translate to real
 * points using:
 *   x = rl + dr * \xi
 * where rl is the left (real) end of the cell, dr is the sell size,
 * and \xi \in (0, 1) is the canonical variable.
 * Over cell c, a real point translates to a canonical (0, 1)
 * with the inverse trnasfomation, i.e.,
 *   z = (x - cl) / dc
 * where cl is the left end of the cell c, dc is the size
 * and z \in (0, 1) is the canonical variable over cell c.
 * Substituting x from the first formula, we have
 *   z = (rl - cl) / dc + (dr / dc) * \xi
 * Therefore,
 *   xl    = (rl - cl) / dc
 *   scale = (dr / dc)
 */
template<typename P>
struct linear_integrator
{
  static constexpr auto pleg0  = basis::linear<P>::pleg0;
  static constexpr auto pleg1  = basis::linear<P>::pleg1;
  static constexpr auto pwav0L = basis::linear<P>::pwav0L;
  static constexpr auto pwav0R = basis::linear<P>::pwav0R;
  static constexpr auto pwav1L = basis::linear<P>::pwav1L;
  static constexpr auto pwav1R = basis::linear<P>::pwav1R;
  static constexpr auto ibas0  = basis::linear<P>::ibas0;
  static constexpr auto ibas1  = basis::linear<P>::ibas1;
  static constexpr auto iwav0L = basis::linear<P>::iwav0L;
  static constexpr auto iwav1R = basis::linear<P>::iwav1R;

  // Gauss-Legendre quadrature on (0, 1), 2 points
  static P constexpr x0 = 0.21132486540518712;
  static P constexpr x1 = 0.78867513459481288;
  static P constexpr w  = 0.5;
  // Gauss-Legendre quadrature on (0, 0.5)
  static P constexpr x0L = 0.5 * x0;
  static P constexpr x1L = 0.5 * x1;
  static P constexpr wL  = 0.5 * w;
  // Gauss-Legendre quadrature on (0.5, 1)
  static P constexpr x0R = 0.5 + x0L;
  static P constexpr x1R = 0.5 + x1L;
  static P constexpr wR  = wL;

  // level 0 vs. level 0
  static constexpr void lb00(P mat[])
  {
    mat[0] = w * (pleg0(x0) * ibas0(x0) + pleg0(x1) * ibas0(x1));
    mat[1] = w * (pleg1(x0) * ibas0(x0) + pleg1(x1) * ibas0(x1));
    mat[2] = w * (pleg0(x0) * ibas1(x0) + pleg0(x1) * ibas1(x1));
    mat[3] = w * (pleg1(x0) * ibas1(x0) + pleg1(x1) * ibas1(x1));
  }
  // projection legendre basis (level 0) vs. interpolation level 1
  static constexpr void li01(P mat[])
  {
    mat[0] = wL * (pleg0(x0L) * iwav0L(x0L) + pleg0(x1L) * iwav0L(x1L));
    mat[1] = wL * (pleg1(x0L) * iwav0L(x0L) + pleg1(x1L) * iwav0L(x1L));
    mat[2] = wR * (pleg0(x0R) * iwav1R(x0R) + pleg0(x1R) * iwav1R(x1R));
    mat[3] = wR * (pleg1(x0R) * iwav1R(x0R) + pleg1(x1R) * iwav1R(x1R));
  }
  // projection legendre basis (level 0) vs. interpolation level x
  static constexpr void li0x(P xl, P scale, std::array<P, 4> &mat)
  {
    mat[0] = wL * (pleg0(xl + scale * x0L) * iwav0L(x0L) + pleg0(xl + scale * x1L) * iwav0L(x1L));
    mat[1] = wL * (pleg1(xl + scale * x0L) * iwav0L(x0L) + pleg1(xl + scale * x1L) * iwav0L(x1L));
    mat[2] = wR * (pleg0(xl + scale * x0R) * iwav1R(x0R) + pleg0(xl + scale * x1R) * iwav1R(x1R));
    mat[3] = wR * (pleg1(xl + scale * x0R) * iwav1R(x0R) + pleg1(xl + scale * x1R) * iwav1R(x1R));
  }
  // projection basis (level 1) vs. interpolation level 0
  static constexpr void wb10(P mat[])
  {
    std::fill_n(mat, 4, P{0});
    // the wavelets on level 1 are orthogonal to any linear function that spans the whole domain
  }
  // projection basis (level x) vs. interpolation level 0
  static constexpr void wbx0(P xl, P scale, std::array<P, 4> &mat)
  {
    mat[0] = wL * (pwav0L(x0L) * ibas0(xl + scale * x0L) + pwav0L(x1L) * ibas0(xl + scale * x1L)) //
             + wR * (pwav0R(x0R) * ibas0(xl + scale * x0R) + pwav0R(x1R) * ibas0(xl + scale * x1R));
    mat[1] = wL * (pwav1L(x0L) * ibas0(xl + scale * x0L) + pwav1L(x1L) * ibas0(xl + scale * x1L)) //
             + wR * (pwav1R(x0R) * ibas0(xl + scale * x0R) + pwav1R(x1R) * ibas0(xl + scale * x1R));
    mat[2] = wL * (pwav0L(x0L) * ibas1(xl + scale * x0L) + pwav0L(x1L) * ibas1(xl + scale * x1L)) //
             + wR * (pwav0R(x0R) * ibas1(xl + scale * x0R) + pwav0R(x1R) * ibas1(xl + scale * x1R));
    mat[3] = wL * (pwav1L(x0L) * ibas1(xl + scale * x0L) + pwav1L(x1L) * ibas1(xl + scale * x1L)) //
             + wR * (pwav1R(x0R) * ibas1(xl + scale * x0R) + pwav1R(x1R) * ibas1(xl + scale * x1R));
  }
  // wavelet-wavelet, matching support
  static constexpr void wixx(std::array<P, 4> &mat)
  {
    mat[0] = wL * (pwav0L(x0L) * iwav0L(x0L) + pwav0L(x1L) * iwav0L(x1L));
    mat[1] = wL * (pwav1L(x0L) * iwav0L(x0L) + pwav1L(x1L) * iwav0L(x1L));
    mat[2] = wR * (pwav0R(x0R) * iwav1R(x0R) + pwav0R(x1R) * iwav1R(x1R));
    mat[3] = wR * (pwav1R(x0R) * iwav1R(x0R) + pwav1R(x1R) * iwav1R(x1R));
  }
  // wavelet-wavelet, projection has larger support
  static constexpr void wip(P xl, P scale, std::array<P, 4> &mat)
  {
    if (xl + P{0.25} * scale < 0.5) // interp-basis is on the left of the proj. basis
    {
      mat[0] = wL * (pwav0L(xl + scale * x0L) * iwav0L(x0L) + pwav0L(xl + scale * x1L) * iwav0L(x1L));
      mat[1] = wL * (pwav1L(xl + scale * x0L) * iwav0L(x0L) + pwav1L(xl + scale * x1L) * iwav0L(x1L));
      mat[2] = wR * (pwav0L(xl + scale * x0R) * iwav1R(x0R) + pwav0L(xl + scale * x1R) * iwav1R(x1R));
      mat[3] = wR * (pwav1L(xl + scale * x0R) * iwav1R(x0R) + pwav1L(xl + scale * x1R) * iwav1R(x1R));
    }
    else
    {
      mat[0] = wL * (pwav0R(xl + scale * x0L) * iwav0L(x0L) + pwav0R(xl + scale * x1L) * iwav0L(x1L));
      mat[1] = wL * (pwav1R(xl + scale * x0L) * iwav0L(x0L) + pwav1R(xl + scale * x1L) * iwav0L(x1L));
      mat[2] = wR * (pwav0R(xl + scale * x0R) * iwav1R(x0R) + pwav0R(xl + scale * x1R) * iwav1R(x1R));
      mat[3] = wR * (pwav1R(xl + scale * x0R) * iwav1R(x0R) + pwav1R(xl + scale * x1R) * iwav1R(x1R));
    }
  }
  // wavelet-wavelet, interpolation has larger support
  static constexpr void wii(P xl, P scale, std::array<P, 4> &mat)
  {
    if (xl + P{0.25} * scale < 0.5) // proj-basis is on the left of the interp. basis
    {
      mat[0] = wL * (pwav0L(x0L) * iwav0L(xl + scale * x0L) + pwav0L(x1L) * iwav0L(xl + scale * x1L)) //
               + wR * (pwav0R(x0R) * iwav0L(xl + scale * x0R) + pwav0R(x1R) * iwav0L(xl + scale * x1R));
      mat[1] = wL * (pwav1L(x0L) * iwav0L(xl + scale * x0L) + pwav1L(x1L) * iwav0L(xl + scale * x1L)) //
               + wR * (pwav1R(x0R) * iwav0L(xl + scale * x0R) + pwav1R(x1R) * iwav0L(xl + scale * x1R));
      mat[2] = 0; // the left part of the right interp. wavelet is zero
      mat[3] = 0;
    }
    else // proj-basis is on the right of the interp basis
    {
      mat[0] = 0; // right part of the left interp. wavelet is zero
      mat[1] = 0;
      mat[2] = wL * (pwav0L(x0L) * iwav1R(xl + scale * x0L) + pwav0L(x1L) * iwav1R(xl + scale * x1L)) //
               + wR * (pwav0R(x0R) * iwav1R(xl + scale * x0R) + pwav0R(x1R) * iwav1R(xl + scale * x1R));
      mat[3] = wL * (pwav1L(x0L) * iwav1R(xl + scale * x0L) + pwav1L(x1L) * iwav1R(xl + scale * x1L)) //
               + wR * (pwav1R(x0R) * iwav1R(xl + scale * x0R) + pwav1R(x1R) * iwav1R(xl + scale * x1R));
    }
  }
};

template<int order, typename precision>
void wavelet_interp1d<order, precision>::make_wavelet_wmat0(
    std::array<precision, pterms> const &x, precision mat[])
{
  if constexpr (order == 1)
  {
    mat[0] = basis::linear<precision>::pleg0(x[0]);
    mat[1] = basis::linear<precision>::pleg0(x[1]);
    mat[2] = basis::linear<precision>::pleg1(x[0]);
    mat[3] = basis::linear<precision>::pleg1(x[1]);
  }
}

template<int order, typename precision>
void wavelet_interp1d<order, precision>::make_wavelet_imat0(
    std::array<precision, pterms> const &x, precision mat[])
{
  if constexpr (order == 1)
  {
    mat[0] = -basis::linear<precision>::ibas0(x[0]);
    mat[1] = -basis::linear<precision>::ibas0(x[1]);
    mat[2] = -basis::linear<precision>::ibas1(x[0]);
    mat[3] = -basis::linear<precision>::ibas1(x[1]);
  }
}

template<int order, typename precision>
void wavelet_interp1d<order, precision>::make_wavelet_wmat(
    std::array<precision, pterms> const &x, precision mat[])
{
  if constexpr (order == 1)
  {
    for (int i = 0; i < 2; i++)
      if (x[i] < 0.5) // using the left sections of the wavelets
      {
        mat[i]     = basis::linear<precision>::pwav0L(x[i]);
        mat[i + 2] = basis::linear<precision>::pwav1L(x[i]);
      }
      else // using the right section
      {
        mat[i]     = basis::linear<precision>::pwav0R(x[i]);
        mat[i + 2] = basis::linear<precision>::pwav1R(x[i]);
      }
  }
}

template<int order, typename precision>
void wavelet_interp1d<order, precision>::make_wavelet_imat(
    std::array<precision, pterms> const &x, precision mat[])
{
  if constexpr (order == 1)
  {
    for (int i = 0; i < 2; i++)
      if (x[i] < 0.5) // using the left sections of the wavelets
      {
        mat[i]     = -basis::linear<precision>::iwav0L(x[i]);
        mat[i + 2] = -basis::linear<precision>::iwav1L(x[i]);
      }
      else // using the right section
      {
        mat[i]     = -basis::linear<precision>::iwav0R(x[i]);
        mat[i + 2] = -basis::linear<precision>::iwav1R(x[i]);
      }
  }
}

// the algorithms behind the 3 prepare methods are similar
// the ideal is to allocate the matrix and write over the blocks
// the goal is to do so sequentially with minimal indexing work
template<int order, typename precision>
void wavelet_interp1d<order, precision>::prepare_proj2node()
{
  proj2node_.resize(conn->num_connections() * matsize);
  auto em = proj2node_.data();

  // functions 0/1 at nodes 0/1
  std::array<precision, matsize> local_mat;
  std::array<precision, pterms> x;
  std::array<precision, pterms> nx;

  for (int row = 0; row < conn->num_rows(); row++)
  {
    std::copy_n(nodes_.begin() + pterms * row, pterms, x.begin());

    // cells 0 and 1 are always connected
    make_wavelet_wmat0(x, em);
    em += matsize;
    make_wavelet_wmat(x, em);
    em += matsize;

    int lbegin = 2; // first cell on each level

    precision scale     = s2;
    precision cell_size = 0.5;

    // using the wavelet functions
    int const row_start = conn->row_begin(row) + 2;
    int const row_end   = conn->row_end(row);
    for (int c = row_start; c < row_end; c++)
    {
      int const col = (*conn)[c]; // connected cell

      // move to the next level
      while (col >= 2 * lbegin)
      {
        lbegin *= 2;
        scale *= s2;
        cell_size *= 0.5;
      }

      precision xl = cell_size * (col - lbegin);

#pragma omp simd
      for (int i = 0; i < pterms; i++)
        nx[i] = (x[i] - xl) / cell_size;

      make_wavelet_wmat(nx, local_mat.data());
      for (int i = 0; i < pterms; i++)
        if (nx[i] < precision{0} or nx[i] > precision{1})
          for (int j = 0; j < pterms; j++)
            local_mat[i + j * pterms] = precision{0};
        else
          for (int j = 0; j < pterms; j++)
            local_mat[i + j * pterms] *= scale;

      em = std::copy(local_mat.begin(), local_mat.end(), em);
    }
  }
}

template<int order, typename precision>
void wavelet_interp1d<order, precision>::prepare_node2hier()
{
  node2hier_.resize(conn->num_connections() * matsize);

  // computing the hierarchical coefficients uses only the lower triangular
  // part of the connectivity pattern with implicit identity blocks
  // along the diagona

  // functions 0/1 at nodes 0/1
  std::array<precision, matsize> local_mat;
  std::array<precision, pterms> x;
  std::array<precision, pterms> nx;

  // row zero is irrelevant (won't be used)
  auto em = node2hier_.data() + matsize * conn->row_begin(1);

  // row one is trivial as it contains only one block
  std::copy_n(nodes_.begin() + pterms, pterms, x.begin());
  make_wavelet_imat0(x, em);
  // jump to row 2
  em = node2hier_.data() + matsize * conn->row_begin(2);

  for (int row = 2; row < conn->num_rows(); row++)
  {
    std::copy_n(nodes_.begin() + row * pterms, pterms, x.begin());

    // cell 0 is always connected
    make_wavelet_imat0(x, em);
    em += matsize;

    // cell 1 is always connected
    make_wavelet_imat(x, em);
    em += matsize;

    int lbegin = 2; // level begin, first cell on each level

    precision cell_size = 0.5;

    int const row_start = conn->row_begin(row) + 2;
    int const row_diag  = conn->row_diag(row);
    for (int c = row_start; c < row_diag; c++)
    {
      int const col = (*conn)[c]; // connected cell

      // if reached the next leve, adjust constants
      if (col >= 2 * lbegin)
      {
        lbegin *= 2;      // adjust starting index
        cell_size *= 0.5; // cells shrink in size
      }

      precision xl = cell_size * (col - lbegin);

#pragma omp simd
      for (int i = 0; i < pterms; i++)
        nx[i] = (x[i] - xl) / cell_size;

      make_wavelet_imat(nx, local_mat.data());
      for (int i = 0; i < pterms; i++)
        if (nx[i] < precision{0} or nx[i] > precision{1})
          for (int j = 0; j < pterms; j++)
            local_mat[i + j * pterms] = precision{0};

      em = std::copy(local_mat.begin(), local_mat.end(), em);
    }

    // ignore the rest of the pattern, won't be used
    em += matsize * (conn->row_end(row) - row_diag);
  }
  expect(em == (node2hier_.data() + node2hier_.size()));
}

template<int order, typename precision>
void wavelet_interp1d<order, precision>::prepare_hier2proj()
{
  hier2proj_.resize(conn->num_connections() * matsize);

  // unscaled values for the integrals
  std::array<precision, matsize> local_mat;

  // row zero, first two cells have global support
  linear_integrator<precision>::lb00(hier2proj_.data());
  linear_integrator<precision>::li01(hier2proj_.data() + matsize);

  auto em = hier2proj_.data() + 2 * matsize;

  // if a block is already computed in local_mat, scale by s and add to
  // the hier2proj using the iterator em
  auto add_block = [&](precision s)
      -> void {
#pragma omp simd
    for (int i = 0; i < matsize; i++)
      em[i] = local_mat[i] * s;
    em += matsize;
  };

  int lbegin = 2; // level begin index

  precision w = 0.5; // quadrature scale

  int row_start = conn->row_begin(0) + 2; // remainder of the row
  int row_end   = conn->row_end(0);
  for (int c = row_start; c < row_end; c++)
  {
    int const col = (*conn)[c]; // connected cell

    if (col >= 2 * lbegin)
    {
      lbegin *= 2; // reset the index of first cell in the level
      w *= 0.5;    // effective size of integration domain drops by 2
    }

    precision xl = (col - lbegin) * w;

    linear_integrator<precision>::li0x(xl, w, local_mat);
    add_block(w);
  }

  // row one, first 2 cells use full support (no-scale)
  linear_integrator<precision>::wb10(em);
  em += matsize;

  linear_integrator<precision>::wixx(local_mat);
  em = std::copy(local_mat.begin(), local_mat.end(), em);

  lbegin = 2;
  w      = 0.5; // quadrature scale

  for (int c = conn->row_begin(1) + 2; c < conn->row_end(1); c++)
  {
    int const col = (*conn)[c]; // connected cell

    if (col >= 2 * lbegin)
    {
      lbegin *= 2; // reset the index of first cell in the level
      w *= 0.5;    // effective size of integration domain drops by 2
    }

    precision xl = (col - lbegin) * w;

    linear_integrator<precision>::wip(xl, w, local_mat);
    add_block(w);
  }

  // rows 2 and onwards
  int pbegin   = 2;   // index of the level of proj. functions
  precision ps = s2;  // projection basis function scale
  precision pw = 0.5; // proj. domain scale

  for (int row = 2; row < conn->num_rows(); row++)
  {
    if (row >= 2 * pbegin)
    {
      pbegin *= 2;
      ps *= s2;
      pw *= 0.5;
    }

    // 1. handle functions larger than this one
    precision rl = pw * (row - pbegin);

    // starting the row with cells with larger support, the intergation
    // will happen over the cell of the row, hence, the integration
    // scale factor will be the length of the row cell times the scale
    // of the row (projection) function
    precision iscale = ps * pw;

    linear_integrator<precision>::wbx0(rl, pw, local_mat);
    add_block(iscale);

    linear_integrator<precision>::wii(rl, pw, local_mat);
    add_block(iscale);

    lbegin = 2;
    w      = 0.5; // quadrature scale for the columns

    for (int c = conn->row_begin(row) + 2; c < conn->row_diag(row); c++)
    {
      int const col = (*conn)[c]; // connected cell
      if (col >= 2 * lbegin)
      {
        lbegin *= 2; // reset the index of first cell in the level
        w *= 0.5;    // effective size of integration domain drops by 2
      }

      precision cl = (col - lbegin) * w;

      linear_integrator<precision>::wii((rl - cl) / w, pw / w, local_mat);
      add_block(iscale);
    }

    // 2. the self-connection is handled here
    linear_integrator<precision>::wixx(local_mat);
    add_block(iscale);

    // 3. handle functions with smaller support
    // now switching the logic, the row cell will be the larger cell
    // there's only one connection at this level, so updating the variables
    lbegin *= 2;
    w *= 0.5;

    for (int c = conn->row_diag(row) + 1; c < conn->row_end(row); c++)
    {
      int const col = (*conn)[c]; // connected cell
      if (col >= 2 * lbegin)
      {
        lbegin *= 2; // reset the index of first cell in the level
        w *= 0.5;    // effective size of integration domain drops by 2
      }

      precision cl = (col - lbegin) * w;

      linear_integrator<precision>::wip((cl - rl) / pw, w / pw, local_mat);
      add_block(w * ps);
    }
  } // done with this row
}

template<int order, typename precision>
void wavelet_interp1d<order, precision>::cache_nodes()
{
  if constexpr (order == 1)
  {
    int const mlevel = conn->max_loaded_level();
    nodes_.resize(pterms * conn->num_rows());

    nodes_[0] = precision{1.0} / precision{3.0};
    nodes_[1] = precision{2.0} / precision{3.0};
    if (mlevel == 0)
      return;

    nodes_[2] = precision{1.0} / precision{6.0};
    nodes_[3] = precision{5.0} / precision{6.0};

    int num_points = 2; // number of points on level 1

    auto em = nodes_.begin() + 4;

    precision num  = 0.0;       // numerator
    precision step = 1.0 / 6.0; // denominator
    for (int l = 2; l <= mlevel; l++)
    {
      num_points *= 2; // double the level above
      step *= 0.5;     // denominator drops by factor 2
      num = 1;
      for (int p = 0; p < num_points; p += 2)
      {
        *em++ = num * step;
        num += 4;
        *em++ = num * step;
        num += 2;
      }
    }
    expect(em == nodes_.end());
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template class wavelet_interp1d<1, double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class wavelet_interp1d<1, float>;
#endif

#endif // KRON_MODE_GLOBAL
} // namespace asgard
