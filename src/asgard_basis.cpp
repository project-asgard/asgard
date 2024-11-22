#include "asgard_basis.hpp"
#include "asgard_distribution.hpp" // for node_out

namespace asgard
{
// generate_multi_wavelets routine creates wavelet basis (phi_co)
// then uses these to generate the two-scale coefficients which can be
// used (outside of this routine) to construct the forward multi-wavelet
// transform
template<typename P>
std::array<fk::matrix<P>, 4> generate_multi_wavelets(int const degree)
{
  expect(degree >= 0);

  int const pdof = degree + 1;

  // These are the function outputs
  // g0,g1,h0, and h1 are two-scale coefficients
  // The returned phi_co is the wavelet basis
  // scalet_coefficients are the scaling function basis
  //   -- the scalet coefficients form the legendre basis
  //      from a monomial basis

  // hard-cording degree 0, 1, 2 (mostly for less rounding)
  if (degree <= 2)
  {
    constexpr P s2 = 1.41421356237309505;

    switch (degree)
    {
    case 0: {
      P const is2 = 1 / s2;
      fk::matrix<P> h0 = {{is2,},};
      fk::matrix<P> h1 = {{is2,},};
      fk::matrix<P> g0 = {{-is2,},};
      fk::matrix<P> g1 = {{is2,},};
      return {h0, h1, g0, g1};
    }
    case 1: {
      P const is2  = 1 / s2;
      P const is22 = 1 / (2 * s2);
      P const is6  = std::sqrt(P{6}) / 4;
      fk::matrix<P> h0 = {{is2, 0}, {-is6, is22}};
      fk::matrix<P> h1 = {{is2, 0}, {is6, is22}};
      fk::matrix<P> g0 = {{0, -is2}, {is22, is6}};
      fk::matrix<P> g1 = {{0, is2}, {-is22, is6}};
      return {h0, h1, g0, g1};
    }
    case 2: {
      P const is2  = 1 / s2;
      P const is22 = 1 / (2 * s2);
      P const is24 = 1 / (4 * s2);
      P const is6  = std::sqrt(P{6}) / 4;
      P const is30 = 15 / (P{4} * std::sqrt(P{30}));
      fk::matrix<P> h0 = {{is2, 0, 0}, {-is6, is22, 0}, {0, -is30, is24}};
      fk::matrix<P> h1 = {{is2, 0, 0}, {is6, is22, 0}, {0, is30, is24}};
      fk::matrix<P> g0 = {{0, 0, -is2}, {0, is24, is30}, {-is22, -is6, 0}};
      fk::matrix<P> g1 = {{0, 0, is2}, {0, -is24, is30}, {is22, -is6, 0}};
      return {h0, h1, g0, g1};
    }
    default:
      break;
    };
  }

  fk::matrix<P> g0(pdof, pdof);
  fk::matrix<P> g1(pdof, pdof);
  fk::matrix<P> h0(pdof, pdof);
  fk::matrix<P> h1(pdof, pdof);

  basis::canonical_integrator quad(degree);

  // those are the transposes compared to the matrices used in the rest of the code
  auto leg = basis::legendre_poly<P>(degree);
  auto wav = basis::wavelet_poly(leg, quad);

  P const  s2 = std::sqrt(P{2});
  P const is2 = P{1} / s2;

  fk::matrix<P> scalets(pdof, pdof);
  for (auto i : indexof<int>(pdof))
    for (auto j : indexof<int>(pdof))
      scalets(i, j) = s2 * leg[i][degree - j];

  fk::matrix<P> phi_co(pdof * 2, pdof);
  for (auto i : indexof<int>(pdof))
    for (auto j : indexof<int>(pdof))
      phi_co(i, j) = wav[i][degree - j];
  for (auto i : indexof<int>(pdof))
    for (auto j : indexof<int>(pdof))
      phi_co(i + pdof, j) = wav[i][pdof + degree - j];

  // Calculate Two-Scale Coefficients

  // Sums to directly generate H0, H1, G0, G1
  //  H0 and H1 are the "coarsening coefficients"
  //  These describe how two adjacent locations of a higher (finer resolution)
  //  level sum to give a lower (more coarse resolution) level coefficients
  //  G0 and G1 are the "refining or detail coefficients"
  //  These describe how lower level (more coarse resolution)
  //  is split into two higher (finer resolution) level coefficients
  //  H0 is the inner product of the scaling functions of two successive
  //   levels - thus the difference in roots
  // elem_1 is the scalet functions on (-1,0)
  // elem_2 is the scalet function of a lower level and therefore spans (-1,1)
  //  H1 is also the inner product of the scaling functions of two successive
  //   levels - thus the difference in roots
  // elem_3 is the scalet functions on (0,1)
  //  G0 is the inner product of the wavelet functions of one level
  //   with the scalet functions of a lower level
  //   - thus the difference in roots
  // elem_4 is the wavelet functions on (-1,0)
  //  G1 is also the inner product of the wavelet functions of one level
  //   with the scalet functions of a lower level
  // elem_5 is the scalet functions on (0,1)

  // if you have a function represented on a finer grid (say 2 n cells) with
  // a set of legendre coefficients per cell
  // representing the function using the n cell wavelets means multiplying by
  // the adjacent cells by G0 and G1 and adding them together
  // the remainder is formed by multiplying by H0 an H1 and adding it up
  // no we have the wavelets at level n and the corresponding remainder going up
  // on level 0, there will be just a remainders

  auto leg2 = basis::legendre_poly<P, basis::integ_range::right>(degree);
  P s = 1.0;

  for (int row = 0; row < pdof; ++row)
  {
    for (int col = 0; col < row; ++col)
    {
      P const it = quad.integrate_right(leg2[col], leg[row]);

      h1(row, col) = it;
      h0(row, col) = ((row - col) % 2 == 0) ? it : -it;
    }

    h0(row, row) = is2 / s;
    h1(row, row) = h0(row, row);

    s *= 2;
  }

  for (int row = 0; row < pdof; ++row)
  {
    for (int col = degree - row; col < pdof; ++col)
    {
      P const it = is2 * quad.integrate_right(leg2[col], wav[row] + pdof);

      g1(row, col) = it;
      g0(row, col) = ((col - row + degree) % 2 == 0) ? -it : it;
    }
  }

  P constexpr tol = (std::is_same_v<P, double>) ? 1.e-12 : 1.e-4;

  auto const normalize = [&](fk::matrix<P> &matrix) -> void {
    for (auto &m : matrix)
      if (std::abs(m) < tol)
        m = 0;
  };
  normalize(h0);
  normalize(h1);
  normalize(g0);
  normalize(g1);

  return {h0, h1, g0, g1};
}

template<typename R>
fk::matrix<R> operator_two_scale(int const degree, int const num_levels)
{
  expect(degree >= 0);
  expect(num_levels > 1);

  int const pdof      = degree + 1;
  int const max_level = fm::ipow2(num_levels);

  // this is to get around unused warnings
  // because can't unpack only some args w structured binding (until c++20)
  auto const [h0, h1, g0, g1] = generate_multi_wavelets<R>(degree);

  fk::matrix<R> fmwt(pdof * max_level, pdof * max_level);

  fk::matrix<R> const h_block = fk::matrix<R>(h0.nrows(), h0.ncols() * 2)
                                    .set_submatrix(0, 0, h0)
                                    .set_submatrix(0, h0.ncols(), h1);
  fk::matrix<R> const g_block = fk::matrix<R>(g0.nrows(), g0.ncols() * 2)
                                    .set_submatrix(0, 0, g0)
                                    .set_submatrix(0, g0.ncols(), g1);

  // set the top vertical half of fmwt along the block diagonal with h_block
  // the lower half is set in the same manner, but with g_block
  for (int i = 0; i < max_level / 2; ++i)
  {
    fmwt.set_submatrix(pdof * i, 2 * pdof * i, h_block);
    fmwt.set_submatrix(pdof * (i + max_level / 2), 2 * pdof * i, g_block);
  }

  fk::matrix<R> fmwt_comp = eye<R>(pdof * max_level);

  int const n = std::floor(std::log2(max_level));
  for (int j = 1; j <= n; j++)
  {
    fk::matrix<R> cfmwt(pdof * max_level, pdof * max_level);
    if (j == 1)
    {
      cfmwt = fmwt;
    }
    else
    {
      int const cn = fm::ipow2(n - j + 1) * pdof;

      std::fill(cfmwt.begin(), cfmwt.end(), 0.0);
      cfmwt.set_submatrix(cn, cn, eye<R>(pdof * max_level - cn));
      cfmwt.set_submatrix(
          0, 0,
          fk::matrix<R, mem_type::const_view>(fmwt, 0, cn / 2 - 1, 0, cn - 1));
      cfmwt.set_submatrix(cn / 2, 0,
                          fk::matrix<R, mem_type::const_view>(
                              fmwt, pdof * max_level / 2,
                              pdof * max_level / 2 + cn / 2 - 1, 0, cn - 1));
    }
    fmwt_comp = cfmwt * fmwt_comp;
  }
  std::transform(fmwt_comp.begin(), fmwt_comp.end(), fmwt_comp.begin(),
                 [](R &elem) { return std::abs(elem) < 1e-12 ? 0.0 : elem; });
  return fmwt_comp;
}

#ifdef ASGARD_ENABLE_DOUBLE
template std::array<fk::matrix<double>, 4>
generate_multi_wavelets(int const degree);
template fk::matrix<double>
operator_two_scale(int const degree, int const num_levels);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template std::array<fk::matrix<float>, 4>
generate_multi_wavelets(int const degree);
template fk::matrix<float>
operator_two_scale(int const degree, int const num_levels);
#endif

namespace basis
{
template<typename P, resource resrc>
wavelet_transform<P, resrc>::wavelet_transform(int const max_level_in,
                                               int const max_degree,
                                               verbosity_level verb)
    : max_level(max_level_in), degree(max_degree), dense_blocks_(max_level * 2)
{
  // this is to get around unused warnings
  // because can't unpack only some args w structured binding (until
  // c++20)
  int const pdof = degree + 1;

  auto const [h0, h1, g0, g1] = generate_multi_wavelets<P>(degree);

  int const fmwt_size = pdof * fm::ipow2(max_level);

  std::vector<fk::matrix<P>> block_builder(max_level * 2);

  fk::matrix<P> g_mat(pdof, fmwt_size);
  fk::matrix<P> h_mat = fk::matrix<P>(pdof, fmwt_size)
                            .set_submatrix(0, 0, eye<P>(pdof));

  // main loop - build the blocks with small gemms
  for (auto j = max_level - 1; j >= 0; --j)
  {
    auto const num_cells   = fm::ipow2(j);
    auto const block_ncols = fmwt_size / num_cells;
    auto const ncols_h     = block_ncols / 2;

    fk::matrix<P> h_tmp(pdof, ncols_h);
    fk::matrix<P, mem_type::view> h_view(h_mat, 0, pdof - 1, 0, ncols_h - 1);
    h_tmp = h_view;

    fk::matrix<P, mem_type::view> g_view(g_mat, 0, pdof - 1, 0, ncols_h - 1);
    fm::gemm(g0, h_tmp, g_view);

    fk::matrix<P, mem_type::view> g_view_2(g_mat, 0, pdof - 1, ncols_h,
                                           ncols_h * 2 - 1);
    fm::gemm(g1, h_tmp, g_view_2);

    fm::gemm(h0, h_tmp, h_view);

    fk::matrix<P, mem_type::view> h_view_2(h_mat, 0, pdof - 1, ncols_h,
                                           ncols_h * 2 - 1);
    fm::gemm(h1, h_tmp, h_view_2);

    fk::matrix<P, mem_type::const_view> const g_block(g_mat, 0, pdof - 1, 0,
                                                      block_ncols - 1);
    block_builder[j * 2 + 1].clear_and_resize(pdof, block_ncols) = g_block;

    fk::matrix<P, mem_type::const_view> const h_block(h_mat, 0, pdof - 1, 0,
                                                      block_ncols - 1);
    block_builder[j * 2].clear_and_resize(pdof, block_ncols) = h_block;
  }

  // how much space are we using?
  auto const num_elems =
      std::accumulate(block_builder.begin(), block_builder.end(), 0,
                      [](int64_t const sum, auto const &matrix) {
                        return sum + static_cast<int64_t>(matrix.size());
                      });

  if (verb == verbosity_level::high)
    node_out() << "  basis operator allocation (MB): " << get_MB<P>(num_elems)
               << '\n';

  // copy to device if necessary
  expect(block_builder.size() == dense_blocks_.size());
  for (auto i = 0; i < static_cast<int>(block_builder.size()); ++i)
  {
    if constexpr (resrc == resource::host)
    {
      dense_blocks_[i].clear_and_resize(block_builder[i].nrows(),
                                        block_builder[i].ncols()) =
          block_builder[i];
    }
    else
    {
      dense_blocks_[i]
          .clear_and_resize(block_builder[i].nrows(), block_builder[i].ncols())
          .transfer_from(block_builder[i]);
    }
  }
}

template<typename P, resource resrc>
template<mem_type omem>
fk::vector<P, mem_type::owner, resrc> wavelet_transform<P, resrc>::apply(
    fk::vector<P, omem, resrc> const &coefficients, int const level,
    basis::side const transform_side,
    basis::transpose const transform_trans) const
{
  expect(level >= 0);
  expect(level <= max_level);

  auto const ncols =
      transform_side == basis::side::right ? coefficients.size() : 1;
  auto const nrows =
      transform_side == basis::side::right ? 1 : coefficients.size();
  auto const as_matrix = apply(
      fk::matrix<P, mem_type::const_view, resrc>(coefficients, nrows, ncols, 0),
      level, transform_side, transform_trans);
  return fk::vector<P, mem_type::owner, resrc>(as_matrix);
}
template<typename P, resource resrc>
template<mem_type omem>
fk::matrix<P, mem_type::owner, resrc> wavelet_transform<P, resrc>::apply(
    fk::matrix<P, omem, resrc> const &coefficients, int const level,
    basis::side const transform_side,
    basis::transpose const transform_trans) const
{
  expect(level >= 0);
  if (level == 0)
  {
    return fk::matrix<P, mem_type::owner, resrc>(coefficients);
  }

  expect(level <= max_level);

  int pdof = degree + 1;

  auto const op_size = fm::ipow2(level) * pdof;
  if (transform_side == basis::side::right)
  {
    expect(coefficients.ncols() == op_size);
  }
  else
  {
    expect(coefficients.nrows() == op_size);
  }

  int const rows_y =
      transform_side == basis::side::left ? op_size : coefficients.nrows();
  int const cols_y =
      transform_side == basis::side::left ? coefficients.ncols() : op_size;
  fk::matrix<P, mem_type::owner, resrc> transformed(rows_y, cols_y);

  // first, coarsest level
  auto const do_trans =
      transform_trans == basis::transpose::trans ? true : false;
  auto const &first_block = dense_blocks_[(max_level - level) * 2];

  if (transform_side == basis::side::left)
  {
    if (transform_trans == basis::transpose::trans)
    {
      fk::matrix<P, mem_type::const_view, resrc> const B(
          coefficients, 0, pdof - 1, 0, coefficients.ncols() - 1);
      fk::matrix<P, mem_type::view, resrc> C(transformed, 0, op_size - 1, 0,
                                             coefficients.ncols() - 1);

      fm::gemm(first_block, B, C, do_trans);
    }
    else
    {
      fk::matrix<P, mem_type::const_view, resrc> const B(
          coefficients, 0, op_size - 1, 0, coefficients.ncols() - 1);
      fk::matrix<P, mem_type::view, resrc> C(transformed, 0, pdof - 1, 0,
                                             coefficients.ncols() - 1);

      fm::gemm(first_block, B, C, do_trans);
    }
  }
  else
  {
    if (transform_trans == basis::transpose::trans)
    {
      fk::matrix<P, mem_type::const_view, resrc> const A(
          coefficients, 0, coefficients.nrows() - 1, 0, op_size - 1);
      fk::matrix<P, mem_type::view, resrc> C(
          transformed, 0, coefficients.nrows() - 1, 0, pdof - 1);

      fm::gemm(A, first_block, C, false, do_trans);
    }
    else
    {
      fk::matrix<P, mem_type::const_view, resrc> const A(
          coefficients, 0, coefficients.nrows() - 1, 0, pdof - 1);
      fk::matrix<P, mem_type::view, resrc> C(
          transformed, 0, coefficients.nrows() - 1, 0, op_size - 1);

      fm::gemm(A, first_block, C, false, do_trans);
    }
  }

  // remaining levels
  auto const block_offset = (max_level - level) * 2 + 1;
  auto degree_start       = pdof;

  for (auto i = 0; i < level; ++i)
  {
    auto const num_cells = fm::ipow2(i);
    auto const cell_size = op_size / num_cells;

    auto const &current_block = dense_blocks_[block_offset + i * 2];

    P const alpha = 1.0;
    P const beta  = 1.0;

    for (auto j = 0; j < num_cells; ++j)
    {
      auto const cell_start = j * cell_size;
      auto const cell_end   = cell_start + cell_size - 1;
      auto const degree_end = degree_start + pdof - 1;

      if (transform_side == basis::side::left)
      {
        if (transform_trans == basis::transpose::trans)
        {
          fk::matrix<P, mem_type::view, resrc> C(
              transformed, cell_start, cell_end, 0, coefficients.ncols() - 1);
          fk::matrix<P, mem_type::const_view, resrc> const B(
              coefficients, degree_start, degree_end, 0,
              coefficients.ncols() - 1);

          fm::gemm(current_block, B, C, do_trans, false, alpha, beta);
        }
        else
        {
          fk::matrix<P, mem_type::view, resrc> C(transformed, degree_start,
                                                 degree_end, 0,
                                                 coefficients.ncols() - 1);
          fk::matrix<P, mem_type::const_view, resrc> const B(
              coefficients, cell_start, cell_end, 0, coefficients.ncols() - 1);

          fm::gemm(current_block, B, C, do_trans, false, alpha, beta);
        }
      }
      else
      {
        if (transform_trans == basis::transpose::trans)
        {
          fk::matrix<P, mem_type::view, resrc> C(transformed, 0,
                                                 coefficients.nrows() - 1,
                                                 degree_start, degree_end);
          fk::matrix<P, mem_type::const_view, resrc> const A(
              coefficients, 0, coefficients.nrows() - 1, cell_start, cell_end);
          fm::gemm(A, current_block, C, false, do_trans, alpha, beta);
        }
        else
        {
          fk::matrix<P, mem_type::view, resrc> C(
              transformed, 0, coefficients.nrows() - 1, cell_start, cell_end);
          fk::matrix<P, mem_type::const_view, resrc> const A(
              coefficients, 0, coefficients.nrows() - 1, degree_start,
              degree_end);

          fm::gemm(A, current_block, C, false, do_trans, alpha, beta);
        }
      }
      degree_start = degree_end + 1;
    }
  }

  return transformed;
}

#ifdef ASGARD_ENABLE_DOUBLE
template class wavelet_transform<double, resource::host>;
#ifdef ASGARD_USE_CUDA
template class wavelet_transform<double, resource::device>;
#endif

template fk::vector<double, mem_type::owner, resource::host>
wavelet_transform<double, resource::host>::apply(
    fk::vector<double, mem_type::owner, resource::host> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::vector<double, mem_type::owner, resource::host>
wavelet_transform<double, resource::host>::apply(
    fk::vector<double, mem_type::const_view, resource::host> const
        &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::matrix<double, mem_type::owner, resource::host>
wavelet_transform<double, resource::host>::apply(
    fk::matrix<double, mem_type::owner, resource::host> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
#ifdef ASGARD_USE_CUDA
template fk::vector<double, mem_type::owner, resource::device>
wavelet_transform<double, resource::device>::apply(
    fk::vector<double, mem_type::owner, resource::device> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::matrix<double, mem_type::owner, resource::device>
wavelet_transform<double, resource::device>::apply(
    fk::matrix<double, mem_type::const_view, resource::device> const
        &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::matrix<double, mem_type::owner, resource::device>
wavelet_transform<double, resource::device>::apply(
    fk::matrix<double, mem_type::owner, resource::device> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
#endif
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class wavelet_transform<float, resource::host>;
#ifdef ASGARD_USE_CUDA
template class wavelet_transform<float, resource::device>;
#endif

template fk::vector<float, mem_type::owner, resource::host>
wavelet_transform<float, resource::host>::apply(
    fk::vector<float, mem_type::owner, resource::host> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;

template fk::vector<float, mem_type::owner, resource::host>
wavelet_transform<float, resource::host>::apply(
    fk::vector<float, mem_type::const_view, resource::host> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::matrix<float, mem_type::owner, resource::host>
wavelet_transform<float, resource::host>::apply(
    fk::matrix<float, mem_type::owner, resource::host> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::matrix<float, mem_type::owner, resource::host>
wavelet_transform<float, resource::host>::apply(
    fk::matrix<float, mem_type::const_view, resource::host> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;

#ifdef ASGARD_USE_CUDA
template fk::vector<float, mem_type::owner, resource::device>
wavelet_transform<float, resource::device>::apply(
    fk::vector<float, mem_type::owner, resource::device> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::vector<float, mem_type::owner, resource::device>
wavelet_transform<float, resource::device>::apply(
    fk::vector<float, mem_type::const_view, resource::device> const
        &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::matrix<float, mem_type::owner, resource::device>
wavelet_transform<float, resource::device>::apply(
    fk::matrix<float, mem_type::owner, resource::device> const &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
template fk::matrix<float, mem_type::owner, resource::device>
wavelet_transform<float, resource::device>::apply(
    fk::matrix<float, mem_type::const_view, resource::device> const
        &coefficients,
    int const level, basis::side const transform_side,
    basis::transpose const transform_trans) const;
#endif
#endif

} // namespace basis
} // namespace asgard
