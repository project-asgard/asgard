#include "asgard_transformations.hpp"

#include "asgard_small_mats.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace asgard
{

template<typename P>
legendre_basis<P>::legendre_basis(int degree) : pdof(degree + 1) {

  auto const quad_vals = legendre_weights(degree, -1.0, 1.0);

  num_quad = quad_vals[0].size();

  auto [lP_L, lPP_L] = legendre_vals({-1.0}, degree);
  auto [lP_R, lPP_R] = legendre_vals({+1.0}, degree);

  auto [lP, lPP] = legendre_vals(quad_vals[0], degree);

  // we need to keep, the quadrature points and weights, 4 matrices corresponding
  // to the edge fluxes on the left and right, 2 matrices corresponding to
  // the values of the legendre polynomials and derivatives at the quadrature points
  data_.resize(2 * num_quad + 4 * pdof * pdof + 3 * pdof * num_quad + 2 * pdof);

  { // create a scope, so that d is not visible outside of this scope, it is a temp variable
    P *d = data_.data();
    qp = std::exchange(d, d + num_quad); // dedicate num_quad entries to qp
    qw = std::exchange(d, d + num_quad);

    to_left    = std::exchange(d, d + pdof * pdof);
    from_left  = std::exchange(d, d + pdof * pdof);
    from_right = std::exchange(d, d + pdof * pdof);
    to_right   = std::exchange(d, d + pdof * pdof);

    leg  = std::exchange(d, d + pdof * num_quad);
    legw = std::exchange(d, d + pdof * num_quad);
    der  = std::exchange(d, d + pdof * num_quad);

    leg_left  = std::exchange(d, d + pdof);
    leg_right = std::exchange(d, d + pdof);

    expect(static_cast<size_t>(std::distance(data_.data(), d)) == data_.size());
  }

  // copy the values returned by legendre into the locals
  std::copy_n(quad_vals[0].data(), num_quad, qp);
  std::copy_n(quad_vals[1].data(), num_quad, qw);

  if constexpr (std::is_same_v<P, double>)
  {
    smmat::gemm_outer_inc(pdof, lP_L.data(), lP_L.data(), to_left);
    smmat::gemm_outer_inc(pdof, lP_L.data(), lP_R.data(), from_left);
    smmat::gemm_outer_inc(pdof, lP_R.data(), lP_L.data(), from_right);
    smmat::gemm_outer_inc(pdof, lP_R.data(), lP_R.data(), to_right);

    std::copy_n(lP.data(), num_quad * pdof, leg);
    smmat::col_scal(num_quad, pdof, P{0.5}, qw, leg, legw);
    std::copy_n(lPP.data(), num_quad * pdof, der);
  }
  else
  {
    std::vector<double> ddata(4 * pdof * pdof + 3 * pdof * num_quad);

    double *d = ddata.data();

    smmat::gemm_outer_inc(pdof, lP_L.data(), lP_L.data(), std::exchange(d, d + pdof * pdof));
    smmat::gemm_outer_inc(pdof, lP_L.data(), lP_R.data(), std::exchange(d, d + pdof * pdof));
    smmat::gemm_outer_inc(pdof, lP_R.data(), lP_L.data(), std::exchange(d, d + pdof * pdof));
    smmat::gemm_outer_inc(pdof, lP_R.data(), lP_R.data(), std::exchange(d, d + pdof * pdof));

    std::copy_n(lP.data(), num_quad * pdof, std::exchange(d, d + pdof * num_quad));
    smmat::col_scal(num_quad, pdof, 0.5, quad_vals[1].data(),
                    ddata.data() + 4 * pdof * pdof,
                    std::exchange(d, d + pdof * num_quad));
    std::copy_n(lPP.data(), num_quad * pdof, std::exchange(d, d + pdof * num_quad));

    std::copy(ddata.begin(), ddata.end(), to_left);
  }

  std::copy_n(lP_L.data(), pdof, leg_left);
  std::copy_n(lP_R.data(), pdof, leg_right);
}

template<typename P>
void legendre_basis<P>::interior_quad(
    P xleft, P xright, int level, std::vector<P> &pnts)
{
  int const num_cells = fm::ipow2(level);

  P const dx = (xright - xleft) / num_cells;

  pnts.resize(num_cells * num_quad);

  #pragma omp parallel for
  for (int i = 0; i < num_cells; i++) {
    P const l = xleft + i * dx; // left edge of cell i
    for (int k = 0; k < num_quad; k++)
      pnts[i * num_quad + k] = (0.5 * qp[k] + 0.5) * dx + l;
  }
}

template<typename P>
std::vector<P> legendre_basis<P>::project(
    bool is_interior, int level, P dsqrt, P alpha, std::vector<P> const &raw_data) const
{
  alpha *= dsqrt * fm::powi(P{0.707106781186547}, level);
  int const num_cells = fm::ipow2(level);

  span2d<P const> raw;
  if (is_interior)
    raw = span2d<P const>(num_quad, num_cells, raw_data.data());
  else
    raw = span2d<P const>(num_quad + 1, num_cells, raw_data.data() + 1);

  std::vector<P> lgn(num_cells * pdof);
  span2d<P> leg_basis(pdof, num_cells, lgn.data());

  #pragma omp parallel for
  for (int i = 0; i < num_cells; i++) {
    smmat::gemtv(num_quad, pdof, legw, raw[i], leg_basis[i]);
    smmat::scal(pdof, alpha, leg_basis[i]);
  }

  return lgn;
}

template<typename P>
std::vector<P> legendre_basis<P>::project(int level, P dsqrt, P alpha) const
{
  alpha *= dsqrt * fm::powi(P{0.707106781186547}, level);

  int const num_cells = fm::ipow2(level);

  std::vector<P> lgn(num_cells * pdof);

  #pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
    lgn[i * pdof] = alpha;

  return lgn;
}

template<typename P>
std::vector<P> legendre_basis<P>::project(
    bool is_interior, int level, P dsqrt, std::vector<P> const &raw_data1, std::vector<P> &raw_data2) const
{
  dsqrt *= fm::powi(P{0.707106781186547}, level);
  int const num_cells = fm::ipow2(level);

  span2d<P const> raw1;
  if (is_interior)
    raw1 = span2d<P const>(num_quad, num_cells, raw_data1.data());
  else
    raw1 = span2d<P const>(num_quad + 1, num_cells, raw_data1.data() + 1);

  span2d<P> raw2;
  if (is_interior)
    raw2 = span2d<P>(num_quad, num_cells, raw_data2.data());
  else
    raw2 = span2d<P>(num_quad + 1, num_cells, raw_data2.data() + 1);

  std::vector<P> lgn(num_cells * pdof);
  span2d<P> leg_basis(pdof, num_cells, lgn.data());

  #pragma omp parallel for
  for (int i = 0; i < num_cells; i++) {
    for (int p = 0; p < num_quad; p++)
      raw2[i][p] *= raw1[i][p];
    smmat::gemtv(num_quad, pdof, legw, raw2[i], leg_basis[i]);
    smmat::scal(pdof, dsqrt, leg_basis[i]);
  }

  return lgn;
}

template<typename P>
template<data_mode action>
void hierarchy_manipulator<P>::project_separable(
    separable_func<P> const &sep,
    sparse_grid const &grid, mass_diag<P> const &mass,
    P time, P alpha, P f[]) const
{
  int const num_dims = grid.num_dims();
  for (int d : iindexof(num_dims))
  {
    if (sep.is_const(d)) {
      project1d_c(sep.cdomain(d), mass[d], d, grid.current_level(d), pf[d]);
    } else {
      project1d_f([&](std::vector<P> const &x, std::vector<P> &fx)
          -> void {
        sep.fdomain(d, x, time, fx);
      }, mass[d], d, grid.current_level(d), pf[d]);
    }
  }

  P const tmult = (sep.ftime()) ? sep.ftime()(time) : P{1};

  int const pdof = degree_ + 1;

  #pragma omp parallel for
  for (int64_t j = 0; j < grid.num_indexes(); j++)
  {
    P *proj = f + j * block_size_;

    std::array<P const *, max_num_dimensions> data1d;

    int const *idx = grid[j];
    for (int d : iindexof(num_dims))
      data1d[d] = pf[d].data() + idx[d] * pdof;

    std::array<int, max_num_dimensions> v;
    std::fill_n(v.begin(), num_dims, 0);

    int i = 0;

    bool is_in = true;
    int c = 0;
    while (is_in or c > 0)
    {
      if (is_in)
      {
        P val = tmult;
        for (int d = 0; d < num_dims; d++)
          val *= data1d[d][ v[d] ];

        c = num_dims - 1;
        v[c]++;

        if constexpr (action == data_mode::replace)
          proj[i++] = val;
        else if constexpr (action == data_mode::scal_rep)
          proj[i++] = alpha * val;
        else if constexpr (action == data_mode::increment)
          proj[i++] += val;
        else if constexpr (action == data_mode::scal_inc)
          proj[i++] += alpha * val;
      }
      else
      {
        std::fill(v.begin() + c, v.begin() + num_dims, 0);
        v[--c]++;
      }

      is_in = (v[c] < pdof);
    }
  }
}

template<typename P>
void hierarchy_manipulator<P>::reconstruct1d(
    int const nbatch, int const level, span2d<P> data) const
{
  expect(data.num_strips() == nbatch * fm::ipow2(level));
  expect(data.stride() == (degree_ + 1));

  if (level == 0)
    return; // the hierarchical form is just scaled/normalized

  switch (degree_)
  {
    case 0:
      reconstruct1d<0>(nbatch, level, data);
      break;
    case 1:
      reconstruct1d<1>(nbatch, level, data);
      break;
    default:
      reconstruct1d<-1>(nbatch, level, data);
      break;
  }
}

template<typename P>
template<int tdegree>
void hierarchy_manipulator<P>::reconstruct1d(
    int const nbatch, int level, span2d<P> data) const
{
  int const ssize = (degree_ + 1); // strip size
  P constexpr s22 = 0.5 * s2;
  P constexpr is2h = 0.5 * is2;
  P constexpr is64  = s6 / 4.0;

  auto prj1 = [&](P const left[], P const right[], P out_left[], P out_right[])
  {
    for (int b : indexof<int>(nbatch))
    {
      switch (tdegree)
      {
        case 0:
          out_left[b]  = s22 * left[b] - s22 * right[b];
          out_right[b] = s22 * left[b] + s22 * right[b];
          break;
        case 1:
          out_left[2*b]   = is2 * left[2*b] - is64 * left[2*b+1] +                    is2h * right[2*b+1];
          out_left[2*b+1] =                   is2h * left[2*b+1] - is2 * right[2*b] + is64 * right[2*b+1];

          out_right[2*b]   = is2 * left[2*b] + is64 * left[2*b+1]                    - is2h * right[2*b+1];
          out_right[2*b+1] =                 + is2h * left[2*b+1] + is2 * right[2*b] + is64 * right[2*b+1];
          break;
        default:
          smmat::gemtv(ssize, tmatup, left + b * ssize, out_left + b * ssize);
          smmat::gemtv1(ssize, tmatlev, right + b * ssize, out_left + b * ssize);
          smmat::gemtv(ssize, tmatup + ssize * ssize, left + b * ssize, out_right + b * ssize);
          smmat::gemtv1(ssize, tmatlev + ssize * ssize, right + b * ssize, out_right + b * ssize);
          break;
      };
    }
  };

  twork.resize(data.stride() * data.num_strips());
  pwork.resize(twork.size());

  span2d<P> work0(data.stride(), data.num_strips(), twork.data());
  span2d<P> work1(data.stride(), data.num_strips(), pwork.data());

  prj1(data[0], data[nbatch], work0[0], work0[nbatch]);
  --level;

  int num = 2;

  while (level > 0)
  {
#pragma omp parallel for
    for (int i = 0; i < num; i++)
      prj1(work0[i * nbatch], data[(num + i) * nbatch],
           work1[2 * i * nbatch], work1[(2 * i + 1) * nbatch]);
    std::swap(work0, work1);
    num *= 2;
    --level;
  }

  std::copy_n(work0[0], data.num_strips() * ssize, data[0]);
}

template<typename P>
void hierarchy_manipulator<P>::project1d(
    int dim, int level, std::vector<P> const &vals,
    block_diag_matrix<P> const &mass, std::vector<P> &cells) const
{
  int const num_cells = fm::ipow2(level);

  int const num_quad = quad.stride();
  int const pdof     = degree_ + 1;

  expect(vals.size() == static_cast<size_t>(num_cells * num_quad));

  cells.resize(pdof * num_cells);

  // doing the hierarchical projection, we must normalize the Legendre polynomial to unit l-2 norm
  P const scale = std::pow(is2, level + 1) * std::sqrt(dmax[dim] - dmin[dim]);

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
  {
    smmat::gemv(pdof, num_quad, leg_vals[0], vals.data() + i * num_quad,
                cells.data() + i * pdof);
    smmat::scal(pdof, scale, cells.data() + i * pdof);
  }

  if (mass)
    mass.solve(pdof, cells);
}

template<typename P>
template<int tdegree, typename hierarchy_manipulator<P>::operation op>
void hierarchy_manipulator<P>::apply_transform(P const *trans, int level, P src[], P dest[]) const
{
  if constexpr (op == operation::custom_unitary or op == operation::custom_non_unitary) {
    expect(trans != nullptr);
  }

  int const pdof = degree_ + 1; // polynomial degree of freedom

  // only used by the custom transform for level > 1
  std::vector<P> ctrans;
  P *cupper = nullptr, *clower = nullptr;

  auto last2block = [&](P const raw[], P fin[]) -> void
    {
      if constexpr (op == operation::transform) {
        if constexpr (tdegree == 0)
        {
          P constexpr s22 = 0.5 * s2;
          fin[0] = s22 * raw[0] + s22 * raw[1];
          fin[1] = -s22 * raw[0] + s22 * raw[1];
        }
        else if constexpr (tdegree == 1)
        {
          P constexpr is2h = 0.5 * is2;
          P constexpr is64 = s6 / 4.0;

          fin[0] =  is2  * raw[0]                 + is2  * raw[2];
          fin[1] = -is64 * raw[0] + is2h * raw[1] + is64 * raw[2] + is2h * raw[3];
          fin[2] =                - is2  * raw[1]                 + is2  * raw[3];
          fin[3] =  is2h * raw[0] + is64 * raw[1] - is2h * raw[2] + is64 * raw[3];
        }
        else
        {
          int const n = 2 * pdof;
          smmat::gemv(n, n, tmats.data(), raw, fin);
        }
      } else if constexpr (op == operation::custom_unitary or op == operation::custom_non_unitary) {
        int const n = 2 * pdof;
        smmat::gemv(n, n, trans, raw, fin);
      }
    };

  auto merge2blocks = [&](P const raw[], P upper[], P fin[]) -> void
    {
      if constexpr (op == operation::transform) {
        if constexpr (tdegree == 0)
        {
          P constexpr s22 = 0.5 * s2;
          P const r0 = raw[0];
          P const r1 = raw[1];
          upper[0] =  s22 * r0 + s22 * r1;
          fin[0]   = -s22 * r0 + s22 * r1;
        }
        else if constexpr (tdegree == 1)
        {
          P constexpr is2h = 0.5 * is2;
          P constexpr is64  = s6 / 4.0;
          P const r0 = raw[0];
          P const r1 = raw[1];
          P const r2 = raw[2];
          P const r3 = raw[3];
          upper[0] =  is2  * r0             + is2  * r2;
          upper[1] = -is64 * r0 + is2h * r1 + is64 * r2 + is2h * r3;
          fin[0]   =            - is2  * r1             + is2  * r3;
          fin[1]   =  is2h * r0 + is64 * r1 - is2h * r2 + is64 * r3;
        }
        else
        {
          smmat::gemv(pdof, 2 * pdof, tmatup,  raw, upper);
          smmat::gemv(pdof, 2 * pdof, tmatlev, raw, fin);
        }
      } else if constexpr (op == operation::custom_unitary or op == operation::custom_non_unitary) {
        smmat::gemv(pdof, 2 * pdof, cupper, raw, upper);
        smmat::gemv(pdof, 2 * pdof, clower, raw, fin);
      }
    };

  auto process_level = [&](int const num_upper_cells, P const raw[], P upper[], P fin[]) -> void
    {
      #pragma omp parallel for
      for (int i = 0; i < num_upper_cells; i++)
        merge2blocks(raw + 2 * pdof * i, upper + i * pdof, fin + i * pdof);
    };

  switch (level)
  {
  case 0:
    std::copy_n(src, pdof, dest);
    return;
  case 1:
    last2block(src, dest);
    return;
  default:
    break;
  }

  // there is work to be done, if using a custom transform the blocks will be
  // rearranged here to allow for better utilization of simd (hopefully)
  if constexpr (op == operation::custom_unitary or op == operation::custom_non_unitary)
  {
    int const pdof2 = pdof * pdof;
    ctrans.resize(4 * pdof2);
    cupper = ctrans.data();
    clower = ctrans.data() + 2 * pdof2;

    smmat::matrix<P const> const tansf(2 * pdof, trans);
    smmat::matrix<P> to_upper(pdof, cupper);
    smmat::matrix<P> to_lower(pdof, clower);

    for (int r = 0; r < pdof; r++) {
      for (int c = 0; c < 2 * pdof; c++) {
        to_upper(r, c) = tansf(r, c);
        to_lower(r, c) = tansf(r + pdof, c);
      }
    }
  }

  int num = fm::ipow2(level - 1); // number of cells on the current level
  twork.resize(num * pdof); // scratch space

  P *s0 = src;
  P *s1 = twork.data();

  while (--level > 0)
  {
    process_level(num, s0, s1, dest + num * pdof);
    std::swap(s0, s1);

    num /= 2; // consider the cells on the upper level
  }

  last2block(s0, dest);
}

template<typename P>
block_sparse_matrix<P>
hierarchy_manipulator<P>::diag2hierarchical(block_diag_matrix<P> const &diag,
                                            int const level,
                                            connection_patterns const &conns) const
{
  block_sparse_matrix<P> col = make_block_sparse_matrix(conns, connect_1d::hierarchy::col_volume);
  block_sparse_matrix<P> res = make_block_sparse_matrix(conns, connect_1d::hierarchy::volume);

  constexpr operation op = operation::transform;
  switch (degree_)
  {
  case 0:
    col_project_vol<0, op>(nullptr, diag, level, conns, col);
    row_project_any<0, op>(nullptr, col, level, conns, res);
    break;
  case 1:
    col_project_vol<1, op>(nullptr, diag, level, conns, col);
    row_project_any<1, op>(nullptr, col, level, conns, res);
    break;
  default:
    col_project_vol<-1, op>(nullptr, diag, level, conns, col);
    row_project_any<-1, op>(nullptr, col, level, conns, res);
    break;
  };

  return res;
}

template<typename P>
block_sparse_matrix<P>
hierarchy_manipulator<P>::tri2hierarchical(block_tri_matrix<P> const &tri,
                                           int const level,
                                           connection_patterns const &conns) const
{
  block_sparse_matrix<P> col = make_block_sparse_matrix(conns, connect_1d::hierarchy::col_full);
  block_sparse_matrix<P> res = make_block_sparse_matrix(conns, connect_1d::hierarchy::full);

  constexpr operation op = operation::transform;
  switch (degree_)
  {
  case 0:
    col_project_full<0, op>(nullptr, tri, level, conns, col);
    row_project_any<0, op>(nullptr, col, level, conns, res);
    break;
  case 1:
    col_project_full<1, op>(nullptr, tri, level, conns, col);
    row_project_any<1, op>(nullptr, col, level, conns, res);
    break;
  default:
    col_project_full<-1, op>(nullptr, tri, level, conns, col);
    row_project_any<-1, op>(nullptr, col, level, conns, res);
    break;
  };

  return res;
}

template<typename P>
template<int tdegree, typename hierarchy_manipulator<P>::operation op>
void hierarchy_manipulator<P>::col_project_full(P const *trans,
                                                block_tri_matrix<P> const &tri,
                                                int const level,
                                                connection_patterns const &conns,
                                                block_sparse_matrix<P> &sp) const
{
  expect(connect_1d::hierarchy::col_full == sp);
#ifdef _OPENMP
  int const max_threads = omp_get_max_threads();
#else
  int const max_threads = 1;
#endif
  if (static_cast<int>(colblocks.size()) < max_threads)
    colblocks.resize(max_threads);

  P constexpr s22 = 0.5 * s2;
  P constexpr is2h = 0.5 * is2;
  P constexpr is64  = s6 / 4.0;
  P const h0[4] = {is2, -is64, 0, is2h};
  P const h1[4] = {is2, is64, 0, is2h};
  P const w0[4] = {0, is2h, -is2, is64};
  P const w1[4] = {0, -is2h, is2, is64};

  // small matrices can be cached on the stack for faster access
  P cc[4], c0[4], c1[4], c2[4], c3[4];

  int const pdof  = degree_ + 1;
  int const pdof2 = pdof * pdof;

  std::vector<P> custom;
  if constexpr (op == operation::custom_unitary) {
    expect(trans != nullptr);
    if constexpr (tdegree == 0) {
      cc[0] = trans[0];
      cc[1] = trans[2];
      cc[2] = trans[1];
      cc[3] = trans[3];
    } else if constexpr (tdegree == 1) {
      c0[0] = trans[ 0]; c0[1] = trans[ 1]; c0[2] = trans[ 4]; c0[3] = trans[ 5];
      c1[0] = trans[ 8]; c1[1] = trans[ 9]; c1[2] = trans[12]; c1[3] = trans[13];
      c2[0] = trans[ 2]; c2[1] = trans[ 3]; c2[2] = trans[ 6]; c2[3] = trans[ 7];
      c3[0] = trans[10]; c3[1] = trans[11]; c3[2] = trans[14]; c3[3] = trans[15];
    } else {
      custom.resize(4 * pdof2);
      smmat::matrix<P const> transf(2 * pdof, trans);
      smmat::matrix<P> pc0(pdof, custom.data());
      smmat::matrix<P> pc1(pdof, custom.data() + pdof2);
      smmat::matrix<P> pc2(pdof, custom.data() + 2 * pdof2);
      smmat::matrix<P> pc3(pdof, custom.data() + 3 * pdof2);
      for (int r = 0; r < pdof; r++) {
        for (int c = 0; c < pdof; c++) {
          pc0(r, c) = transf(r, c);
          pc1(r, c) = transf(r, c + pdof);
          pc2(r, c) = transf(r + pdof, c);
          pc3(r, c) = transf(r + pdof, c + pdof);
        }
      }
    }
  } else if constexpr (op == operation::custom_non_unitary) {
    // in the non-unitary case, the forward and inverse transforms use different matrices
    // and we do not transpose in the application of the blocks
    expect(trans != nullptr);
    if constexpr (tdegree == 0) {
      std::copy_n(trans, 4, cc);
    } else if constexpr (tdegree == 1) {
      c0[0] = trans[ 0]; c0[1] = trans[ 1]; c0[2] = trans[ 4]; c0[3] = trans[ 5];
      c1[0] = trans[ 2]; c1[1] = trans[ 3]; c1[2] = trans[ 6]; c1[3] = trans[ 7];
      c2[0] = trans[ 8]; c2[1] = trans[ 9]; c2[2] = trans[12]; c2[3] = trans[13];
      c3[0] = trans[10]; c3[1] = trans[11]; c3[2] = trans[14]; c3[3] = trans[15];
    } else {
      custom.resize(4 * pdof2);
      smmat::matrix<P const> transf(2 * pdof, trans);
      smmat::matrix<P> pc0(pdof, custom.data());
      smmat::matrix<P> pc1(pdof, custom.data() + pdof2);
      smmat::matrix<P> pc2(pdof, custom.data() + 2 * pdof2);
      smmat::matrix<P> pc3(pdof, custom.data() + 3 * pdof2);
      for (int r = 0; r < pdof; r++) {
        for (int c = 0; c < pdof; c++) {
          pc0(r, c) = transf(r, c);
          pc1(r, c) = transf(r + pdof, c);
          pc2(r, c) = transf(r, c + pdof);
          pc3(r, c) = transf(r + pdof, c + pdof);
        }
      }
    }
  }

  // project cells left/right with index 2n and 2n+1 at level L
  // to cells n at the hierarchical level L-1, stored in out
  // upper cells at level L-1, stored in upper
  // see the block-diagonal overload too
  auto apply = [&](P const *left, P const *right, P *out, P *upper)
  {
    if constexpr (op == operation::transform) {
      if constexpr (tdegree == 0)
        *out = -s22 * (*left) + s22 * (*right);
      else if constexpr (tdegree == 1)
        smmat::gemm_pairt(2, left, w0, right, w1, out);
      else
        smmat::gemm_pairt(pdof, left, tmatlev, right, tmatlev + pdof2, out);

      if constexpr (tdegree == 0)
        *upper = s22 * (*left) + s22 * (*right);
      else if constexpr (tdegree == 1)
        smmat::gemm_pairt(2, left, h0, right, h1, upper);
      else
        smmat::gemm_pairt(pdof, left, tmatup, right, tmatup + pdof2, upper);
    } else if constexpr (op == operation::custom_unitary) {
      if constexpr (tdegree == 0)
        *upper = (*left) * cc[0] + (*right) * cc[1];
      else if constexpr (tdegree == 1)
        smmat::gemm_pairt(2, left, c0, right, c1, upper);
      else
        smmat::gemm_pairt(pdof, left, custom.data(),
                          right, custom.data() + pdof2, upper);

      if constexpr (tdegree == 0)
        *out = (*left) * cc[2] + (*right) * cc[3];
      else if constexpr (tdegree == 1)
        smmat::gemm_pairt(2, left, c2, right, c3, out);
      else
        smmat::gemm_pairt(pdof, left, custom.data() + 2 * pdof2,
                          right, custom.data() + 3 * pdof2, out);
    } else if constexpr (op == operation::custom_non_unitary) {
      if constexpr (tdegree == 0)
        *upper = (*left) * cc[0] + (*right) * cc[1];
      else if constexpr (tdegree == 1)
        smmat::gemm_pair(2, left, c0, right, c1, upper);
      else
        smmat::gemm_pair(pdof, left, custom.data(),
                         right, custom.data() + pdof2, upper);

      if constexpr (tdegree == 0)
        *out = (*left) * cc[2] + (*right) * cc[3];
      else if constexpr (tdegree == 1)
        smmat::gemm_pair(2, left, c2, right, c3, out);
      else
        smmat::gemm_pair(pdof, left, custom.data() + 2 * pdof2,
                         right, custom.data() + 3 * pdof2, out);
    }
  };

  connect_1d const &conn = conns(sp);

  int const nrows = fm::ipow2(level);

  if (nrows == 1) // special case, single cell
  {
    std::copy_n(tri.diag(0), pdof2, sp[0]);
    return;
  }

  // using 5 scratch small matrices, left/right wleft/wright that get swapped
  // also one zero matrix
  colblocks[0].resize(5 * pdof2);
  P *left   = colblocks[0].data();
  P *wleft  = left + pdof2;
  P *right  = wleft + pdof2;
  P *wright = right + pdof2;

  if (nrows == 2) // special case, two cells
  {
    // tri.lower(0) is actually the same as tri.upper(0) (must add them)
    std::copy_n(tri.lower(0), pdof2, right);
    for (int i : indexof<int>(pdof2))
      right[i] += tri.upper(0)[i];
    apply(tri.diag(0), right, sp[1], sp[0]);
    std::copy_n(tri.lower(1), pdof2, left);
    for (int i : indexof<int>(pdof2))
      left[i] += tri.upper(1)[i];
    apply(left, tri.diag(1), sp[conn.row_begin(1) + 1], sp[conn.row_begin(1)]);
    return;
  }

  // number of rows in the extened scatch space pattern
  int const lrows = fm::ipow2(level + 1) - 2; // using power series law

  // the 3 diagonals yield 2 entries per-column
  // the periodic boundary gives special cases for the first and last rows
  P *zero = wright + pdof2; // TODO: get rid of zero
  std::fill_n(zero, pdof2, P{0});
  {
    // working on row 0
    // then we carry the right-most entry and the left-most entry per level
    int orow = lrows - nrows; // out-row
    int num  = nrows; // num/nrows number of rows at this level
    int j    = conn.row_end(orow); // right-most entry, keep cycling on j
    while (conn[--j] != num - 1);
    apply(zero, tri.lower(0), sp[j], right);
    num /= 2; // move up one level
    while (conn[--j] != num); // find mid-point
    apply(tri.diag(0), tri.upper(0), sp[j], left);
    while (num > 2)
    {
      while (conn[--j] != num - 1);
      apply(zero, right, sp[j], wright);
      num /= 2;
      while (conn[--j] != num);
      apply(left, zero, sp[j], wleft);
      std::swap(left, wleft);
      std::swap(right, wright);
    }
    // last column, num == 2
    int spr = conn.row_begin(orow); // out-row-sparse-offset
    apply(left, right, sp[spr + 1], sp[spr]);
  }{
    // working on row nrow - 1
    // must work with the sparsity pattern
    int orow = lrows - 1; // out-row
    int num  = nrows; // number of rows at this level
    int j    = conn.row_end(orow); // index of entry of interest
    while (conn[--j] != num - 1);
    apply(tri.lower(nrows - 1), tri.diag(nrows - 1), sp[j], right);
    num /= 2;
    while (conn[--j] != num);
    apply(tri.upper(nrows - 1), zero, sp[j], left);
    while (num > 2)
    {
      apply(zero, right, sp[j - 1], wright);
      num /= 2;
      while (conn[--j] != num);
      apply(left, zero, sp[j], wleft);
      std::swap(left, wleft);
      std::swap(right, wright);
    }
    // last column, num == 2
    int spr = conn.row_begin(orow); // out-row-sparse-offset
    apply(left, right, sp[spr + 1], sp[spr]);
  }

  // handle the middle rows of the matrix, no need to worry about the boundary

  // the column patters is actually denser ... need to pad
  // or use a completely different algorithm ....

  int threadid = 0;
#pragma omp parallel
  {
    int tid;
#pragma omp critical
    tid = threadid++;

    // scratch space per thread-id
    colblocks[tid].resize(5 * pdof * pdof);
    P *L  = colblocks[tid].data();
    P *R  = L + pdof2;
    P *wL = R + pdof2;
    P *wR = wL + pdof2;
    P *Z  = wR + pdof2;
    std::fill_n(Z, pdof2, P{0});

#pragma omp for
    for (int r = 1; r < nrows - 1; r++)
    { // initiate new transform for this row, reduce 3 columns to 2
      // if the diagonal entry is even, i.e., 2k for some k, we need to group
      //    (0, lower) (diag, upper) -> entries at num + (k-1, k) and upper level (k-1, k)
      // if the diagonal entry is odd, i.e., 2k-1 for some k, we group
      //    (lower, diag) (upper, 0) -> entries at num + (k-1, k) and upper level (k-1, k)
      int cs  = r % 2; // cases, cs indicates if r is even or odd
      int orow = lrows - nrows + r; // out-row
      int num = nrows; // number of entries for this row
      int k   = num / 2 + ((cs == 0) ? (r / 2) : ((r + 1) / 2)); // from above (k-1, k)
      int j   = conn.row_end(orow);
      while (conn[--j] != k); // move to where j == k
      if (cs == 0) {
        apply(Z, tri.lower(r), sp[j - 1], L);
        apply(tri.diag(r), tri.upper(r), sp[j], R);
      } else {
        apply(tri.lower(r), tri.diag(r), sp[j - 1], L);
        apply(tri.upper(r), Z, sp[j], R);
      }
      num /= 2;
      // here cs becomes "column-count"
      // we have two columns next to each other and we must process
      //  (0, L), (R, 0) -> (L, R), which yields two columns again
      //  (L, R) -> L, which merges the two into one column
      // we loop until we either reach top level or the 2 columns merge into 1
      cs = 2; // column count 2
      while (num > 2)
      {
        int c = conn[j - 1]; // column for left
        // if c is even, then left/right merge into one
        // if c is odd, left/right remain split
        if (c % 2 == 0) {
          k = c / 2;
          while (conn[--j] != k); // move to where j == k
          apply(L, R, sp[j], wL);
          std::swap(L, wL);
          num /= 2;
          cs = 1;
          break;
        }
        else
        {
          k = 1 + c / 2;
          while (conn[--j] != k); // move to where j == k
          apply(Z, L, sp[j - 1], wL);
          apply(R, Z, sp[j], wR);
          std::swap(L, wL);
          std::swap(R, wR);
          num /= 2;
        }
      }
      // working with a single column located in L pointer but could be left or right cell
      // i.e., we can have (L, 0) or (0, L)
      while (num > 2)
      {
        int c = conn[j];
        k = c / 2;
        if (c % 2 == 0) { // (L, 0), L is the left entry
          while (conn[--j] != k); // move to where j == k
          apply(L, Z, sp[j], wL);
        } else { // (0, L), L is the right entry
          while (conn[--j] != k); // move to where j == k
          apply(Z, L, sp[j], wL);
        }
        std::swap(L, wL);
        num /= 2;
      }
      // working on the last two columns
      if (cs == 1) { // one column case
        if (conn[j] == 2) { // last written to column either 2 or 3
          apply(L, Z, sp[j - 1], sp[j - 2]);
        } else { // column 3
          while (conn[--j] != 1);
          apply(Z, L, sp[j], sp[j - 1]);
        }
      } else {
        while (conn[--j] != 1);
        apply(L, R, sp[j], sp[j - 1]);
      }
    } // #pragma omp for
  } // #pragma omp parallel
}

template<typename P>
template<int tdegree, typename hierarchy_manipulator<P>::operation op>
void hierarchy_manipulator<P>::col_project_vol(
    P const *trans, block_diag_matrix<P> const &diag, int const level,
    connection_patterns const &conns, block_sparse_matrix<P> &sp) const
{
  expect(connect_1d::hierarchy::col_volume == sp);
#ifdef _OPENMP
  int const max_threads = omp_get_max_threads();
#else
  int const max_threads = 1;
#endif
  if (static_cast<int>(colblocks.size()) < max_threads)
    colblocks.resize(max_threads);

  P constexpr s22 = 0.5 * s2;
  P constexpr is2h = 0.5 * is2;
  P constexpr is64 = s6 / 4.0;
  P const h0[4] = {is2, -is64, 0, is2h};
  P const h1[4] = {is2, is64, 0, is2h};
  P const w0[4] = {0, is2h, -is2, is64};
  P const w1[4] = {0, -is2h, is2, is64};

  // small matrices can be cached on the stack for faster access
  P cc[4], c0[4], c1[4], c2[4], c3[4];

  int const pdof  = degree_ + 1;
  int const pdof2 = pdof * pdof;

  std::vector<P> custom;
  if constexpr (op == operation::custom_unitary) {
    expect(trans != nullptr);
    if constexpr (tdegree == 0) {
      cc[0] = trans[0];
      cc[1] = trans[2];
      cc[2] = trans[1];
      cc[3] = trans[3];
    } else if constexpr (tdegree == 1) {
      c0[0] = trans[ 0]; c0[1] = trans[ 1]; c0[2] = trans[ 4]; c0[3] = trans[ 5];
      c1[0] = trans[ 8]; c1[1] = trans[ 9]; c1[2] = trans[12]; c1[3] = trans[13];
      c2[0] = trans[ 2]; c2[1] = trans[ 3]; c2[2] = trans[ 6]; c2[3] = trans[ 7];
      c3[0] = trans[10]; c3[1] = trans[11]; c3[2] = trans[14]; c3[3] = trans[15];
    } else {
      custom.resize(4 * pdof2);
      smmat::matrix<P const> transf(2 * pdof, trans);
      smmat::matrix<P> pc0(pdof, custom.data());
      smmat::matrix<P> pc1(pdof, custom.data() + pdof2);
      smmat::matrix<P> pc2(pdof, custom.data() + 2 * pdof2);
      smmat::matrix<P> pc3(pdof, custom.data() + 3 * pdof2);
      for (int r = 0; r < pdof; r++) {
        for (int c = 0; c < pdof; c++) {
          pc0(r, c) = transf(r, c);
          pc1(r, c) = transf(r, c + pdof);
          pc2(r, c) = transf(r + pdof, c);
          pc3(r, c) = transf(r + pdof, c + pdof);
        }
      }
    }
  } else if constexpr (op == operation::custom_non_unitary) {
    // in the non-unitary case, the forward and inverse transforms use different matrices
    // and we do not transpose in the application of the blocks
    expect(trans != nullptr);
    if constexpr (tdegree == 0) {
      std::copy_n(trans, 4, cc);
    } else if constexpr (tdegree == 1) {
      c0[0] = trans[ 0]; c0[1] = trans[ 1]; c0[2] = trans[ 4]; c0[3] = trans[ 5];
      c1[0] = trans[ 2]; c1[1] = trans[ 3]; c1[2] = trans[ 6]; c1[3] = trans[ 7];
      c2[0] = trans[ 8]; c2[1] = trans[ 9]; c2[2] = trans[12]; c2[3] = trans[13];
      c3[0] = trans[10]; c3[1] = trans[11]; c3[2] = trans[14]; c3[3] = trans[15];
    } else {
      custom.resize(4 * pdof2);
      smmat::matrix<P const> transf(2 * pdof, trans);
      smmat::matrix<P> pc0(pdof, custom.data());
      smmat::matrix<P> pc1(pdof, custom.data() + pdof2);
      smmat::matrix<P> pc2(pdof, custom.data() + 2 * pdof2);
      smmat::matrix<P> pc3(pdof, custom.data() + 3 * pdof2);
      for (int r = 0; r < pdof; r++) {
        for (int c = 0; c < pdof; c++) {
          pc0(r, c) = transf(r, c);
          pc1(r, c) = transf(r + pdof, c);
          pc2(r, c) = transf(r, c + pdof);
          pc3(r, c) = transf(r + pdof, c + pdof);
        }
      }
    }
  }

  // given a left/right cells at some level L, this computes out as the corresponding entry
  // at level L-1 and the upper which is the non-hierarchical cell at level L-1
  // the cell index of left/right should be 2n and 2n+1, while out and upper have index n
  auto apply = [&](P const *left, P const *right, P *out, P *upper)
  {
    if constexpr (op == operation::transform) {
      if constexpr (tdegree == 0)
        *out = -s22 * (*left) + s22 * (*right);
      else if constexpr (tdegree == 1)
        smmat::gemm_pairt(2, left, w0, right, w1, out);
      else
        smmat::gemm_pairt(pdof, left, tmatlev, right, tmatlev + pdof2, out);

      if constexpr (tdegree == 0)
        *upper = s22 * (*left) + s22 * (*right);
      else if constexpr (tdegree == 1)
        smmat::gemm_pairt(2, left, h0, right, h1, upper);
      else
        smmat::gemm_pairt(pdof, left, tmatup, right, tmatup + pdof2, upper);
    } else if constexpr (op == operation::custom_unitary) {
      if constexpr (tdegree == 0)
        *upper = (*left) * cc[0] + (*right) * cc[1];
      else if constexpr (tdegree == 1)
        smmat::gemm_pairt(2, left, c0, right, c1, upper);
      else
        smmat::gemm_pairt(pdof, left, custom.data(),
                          right, custom.data() + pdof2, upper);

      if constexpr (tdegree == 0)
        *out = (*left) * cc[2] + (*right) * cc[3];
      else if constexpr (tdegree == 1)
        smmat::gemm_pairt(2, left, c2, right, c3, out);
      else
        smmat::gemm_pairt(pdof, left, custom.data() + 2 * pdof2,
                          right, custom.data() + 3 * pdof2, out);
    } else if constexpr (op == operation::custom_non_unitary) {
      if constexpr (tdegree == 0)
        *upper = (*left) * cc[0] + (*right) * cc[1];
      else if constexpr (tdegree == 1)
        smmat::gemm_pair(2, left, c0, right, c1, upper);
      else
        smmat::gemm_pair(pdof, left, custom.data(),
                         right, custom.data() + pdof2, upper);

      if constexpr (tdegree == 0)
        *out = (*left) * cc[2] + (*right) * cc[3];
      else if constexpr (tdegree == 1)
        smmat::gemm_pair(2, left, c2, right, c3, out);
      else
        smmat::gemm_pair(pdof, left, custom.data() + 2 * pdof2,
                         right, custom.data() + 3 * pdof2, out);
    }
  };

  connect_1d const &conn = conns(sp);

  int const nrows = diag.nrows();

  if (nrows == 1) // special case, single cell
  {
    std::copy_n(diag[0], pdof2, sp[0]);
    return;
  }

  if (nrows == 2) // special case, two cells
  {
    // using 5 scratch small matrices
    // the block algorithm uses only 1 but keep it consistent with the tri-diagonal
    colblocks[0].resize(5 * pdof2);
    P *zero = colblocks[0].data(); // TODO: get rid of zero
    std::fill_n(zero, pdof2, P{0});

    apply(diag[0], zero, sp[1], sp[0]);
    apply(zero, diag[1], sp[conn.row_begin(1) + 1], sp[conn.row_begin(1)]);
    return;
  }

  // number of rows in the extended pattern used for scratch space
  int const lrows = fm::ipow2(level + 1) - 2; // using power series law

  int threadid = 0;
#pragma omp parallel
  {
    int tid;
#pragma omp critical
    tid = threadid++;

    // setup scratch space, 5 matrices are needed for tri-diagonal alg.
    // the diag algorithm uses fewer entries, but we resize to the large one
    colblocks[tid].resize(5 * pdof * pdof);
    P *D  = colblocks[tid].data();
    P *wD = D + pdof2;
    P *Z  = wD + pdof2;
    std::fill_n(Z, pdof2, P{0});

#pragma omp for
    for (int r = 0; r < nrows; r++)
    { // initiate new transform for this row, work with one entry per level
      // if the diagonal entry is even, i.e., 2k for some k, we have
      //    (diag, 0) and new entry at k
      // if the diagonal entry is odd, i.e., 2k + 1 for some k, we have
      //    (0, diag) and new entry at k

      int cs = r % 2; // indicates even of odd
      int num = nrows; // counts the levels
      int k = num / 2 + r / 2; // next entry
      int orow = lrows - nrows + r; // out-row
      int j    = conn.row_end(orow);
      while (conn[--j] != k); // move to where j == k

      if (cs == 0)
        apply(diag[r], Z, sp[j], D);
      else
        apply(Z, diag[r], sp[j], D);
      num /= 2;

      while (num > 2)
      {
        cs = k % 2;
        k = k / 2;
        while (conn[--j] != k); // move to where j == k

        if (cs == 0)
          apply(D, Z, sp[j], wD);
        else
          apply(Z, D, sp[j], wD);

        std::swap(D, wD);
        num /= 2;
      }

      // do the last two columns
      while (conn[--j] != 1);
      if (k == 2)
        apply(D, Z, sp[j], sp[j - 1]);
      else
        apply(Z, D, sp[j], sp[j - 1]);

    } // #pragma omp for
  } // #pragma omp parallel
}

template<typename P>
template<int tdegree, typename hierarchy_manipulator<P>::operation op>
void hierarchy_manipulator<P>::row_project_any(
    P const *trans, block_sparse_matrix<P> &col, int const level,
    connection_patterns const &conn, block_sparse_matrix<P> &sp) const
{
  expect(connect_1d::hierarchy::col_full == col or
         connect_1d::hierarchy::col_volume == col);
  expect(connect_1d::hierarchy::full == sp or
         connect_1d::hierarchy::volume == sp);

  P constexpr s22 = 0.5 * s2;
  P constexpr is2h = 0.5 * is2;
  P constexpr is64  = s6 / 4.0;
  P const h0[4] = {is2, -is64, 0, is2h};
  P const h1[4] = {is2, is64, 0, is2h};
  P const w0[4] = {0, is2h, -is2, is64};
  P const w1[4] = {0, -is2h, is2, is64};

  P cc[4], c0[4], c1[4], c2[4], c3[4];

  int const pdof  = degree_ + 1;
  int const pdof2 = pdof * pdof;

  std::vector<P> custom;
  if constexpr (op == operation::custom_unitary or op == operation::custom_non_unitary) {
    expect(trans != nullptr);
    if constexpr (tdegree == 0) {
      cc[0] = trans[0]; cc[1] = trans[2]; cc[2] = trans[1]; cc[3] = trans[3];
    } else if constexpr (tdegree == 1) {
      c0[0] = trans[ 0]; c0[1] = trans[ 1]; c0[2] = trans[ 4]; c0[3] = trans[ 5];
      c1[0] = trans[ 8]; c1[1] = trans[ 9]; c1[2] = trans[12]; c1[3] = trans[13];
      c2[0] = trans[ 2]; c2[1] = trans[ 3]; c2[2] = trans[ 6]; c2[3] = trans[ 7];
      c3[0] = trans[10]; c3[1] = trans[11]; c3[2] = trans[14]; c3[3] = trans[15];
    } else {
      custom.resize(4 * pdof2);
      smmat::matrix<P const> transf(2 * pdof, trans);
      smmat::matrix<P> pc0(pdof, custom.data());
      smmat::matrix<P> pc1(pdof, custom.data() + pdof2);
      smmat::matrix<P> pc2(pdof, custom.data() + 2 * pdof2);
      smmat::matrix<P> pc3(pdof, custom.data() + 3 * pdof2);
      for (int r = 0; r < pdof; r++) {
        for (int c = 0; c < pdof; c++) {
          pc0(r, c) = transf(r, c);
          pc1(r, c) = transf(r, c + pdof);
          pc2(r, c) = transf(r + pdof, c);
          pc3(r, c) = transf(r + pdof, c + pdof);
        }
      }
    }
  }

  // given a left/right cells at some level L, this computes out as the corresponding entry
  // at level L-1 and the upper which is the non-hierarchical cell at level L-1
  // the cell index of left/right should be 2n and 2n+1, while out and upper have index n
  auto apply = [&](P const *left, P const *right, P *out, P *upper)
  {
    if constexpr (op == operation::transform) {
      if constexpr (tdegree == 0)
        *out = -s22 * (*left) + s22 * (*right);
      else if constexpr (tdegree == 1)
        smmat::gemm_pair(2, w0, left, w1, right, out);
      else
        smmat::gemm_pair(pdof, tmatlev, left, tmatlev + pdof2, right, out);

      if constexpr (tdegree == 0)
        *upper = s22 * (*left) + s22 * (*right);
      else if constexpr (tdegree == 1)
        smmat::gemm_pair(2, h0, left, h1, right, upper);
      else
        smmat::gemm_pair(pdof, tmatup, left, tmatup + pdof2, right, upper);
    } else if constexpr (op == operation::custom_unitary or op == operation::custom_non_unitary) {
      if constexpr (tdegree == 0)
        *upper = cc[0] * (*left) + cc[1] * (*right);
      else if constexpr (tdegree == 1)
        smmat::gemm_pair(2, c0, left, c1, right, upper);
      else
        smmat::gemm_pair(pdof, custom.data(), left,
                         custom.data() + pdof2, right, upper);

      if constexpr (tdegree == 0)
        *out = cc[2] * (*left) + cc[3] * (*right);
      else if constexpr (tdegree == 1)
        smmat::gemm_pair(2, c2, left, c3, right, out);
      else
        smmat::gemm_pair(pdof, custom.data() + 2 * pdof2, left,
                         custom.data() + 3 * pdof2, right, out);
    }
  };

  connect_1d const &fconn = conn(sp);
  connect_1d const &tconn = conn(col);

  int nrows = fm::ipow2(level);
  if (nrows == 1) // one cell/one row
  {
    std::copy_n(col[0], pdof2, sp[0]);
    return;
  }
  if (nrows == 2) // two cells, very simple
  {
    // work on 4 entries as the matrix is 2 by 2
    apply(col[0], col[tconn.row_begin(1)], sp[fconn.row_begin(1)], sp[0]);
    apply(col[1], col[tconn.row_begin(1) + 1], sp[fconn.row_begin(1) + 1], sp[1]);
    return;
  }

  // effective number of rows in the extended pattern
  int const lrows = fm::ipow2(level + 1) - 2;

  int trows = lrows - nrows; // row-offset, current level

  while (nrows > 2)
  {
    nrows /= 2; // handling bottom nrows/2 rows
    trows -= nrows;

#pragma omp parallel for
    for (int r = 0; r < nrows; r++)
    {
      int const fout = nrows + r; // final row in sp
      int const tout = trows + r; // scratch row in col
      int const cl   = 2 * nrows - 2 + 2 * r; // first input row
      int const cr   = 2 * nrows - 2 + 2 * r + 1; // second input row

      int jt = tconn.row_begin(tout) - 1;
      int jl = tconn.row_begin(cl) - 1;
      int jr = tconn.row_begin(cr) - 1;
      for (int j = fconn.row_begin(fout); j < fconn.row_end(fout); j++)
      {
        // process a full row, go over the columns and match the pattern
        int c = fconn[j];
        while (tconn[++jt] != c);
        while (++jr, tconn[++jl] != c); // the two rows must have identical pattern
        expect(tconn[jr] == c); // TODO: shold not be needed
        expect(jt < tconn.row_end(tout));
        expect(jl < tconn.row_end(cl));
        expect(jr < tconn.row_end(cr));
        apply(col[jl], col[jr], sp[j], col[jt]);
      }
    }
  }

  if (nrows == 2) // last two cells, all rows are dense
  {
    int r1 = tconn.row_begin(1);
    expect(r1 == fconn.row_begin(1));
    for (int j = 0; j < tconn.row_end(0); j++)
      apply(col[j], col[r1 + j], sp[r1 + j], sp[j]);

    return;
  }
}

template<typename P>
void hierarchy_manipulator<P>::prepare_quadrature(int d, int num_cells) const
{
  int const num_quad = quad.stride();

  // if quadrature is already set for the correct level, no need to do anything
  // this assumes that the min/max of the domain does not change
  if (quad_points[d].size() == static_cast<size_t>(num_quad * num_cells))
    return;

  quad_points[d].resize(num_quad * num_cells);

  P const cell_size = (dmax[d] - dmin[d]) / num_cells;

  P mid       = dmin[d] + 0.5 * cell_size;
  P const slp = 0.5 * cell_size;

  P *iq = quad_points[d].data();
  for (int i : indexof<int>(num_cells))
  {
    ignore(i);
    for (int j : indexof<int>(num_quad))
      iq[j] = slp * quad[points][j] + mid;
    mid += cell_size;
    iq  += num_quad;
  }
}

template<typename P>
void hierarchy_manipulator<P>::setup_projection_matrices()
{
  int const num_quad = quad.stride();

  // leg_vals is a small matrix with the values of Legendre polynomials
  // scaled by the quadrature weights
  // the final structure is such that small matrix leg_vals times the
  // vector of f(x_i) at quadrature points x_i will give us the projection
  // of f onto the Legendre polynomial basis
  // scaled by the l-2 volume of the cell, this is the local projection of f(x)
  // leg_unscal is the transpose of leg_vals and unscaled by the quadrature w.
  // if rho(x_i) are local values of the mass weight, the local mass matrix is
  // leg_vals * diag(rho(x_i)) * leg_unscal
  leg_vals   = vector2d<P>(degree_ + 1, num_quad);
  leg_unscal = vector2d<P>(num_quad, degree_ + 1);

  P const *qpoints = quad[points];
  // using the recurrence: L_n = ((2n - 1) L_{n-1} - (n - 1) L_{n-2}) / n
  for (int i : indexof<int>(num_quad))
  {
    P *l = leg_vals[i];
    l[0] = 1.0;
    if (degree_ > 0)
      l[1] = qpoints[i];
    for (int j = 2; j <= degree_; j++)
      l[j] = ((2 * j - 1) * qpoints[i] * l[j-1] - (j - 1) * l[j-2]) / P(j);
  }

  for (int j = 0; j <= degree_; j++)
  {
    P const scale = std::sqrt( (2 * j + 1) / P(2) );
    for (int i : indexof<int>(num_quad))
      leg_unscal[j][i] = scale * leg_vals[i][j];
    for (int i : indexof<int>(num_quad))
      leg_vals[i][j] *= scale * quad[weights][i];
  }

  if (degree_ >= 2) // need transformation matrices, degree_ <= 1 is hard-coded
  {
    // work on the transformation matrices
    auto rawmats = basis::generate_multi_wavelets(degree_);
    int const pdof = degree_ + 1;
    // copy the matrices twice, once for level 1->0 and once for generic levels
    tmats.resize(8 * pdof * pdof);
    auto ip = tmats.data();
    for (int i : indexof<int>(pdof)) {
      ip = std::copy_n(rawmats[0].data() + pdof * i, pdof, ip);
      ip = std::copy_n(rawmats[2].data() + pdof * i, pdof, ip);
    }
    for (int i : indexof<int>(pdof)) {
      ip = std::copy_n(rawmats[1].data() + pdof * i, pdof, ip);
      ip = std::copy_n(rawmats[3].data() + pdof * i, pdof, ip);
    }

    tmatup = ip;
    tmatlev = tmatup + 2 * pdof * pdof;

    for (int j : indexof<int>(4))
      for (int i : indexof<int>(pdof))
        ip = std::copy_n(rawmats[j].data() + i * pdof, pdof, ip);
  }
}

#define instantiate_multi(prec, deg) \
  template void hierarchy_manipulator<prec>::col_project_full<deg, hierarchy_manipulator<prec>::operation::transform>( \
      prec const *, block_tri_matrix<prec> const &, int const, connection_patterns const &, \
      block_sparse_matrix<prec> &) const; \
  template void hierarchy_manipulator<prec>::col_project_full<deg, hierarchy_manipulator<prec>::operation::custom_unitary>( \
      prec const *, block_tri_matrix<prec> const &, int const, connection_patterns const &, \
      block_sparse_matrix<prec> &) const; \
  template void hierarchy_manipulator<prec>::col_project_full<deg, hierarchy_manipulator<prec>::operation::custom_non_unitary>( \
      prec const *, block_tri_matrix<prec> const &, int const, connection_patterns const &, \
      block_sparse_matrix<prec> &) const; \
  template void hierarchy_manipulator<prec>::col_project_vol<deg, hierarchy_manipulator<prec>::operation::transform>( \
      prec const *, block_diag_matrix<prec> const &, int const, connection_patterns const &, \
      block_sparse_matrix<prec> &) const; \
  template void hierarchy_manipulator<prec>::col_project_vol<deg, hierarchy_manipulator<prec>::operation::custom_unitary>( \
      prec const *, block_diag_matrix<prec> const &, int const, connection_patterns const &, \
      block_sparse_matrix<prec> &) const; \
  template void hierarchy_manipulator<prec>::col_project_vol<deg, hierarchy_manipulator<prec>::operation::custom_non_unitary>( \
      prec const *, block_diag_matrix<prec> const &, int const, connection_patterns const &, \
      block_sparse_matrix<prec> &) const; \
  template void hierarchy_manipulator<prec>::row_project_any<deg, hierarchy_manipulator<prec>::operation::transform>( \
      prec const *, block_sparse_matrix<prec> &, int const, connection_patterns const &, \
      block_sparse_matrix<prec> &) const; \
  template void hierarchy_manipulator<prec>::row_project_any<deg, hierarchy_manipulator<prec>::operation::custom_unitary>( \
      prec const *, block_sparse_matrix<prec> &, int const, connection_patterns const &, \
      block_sparse_matrix<prec> &) const; \
  template void hierarchy_manipulator<prec>::apply_transform<deg, hierarchy_manipulator<prec>::operation::transform>( \
      prec const *trans, int level, prec src[], prec dest[]) const; \
  template void hierarchy_manipulator<prec>::apply_transform<deg, hierarchy_manipulator<prec>::operation::custom_unitary>( \
      prec const *trans, int level, prec src[], prec dest[]) const; \
  template void hierarchy_manipulator<prec>::apply_transform<deg, hierarchy_manipulator<prec>::operation::custom_non_unitary>( \
      prec const *trans, int level, prec src[], prec dest[]) const; \

#ifdef ASGARD_ENABLE_DOUBLE
template struct legendre_basis<double>;
template class hierarchy_manipulator<double>;

template void hierarchy_manipulator<double>::project_separable<data_mode::replace>(
    separable_func<double> const &sep,
    sparse_grid const &grid, mass_diag<double> const &mass,
    double time, double alpha, double f[]) const;
template void hierarchy_manipulator<double>::project_separable<data_mode::scal_rep>(
    separable_func<double> const &sep,
    sparse_grid const &grid, mass_diag<double> const &mass,
    double time, double alpha, double f[]) const;
template void hierarchy_manipulator<double>::project_separable<data_mode::increment>(
    separable_func<double> const &sep,
    sparse_grid const &grid, mass_diag<double> const &mass,
    double time, double alpha, double f[]) const;
template void hierarchy_manipulator<double>::project_separable<data_mode::scal_inc>(
    separable_func<double> const &sep,
    sparse_grid const &grid, mass_diag<double> const &mass,
    double time, double alpha, double f[]) const;

instantiate_multi(double, 0);
instantiate_multi(double, 1);
instantiate_multi(double, -1);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct legendre_basis<float>;
template class hierarchy_manipulator<float>;

template void hierarchy_manipulator<float>::project_separable<data_mode::replace>(
    separable_func<float> const &sep,
    sparse_grid const &grid, mass_diag<float> const &mass,
    float time, float alpha, float f[]) const;
template void hierarchy_manipulator<float>::project_separable<data_mode::scal_rep>(
    separable_func<float> const &sep,
    sparse_grid const &grid, mass_diag<float> const &mass,
    float time, float alpha, float f[]) const;
template void hierarchy_manipulator<float>::project_separable<data_mode::increment>(
    separable_func<float> const &sep,
    sparse_grid const &grid, mass_diag<float> const &mass,
    float time, float alpha, float f[]) const;
template void hierarchy_manipulator<float>::project_separable<data_mode::scal_inc>(
    separable_func<float> const &sep,
    sparse_grid const &grid, mass_diag<float> const &mass,
    float time, float alpha, float f[]) const;

instantiate_multi(float, 0);
instantiate_multi(float, 1);
instantiate_multi(float, -1);
#endif

} // namespace asgard
