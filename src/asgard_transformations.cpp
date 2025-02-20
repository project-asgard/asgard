#include "asgard_transformations.hpp"

#include "asgard_small_mats.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace asgard
{
// combine components and create the portion of the multi-d vector associated
// with the provided start and stop element bounds (inclusive)
template<typename P>
void combine_dimensions(int const degree, elements::table const &table,
                        int const start_element, int const stop_element,
                        std::vector<std::vector<P>> const &vectors,
                        P combined[])
{
  int const num_dims = static_cast<int>(vectors.size());
  expect(num_dims > 0);
  expect(start_element >= 0);
  expect(stop_element >= start_element);
  expect(stop_element < table.size());

  int const pdof        = degree + 1;
  int64_t const mdblock = fm::ipow(pdof, num_dims);

  for (int cell = start_element; cell <= stop_element; cell++)
  {
    fk::vector<int> const coords = table.get_coords(cell);

    std::array<int64_t, max_num_dimensions> offset1d;
    for (int d : indexof<int>(num_dims))
      offset1d[d] = pdof * elements::get_1d_index(coords(d), coords(d + num_dims));

    for (int64_t i : indexof(mdblock))
    {
      int64_t t = i;
      combined[i] = vectors.back()[offset1d[num_dims - 1] + t % pdof];
      t /= pdof;
      for (int j = num_dims - 2; j >= 0; j--)
      {
        combined[i] *= vectors[j][offset1d[j] + t % pdof];
        t /= pdof;
      }
    }

    combined += mdblock;
  }
}

template<typename P>
legendre_basis<P>::legendre_basis(int degree) : pdof(degree + 1) {

  static auto const legendre_values =
      legendre_weights<P>(degree, -1.0, 1.0);
  auto const &points  = legendre_values[0];
  auto const &weights = legendre_values[1];

  num_quad = weights.size();

  auto [lP_L, lPP_L] = legendre(fk::vector<P>{-1}, degree);
  auto [lP_R, lPP_R] = legendre(fk::vector<P>{+1}, degree);

  auto [lP, lPP] = legendre(points, degree);

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
  std::copy_n(points.data(), num_quad, qp);
  std::copy_n(weights.data(), num_quad, qw);

  smmat::gemm_outer_inc(pdof, lP_L.data(), lP_L.data(), to_left);
  smmat::gemm_outer_inc(pdof, lP_L.data(), lP_R.data(), from_left);
  smmat::gemm_outer_inc(pdof, lP_R.data(), lP_L.data(), from_right);
  smmat::gemm_outer_inc(pdof, lP_R.data(), lP_R.data(), to_right);

  std::copy_n(lP.data(), num_quad * pdof, leg);
  smmat::col_scal(num_quad, pdof, P{0.5}, qw, leg, legw);
  std::copy_n(lPP.data(), num_quad * pdof, der);

  std::copy_n(lP_L.data(), pdof, leg_left);
  std::copy_n(lP_R.data(), pdof, leg_right);
}

template<typename P>
void legendre_basis<P>::project(bool is_interior, int level, std::vector<P> const &raw_data,
                                std::vector<P> &lgn) const {

  int const num_cells = fm::ipow2(level);

  span2d<P const> raw;
  if (is_interior)
    raw = span2d<P const>(pdof, num_cells, raw_data.data());
  else
    raw = span2d<P const>(pdof + 1, num_cells, raw_data.data() + 1);

  lgn.resize(num_cells * pdof);
  span2d<P> leg_basis(pdof, num_cells, lgn.data());

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++) {
    smmat::gemtv(num_quad, pdof, legw, raw[i], leg_basis[i]);
  }
}

template<typename P>
mass_matrix<P> hierarchy_manipulator<P>::make_mass(int dim, int level) const
{
  int const num_cells = fm::ipow2(level);
  int const num_quad  = leg_unscal.stride();
  int const pdof      = degree_ + 1;

  mass_matrix<P> mat(pdof * pdof, num_cells);

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
  {
    smmat::gemm3(pdof, num_quad, leg_vals[0], quad_dv[dim].data() + i * num_quad,
                 leg_unscal[0], mat[i]);
  }

  switch (degree_)
  {
  case 0:
#pragma omp parallel for
    for (int i = 0; i < num_cells; i++)
      mat[i][0] = P{1} / mat[i][0];
    break;
  case 1:
#pragma omp parallel for
    for (int i = 0; i < num_cells; i++)
      smmat::inv2by2(mat[i]);
    break;
  default:
#pragma omp parallel for
    for (int i = 0; i < num_cells; i++)
      smmat::potrf(pdof, mat[i]);
    break;
  }

  return mat;
}

template<typename P>
void hierarchy_manipulator<P>::reconstruct1d(
    int const nbatch, int const level, span2d<P> data) const
{
  expect(data.num_strips() == nbatch * fm::ipow2(level));
  expect(data.stride() == (degree_ + 1));

  if (level == 0)
    return; // the hierarchical form is just scaled/normalized

  stage0.resize(data.stride() * data.num_strips());
  stage1.resize(stage0.size());

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
          smmat::gemtv(ssize, pmatup, left + b * ssize, out_left + b * ssize);
          smmat::gemtv1(ssize, pmatlev, right + b * ssize, out_left + b * ssize);
          smmat::gemtv(ssize, pmatup + ssize * ssize, left + b * ssize, out_right + b * ssize);
          smmat::gemtv1(ssize, pmatlev + ssize * ssize, right + b * ssize, out_right + b * ssize);
          break;
      };
    }
  };

  span2d<P> work0(data.stride(), data.num_strips(), stage0.data());
  span2d<P> work1(data.stride(), data.num_strips(), stage1.data());

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
template<bool skip_hierarchy>
void hierarchy_manipulator<P>::project1d(int d, int level, P const dsize, level_mass_matrces<P> const &mass) const
{
  int const num_cells = fm::ipow2(level);

  int const num_quad = quad.stride();
  int const pdof     = degree_ + 1;

  expect(fvals.size() == static_cast<size_t>(num_cells * num_quad));

  stage0.resize(pdof * num_cells);

  // doing the hierarchical projection, we must normalize the Legendre polynomial to unit l-2 norm
  P const scale = std::pow(is2, level + 1) * std::sqrt(dsize);

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
  {
    smmat::gemv(pdof, num_quad, leg_vals[0], &fvals[i * num_quad],
                &stage0[i * pdof]);
    smmat::scal(pdof, scale, &stage0[i * pdof]);
  }

  if (mass.has_level(level))
    invert_mass(pdof, mass[level], stage0.data());

  if constexpr (skip_hierarchy)
    return;

  pf[d].resize(pdof * num_cells);

  // stage0 contains the projection data per-cell
  // pf has the correct size to take the data, so project all levels up
  switch (degree_)
  { // hardcoded degrees first, the default uses the projection matrices
  case 0:
    projectlevels<0>(d, level);
    break;
  case 1:
    projectlevels<1>(d, level);
    break;
  default:
    projectlevels<-1>(d, level);
  };
}

template<typename P>
void hierarchy_manipulator<P>::project1d(int d, int level, P const dsize, block_diag_matrix<P> const &mass) const
{
  int const num_cells = fm::ipow2(level);

  int const num_quad = quad.stride();
  int const pdof     = degree_ + 1;

  expect(fvals.size() == static_cast<size_t>(num_cells * num_quad));

  stage0.resize(pdof * num_cells);

  // doing the hierarchical projection, we must normalize the Legendre polynomial to unit l-2 norm
  P const scale = std::pow(is2, level + 1) * std::sqrt(dsize);

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
  {
    smmat::gemv(pdof, num_quad, leg_vals[0], &fvals[i * num_quad],
                &stage0[i * pdof]);
    smmat::scal(pdof, scale, &stage0[i * pdof]);
  }

  if (mass)
    mass.solve(pdof, stage0);

  // std::cout << " --- num-cells = " << num_cells << " pdof = " << pdof << "\n";
  // for (auto &a : stage0)
  //   std::cout << a << "\n";

  pf[d].resize(pdof * num_cells);

  // stage0 contains the projection data per-cell
  // pf has the correct size to take the data, so project all levels up
  switch (degree_)
  { // hardcoded degrees first, the default uses the projection matrices
  case 0:
    projectlevels<0>(d, level);
    break;
  case 1:
    projectlevels<1>(d, level);
    break;
  default:
    projectlevels<-1>(d, level);
  };
}

template<typename P>
template<int tdegree>
void hierarchy_manipulator<P>::projectup2(P const *raw, P *fin) const
{
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

    fin[0] = is2 * raw[0]                   + is2 * raw[2];
    fin[1] = -is64 * raw[0] + is2h * raw[1] + is64 * raw[2] + is2h * raw[3];
    fin[2] = -is2 * raw[1] + is2 * raw[3];
    fin[3] = is2h * raw[0] + is64 * raw[1] - is2h * raw[2] + is64 * raw[3];
  }
  else
  {
    int const n = 2 * (degree_ + 1);
    smmat::gemv(n, n, pmats.data(), raw, fin);
  }
}

template<typename P>
template<int tdegree>
void hierarchy_manipulator<P>::projectup(int num_final, P const *raw, P *upper, P *fin) const
{
  if constexpr (tdegree == 0)
  {
#pragma omp parallel for
    for (int i = 0; i < num_final; i++)
    {
      P constexpr s22 = 0.5 * s2;
      P const r0 = raw[2 * i];
      P const r1 = raw[2 * i + 1];
      upper[i] = s22 * r0 + s22 * r1;
      fin[i]   = -s22 * r0 + s22 * r1;
    }
  }
  else if constexpr (tdegree == 1)
  {
#pragma omp parallel for
    for (int i = 0; i < num_final; i++)
    {
      P constexpr is2h = 0.5 * is2;
      P constexpr is64  = s6 / 4.0;
      P const r0 = raw[4 * i];
      P const r1 = raw[4 * i + 1];
      P const r2 = raw[4 * i + 2];
      P const r3 = raw[4 * i + 3];
      upper[2 * i]     = is2 * r0                   + is2 * r2;
      upper[2 * i + 1] = -is64 * r0 + is2h * r1 + is64 * r2 + is2h * r3;
      fin[2 * i]       = -is2 * r1 + is2 * r3;
      fin[2 * i + 1]   = is2h * r0 + is64 * r1 - is2h * r2 + is64 * r3;
    }
  }
  else
  {
    int const pdof = degree_ + 1;
#pragma omp parallel for
    for (int i = 0; i < num_final; i++)
    {
      smmat::gemv(pdof, 2 * pdof, pmatup, &raw[2 * pdof * i], &upper[i * pdof]);
      smmat::gemv(pdof, 2 * pdof, pmatlev, &raw[2 * pdof * i], &fin[i * pdof]);
    }
  }
}

template<typename P>
template<int tdegree>
void hierarchy_manipulator<P>::projectlevels(int d, int level) const
{
  switch (level)
  {
  case 0:
    std::copy(stage0.begin(), stage0.end(), pf[d].begin()); // nothing to project upwards
    break;
  case 1:
    projectup2<tdegree>(stage0.data(), pf[d].data()); // level 0 and 1
    break;
  default: {
      stage1.resize(stage0.size() / 2);
      int const pdof = degree_ + 1;

      P *w0  = stage0.data();
      P *w1  = stage1.data();
      int num = static_cast<int>(pf[d].size() / (2 * pdof));
      P *fin = pf[d].data() + num * pdof;
      for (int l = level; l > 1; l--)
      {
        projectup<tdegree>(num, w0, w1, fin);
        std::swap(w0, w1);
        num /= 2;
        fin -= num * pdof;
      }
      projectup2<tdegree>(w0, pf[d].data());
    }
  }
}

template<typename P>
block_sparse_matrix<P>
hierarchy_manipulator<P>::diag2hierarchical(block_diag_matrix<P> const &diag,
                                            int const level,
                                            connection_patterns const &conns) const
{
  block_sparse_matrix<P> col = make_block_sparse_matrix(conns, connect_1d::hierarchy::col_volume);
  block_sparse_matrix<P> res = make_block_sparse_matrix(conns, connect_1d::hierarchy::volume);

  switch (degree_)
  {
  case 0:
    col_project_full<0>(diag, level, conns, col);
    row_project_full<0>(col, level, conns, res);
    break;
  case 1:
    col_project_full<1>(diag, level, conns, col);
    row_project_full<1>(col, level, conns, res);
    break;
  default:
    col_project_full<-1>(diag, level, conns, col);
    row_project_full<-1>(col, level, conns, res);
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

  switch (degree_)
  {
  case 0:
    col_project_full<0>(tri, level, conns, col);
    row_project_full<0>(col, level, conns, res);
    break;
  case 1:
    col_project_full<1>(tri, level, conns, col);
    row_project_full<1>(col, level, conns, res);
    break;
  default:
    col_project_full<-1>(tri, level, conns, col);
    row_project_full<-1>(col, level, conns, res);
    break;
  };

  return res;
}

template<typename P>
template<int tdegree>
void hierarchy_manipulator<P>::col_project_full(block_tri_matrix<P> const &tri,
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

  int const pdof  = degree_ + 1;
  int const pdof2 = pdof * pdof;

  // project cells left/right with index 2n and 2n+1 at level L
  // to cells n at the hierarchical level L-1, stored in out
  // upper cells at level L-1, stored in upper
  // see the block-diagonal overload too
  auto apply = [&](P const *left, P const *right, P *out, P *upper)
  {
    if constexpr (tdegree == 0)
      *out = -s22 * (*left) + s22 * (*right);
    else if constexpr (tdegree == 1)
      smmat::gemm_pairt(2, left, w0, right, w1, out);
    else
      smmat::gemm_pairt(pdof, left, pmatlev, right, pmatlev + pdof2, out);

    if constexpr (tdegree == 0)
      *upper = s22 * (*left) + s22 * (*right);
    else if constexpr (tdegree == 1)
      smmat::gemm_pairt(2, left, h0, right, h1, upper);
    else
      smmat::gemm_pairt(pdof, left, pmatup, right, pmatup + pdof2, upper);
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
template<int tdegree>
void hierarchy_manipulator<P>::col_project_full(block_diag_matrix<P> const &diag,
                                                int const level,
                                                connection_patterns const &conns,
                                                block_sparse_matrix<P> &sp) const
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

  int const pdof  = degree_ + 1;
  int const pdof2 = pdof * pdof;

  // given a left/right cells at some level L, this computes out as the corresponding entry
  // at level L-1 and the upper which is the non-hierarchical cell at level L-1
  // the cell index of left/right should be 2n and 2n+1, while out and upper have index n
  auto apply = [&](P const *left, P const *right, P *out, P *upper)
  {
    if constexpr (tdegree == 0)
      *out = -s22 * (*left) + s22 * (*right);
    else if constexpr (tdegree == 1)
      smmat::gemm_pairt(2, left, w0, right, w1, out);
    else
      smmat::gemm_pairt(pdof, left, pmatlev, right, pmatlev + pdof2, out);

    if constexpr (tdegree == 0)
      *upper = s22 * (*left) + s22 * (*right);
    else if constexpr (tdegree == 1)
      smmat::gemm_pairt(2, left, h0, right, h1, upper);
    else
      smmat::gemm_pairt(pdof, left, pmatup, right, pmatup + pdof2, upper);
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
template<int tdegree>
void hierarchy_manipulator<P>::row_project_full(
    block_sparse_matrix<P> &col,
    int const level,
    connection_patterns const &conn,
    block_sparse_matrix<P> &sp) const
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

  int const pdof  = degree_ + 1;
  int const pdof2 = pdof * pdof;

  // given a left/right cells at some level L, this computes out as the corresponding entry
  // at level L-1 and the upper which is the non-hierarchical cell at level L-1
  // the cell index of left/right should be 2n and 2n+1, while out and upper have index n
  auto apply = [&](P const *left, P const *right, P *out, P *upper)
  {
    if constexpr (tdegree == 0)
      *out = -s22 * (*left) + s22 * (*right);
    else if constexpr (tdegree == 1)
      smmat::gemm_pair(2, w0, left, w1, right, out);
    else
      smmat::gemm_pair(pdof, pmatlev, left, pmatlev + pdof2, right, out);

    if constexpr (tdegree == 0)
      *upper = s22 * (*left) + s22 * (*right);
    else if constexpr (tdegree == 1)
      smmat::gemm_pair(2, h0, left, h1, right, upper);
    else
      smmat::gemm_pair(pdof, pmatup, left, pmatup + pdof2, right, upper);
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

  if (degree_ >= 2) // need projection matrices, degree_ <= 1 are hard-coded
  {
    auto rawmats = generate_multi_wavelets<P>(degree_);
    int const pdof = degree_ + 1;
    // copy the matrices twice, once for level 1->0 and once for generic levels
    pmats.resize(8 * pdof * pdof);
    auto ip = pmats.data();
    for (int i : indexof<int>(pdof)) {
      ip = std::copy_n(rawmats[0].data(0, i), pdof, ip);
      ip = std::copy_n(rawmats[2].data(0, i), pdof, ip);
    }
    for (int i : indexof<int>(pdof)) {
      ip = std::copy_n(rawmats[1].data(0, i), pdof, ip);
      ip = std::copy_n(rawmats[3].data(0, i), pdof, ip);
    }

    pmatup = ip;
    pmatlev = pmatup + 2 * pdof * pdof;

    for (int j : indexof<int>(4))
      for (int i : indexof<int>(pdof))
        ip = std::copy_n(rawmats[j].data(0, i), pdof, ip);
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct legendre_basis<double>;
template class hierarchy_manipulator<double>;

template void hierarchy_manipulator<double>::project1d<true>(
    int, int, double, level_mass_matrces<double> const &) const;
template void hierarchy_manipulator<double>::project1d<false>(
    int, int, double, level_mass_matrces<double> const &) const;

template void hierarchy_manipulator<double>::projectlevels<0>(int, int) const;
template void hierarchy_manipulator<double>::projectlevels<1>(int, int) const;
template void hierarchy_manipulator<double>::projectlevels<-1>(int, int) const;

template void combine_dimensions(
  int const, elements::table const &, int const, int const,
  std::vector<std::vector<double>> const &, double[]);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct legendre_basis<float>;
template class hierarchy_manipulator<float>;

template void hierarchy_manipulator<float>::project1d<true>(
    int, int, float, level_mass_matrces<float> const &) const;
template void hierarchy_manipulator<float>::project1d<false>(
    int, int, float, level_mass_matrces<float> const &) const;

template void hierarchy_manipulator<float>::projectlevels<0>(int, int) const;
template void hierarchy_manipulator<float>::projectlevels<1>(int, int) const;
template void hierarchy_manipulator<float>::projectlevels<-1>(int, int) const;

template void combine_dimensions(
  int const, elements::table const &, int const, int const,
  std::vector<std::vector<float>> const &, float[]);
#endif

} // namespace asgard
