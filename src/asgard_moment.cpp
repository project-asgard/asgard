#include "asgard_moment.hpp"
#include "asgard_small_mats.hpp"

namespace asgard
{

template<typename P>
moments1d<P>::moments1d(int num_mom, int degree, int max_level,
                        std::vector<dimension<P>> const &dims)
  : num_mom_(num_mom), num_dims_(static_cast<int>(dims.size())),
    degree_(degree)
{
  expect(num_dims_ > 1); // cannot have moments for 1D
  expect(num_dims_ <= 4); // cannot have more than 3 velocity dims
  P constexpr s2 = 1.41421356237309505; // sqrt(2.0)

  std::array<P, max_num_dimensions> dim_scale;
  for (int d : iindexof(num_dims_))
    dim_scale[d] = dims[d].domain_max - dims[d].domain_min;

  // Legendre and wavelet polynomials
  vector2d<P> pleg = basis::legendre_poly<P>(degree_);
  vector2d<P> pwav = basis::wavelet_poly(pleg, degree_);

  int const pdof   = degree_ + 1;
  int const nblock = num_mom_ * pdof;
  int const nump   = fm::ipow2(max_level);

  for (int d : iindexof(num_dims_))
  {
    if (d == 0)
      continue; // first dimension is the position

    integ[d] = vector2d<P>(nblock, nump);

    P const amin = dims[d].domain_min;
    P const amax = dims[d].domain_max;

    // global scale factor, the basis is unit normalized
    P const scale = 1.0 / (s2 * std::sqrt(amax - amin));

#pragma omp parallel
    {
      basis::canonical_integrator quad(num_mom_, degree_);
      std::vector<P> work(4 * quad.left_nodes().size());

#pragma omp for
      for (int i = 1; i < nump; i++)
      {
        int const level  = fm::intlog2(i);   // previous level of i
        int const istart = fm::ipow2(level); // first index on this level

        P const dx = (amax - amin) / istart; // cell size
        P const a  = amin + (i - istart) * dx;

        span2d<P> block(pdof, num_mom_, integ[d][i]);
        integrate(quad, a, a + dx, dims[d].volume_jacobian_dV, pwav, work, block);

        P s = ((level > 1) ? fm::powi<P>(s2, level -1) : P{1}) * scale;
        smmat::scal(pdof * num_mom_, s, integ[d][i]);
      }

#pragma omp single
      {
        integrate(quad, amin, amax, dims[d].volume_jacobian_dV, pleg, work,
                  span2d<P>(pdof, num_mom_, integ[d][0]));
        smmat::scal(pdof * num_mom_, scale, integ[d][0]);
      }
    }
  }
}

template<typename P>
moments1d<P>::moments1d(int num_mom, int degree, int max_level, pde_domain<P> const &domain)
  : num_mom_(num_mom), num_dims_(static_cast<int>(domain.num_dims())),
    num_pos_(domain.num_pos()), degree_(degree)
{
  rassert(domain.num_pos() <= 1, "moments not implemented for 2 or 3 position dimensions (yet)");
  rassert(domain.num_pos() == 1, "moments not implemented for 0 position dimensions (yet)");
  P constexpr s2 = 1.41421356237309505; // sqrt(2.0)

  // Legendre and wavelet polynomials
  vector2d<P> pleg = basis::legendre_poly<P>(degree_);
  vector2d<P> pwav = basis::wavelet_poly(pleg, degree_);

  int const pdof   = degree_ + 1;
  int const nblock = num_mom_ * pdof;
  int const nump   = fm::ipow2(max_level);

  for (int d : iindexof(num_dims_))
  {
    if (d < domain.num_pos())
      continue; // ignore the position dimensions

    integ[d] = vector2d<P>(nblock, nump);

    P const amin = domain.xleft(d);
    P const amax = domain.xright(d);

    // global scale factor, the basis is unit normalized
    P const scale = 1.0 / (s2 * std::sqrt(amax - amin));

#pragma omp parallel
    {
      basis::canonical_integrator quad(num_mom_, degree_);
      std::vector<P> work(4 * quad.left_nodes().size());

#pragma omp for
      for (int i = 1; i < nump; i++)
      {
        int const level  = fm::intlog2(i);   // previous level of i
        int const istart = fm::ipow2(level); // first index on this level

        P const dx = (amax - amin) / istart; // cell size
        P const a  = amin + (i - istart) * dx;

        span2d<P> block(pdof, num_mom_, integ[d][i]);
        integrate(quad, a, a + dx, nullptr, pwav, work, block);

        P s = ((level > 1) ? fm::powi<P>(s2, level -1) : P{1}) * scale;
        smmat::scal(pdof * num_mom_, s, integ[d][i]);
      }

#pragma omp single
      {
        integrate(quad, amin, amax, nullptr, pleg, work,
                  span2d<P>(pdof, num_mom_, integ[d][0]));
        smmat::scal(pdof * num_mom_, scale, integ[d][0]);
      }
    }
  }
}

template<typename P>
void moments1d<P>::integrate(
    basis::canonical_integrator const &quad, P a, P b, g_func_type<P> const &dv,
    vector2d<P> const &basis, std::vector<P> &work, span2d<P> intg) const
{
  expect(work.size() == 4 * quad.left_nodes().size());
  expect(basis.num_strips() == degree_ + 1);
  expect(intg.stride() == degree_ + 1);
  expect(intg.num_strips() == num_mom_);

  size_t nquad = quad.left_nodes().size(); // num-quad-points

  // holds the values of the moment-weight, e.g., 1, v, v^2 ...
  P *ml = work.data();
  P *mr = ml + nquad;
  // holds the values of the v-nodes, e.g., v_1, v_2 ...
  P *nl = mr + nquad;
  P *nr = nl + nquad;

  std::copy_n(quad.left_nodes().begin(), nquad, nl);
  std::copy_n(quad.right_nodes().begin(), nquad, nr);

  P const scal = b - a; // domain scale
  { // convert the canonical interval to (b, a)
    P slope = 0.5 * scal, intercept = 0.5 * (b + a);
    for (int i : iindexof(2 * nquad))
      nl[i] = slope * nl[i] + intercept;
  }

  // setting the zeroth moment
  if (dv) // if using non-Cartesian coords
    for (int i : iindexof(2 * nquad))
      ml[i] = dv(nl[i], 0);
  else
    std::fill_n(ml, 2 * nquad, 1.0);

  for (int moment : iindexof(num_mom_))
  {
    if (moment > 0)
      for (int i : iindexof(2 * nquad)) // does both left/right parts
        ml[i] *= nl[i];

    P *ii = intg[moment];
    if (basis.stride() == degree_ + 1)
      for (int d : iindexof(degree_ + 1))
        *ii++ = scal * quad.integrate_lmom(ml, mr, basis[d]);
    else
      for (int d : iindexof(degree_ + 1))
        *ii++ = scal * quad.integrate_wmom(ml, mr, basis[d]);
  }
}

template<typename P>
void moments1d<P>::project_moments(
    int const dim0_level, std::vector<P> const &state,
    elements::table const &etable, std::vector<P> &moments) const
{
  tools::time_event performance("moments project");

  int const mom_outs = 1 + (num_dims_ - 1) * (num_mom_ - 1);

  int const pdof = degree_ + 1;
  int const nout = fm::ipow2(dim0_level);
  if (moments.empty())
    moments.resize(nout * mom_outs * pdof);
  else {
    moments.resize(nout * mom_outs * pdof);
    std::fill(moments.begin(), moments.end(), P{0});
  }

  vector2d<int> cells = etable.get_cells();

  auto const ncells = cells.num_strips();

  int64_t const tsize = fm::ipow(pdof, num_dims_);

  span2d<P const> x(tsize, ncells, state.data());

  span2d<P> smom(mom_outs * pdof, nout, moments.data());

  std::vector<P> work; // persistent workspace

  switch (num_dims_) {
    case 2:
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<2>(x[i], idx, span2d<P>(pdof, mom_outs, smom[idx[0]]), work);
      }
      break;
    case 3:
      work.resize(pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<3>(x[i], idx, span2d<P>(pdof, mom_outs, smom[idx[0]]), work);
      }
      break;
    case 4:
      work.resize(pdof * pdof * pdof + pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<4>(x[i], idx, span2d<P>(pdof, mom_outs, smom[idx[0]]), work);
      }
      break;
  }
}

template<typename P>
void moments1d<P>::project_moments(
    sparse_grid const &grid, std::vector<P> const &state, std::vector<P> &moments) const
{
  tools::time_event performance("moments project");

  int const mom_outs = 1 + (num_dims_ - 1) * (num_mom_ - 1);

  int const pdof = degree_ + 1;
  int const nout = fm::ipow2(grid.current_level(0));
  if (moments.empty())
    moments.resize(nout * mom_outs * pdof);
  else {
    moments.resize(nout * mom_outs * pdof);
    std::fill(moments.begin(), moments.end(), P{0});
  }

  auto const ncells = grid.num_indexes();

  int64_t const tsize = fm::ipow(pdof, num_dims_);

  span2d<P const> x(tsize, ncells, state.data());

  span2d<P> smom(mom_outs * pdof, nout, moments.data());

  std::vector<P> work; // persistent workspace

  switch (num_dims_) {
    case 2:
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = grid[i];
        project_cell<2>(x[i], idx, span2d<P>(pdof, mom_outs, smom[idx[0]]), work);
      }
      break;
    case 3:
      work.resize(pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = grid[i];
        project_cell<3>(x[i], idx, span2d<P>(pdof, mom_outs, smom[idx[0]]), work);
      }
      break;
    case 4:
      work.resize(pdof * pdof * pdof + pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = grid[i];
        project_cell<4>(x[i], idx, span2d<P>(pdof, mom_outs, smom[idx[0]]), work);
      }
      break;
  }
}

template<typename P>
template<int ndims>
void moments1d<P>::project_cell(P const x[], int const idx[], span2d<P> moments,
                                std::vector<P> &work) const
{
  int const pdof = degree_ + 1;
  if constexpr (ndims == 2) // reducing only one dimension
  {
    for (int m : iindexof(num_mom_))
    {
      P const *wm = integ[1][idx[1]] + m * pdof; // moment weights
      P *mout = moments[m];
      for (int i = 0; i < pdof; i++)
      {
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * x[i * pdof + j];
      }
    }
  }
  else if constexpr (ndims == 3) // reducing 2 dimensions, using work as temp storage
  {
    expect(work.size() == static_cast<size_t>(pdof * pdof));
    int pow = 0;

    P const *wm = integ[2][idx[2]];
    for (int i = 0; i < pdof * pdof; i++)
    {
      work[i] = 0;
      for (int j = 0; j < pdof; j++)
        work[i] += wm[j] * x[i * pdof + j];
    }

    wm = integ[1][idx[1]];

    P *mout = moments[0];
    for (int i = 0; i < pdof; i++)
      for (int j = 0; j < pdof; j++)
        mout[i] += wm[j] * work[i * pdof + j];

    for (int m = 1; m < moments.num_strips(); m += 2)
    {
      pow += 1;

      wm = integ[2][idx[2]];
      for (int i = 0; i < pdof * pdof; i++)
      {
        work[i] = 0;
        for (int j = 0; j < pdof; j++)
          work[i] += wm[j] * x[i * pdof + j];
      }

      wm = integ[1][idx[1]] + pow * pdof;
      mout = moments[m];
      for (int i = 0; i < pdof; i++)
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * work[i * pdof + j];

      wm = integ[2][idx[2]] + pow * pdof;
      for (int i = 0; i < pdof * pdof; i++)
      {
        work[i] = 0;
        for (int j = 0; j < pdof; j++)
          work[i] += wm[j] * x[i * pdof + j];
      }

      wm = integ[1][idx[1]];
      mout = moments[m + 1];
      for (int i = 0; i < pdof; i++)
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * work[i * pdof + j];
    }
  }
  else if constexpr (ndims == 4) // reducing 3 dims, using work in 2 stages
  {
    expect(work.size() == static_cast<size_t>(pdof * pdof * pdof + pdof * pdof));
    int pow = 0;

    P const *wm = integ[3][idx[3]];
    for (int i = 0; i < pdof * pdof * pdof; i++)
    {
      work[i] = 0;
      for (int j = 0; j < pdof; j++)
        work[i] += wm[j] * x[i * pdof + j];
    }

    wm = integ[2][idx[2]];
    P *t = work.data() + pdof * pdof * pdof;

    for (int i = 0; i < pdof * pdof; i++)
    {
      t[i] = 0;
      for (int j = 0; j < pdof; j++)
        t[i] += wm[j] * work[i * pdof + j];
    }

    wm = integ[1][idx[1]];
    P *mout = moments[0];
    for (int i = 0; i < pdof; i++)
      for (int j = 0; j < pdof; j++)
        mout[i] += wm[j] * t[i * pdof + j];

    for (int m = 1; m < moments.num_strips(); m += 3)
    {
      pow += 1;

      wm = integ[3][idx[3]];
      for (int i = 0; i < pdof * pdof * pdof; i++)
      {
        work[i] = 0;
        for (int j = 0; j < pdof; j++)
          work[i] += wm[j] * x[i * pdof + j];
      }

      wm = integ[2][idx[2]];
      for (int i = 0; i < pdof * pdof; i++)
      {
        t[i] = 0;
        for (int j = 0; j < pdof; j++)
          t[i] += wm[j] * work[i * pdof + j];
      }

      wm = integ[1][idx[1]] + pow * pdof;
      mout = moments[m];
      for (int i = 0; i < pdof; i++)
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * t[i * pdof + j];

      // second dim
      wm = integ[3][idx[3]];
      for (int i = 0; i < pdof * pdof * pdof; i++)
      {
        work[i] = 0;
        for (int j = 0; j < pdof; j++)
          work[i] += wm[j] * x[i * pdof + j];
      }

      wm = integ[2][idx[2]] + pow * pdof;
      for (int i = 0; i < pdof * pdof; i++)
      {
        t[i] = 0;
        for (int j = 0; j < pdof; j++)
          t[i] += wm[j] * work[i * pdof + j];
      }

      wm = integ[1][idx[1]];
      mout = moments[m + 1];
      for (int i = 0; i < pdof; i++)
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * t[i * pdof + j];

      // third dim
      wm = integ[3][idx[3]] + pow * pdof;
      for (int i = 0; i < pdof * pdof * pdof; i++)
      {
        work[i] = 0;
        for (int j = 0; j < pdof; j++)
          work[i] += wm[j] * x[i * pdof + j];
      }

      wm = integ[2][idx[2]];
      for (int i = 0; i < pdof * pdof; i++)
      {
        t[i] = 0;
        for (int j = 0; j < pdof; j++)
          t[i] += wm[j] * work[i * pdof + j];
      }

      wm = integ[1][idx[1]];
      mout = moments[m + 2];
      for (int i = 0; i < pdof; i++)
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * t[i * pdof + j];
    }
  }
}

template<typename P>
void moments1d<P>::project_moment(
    int const mom, int const dim0_level, std::vector<P> const &state,
    elements::table const &etable, std::vector<P> &moment) const
{
  tools::time_event performance("moment project");

  int const pdof = degree_ + 1;
  int const nout = fm::ipow2(dim0_level);
  if (moment.empty())
    moment.resize(nout * pdof);
  else {
    moment.resize(nout * pdof);
    std::fill(moment.begin(), moment.end(), P{0});
  }

  vector2d<int> cells = etable.get_cells();

  auto const ncells = cells.num_strips();

  int64_t const tsize = fm::ipow(pdof, num_dims_);

  span2d<P const> x(tsize, ncells, state.data());

  span2d<P> smom(pdof, nout, moment.data());

  std::vector<P> work; // persistent workspace

  switch (num_dims_) {
    case 2:
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<2>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
    case 3:
      work.resize(pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<3>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
    case 4:
      work.resize(pdof * pdof * pdof + pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<4>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
  }
}

template<typename P>
void moments1d<P>::project_moment(
    int const mom, sparse_grid const &grid, std::vector<P> const &state,
    std::vector<P> &moment) const
{
  tools::time_event performance("moment project");

  int const pdof = degree_ + 1;
  int const nout = fm::ipow2(grid.current_level(0));
  if (moment.empty())
    moment.resize(nout * pdof);
  else {
    moment.resize(nout * pdof);
    std::fill(moment.begin(), moment.end(), P{0});
  }

  auto const ncells = grid.num_indexes();

  int64_t const tsize = fm::ipow(pdof, num_dims_);

  span2d<P const> x(tsize, ncells, state.data());

  span2d<P> smom(pdof, nout, moment.data());

  std::vector<P> work; // persistent workspace

  switch (num_dims_) {
    case 2:
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = grid[i];
        project_cell<2>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
    case 3:
      work.resize(pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = grid[i];
        project_cell<3>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
    case 4:
      work.resize(pdof * pdof * pdof + pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = grid[i];
        project_cell<4>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
  }
}

template<typename P>
template<int ndims>
void moments1d<P>::project_cell(
    int const mom, P const x[], int const idx[], P moment[], std::vector<P> &work) const
{
  int const pdof = degree_ + 1;
  if constexpr (ndims == 2) // reducing only one dimension
  {
    P const *wm = integ[1][idx[1]] + mom * pdof; // moment weights
    for (int i = 0; i < pdof; i++)
    {
      for (int j = 0; j < pdof; j++)
        moment[i] += wm[j] * x[i * pdof + j];
    }
  }
  else if constexpr (ndims == 3) // reducing 2 dimensions, using work as temp storage
  {
    expect(work.size() == static_cast<size_t>(pdof * pdof));
    int p1 = 0, p2 = 0;
    if (mom > 0) {
      switch (mom) {
        case 1:
          p1 = 1;
          break;
        case 2:
          p2 = 1;
          break;
        case 3:
          p1 = 2;
          break;
        case 4:
          p2 = 2;
          break;
      }
    }

    P const *wm = integ[2][idx[2]] + p2 * pdof; // moment weights

    for (int i = 0; i < pdof * pdof; i++)
    {
      work[i] = 0;
      for (int j = 0; j < pdof; j++)
        work[i] += wm[j] * x[i * pdof + j];
    }

    wm = integ[1][idx[1]] + p1 * pdof;
    for (int i = 0; i < pdof; i++)
    {
      for (int j = 0; j < pdof; j++)
        moment[i] += wm[j] * work[i * pdof + j];
    }
  }
  else if constexpr (ndims == 4) // reducing 3 dims, using work in 2 stages
  {
    expect(work.size() == static_cast<size_t>(pdof * pdof * pdof + pdof * pdof));
    int p1 = 0, p2 = 0, p3 = 0;
    if (mom > 0) {
      switch (mom) {
        case 1:
          p1 = 1;
          break;
        case 2:
          p2 = 1;
          break;
        case 3:
          p3 = 1;
          break;
        case 4:
          p1 = 2;
          break;
        case 5:
          p2 = 2;
          break;
        case 6:
          p3 = 2;
          break;
      }
    }

    P const *wm = integ[3][idx[3]] + p3 * pdof; // moment weights

    for (int i = 0; i < pdof * pdof * pdof; i++)
    {
      work[i] = 0;
      for (int j = 0; j < pdof; j++)
        work[i] += wm[j] * x[i * pdof + j];
    }

    wm = integ[2][idx[2]] + p2 * pdof;
    P *t = work.data() + pdof * pdof * pdof;

    for (int i = 0; i < pdof * pdof; i++)
    {
      t[i] = 0;
      for (int j = 0; j < pdof; j++)
        t[i] += wm[j] * work[i * pdof + j];
    }

    wm = integ[1][idx[1]] + p1 * pdof;
    for (int i = 0; i < pdof; i++)
    {
      for (int j = 0; j < pdof; j++)
        moment[i] += wm[j] * t[i * pdof + j];
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template class moments1d<double>;
#endif
#ifdef ASGARD_ENABLE_FLOAT
template class moments1d<float>;
#endif

} // namespace asgard
