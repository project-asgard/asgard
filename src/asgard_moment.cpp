#include "asgard_moment.hpp"
#include "asgard_coefficients_mats.hpp"

namespace asgard
{

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

  // for (int d = 1; d < 3; d++) {
  //   std::cout << " ============ dim = " << d << " ============\n";
  //   for (int i = 0; i < nump; i++) {
  //     for (int j = 0; j < pdof; j++) {
  //       for (int k = 0; k < num_mom_; k++) {
  //         std::cout << std::setw(12) << integ[d][i][k * pdof + j] << "    ";
  //       }
  //       std::cout << '\n';
  //     }
  //   }
  // }
  // std::cout << "\n ============================== \n";
}

template<typename P>
void moments1d<P>::integrate(
    basis::canonical_integrator const &quad, P a, P b, scalar_func<P> const &dv,
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
      ml[i] = dv(nl[i]);
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

template<typename P>
moment_manager<P>::moment_manager(moments_list &&mlist_in, std::vector<moments_list> &&mom_groups)
    : mlist(std::move(mlist_in))
{
  if (not mom_groups.empty()) {
    groups_.reserve(mom_groups.size());
    for (auto const &mgroup : mom_groups)
      groups_.push_back( mlist.find_as_subset_of(mgroup) );
  }

  pos_grid.generation_ = -1;
}

template<typename P>
moment_manager<P>::moment_manager(pde_domain<P> const &domain, int degree,
                                  moments_list &&mlist_in,
                                  std::vector<moments_list> &&mom_groups)
    : moment_manager(std::move(mlist_in), std::move(mom_groups))
{
  if (mlist.empty()) // no moments, nothing more to set
    return;

  num_dims_ = domain.num_dims();
  num_vel_  = domain.num_vel();
  pdof      = degree + 1;

  pos_block  = (domain.num_pos() == 0) ? 0 :fm::ipow(pdof, domain.num_pos());
  vel_block  = fm::ipow(pdof, domain.num_vel());
  full_block = fm::ipow(pdof, domain.num_dims());

  pos_grid.iset_.num_dimensions_ = domain.num_pos();

  dim_level.fill(moment_level::zero);

  moment const max_moms = mlist.max_moment();

  // this constructor assumes no mass and the degree is high enough
  // to capture all moments into the zero-level element
  expect(pdof > max_moms.pows[0] and pdof > max_moms.pows[1] and pdof > max_moms.pows[2]);

  for (int d = 0; d < num_vel_; d++)
    set_level_zero(domain, max_moms, d);
}

template<typename P>
moment_manager<P>::moment_manager(pde_domain<P> const &domain, int max_level,
                                  hierarchy_manipulator<P> const &hier,
                                  moments_list &&mlist_in,
                                  std::vector<moments_list> &&mom_groups)
    : moment_manager(std::move(mlist_in), std::move(mom_groups))
{
  if (mlist.empty()) // no moments, nothing more to set
    return;

  num_dims_ = domain.num_dims();
  num_vel_  = domain.num_vel();
  pdof      = hier.degree() + 1;

  pos_block  = (domain.num_pos() == 0) ? 0 : fm::ipow(pdof, domain.num_pos());
  vel_block  = fm::ipow(pdof, domain.num_vel());
  full_block = fm::ipow(pdof, domain.num_dims());

  pos_grid.iset_.num_dimensions_ = domain.num_pos();

  dim_level.fill(moment_level::zero);

  moment const max_moms = mlist.max_moment();

  if (pdof <= max_moms.pows[0] or pdof <= max_moms.pows[1] or pdof <= max_moms.pows[2])
    all_levels_zero = false;

  rhs_raw_data<P> coeff;
  for (int d = 0; d < num_vel_; d++) {
    if (pdof > max_moms.pows[d])
      set_level_zero(domain, max_moms, d);
    else
      set_mass(d, domain.xleft(domain.num_pos() + d), domain.xright(domain.num_pos() + d),
               max_level, hier, coeff);
  }
}

template<typename P>
void moment_manager<P>::set_level_zero(pde_domain<P> const &domain, moment const &max_moms, int dim)
{
  dim_level[dim] = moment_level::zero;

  // TODO: can reuse some of these, maybe take in a Legendre basis
  auto [quadp, quadw]  = legendre_weights(pdof - 1, -1, 1);
  auto [lvals, lprime] = legendre_vals(quadp, pdof - 1);
  ignore(lprime);
  int const num_quad = static_cast<int>(quadw.size());
  std::vector<double> legw(2 * pdof * num_quad);
  double *legws = legw.data() + pdof * num_quad; // scaled points and weights
  smmat::col_scal(num_quad, pdof, quadw.data(), lvals.data(), legw.data());

  rhs_raw_data<double> rhs_raw;
  rhs_raw.pnts.resize(num_quad);
  rhs_raw.vals.resize(rhs_raw.pnts.size());

  std::vector<double> work;
  if constexpr (is_float<P>)
    work.resize(pdof); // scratch space to convert to float

  // setting up quadrature points over all cells in the given dimension
  double const xleft = domain.xleft(domain.num_pos() + dim);
  double const dx    = (domain.xright(domain.num_pos() + dim) - xleft);

  std::copy_n(legw.begin(), pdof * num_quad, legws);
  smmat::scal(pdof * num_quad, 0.5 * std::sqrt(dx), legws);

  for (int k = 0; k < num_quad; k++)
    rhs_raw.pnts[k] = (0.5 * quadp[k] + 0.5) * dx + xleft;

  integ[dim] = vector2d<P>(pdof, max_moms.pows[dim] + 1);
  for (int m = 0; m <= max_moms.pows[dim]; m++)
  {
    switch (m) {
    case 0:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad; i++)
        rhs_raw.vals[i] = 1.0;
      break;
    case 1:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad; i++)
        rhs_raw.vals[i] = rhs_raw.pnts[i];
      break;
    case 2:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad; i++)
        rhs_raw.vals[i] = rhs_raw.pnts[i] * rhs_raw.pnts[i];
      break;
    case 3:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad; i++)
        rhs_raw.vals[i] = rhs_raw.pnts[i] * rhs_raw.pnts[i] * rhs_raw.pnts[i];
      break;
    default:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad; i++)
        rhs_raw.vals[i] = fm::powi(rhs_raw.pnts[i], m);
      break;
    };

    if constexpr (is_double<P>) {
      smmat::gemtv(num_quad, pdof, legws, rhs_raw.vals.data(), integ[dim][m]);
    } else {
      // convert to single precision
      smmat::gemtv(num_quad, pdof, legws, rhs_raw.vals.data(), work.data());
      std::copy_n(work.data(), pdof, integ[dim][m]);
    }
  }
}

template<typename P>
void moment_manager<P>::set_mass(
    int dim, P xleft, P xright, int max_level,
    hierarchy_manipulator<P> const &hier, rhs_raw_data<P> &coeff)
{
  dim_level[dim] = moment_level::all;

  int const num_cells  = fm::ipow2(max_level);
  int const max_moment = mlist.max_moment(dim);

  // TODO: can reuse some of these, maybe take in a Legendre basis
  auto [quadp, quadw]  = legendre_weights(pdof - 1, -1, 1);
  auto [lvals, lprime] = legendre_vals(quadp, pdof - 1);
  ignore(lprime);
  int const num_quad = static_cast<int>(quadw.size());
  std::vector<double> legw(2 * pdof * num_quad);
  double *legws = legw.data() + pdof * num_quad; // scaled points and weights
  smmat::col_scal(num_quad, pdof, quadw.data(), lvals.data(), legw.data());

  if (coeff.vals.empty())
    coeff.vals.resize(num_quad * num_cells, 1);

  rhs_raw_data<double> rhs_raw;
  rhs_raw.pnts.resize(num_quad * num_cells);
  rhs_raw.vals.resize(rhs_raw.pnts.size());
  span2d<double> rhs_vals(num_quad, num_cells, rhs_raw.vals.data());

  vector2d<double> cell_moments(pdof, num_cells);
  std::vector<float> work;
  if constexpr (is_float<P>)
    work.resize(num_cells * pdof); // scratch space to convert to float

  // setting up quadrature points over all cells in the given dimension
  double const dx = (xright - xleft) / static_cast<double>(num_cells);

  std::copy_n(legw.begin(), pdof * num_quad, legws);
  smmat::scal(pdof * num_quad, 0.5 * std::sqrt(dx), legws);

  #pragma omp parallel for
  for (int i = 0; i < num_cells; i++) {
      double const l = xleft + i * dx; // left edge of cell i
      for (int k = 0; k < num_quad; k++)
      rhs_raw.pnts[i * num_quad + k] = (0.5 * quadp[k] + 0.5) * dx + l;
  }

  integ[dim] = vector2d<P>(num_cells * pdof, max_moment + 1);
  for (int m = 0; m <= max_moment; m++)
  {
    switch (m) {
    case 0:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad * num_cells; i++)
        rhs_raw.vals[i] = coeff.vals[i];
    break;
    case 1:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad * num_cells; i++)
        rhs_raw.vals[i] = coeff.vals[i] * rhs_raw.pnts[i];
    break;
    case 2:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad * num_cells; i++)
        rhs_raw.vals[i] = coeff.vals[i] * rhs_raw.pnts[i] * rhs_raw.pnts[i];
    break;
    case 3:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad * num_cells; i++)
        rhs_raw.vals[i] = coeff.vals[i] * rhs_raw.pnts[i] * rhs_raw.pnts[i] * rhs_raw.pnts[i];
    break;
    default:
      ASGARD_OMP_PARFOR_SIMD
      for(int i = 0; i < num_quad * num_cells; i++)
        rhs_raw.vals[i] = coeff.vals[i] * fm::powi(rhs_raw.pnts[i], m);
    break;
    };

    #pragma omp parallel for
    for (int i = 0; i < num_cells; i++)
      smmat::gemtv(num_quad, pdof, legws, rhs_vals[i], cell_moments[i]);

    if constexpr (is_double<P>) {
      hier.transform(max_level, cell_moments[0], integ[dim][m]);
    } else {
      // convert to single precision before transformation
      std::copy_n(cell_moments[0], work.size(), work.data());
      hier.transform(max_level, work.data(), integ[dim][m]);
    }
  }
}

template<typename P>
template<int npos>
void moment_manager<P>::reduce_grid(sparse_grid const &grid) const
{
  expect(npos == pos_grid.num_dims());
  std::vector<int> &pos_indexes = pos_grid.iset_.indexes_;
  pos_indexes.resize(npos, 0); // zero index
  pos_indexes.reserve(grid.num_indexes() * npos);
  pntr.resize(1);
  pntr.reserve(grid.num_indexes() + 1);

  auto position_mismatch = [&](int const idx1[], int const idx2[])
        -> bool {
        if constexpr (npos == 1)
          return (idx1[0] != idx2[0]);
        else if constexpr (npos == 2)
          return (idx1[0] != idx2[0] or idx1[1] != idx2[1]);
        else if constexpr (npos == 3)
          return (idx1[0] != idx2[0] or idx1[1] != idx2[1] or idx1[2] != idx2[2]);
        else
          return false; // unreachable
      };

  int ipos = 0;

  // this loop is sequential (do not use parallel for)
  for (int i = 0; i < grid.num_indexes(); i++)
  {
    if (position_mismatch(pos_grid[ipos], grid[i])) { // found new entry
      pos_indexes.insert(pos_indexes.end(), grid[i], grid[i] + npos);
      pntr.push_back(i);
      ipos++;
    }
  }

  pos_grid.iset_.num_indexes_ = ipos + 1;
  pntr.push_back(grid.num_indexes());
  pos_grid.generation_ = grid.generation();

  // take the highest levels for full-level vectors
  for (int d = 0; d < pos_grid.num_dims(); d++)
    pos_grid.level_[d] = grid.level_[d];
}

template<typename P>
template<int nvel, int tpdof>
void moment_manager<P>::compute(sparse_grid const &grid, moment_id id,
                                std::vector<P> const &state, std::vector<P> &vals) const
{
  int const num = pos_grid.num_indexes();
  vals.resize(pos_block * num);

  moment const mom = mlist[id]; // using this to get the necessary powers

  bool allzero = all_levels_zero;
  std::array<bool, max_mom_dims> lzero;
  if (not allzero) {
    for (int d = 0; d < max_mom_dims; d++)
      lzero[d] = (pdof > mom[d]);
    allzero = lzero[0] and lzero[1] and lzero[2]; // assuming only 3 entries
  }

  if (allzero) { // simple case, consider only zero-th indexes
    #pragma omp parallel for
    for (int i = 0; i < num; i++)
    {
      P const *v1 = integ[0][mom.pows[0]];
      P const *v2 = (nvel >= 1) ? integ[1][mom.pows[1]] : nullptr;
      P const *v3 = (nvel >= 2) ? integ[2][mom.pows[2]] : nullptr;

      P const *in  = state.data() + full_block * pntr[i];
      P *out       = vals.data() + pos_block * i;

      if constexpr (nvel == 1) {
        for (int j = 0; j < pos_block; j++) {
          P sum = 0;
          for (int k = 0; k < tpdof; k++)
            sum += v1[k] * (*in++);
          out[j] = sum;
        }
      } else if constexpr (nvel == 2) {
        for (int j = 0; j < pos_block; j++) {
          P sum1 = 0;
          for (int k1 = 0; k1 < tpdof; k1++) {
            P sum2 = 0;
            for (int k2 = 0; k2 < tpdof; k2++) {
              sum2 += v2[k2] * (*in++);
            }
            sum1 += v1[k1] * sum2;
          }
          out[j] = sum1;
        }
      } else if constexpr (nvel == 3) {
        for (int j = 0; j < pos_block; j++) {
          P sum1 = 0;
          for (int k1 = 0; k1 < tpdof; k1++) {
            P sum2 = 0;
            for (int k2 = 0; k2 < tpdof; k2++) {
              P sum3 = 0;
              for (int k3 = 0; k3 < tpdof; k3++) {
                sum3 += v3[k3] * (*in++);
              }
              sum2 += v2[k2] * sum3;
            }
            sum1 += v1[k1] * sum2;
          }
          out[j] = sum1;
        }
      }
    }
    return;
  }

  int const npos = pos_grid.num_dims();

  #pragma omp parallel for
  for (int i = 0; i < num; i++)
  {
    P *out = vals.data() + pos_block * i;
    std::fill_n(out, pos_block, P{0});

    for (int j = pntr[i]; j < pntr[i + 1]; j++)
    {
      // some directions may have only level zero entries, then if the index is non-zero
      // the moment contribution is zero and the index can be skipped
      if constexpr (nvel == 2) {
        if ((lzero[0] and grid[j][npos] != 0)
            or (lzero[1] and grid[j][npos + 1] != 0))
        continue;
      } else if constexpr (nvel == 3) {
        if ((lzero[0] and grid[j][npos] != 0)
            or (lzero[1] and grid[j][npos + 1] != 0)
              or (lzero[2] and grid[j][npos + 2] != 0))
        continue;
      }

      // if we got here, the j-th index has a contribution to the i-th block
      P const *v1 = integ[0][mom.pows[0]];
      if (dim_level[0] == moment_level::all) {
        v1 += grid[j][npos] * tpdof;
      }
      P const *v2, *v3;
      if constexpr (nvel >= 2) {
        v2 = integ[1][mom.pows[1]];
        if (dim_level[1] == moment_level::all)
          v2 += grid[j][npos + 1] * tpdof;
      }
      if constexpr (nvel >= 3) {
        v3 = integ[2][mom.pows[2]];
        if (dim_level[2] == moment_level::all)
          v3 += grid[j][npos + 2] * tpdof;
      }

      P const *in  = state.data() + full_block * j;

      // TODO: test SIMD directives below, although this is pretty cheap overall
      if constexpr (nvel == 1) {
        // #pragma omp simd
        for (int k = 0; k < pos_block; k++) {
          P sum = 0;
          // #pragma omp simd reduction(+:sum)
          for (int k1 = 0; k1 < tpdof; k1++) {
            sum += v1[k1] * (*in++);
          }
          out[k] += sum;
        }
      } else if constexpr (nvel == 2) {
        for (int k = 0; k < pos_block; k++) {
          P sum1 = 0;
          for (int k1 = 0; k1 < tpdof; k1++) {
            P sum2 = 0;
            for (int k2 = 0; k2 < tpdof; k2++) {
              sum2 += v2[k2] * (*in++);
            }
            sum1 += v1[k1] * sum2;
          }
          out[k] += sum1;
        }
      } else if constexpr (nvel == 3) {
        for (int k = 0; k < pos_block; k++) {
          P sum1 = 0;
          for (int k1 = 0; k1 < tpdof; k1++) {
            P sum2 = 0;
            for (int k2 = 0; k2 < tpdof; k2++) {
              P sum3 = 0;
              for (int k3 = 0; k3 < tpdof; k3++) {
                sum3 += v3[k3] * (*in++);
              }
              sum2 += v2[k2] * sum3;
            }
            sum1 += v1[k1] * sum2;
          }
          out[k] += sum1;
        }
      }

    } // for grid indexes j
  } // for pos_gird indexes i
}

template<typename P>
template<int nvel>
void moment_manager<P>::compute(sparse_grid const &grid, moment_id id,
                                std::vector<P> const &state, std::vector<P> &vals) const
{
  switch (pdof) {
  case 1:
    compute<nvel, 1>(grid, id, state, vals);
    break;
  case 2:
    compute<nvel, 2>(grid, id, state, vals);
    break;
  case 3:
    compute<nvel, 3>(grid, id, state, vals);
    break;
  case 4:
    compute<nvel, 4>(grid, id, state, vals);
    break;
  default:
    // unreachable
    break;
  };
}

template<typename P>
void moment_manager<P>::compute(sparse_grid const &grid, moment_id id,
                                std::vector<P> const &state, std::vector<P> &vals) const
{
  if (pos_grid.generation() != grid.generation()) { // grid changed, must rebuild
    switch (pos_grid.num_dims()) {
    case 1:
      reduce_grid<1>(grid);
      break;
    case 2:
      reduce_grid<2>(grid);
      break;
    case 3:
      reduce_grid<3>(grid);
      break;
    default:
      break;
    };
    pos_grid.generation_ = grid.generation();
  }

  switch (num_vel_) {
  case 1:
    compute<1>(grid, id, state, vals);
    break;
  case 2:
    compute<2>(grid, id, state, vals);
    break;
  case 3:
    compute<3>(grid, id, state, vals);
    break;
  default:
    break;
  };
}

template<typename P>
void moment_manager<P>::cache_moments(
    sparse_grid const &grid, std::vector<P> const &state, int group) const
{
  if (group < 0) { // do all moments
    for (int i : iindexof(mlist.size())) {
      if (mlist[moment_id{i}].action != moment::inactive)
        compute(grid, moment_id{i}, state, raw_vals.get(moment_id{i}));
      full_level.get(moment_id{i}).resize(0); // will be updated upon request
    }
  } else {
    for (auto const &id : groups_[group]) {
      if (mlist[id].action != moment::inactive)
        compute(grid, id, state, raw_vals.get(id));
      full_level.get(id).resize(0);
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template class moments1d<double>;
template class moment_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class moments1d<float>;
template class moment_manager<float>;
#endif

} // namespace asgard
