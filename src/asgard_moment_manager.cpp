#include "asgard_moment_manager.hpp"
#include "asgard_coefficients_mats.hpp"

namespace asgard
{

template<typename P>
moment_manager<P>::moment_manager(moments_list &&mlist_in,
                                  std::vector<moments_list> const &mom_groups)
    : mlist(std::move(mlist_in)), raw_vals(mlist.size()), full_level(mlist.size()),
      interps(mlist.size())
{
  if (not mom_groups.empty()) {
    groups_.reserve(mom_groups.size());
    for (auto const &mgroup : mom_groups)
      groups_.push_back( mgroup.find_as_subset_of(mlist) );
  }

  pos_grid.generation_ = -1;
}

template<typename P>
moment_manager<P>::moment_manager(pde_domain<P> const &domain, int degree,
                                  moments_list &&mlist_in,
                                  std::vector<moments_list> const &mom_groups)
    : moment_manager(std::move(mlist_in), mom_groups)
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

  wav_scale  = 1;
  for (int d : iindexof(pos_grid.num_dims()))
    wav_scale *= (domain.xright(d) - domain.xleft(d));
  wav_scale = P{1} / std::sqrt(wav_scale);

  dim_level.fill(moment_level::zero);

  moment const max_moms = mlist.max_moment();

  // this constructor assumes no mass and the degree is high enough
  // to capture all moments into the zero-level element
  expect(pdof > max_moms.pows[0] and pdof > max_moms.pows[1] and pdof > max_moms.pows[2]);

  legendre_basis<P> basis(pdof - 1);

  for (int d = 0; d < num_vel_; d++)
    set_level_zero(domain, basis, max_moms, d);
}

template<typename P>
moment_manager<P>::moment_manager(pde_domain<P> const &domain, int max_level,
                                  legendre_basis<P> const &basis,
                                  hierarchy_manipulator<P> const &hier,
                                  moments_list &&mlist_in,
                                  std::vector<moments_list> const &mom_groups)
    : moment_manager(std::move(mlist_in), mom_groups)
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
      set_level_zero(domain, basis, max_moms, d);
    else
      set_mass(d, domain.xleft(domain.num_pos() + d), domain.xright(domain.num_pos() + d),
               max_level, basis, hier, 1, coeff);
  }
}

template<typename P>
void moment_manager<P>::set_level_zero(pde_domain<P> const &domain, legendre_basis<P> const &basis,
                                       moment const &max_moms, int dim)
{
  dim_level[dim] = moment_level::zero;

  int const num_quad = basis.num_quad;
  std::vector<double> legws(pdof * num_quad);

  rhs_raw_data<double> rhs_raw;
  rhs_raw.pnts.resize(num_quad);
  rhs_raw.vals.resize(rhs_raw.pnts.size());

  std::vector<double> work;
  if constexpr (is_float<P>)
    work.resize(pdof); // scratch space to convert to float

  // setting up quadrature points over all cells in the given dimension
  double const xleft = domain.xleft(domain.num_pos() + dim);
  double const dx    = (domain.xright(domain.num_pos() + dim) - xleft);

  std::copy_n(basis.legw, pdof * num_quad, legws.data());
  smmat::scal(pdof * num_quad, std::sqrt(dx), legws.data());

  for (int k = 0; k < num_quad; k++)
    rhs_raw.pnts[k] = (0.5 * basis.qp[k] + 0.5) * dx + xleft;

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
      smmat::gemtv(num_quad, pdof, legws.data(), rhs_raw.vals.data(), integ[dim][m]);
    } else {
      // convert to single precision
      smmat::gemtv(num_quad, pdof, legws.data(), rhs_raw.vals.data(), work.data());
      std::copy_n(work.data(), pdof, integ[dim][m]);
    }
  }
}

template<typename P>
void moment_manager<P>::set_mass(
    int dim, P xleft, P xright, int max_level, legendre_basis<P> const &basis,
    hierarchy_manipulator<P> const &hier, P scale, rhs_raw_data<P> &coeff)
{
  dim_level[dim] = moment_level::all;

  int const num_cells  = fm::ipow2(max_level);
  int const max_moment = mlist.max_moment(dim);

  int const num_quad = basis.num_quad;

  std::vector<double> legws(pdof * num_quad);

  if (coeff.vals.empty())
    coeff.vals.resize(num_quad * num_cells, scale);

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

  std::copy_n(basis.legw, pdof * num_quad, legws.data());
  smmat::scal(pdof * num_quad, std::sqrt(dx), legws.data());

  #pragma omp parallel for
  for (int i = 0; i < num_cells; i++) {
      double const l = xleft + i * dx; // left edge of cell i
      for (int k = 0; k < num_quad; k++)
        rhs_raw.pnts[i * num_quad + k] = (0.5 * basis.qp[k] + 0.5) * dx + l;
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
      smmat::gemtv(num_quad, pdof, legws.data(), rhs_vals[i], cell_moments[i]);

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
    tools::time_event performance_("cache all moments");
    for (int i : iindexof(mlist.size())) {
      if (mlist[moment_id{i}].action != moment::inactive) {
        compute(grid, moment_id{i}, state, raw_vals.get(moment_id{i}));
        full_level.get(moment_id{i}).resize(0); // will be updated upon request
        interps.get(moment_id{i}).resize(0);
      }
    }
  } else {
    tools::time_event performance_("cache moments (" + std::to_string(group) + ")");
    for (auto const &id : groups_[group]) {
      if (mlist[id].action != moment::inactive) {
        compute(grid, id, state, raw_vals.get(id));
        full_level.get(id).resize(0);
        interps.get(id).resize(0);
      }
    }
  }
}

template<typename P>
void moment_manager<P>::cache_moment(moment_id id, sparse_grid const &grid,
                                     std::vector<P> const &state)
{
  compute(grid, id, state, raw_vals.get(id));
  full_level.get(id).resize(0);
  interps.get(id).resize(0);
}

template<typename P>
void moment_manager<P>::complete_level(hierarchy_manipulator<P> const &hier,
                                       std::vector<P> const &raw,
                                       std::vector<P> &vals) const
{
  tools::time_event performance_("moment complete level");
  int const num_cells = fm::ipow2(pos_grid.level_[0]);
  if (vals.empty())
    vals.resize(pdof * num_cells);
  else {
    vals.resize(pdof * num_cells);
    std::fill(vals.begin(), vals.end(), P{0});
  }

  for (int i = 0; i < pos_grid.num_indexes(); i++)
    std::copy_n(raw.data() + i * pdof, pdof, vals.data() + pos_grid[i][0] * pdof);

  hier.reconstruct1d(pos_grid.level_[0], vals);
}

template<typename P>
void moment_manager<P>::make_nodal(
    moment_id id, interpolation_manager<P> const &interp, connection_patterns const &conn,
    kronmult::workspace<P> &work, std::vector<P> &workspace) const
{
  interp.pos2nodal(pos_grid, conn, raw_vals[id].data(), wav_scale, workspace, work);

  interps[id].resize(pntr.back() * full_block);

  #pragma omp parallel for
  for (int i = 0; i < pos_grid.num_indexes(); i++)
  {
    P *base = interps[id].data() + pntr[i] * full_block;
    for (int j = 0; j < pos_block; j++)
      std::fill_n(base + j * vel_block, vel_block, workspace[i * pos_block + j]);
    P *out = base + full_block;
    for (int j = pntr[i] + 1; j < pntr[i + 1]; j++)
      out = std::copy_n(base, full_block, out);
  }
}

template<typename P>
void moment_manager<P>::load_interp(
    interpolation_manager<P> const &interp, connection_patterns const &conn,
    kronmult::workspace<P> &work, std::vector<P> &workspace) const
{
  for (int i = 0; i < mlist.size(); i++)
    if (mlist[moment_id{i}].action == moment::interpolatory)
      make_nodal(moment_id{i}, interp, conn, work, workspace);
}

template<typename P>
void moment_manager<P>::load_interp(
    int groupid, interpolation_manager<P> const &interp, connection_patterns const &conn,
    kronmult::workspace<P> &work, std::vector<P> &workspace) const
{
  for (auto id : groups_[groupid])
    if (mlist[id].action == moment::interpolatory)
      make_nodal(id, interp, conn, work, workspace);
}

#ifdef ASGARD_ENABLE_DOUBLE
template class moment_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class moment_manager<float>;
#endif

} // namespace asgard
