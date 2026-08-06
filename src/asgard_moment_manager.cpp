#include "asgard_moment_manager.hpp"
#include "asgard_coefficients_mats.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_tensors.hpp"
#endif

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

  #ifdef ASGARD_USE_GPU
  for (auto &ginterp : gpu_interps)
    ginterp = momentset_gpu<P>(mlist.size());
  #endif

  // start with an invalid generation, triggers update-sync with the full grid
  pos_grid.generation_ = -1;
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
  num_pos_  = domain.num_pos();
  pdof      = hier.degree() + 1;

  vel_block  = fm::ipow(pdof, domain.num_vel());
  full_block = fm::ipow(pdof, domain.num_dims());

  pos_grid.block_size_ = (domain.num_pos() == 0) ? 0 : fm::ipow(pdof, domain.num_pos());
  pos_grid.iset_.num_dimensions_ = domain.num_pos();

  wav_scale  = 1;
  for (int d : iindexof(pos_grid.num_dims())) {
    wav_scale *= (domain.xright(d) - domain.xleft(d));
  }
  wav_scale = P{1} / std::sqrt(wav_scale);

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
void moment_manager<P>::set_poisson(int const max_level, sparse_grid const &grid,
                                    std::array<P, max_num_dimensions> const &xleft,
                                    std::array<P, max_num_dimensions> const &xright,
                                    connection_patterns const &conn,
                                    hierarchy_manipulator<P> const &hier,
                                    build_term_func<P> build_func, prog_opts const &opts)
{
  if (mlist.has_electric()) {
    moment_id const m0 = find_id(moment::zero(num_vel_));
    if (num_pos_ == 1) {
      moment_id const melectric = find_id(moment::electric(dimension_id(0), num_pos_));
      poisson_solver = poisson_1d<P>(hier.degree(), xleft[0], xright[0], grid.current_level(0), m0, melectric);
    } else {
      rassert(opts.poisson_tolerance, "The tolerance for the poisson solver must be provided. "
                                      "Use -poisson-tol or -pt");
      rassert(opts.poisson_iterations, "The maximum number of iterations for the poisson solver must be provided. "
                                       "Use -poisson-iter or -pi");
      poisson_solver = poisson_md<P>(num_pos_, max_level, xleft, xright, conn, hier, mlist, build_func, m0, opts);
    }
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
  #ifdef ASGARD_USE_GPU
  int const num_gpus = compute->num_gpus();
  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    gpu_integ[g][dim] = integ[dim].data_vector();
  }
  #endif
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
  #ifdef ASGARD_USE_GPU
  int const num_gpus = compute->num_gpus();
  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    gpu_integ[g][dim] = integ[dim].data_vector();
  }
  #endif
}

template<typename P>
template<int npos>
void moment_manager<P>::reduce_grid(sparse_grid const &grid) const
{
  assert(npos == pos_grid.num_dims());
  std::vector<int> &pos_indexes = pos_grid.iset_.indexes_;
  pos_indexes.resize(npos, 0); // zero index
  pos_indexes.reserve(grid.num_indexes() * npos);
  pntr.reserve(grid.num_indexes() + 1);
  pntr.resize(1); // 0 -> 0 case

  #ifdef ASGARD_USE_GPU
  std::vector<int> rij;
  rij.reserve(2 * grid.num_indexes());
  rij.resize(2); // 0 -> 0 case
  std::vector<int> rij_zero;
  rij_zero.reserve(2 * grid.num_indexes());
  rij_zero.resize(2); // 0 -> 0 case
  #endif

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
  for (int i = 1; i < grid.num_indexes(); i++)
  {
    if (position_mismatch(pos_grid[ipos], grid[i])) { // found new entry
      pos_indexes.insert(pos_indexes.end(), grid[i], grid[i] + npos);
      pntr.push_back(i);
      ipos++;
      #ifdef ASGARD_USE_GPU
      rij_zero.push_back(ipos);
      rij_zero.push_back(i);
      #endif
    }
    #ifdef ASGARD_USE_GPU
    rij.push_back(ipos);
    rij.push_back(i);
    #endif
  }

  pos_grid.iset_.num_indexes_ = ipos + 1;
  pntr.push_back(grid.num_indexes());
  pos_grid.generation_ = grid.generation();

  // take the highest levels for full-level vectors
  for (int d = 0; d < pos_grid.num_dims(); d++)
    pos_grid.level_[d] = grid.level_[d];

  #ifdef ASGARD_USE_GPU
  int const num_gpus = compute->num_gpus();
  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});
    reduce_ij[g]         = rij;
    reduce_ij_allzero[g] = rij_zero;
  }
  #endif
}

template<typename P>
template<int nvel, int tpdof>
void moment_manager<P>::mcompute(sparse_grid const &grid, moment_id id,
                                 std::vector<P> const &state, std::vector<P> &vals) const
{
  int const num       = pos_grid.num_indexes();
  int const pos_block = pos_grid.block_size();

  moment const mom = mlist[id]; // using this to get the necessary powers
  rassert(not mom.is_electric(), "mcompute shouldn't be called on electric field moments, use solve_poisson instead")

  vals.resize(pos_grid.num_dof());

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
      P const *v1 = integ[0][mom.pows[0]] + grid[j][npos] * tpdof;
      P const *v2, *v3;
      if constexpr (nvel >= 2)
        v2 = integ[1][mom.pows[1]] + grid[j][npos + 1] * tpdof;

      if constexpr (nvel >= 3)
        v3 = integ[2][mom.pows[2]] + grid[j][npos + 2] * tpdof;

      P const *in = state.data() + full_block * j;

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
void moment_manager<P>::mcompute(sparse_grid const &grid, moment_id id,
                                std::vector<P> const &state, std::vector<P> &vals) const
{
  switch (pdof) {
  case 1:
    mcompute<nvel, 1>(grid, id, state, vals);
    break;
  case 2:
    mcompute<nvel, 2>(grid, id, state, vals);
    break;
  case 3:
    mcompute<nvel, 3>(grid, id, state, vals);
    break;
  case 4:
    mcompute<nvel, 4>(grid, id, state, vals);
    break;
  default:
    // unreachable
    break;
  };
}

template<typename P>
void moment_manager<P>::update_position_grid(sparse_grid const &grid) const
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
}

template<typename P>
void moment_manager<P>::mcompute(sparse_grid const &grid, moment_id id,
                                 std::vector<P> const &state, std::vector<P> &vals) const
{
  update_position_grid(grid);

  switch (num_vel_) {
  case 1:
    mcompute<1>(grid, id, state, vals);
    break;
  case 2:
    mcompute<2>(grid, id, state, vals);
    break;
  case 3:
    mcompute<3>(grid, id, state, vals);
    break;
  default:
    break;
  };
}

template<typename P>
void moment_manager<P>::cache_moments(
    group_id group, sparse_grid const &grid, std::vector<P> const &state,
    connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
    interpolation_manager<P> const &interp, kronmult::workspace<P> &work) const
{
  // Compute moments on hierarchical grid
  if (group == group_id::all()) { // do all moments
    tools::time_event performance_("cache all moments");
    for (auto mid : raw_moments_) {
      if (mid == moment_id::unset() or needs_poisson(mid)) continue;
      mcompute(grid, mid, state, raw_vals.get(mid));
      full_level.get(mid).resize(0);
    }
    solve_poisson(grid, conn, hier, interp, work);
  } else {
    tools::time_event performance_("cache moments (" + std::to_string(group()) + ")");
    bool has_poisson = false;
    for (auto mid = first_in(group, raw_moments_);
         *mid != moment_id::unset(); mid++) {
      if (needs_poisson(*mid))
        has_poisson = true;
      else
        mcompute(grid, *mid, state, raw_vals.get(*mid));
      full_level.get(*mid).resize(0);
    }
    if (has_poisson) solve_poisson(grid, conn, hier, interp, work);
  }

  // Poisson moments are already interpolated if they were computed on GPU
  auto skip_poisson = [&](moment_id mid) -> bool {
    #ifdef ASGARD_USE_GPU
      return needs_poisson(mid);
    #else
      std::ignore = mid;
      return false;
    #endif
  };

  // Interpolate moments
  size_t const num_entries = interp.it1.size();
  if (group == group_id::all()) {
    for (auto mid : interp_moments_) {
      if (skip_poisson(mid) or mid == moment_id::unset()) continue;
      make_nodal(mid, interp, hier, work, interp.it1);
    }
  } else {
    for (auto mid = first_in(group, interp_moments_);
         *mid != moment_id::unset(); mid++) {
      if (skip_poisson(*mid)) continue;
      make_nodal(*mid, interp, hier, work, interp.it1);
    }
  }
  interp.it1.resize(num_entries);
}

template<typename P>
void moment_manager<P>::cache_moment(moment_id id, sparse_grid const &grid,
                                     std::vector<P> const &state) const
{
  rassert(not needs_poisson(id), "Electric field moments should not be cached individually.");
  mcompute(grid, id, state, raw_vals.get(id));
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
void moment_manager<P>::solve_poisson(sparse_grid const &grid, connection_patterns const &conn,
                                      hierarchy_manipulator<P> const &hier, interpolation_manager<P> const &interp,
                                      kronmult::workspace<P> &work, bool result_to_cpu) const
{
  std::visit([&](auto &p) {
      if constexpr (std::is_same_v<std::decay_t<decltype(p)>, poisson_1d<P>>) {
        p.solve_periodic(get_cached_level(p.moment0(), hier), full_level.get(p.moment_electric()));
      } else if constexpr (std::is_same_v<std::decay_t<decltype(p)>, poisson_md<P>>) {
        #ifdef ASGARD_USE_GPU
        compute->set_device(gpu::device{0});
        int const num_entries = pos_grid.num_dof();
        std::array<gpu::vector<P>, max_num_gpus> &work1 = interp.gpu_it1;
        assert(work1[0].size() >= num_entries);
        work1[0].copy_from_host(num_entries, raw_vals.get(p.moment0()).data());
        for (moment_id const &mid : p.moments_electric()) {
          if (mid == moment_id::unset()) continue;
          gpu::vector<P> &res = gpu_interps[0][mid];
          res.resize(full_block * grid.num_indexes());
        }
        solve_poisson_gpu(grid, conn, hier, interp, work, result_to_cpu);
        auto find_mom = [this](moment_id mid) -> const mom_on_gpu &
        {
          for (auto const &mom : this->gpu_moments[0]) {
            if (mom.mid == mid)
              return mom;
          }
          throw std::runtime_error("could not find the moment");
        };
        for (moment_id const &mid : p.moments_electric()) {
          if (mid == moment_id::unset()) continue;
          mom_on_gpu const &mom = find_mom(mid);
          if (mom.interp_on_cpu()) continue; // solve_poisson_gpu will already copy if interp_on_cpu() is true
          gpu::vector<P> &res = gpu_interps[0][mid];
          res.copy_to_host(interps.get(mid));
        }
        #else
        std::ignore = interp;
        std::ignore = grid;
        std::ignore = result_to_cpu;
        update_position_grid_dsort();
        p.solve_periodic(raw_vals.get(p.moment0()), raw_vals, pos_grid, conn, work);
        #endif
      }
    }, poisson_solver);
}

#ifdef ASGARD_USE_GPU
template<typename P>
void moment_manager<P>::solve_poisson_gpu(sparse_grid const &grid, connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                                          interpolation_manager<P> const &interp, kronmult::workspace<P> &work, bool result_to_cpu) const
{
  std::visit([&](auto &p) {
      if constexpr (std::is_same_v<std::decay_t<decltype(p)>, poisson_1d<P>>) {
        // Setup
        int64_t const num_entries = pos_grid.num_dof();
        std::array<gpu::vector<P>, max_num_gpus> &work1 = interp.gpu_it1;
        std::array<gpu::vector<P>, max_num_gpus> &work2 = interp.gpu_it2;
        assert(work1[0].size() >= num_entries);
        assert(work2[0].size() >= num_entries);
        gpu::wrap_array<P> w1(work1[0].data(), num_entries);
        gpu::wrap_array<P> w2(work2[0].data(), num_entries);

        // Solve on CPU
        moment_id const m0 = p.moment0();
        moment_id const melectric = p.moment_electric();
        w1.vec.copy_to_host(raw_vals[m0]);
        p.solve_periodic(get_cached_level(m0, hier), full_level.get(melectric));

        // interpolate
        // TODO: interpolation should only happen if set_adapt_weight() uses melectric
        // but there is currently no way to determine this.
        cache_raw_from_level(melectric, hier);
        w1.vec.copy_from_host(raw_vals[melectric].size(), raw_vals[melectric].data()); // Somehow this line messes up bgk test, I suspect w1 is being overwritten and then used somewhere else
        interp.pos2nodal(gpu::device{0}, pos_grid, w1.vec.data(), wav_scale, w2.vec.data(), work);
        gpu::vector<P> &res = gpu_interps[0][melectric];
        res.resize(full_block * grid.num_indexes());
        moment_expand(pdof, pos_grid.num_dims(), num_vel_, reduce_ij[0], w2.vec, res);
        if (result_to_cpu) res.copy_to_host(interps[melectric]);
      } else if constexpr (std::is_same_v<std::decay_t<decltype(p)>, poisson_md<P>>) {
        // Setup
        update_position_grid_dsort();
        int64_t const num_entries = pos_grid.num_dof();
        std::array<gpu::vector<P>, max_num_gpus> &work1 = interp.gpu_it1;
        std::array<gpu::vector<P>, max_num_gpus> &work2 = interp.gpu_it2;
        assert(work1[0].size() >= num_entries);
        assert(work2[0].size() >= num_entries);
        gpu::wrap_array<P> w1(work1[0].data(), num_entries);
        gpu::wrap_array<P> w2(work2[0].data(), num_entries);
        auto find_mom = [this](moment_id mid) -> const mom_on_gpu &
        {
          for (auto const &mom : this->gpu_moments[0]) {
            if (mom.mid == mid)
              return mom;
          }
          throw std::runtime_error("could not find the moment");
        };
        auto interp_func = [&](gpu::vector<P> const &efield, moment_id mid) -> void
        {
          mom_on_gpu const &mom = find_mom(mid);
          if (result_to_cpu or mom.raw_on_cpu()) {
            efield.copy_to_host(this->raw_vals[mid]);
            this->full_level.get(mid).resize(0);
          }
          interp.pos2nodal(gpu::device{0}, this->pos_grid, efield.data(), this->wav_scale, w2.vec.data(), work);
          gpu::vector<P> &res = this->gpu_interps[0][mid];
          res.resize(this->full_block * grid.num_indexes());
          moment_expand(pdof, this->pos_grid.num_dims(), num_vel_, reduce_ij[0], w2.vec, res);
          if (result_to_cpu or mom.interp_on_cpu())
            res.copy_to_host(interps[mid]);
        };
        // Solve
        p.solve_periodic(w1.vec, pos_grid, conn, interp_func, work);
      }
    },
    poisson_solver);
}
#endif // ASGARD_USE_GPU

template<typename P>
void moment_manager<P>::update_position_grid_dsort() const
{
  if (dsort_generation != pos_grid.generation()) {
    pos_grid.dsort_ = dimension_sort(pos_grid.iset_);
    bool constexpr skip_indexes = true; // already loaded in reduce_grid()
    pos_grid.gpu_sync<skip_indexes>();
    dsort_generation = pos_grid.generation();
  }
}

template<typename P>
void moment_manager<P>::make_nodal(
    moment_id id, interpolation_manager<P> const &interp,
    hierarchy_manipulator<P> const &hier,
    kronmult::workspace<P> &kwork, std::vector<P> &workspace) const
{
  update_position_grid_dsort();

  if (num_pos_ == 1 and needs_poisson(id))
    cache_raw_from_level(id, hier);
    
  interp.pos2nodal(pos_grid, raw_vals[id].data(), wav_scale, workspace, kwork);

  interps[id].resize(pntr.back() * full_block);

  int const pos_block = pos_grid.block_size();

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
void moment_manager<P>::compute_interps(
    std::vector<moment_id> const &ids, sparse_grid const &grid,
    std::vector<P> const &state, interpolation_manager<P> const &interp,
    connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
    kronmult::workspace<P> &work) const
{
  size_t const num_entries = interp.it1.size();
  bool has_poisson = false;
  for (auto const &id : ids) {
    if (needs_poisson(id))
      has_poisson = true;
    else
      cache_moment(id, grid, state);
  }
  if (has_poisson) {
    std::visit([&](auto &p) {
      if constexpr (not std::is_same_v<std::decay_t<decltype(p)>, std::monostate>)
        cache_moment(p.moment0(), grid, state);
      }, poisson_solver);
    solve_poisson(grid, conn, hier, interp, work);
  }
  for (auto const &id : ids)
    make_nodal(id, interp, hier, work, interp.it1);
  interp.it1.resize(num_entries);
}

template<typename P>
size_t moment_manager<P>::used_bytes() const {
  return raw_vals.used_bytes() + full_level.used_bytes() + interps.used_bytes();
}

template<typename P>
void moment_manager<P>::set_moment_types(
    std::vector<std::vector<moment_id>> const &raws,
    std::vector<std::vector<moment_id>> const &intps)
{
  assert(not raws.empty());
  size_t const num_groups = raws.size();
  assert(intps.size() == num_groups);

  size_t const num_raws = [&]() -> size_t {
      size_t sum = 0;
      for (auto const &r : raws) sum += r.size();
      return sum;
    }();

  raw_moments_.reserve(num_raws + num_groups);
  for (auto const &rv : raws) {
    for (moment_id mid : rv)
      raw_moments_.push_back(mid);
    raw_moments_.push_back(moment_id::unset());
  }

  size_t const num_interp = [&]() -> size_t {
      size_t sum = 0;
      for (auto const &i : intps) sum += i.size();
      return sum;
    }();

  interp_moments_.reserve(num_interp + num_groups);
  for (auto const &iv : intps) {
    for (moment_id mid : iv)
      interp_moments_.push_back(mid);
    interp_moments_.push_back(moment_id::unset());
  }
}

#ifdef ASGARD_USE_GPU
template<typename P>
void moment_manager<P>::set_moment_distribution(
    std::array<std::vector<std::vector<moment_id>>, max_num_gpus> const &gpu_mom,
    std::vector<std::vector<moment_id>> const &cpu_raw,
    std::vector<std::vector<moment_id>> const &cpu_interp,
    std::vector<std::vector<moment_id>> const &skip_interp)
{
  // avoiding the double-vector, lumping moment for all groups together
  // group-0-moment-0, ..., moment_id::unset(), group-1-moment-0, ...., unset()
  for (int dev : iindexof(max_num_gpus)) {
    auto const &groups = gpu_mom[dev];
    size_t num_moms = 0; // get the total number of moments for this GPU
    for (auto const &m : groups)
      num_moms += m.size();
    if (num_moms == 0) { // nothing to do on this device
      gpu_moments[dev].resize(0);
    } else { // some group has at least one moment
      gpu_moments[dev].reserve(num_moms + groups.size());
      for (auto const &mg : groups) {
        for (auto const &m : mg) // copy the moments for this group
          gpu_moments[dev].push_back(mom_on_gpu{m});
        // using unset moments to indicate the end of the group
        gpu_moments[dev].push_back(mom_on_gpu{});
      }
      assert(num_moms + groups.size() == gpu_moments[dev].size());
    }
  }

  // find the first moment matching the given id and group
  auto find_mom = [&](moment_id mid, group_id group) -> mom_on_gpu &
    {
      int gid = 0;
      for (auto &dev : gpu_moments) {
        for (auto &mom : dev) {
          if (not mom) { // moving to the next group
            gid++;
            continue;
          }
          if (group != group_id{gid})
            continue;
          if (mom.mid == mid)
            return mom;
        }
      }
      throw std::runtime_error("could not find the moment");
    };

  int gid = 0;
  for (auto const &gvec : cpu_raw) {
    for (auto mid : gvec) {
      find_mom(mid, group_id{gid}).set_raw_on_cpu();
    }
    gid++;
  }
  gid = 0;
  for (auto const &gvec : cpu_interp) {
    for (auto mid : gvec)
      find_mom(mid, group_id{gid}).set_interp_on_cpu();
    gid++;
  }
  gid = 0;
  for (auto const &gvec : skip_interp) {
    for (auto mid : gvec) {
      find_mom(mid, group_id{gid}).set_skip_interp();
    }
    gid++;
  }

  // check which group has any interpolation terms, i.e., must sync dsort
  has_interp.resize(gid + 1);
  std::fill(has_interp.begin(), has_interp.end(), false);
  for (auto &dev : gpu_moments) {
    gid = 0;
    for (auto const &mom : dev) {
      if (mom.is_unset()) {
        gid++;
        continue;
      }
      if (not has_interp[gid] and not mom.skip_interp())
        has_interp[gid] = true;
    }
  }
  if (std::none_of(has_interp.begin(), has_interp.end(),
                   [](bool const &v)-> bool { return v; }))
    has_interp.clear();
}

template<typename P>
void moment_manager<P>::prepare_pos_grid_gpu(group_id group, sparse_grid const &grid) const
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

  if (not has_interp.empty()) {
    if (group == group_id::all() or has_interp[group()]) {
      update_position_grid_dsort();
    }
  }
}

template<typename P>
void moment_manager<P>::compute_moments(
    group_id group, sparse_grid const &grid, interpolation_manager<P> const &interp,
    connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
    kronmult::workspace<P> &kwork, gpu::vector<P> const &state) const
{
  static_assert(max_num_gpus == 1, "if multiple GPUs, state has to be an array of vectors");
  // when doing multiple GPUs, spread the state before computing moments
  // which will also allow to avoid the spread when doing term-apply

  std::array<gpu::vector<P>, max_num_gpus> &work1 = interp.gpu_it1;
  std::array<gpu::vector<P>, max_num_gpus> &work2 = interp.gpu_it2;

  prepare_pos_grid_gpu(group, grid);

  int const pos_block = pos_grid.block_size();

  int64_t const num_entries = pos_grid.num_dof();

  int const num_gpus = compute->num_gpus();
  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++)
  {
    if (gpu_moments[g].empty()) continue;

    compute->set_device(gpu::device{g});

    // using work[g] as workspace without resizing
    assert(work1[g].size() >= num_entries);
    gpu::wrap_array<P> w1(work1[g].data(), num_entries);

    // find the begin/end iterators to the moments in the group
    auto im = gpu_moments[g].begin();
    auto iend = gpu_moments[g].end() - 1; // one less, since last entry is unset
    if (group != group_id::all()) { // pick the subgroup
      int gid = 0;
      while (group != group_id{gid}) {
        if (im->is_unset()) gid++;
        im++;
      }
      iend = im;
      while (not iend->is_unset()) ++iend;
    }
    // perform work for all moments from im to iend
    for (; not im->is_unset(); im++) {

      moment const mom = mlist[im->mid]; // using this to get the necessary powers

      if (mom.is_electric()) continue; // skip electric field moments

      gpu::vector<P> &res = gpu_interps[g][im->mid];
      if (not im->skip_interp()) {
        res.resize(full_block * grid.num_indexes());
      }

      std::array<P const *, max_mom_dims> itg =
        {gpu_integ[g][0].data() + mom.pows[0] * integ[0].stride(), nullptr, nullptr};
      for (int i = 1; i < num_vel_; i++)
        itg[i] = gpu_integ[g][i].data() + mom.pows[i] * integ[i].stride();

      bool allzero = all_levels_zero;
      std::array<bool, max_mom_dims> lzero;
      if (not allzero) {
        for (int d = 0; d < max_mom_dims; d++)
          lzero[d] = (pdof > mom[d]);
        allzero = lzero[0] and lzero[1] and lzero[2]; // assuming only 3 entries
      }

      if (allzero) {
        moment_reduce_zero(pdof, pos_block, full_block, num_vel_, reduce_ij_allzero[g],
                           itg, state, w1.vec);
      } else {
        moment_reduce(pdof, pos_block, full_block, pos_grid.num_dims(), num_vel_,
                      lzero, grid.gpu_indexes(), reduce_ij[g], itg, state, w1.vec);
      }

      // at this point, the moment defined on the reduced grid is stored in w.vec

      if (im->raw_on_cpu()) {
        w1.vec.copy_to_host(raw_vals[im->mid]);
        full_level.get(im->mid).resize(0);
      }

      // solve poisson equation while w1 holds density
      if (mom.is_zero()) {
        assert(g == 0); // poisson uses gpu 0 for solve
        solve_poisson_gpu(grid, conn, hier, interp, kwork);
      }

      if (im->skip_interp()) continue;

      // now we have to compute the interpolation
      assert(work2[g].size() >= num_entries);
      gpu::wrap_array<P> w2(work2[g].data(), num_entries);

      interp.pos2nodal(gpu::device{g}, pos_grid, w1.vec.data(), wav_scale, w2.vec.data(), kwork);

      moment_expand(pdof, pos_grid.num_dims(), num_vel_, reduce_ij[g], w2.vec, res);

      if (im->interp_on_cpu()) res.copy_to_host(interps[im->mid]);
    }
  }
}

template<typename P>
void moment_manager<P>::compute_moments(
    std::vector<moment_id> const &mids, sparse_grid const &grid,
    interpolation_manager<P> const &interp, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, kronmult::workspace<P> &kwork,
    gpu::vector<P> const &state, bool result_to_cpu) const
{
  std::array<gpu::vector<P>, max_num_gpus> &work1 = interp.gpu_it1;
  std::array<gpu::vector<P>, max_num_gpus> &work2 = interp.gpu_it2;

  static_assert(max_num_gpus == 1, "if multiple GPUs, state has to be an array of vectors");
  // when doing multiple GPUs, spread the state before computing moments
  // which will also allow to avoid the spread when doing term-apply

  prepare_pos_grid_gpu(group_id::all(), grid);

  // This overload of compute_moments() is only called when there is an adaptive weight
  // set on the pde. All set_adapt_weight() functions are md_functions so interpolation is
  // always needed if this function is called. prepare_pos_grid_gpu will call
  // update_position_grid_dsort() only if one of the moments on the pde is an interpolatory moment;
  // however, this is not sufficient because it is possible for all moments to be separable and still
  // have an adaptive weight which will require interpolation (see pde/two_stream.cpp).
  // 
  // Things are a bit messy in the current state because calling set_adapt_weight()
  // with a moment function has no impact on gpu_mom.skip_interp() or has_interp
  // even though it will require interpolation for the adapt weight moments. It would be possible
  // to change set_adapt_weight() so it impacts gpu_mom.skip_interp() and has_interp; however,
  // this is undesirable because interpolation is only needed during grid refinement and not
  // during regular computation (it is not needed for the other overload of this function but
  // is needed for this overload). For now the easy fix is to call update_position_grid_dsort()
  // explicity in this overload of compute_moments() but we may want to consider how to fix
  // has_interp in the future so that prepare_pos_grid_gpu() works properly without this explicit call.
  // Maybe an additional flag should be added to the gpu_moms for whether or not they are adapt weight
  // moments. This would allow solve_poisson_gpu to determine if it should perform interpolation for the
  // poisson_1d case instead of always interpolating.
  update_position_grid_dsort();

  int const pos_block = pos_grid.block_size();

  int64_t const num_entries = pos_grid.num_dof();

  // using work[0] as workspace without resizing
  compute->set_device(gpu::device{0});
  assert(work1[0].size() >= num_entries);
  gpu::wrap_array<P> w1(work1[0].data(), num_entries);

  bool has_electric = false;
  for (auto im : mids)
  {
    moment const mom = mlist[im];
    if (mom.is_electric()) {
      has_electric = true;
      break;
    }
  }

  // perform work for all moments from im to iend
  for (auto im : mids)
  {
    moment const mom = mlist[im]; // using this to get the necessary powers

    if (mom.is_electric()) continue; // skip electric field moments

    gpu::vector<P> &res = gpu_interps[0][im];
    res.resize(full_block * grid.num_indexes());

    std::array<P const *, max_mom_dims> itg =
      {gpu_integ[0][0].data() + mom.pows[0] * integ[0].stride(), nullptr, nullptr};
    for (int i = 1; i < num_vel_; i++)
      itg[i] = gpu_integ[0][i].data() + mom.pows[i] * integ[i].stride();

    bool allzero = all_levels_zero;
    std::array<bool, max_mom_dims> lzero;
    if (not allzero) {
      for (int d = 0; d < max_mom_dims; d++)
        lzero[d] = (pdof > mom[d]);
      allzero = lzero[0] and lzero[1] and lzero[2]; // assuming only 3 entries
    }

    if (allzero) {
      moment_reduce_zero(pdof, pos_block, full_block, num_vel_, reduce_ij_allzero[0],
                         itg, state, w1.vec);
    } else {
      moment_reduce(pdof, pos_block, full_block, pos_grid.num_dims(), num_vel_,
                    lzero, grid.gpu_indexes(), reduce_ij[0], itg, state, w1.vec);
    }

    // now we have to compute the interpolation
    assert(work2[0].size() >= num_entries);
    gpu::wrap_array<P> w2(work2[0].data(), num_entries);

    interp.pos2nodal(gpu::device{0}, pos_grid, w1.vec.data(), wav_scale, w2.vec.data(), kwork);

    moment_expand(pdof, pos_grid.num_dims(), num_vel_, reduce_ij[0], w2.vec, res);

    if (result_to_cpu) {
      res.copy_to_host(interps[im]);
      full_level.get(im).resize(0);
    }

    // solve poisson equation while w1 holds density
    if (mom.is_zero() and has_electric)
      solve_poisson_gpu(grid, conn, hier, interp, kwork, result_to_cpu);
  }
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template class moment_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class moment_manager<float>;
#endif

} // namespace asgard
