#include "asgard_term_manager.hpp"

#include "asgard_coefficients_mats.hpp" // also brings in small-mats module

#include "asgard_blas.hpp"

namespace asgard
{

template<typename P>
term_entry<P>::term_entry(term_md<P> tin)
  : tmd(std::move(tin)), has_poisson(false)
{
  expect(not tmd.is_chain());
  if (tmd.is_interpolatory()) {
    return; // interpolation poisson dependence goes here
  }

  int const num_dims = tmd.num_dims();
  std::vector<int> active_dirs;
  active_dirs.reserve(num_dims);
  int flux_dir = -1;
  for (int d : iindexof(num_dims))
  {
    auto const &t1d = tmd.dim(d);
    if (not t1d.is_identity()) {
      active_dirs.push_back(d);
      if (t1d.has_flux()) {
        flux_dir = d;
        if (active_dirs.size() > 1)
          std::swap(active_dirs.front(), active_dirs.back());
      }
    }

    has_poisson = has_poisson or has_needs_poisson(t1d);
  }

  perm = kronmult::permutes(active_dirs, flux_dir);
}

template<typename P>
bool term_entry<P>::has_needs_poisson(term_1d<P> const &t1d) {
  auto check_poisson = [](term_1d<P> const &single)
    -> bool {
      return (single.depends() == term_dependence::electric_field or
              single.depends() == term_dependence::electric_field_only);
    };

  if (t1d.is_chain()) {
    for (int i : iindexof(t1d.num_chain()))
      if (check_poisson(t1d[i]))
        return true;
    return false;
  } else {
    return check_poisson(t1d);
  }
}

template<typename P>
term_manager<P>::term_manager(prog_opts const &options, pde_domain<P> const &domain,
                              pde_scheme<P> &pde, sparse_grid const &grid,
                              hierarchy_manipulator<P> const &hier,
                              connection_patterns const &conn)
  : num_dims(domain.num_dims()), max_level(options.max_level()), basis(hier.degree()),
    moms(domain, max_level, basis, hier, std::move(pde.mlist), pde.mom_groups)
#ifdef ASGARD_USE_MPI
    , resources(options.mpicomm)
#endif
{
  if (num_dims == 0)
    return;

  pde.finalize_term_groups(); // if using groups, else this does nothing

  if (pde.mass() and not pde.mass().is_identity())
    mass_term = std::move(pde.mass_);

  std::vector<term_md<P>> &pde_terms = pde.terms_;
  int num_terms = [&]() -> int {
    // get the effective number of terms, chained or not
    int n = 0;
    for (auto const &t : pde_terms)
      n += (t.is_chain()) ? t.num_chain() : 1;
    return n;
  }();

  { // copy over the group ids, keep flattened format
    term_groups.reserve(pde.term_groups.size());
    int ibegin = 0;
    for (auto const &tg : pde.term_groups) {
      int n = 0;
      for (int i : indexrange(tg))
        n += (pde_terms[i].is_chain()) ? pde_terms[i].num_chain() : 1;
      term_groups.emplace_back(ibegin, ibegin + n);
      ibegin += n;
    }
    source_groups.resize(pde.source_groups.size());
    for (int i : iindexof(pde.source_groups))
      source_groups[i].source_range = pde.source_groups[i];
  }

  terms.resize(num_terms);

  {
    bool has_interp = pde.has_interp_funcs;

    auto ir = terms.begin();
    for (int i : iindexof(pde_terms.size()))
    {
      if (pde_terms[i].is_chain()) {
        int const num_chain = pde_terms[i].num_chain();

        // this indicates that t1 and/or t2 workspaces are needed
        if (num_chain >= 2 and t1.empty())
          t1.resize(1);
        if (num_chain >= 3 and t2.empty())
          t2.resize(1);

        has_interp = has_interp or pde_terms[i].chain_[0].is_interpolatory();

        *ir = term_entry<P>(std::move(pde_terms[i].chain_[0]));
        ir++->num_chain = num_chain;
        for (int c = 1; c < num_chain; c++) {
          has_interp = has_interp or pde_terms[i].chain_[c].is_interpolatory();

          *ir = term_entry<P>(std::move(pde_terms[i].chain_[c]));
          ir++->mark_as_chain_link();
        }
      } else {
        has_interp = has_interp or pde_terms[i].is_interpolatory();

        *ir++ = term_entry<P>(std::move(pde_terms[i]));
      }
    }
    if (has_interp)
      interp = interpolation_manager<P>(options, domain, hier, conn);
  }

  int num_bc = 0;

  // check if we need to keep the intermediate terms from matrix builds
  for (auto &tt : terms) {
    int const n = static_cast<int>(tt.tmd.bc_flux_.size());
    tt.bc = indexrange(num_bc, num_bc + n);
    num_bc += n;
  }

  // form groups for the boundary conditions
  if (not term_groups.empty()) {
    int j = 0, bc_begin = 0, bc_end = 0; // index for the boundary conditions
    for (int groupid : iindexof(term_groups)) {
      for (int it : indexrange(term_groups[groupid]))
        bc_end += terms[it].bc.size();
      source_groups[j++].bc_range = irange(bc_begin, bc_end);
      bc_begin = bc_end;
    }
  }

  bcs.reserve(num_bc);
  for (int tid : iindexof(terms)) {
    term_entry<P> &tt = terms[tid];
    int const fdim = tt.tmd.flux_dim();
    tt.flux_dim = fdim;
    for (auto &b : tt.tmd.bc_flux_) {
      bcs.emplace_back(std::move(b));
      bcs.back().term_index = tid;
      for (int d : iindexof(num_dims)) {
        if (bcs.back().flux.chain_level(d) == -1) { // reset to the lowest level
          bcs.back().flux.chain_level(d) = (tt.tmd.dim(d).is_chain())
                                          ? (tt.tmd.dim(d).num_chain() - 1) : 0;
        }
      }
      if (bcs.back().flux.func().ignores_time()) {
        bcs.back().tmode = boundary_entry<P>::time_mode::constant;
      } else {
        if (bcs.back().flux.func_.ftime()) {
          if (bcs.back().flux.func_.cdomain(fdim) == 0) {
            bcs_have_time_dep = true;
            bcs.back().tmode  = boundary_entry<P>::time_mode::time_dependent;
            for (int d : iindexof(num_dims)) {
              rassert(not tt.tmd.dim(d).is_chain(),
                      "cannot use non-separable in time boundary conditions with 1d-chains, "
                      "the purpose of the 1d chain is to pre-compute and cache entries but non-separable "
                      "data cannot be pre-computed, an md-chain must be used instead");
            }
          } else {
            bcs.back().tmode = boundary_entry<P>::time_mode::separable;
          }
        } else {
          // fdim is constant, but the other dirs are non-separable
          bcs.back().tmode = boundary_entry<P>::time_mode::separable;
        }
      }
    }
  }

  // domain left/right bounds
  for (int d : iindexof(num_dims)) {
    xleft[d]  = pde.domain().xleft(d);
    xright[d] = pde.domain().xright(d);
  }

  build_mass_matrices(hier, conn); // large, up to max-level
  rebuild_mass_matrices(grid); // small, up to the current level

  std::vector<separable_func<P>> &sep = pde.sources_sep_;

  int num_sources = 0;
  for (auto const &s : sep) {
    int const dims = s.num_dims();
    rassert(dims == 0 or dims == num_dims, "incorrect dimension set for source");
    if (dims > 0) ++num_sources;
  }

  sources_md = std::move(pde.sources_md_);
  sources.reserve(num_sources);

  for (auto &s : sep) {
    if (s.num_dims() == 0)
      continue;

    if (s.ignores_time() or s.ftime()) {
      // using constant entry
      if (s.ignores_time()) {
        sources.emplace_back(source_entry<P>::time_mode::constant);
        sources.back().func = 0; // no need for a func
      } else {
        sources.emplace_back(source_entry<P>::time_mode::separable);
        sources.back().func = s.ftime();
      }

      for (int d : iindexof(num_dims)) {
        if (s.is_const(d)) {
          sources.back().consts[d]
              = hier.get_project1d_c(s.cdomain(d), mass[d], d, max_level);
        } else {
          sources.back().consts[d] = hier.get_project1d_f(
              [&](std::vector<P> const &x, std::vector<P> &y)-> void { s.fdomain(d, x, 0, y); },
              mass[d], d, max_level);
        }
      }

    } else {
      // non-separable in time
      sources_have_time_dep = true;
      sources.emplace_back(source_entry<P>::time_mode::time_dependent);
      sources.back().func = std::move(s);
    }
  }

  prapare_kron_workspace(grid); // setup kronmult workspace

  has_terms_ = not terms.empty();
  assign_compute_resources();

  // prepare the workspaces for the sources
  // consider only sources that are associated with this MPI rank and not time-dependant
  // the time sources cannot use workspace to accelerate computations
  #ifdef ASGARD_USE_MPI
  auto is_active_src = [&, this](source_entry<P> const &src) -> bool
    {
      if (not resources.owns(src.rec))
        return false;
      return (not src.is_time_dependent());
    };
  auto is_active_bc = [&, this](boundary_entry<P> const &bc) -> bool
    {
      if (not resources.owns(terms[bc.term_index].rec))
        return false;
      return (not bc.is_time_dependent());
    };
  #else
  auto is_active_src = [&](source_entry<P> const &src) -> bool
    {
      return (not src.is_time_dependent());
    };
  auto is_active_bc = [&](boundary_entry<P> const &bc) -> bool
    {
      return (not bc.is_time_dependent());
    };
  #endif

  for (auto const &src : sources)
    if (is_active_src(src)) num_lumped++;

  for (auto const &bc : bcs)
    if (is_active_bc(bc)) num_lumped++;

  if (not source_groups.empty()) { // set sources group by group
    int ibegin = 0, iend = 0;
    for (size_t i = 0; i < source_groups.size(); i++) {
      for (int is : indexrange(source_groups[i].source_range))
        if (is_active_src(sources[is]))
          sources[is].ilump = iend++;
      for (int ib : indexrange(source_groups[i].bc_range))
        if (is_active_bc(bcs[ib]))
          bcs[ib].ilump = iend++;
      source_groups[i].lump_range = irange(ibegin, iend);
      ibegin = iend;
    }
  } else { // no groups, lump everything together
    int j = 0;
    for (auto &src : sources)
      if (is_active_src(src)) src.ilump = j++;
    for (auto &bc : bcs)
      if (is_active_bc(bc)) bc.ilump = j++;
  }
  sweights.reserve(num_lumped); // one weight per lumped source

  // second pass on the problem of assigning workspaces and preparing objects
  // e.g., the needed resources change if this MPI rank has no terms with need
  {
    // set interpolatory properties
    for (int i : indexof(terms)) {
      auto &t = terms[i];
      t.interplan.enable(t.tmd.is_interpolatory());
      if (t.is_interpolatory()) {
        if (t.num_chain == 1) { // single entry
          t.interplan.use_field();
        } else if (t.is_chain_link() and
                   (i+1 == static_cast<int>(terms.size())
                    or not terms[i+1].is_chain_link())) {
          // if part of a chain and the next term is not from the current chain
          // i.e., this is the first link in the chain
          t.interplan.use_field();
        }
      }
    }

    bool has_field_interp = false; // interpolating from a field
    auto it = terms.begin();
    while (it < terms.end())
    {
      #ifdef ASGARD_USE_MPI
      if (not resources.owns(it->rec)) {
        it += it->num_chain;
        continue;
      }
      #endif
      has_field_interp = has_field_interp or it->interplan.uses_field();
      if (it->is_chain_start())
        has_field_interp = has_field_interp or (it + it->num_chain -1)->interplan.uses_field();
      it += it->num_chain;
    }

    if (has_field_interp)
      ifield.resize(1);

    // handle the moment dependence
    std::vector<moment_id> regular_moments;
    std::vector<moment_id> interp_moments;
    regular_moments.reserve(250); // should be more than enough, not a big deal otherwise
    bool has_poisson = false;
    for (auto const &tentry : terms) {
      #ifdef ASGARD_USE_MPI
      if (not resources.owns(tentry.rec))
        continue;
      #endif
      has_poisson = has_poisson or tentry.has_poisson;
      if (tentry.is_separable()) { // only separable terms can have 1D moment deps
        for (int d : iindexof(num_dims)) {
          auto const &mids = tentry.tmd.dim(d).mids_;
          if (not mids.empty()) {
            regular_moments.insert(regular_moments.end(), mids.begin(), mids.end());
          }
        }
      } else if (tentry.interplan.uses_moments()) {
        auto const &mids = tentry.tmd.mids_;
        interp_moments.insert(interp_moments.end(), mids.begin(), mids.end());
      }
    }{
      auto comp_id  = [](moment_id id1, moment_id id2) -> bool { return (id1() < id2()); };
      auto match_id = [](moment_id id1, moment_id id2) -> bool { return (id1() == id2()); };

      std::sort(interp_moments.begin(), interp_moments.end(), comp_id);
      std::sort(regular_moments.begin(), regular_moments.end(), comp_id);

      auto last = std::unique(regular_moments.begin(), regular_moments.end(), match_id);
      regular_moments.erase(last, regular_moments.end());

      last = std::unique(interp_moments.begin(), interp_moments.end(), match_id);
      interp_moments.erase(last, interp_moments.end());

      for (int i : iindexof(moms.num_moments())) {
        if (moms.get_by_id(moment_id{i}).action == moment::moment_type::interpolatory) {
          if (not std::binary_search(interp_moments.begin(), interp_moments.end(),
                                     moment_id{i}, comp_id))
            moms.set_action(moment_id{i}, moment::moment_type::regular);
        }
        if (moms.get_by_id(moment_id{i}).action == moment::moment_type::regular) {
          if (not std::binary_search(regular_moments.begin(), regular_moments.end(),
                                     moment_id{i}, comp_id))
            moms.set_action(moment_id{i}, moment::moment_type::inactive);
        }
      }
    }
    if (has_poisson) {
      if (term_groups.empty())
        has_poisson_.resize(1, true);
      else
        has_poisson_.resize(term_groups.size(), false); // will process groups below
    }
    if (not term_groups.empty()) {
      for (int gid : iindexof(term_groups)) {
        bool needs = false;
        for (int tid : indexrange(term_groups[gid])) {
          #ifdef ASGARD_USE_MPI
          if (not resources.owns(terms[tid].rec))
            continue;
          #endif
          needs = needs or terms[tid].has_poisson;
        }
        if (needs)
          has_poisson_[gid] = needs;
      }
    }
  }
  #ifdef ASGARD_USE_GPU
  kwork.row_map.resize(max_num_gpus);
  #endif
}


template<typename P>
void term_manager<P>::build_const_terms(
    int const tid, sparse_grid const &grid, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, precon_method precon, P alpha)
{
  if (terms[tid].tmd.is_interpolatory()) // skip interpolation terms
    return;

  expect(basis.pdof == hier.degree() + 1);
  expect(not terms[tid].tmd.is_chain());

  auto &tmd = terms[tid];

  bool merging_with_interp = false;
  if ((tmd.is_chain_start() or tmd.is_chain_link())
       and (static_cast<size_t>(tid + 1) < terms.size())
        and terms[tid + 1].is_chain_link()
         and terms[tid + 1].is_interpolatory())
  {
    // there is a potential here to merge this separable term with hier2wav
    merging_with_interp = true;
    for (int d : iindexof(num_dims))
      if (tmd.tmd.dim(d).change() != changes_with::none)
        merging_with_interp = false;
    // if the 1d terms are changing, then skip the merge
    // if everything is constant, we can merge
  }

  if (merging_with_interp)
  {
    constexpr bool merge_with_interp = true;
    terms[tid + 1].interplan.stop_hier();

    std::vector<int> id_dirs;
    id_dirs.reserve(num_dims);
    for (int d : iindexof(num_dims))
    {
      if (tmd.tmd.dim(d).change() == changes_with::time)
        continue;

      rebuild_term1d(terms[tid], d, max_level, conn, hier, precon, alpha, merge_with_interp);
      if (terms[tid].tmd.dim(d).is_identity())
        id_dirs.push_back(d);
    }
    // adjust the kronmult permutations using the fact that the identity directions
    // were replaced by the hier2wav matrix, which is upper hierarchical
    if (not id_dirs.empty())
      terms[tid].perm.prepad_upper(id_dirs);
  }
  else
  {
    for (int d : iindexof(num_dims))
    {
      auto const &t1d = tmd.tmd.dim(d);
      if (t1d.change() == changes_with::time)
        continue;

      int level = grid.current_level(d); // required level

      // terms that don't change should be build only once
      if (t1d.change() == changes_with::none)
        level = max_level; // build up to the max

      rebuild_term1d(terms[tid], d, level, conn, hier, precon, alpha);
    } // move to next dimension d
  }
}

template<typename P>
void term_manager<P>::rebuild_term1d(
    term_entry<P> &tentry, int const dim, int level,
    connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
    precon_method, P, bool merge_with_interp)
{
  int const n = hier.degree() + 1;
  auto &t1d   = tentry.tmd.dim(dim);

  block_diag_matrix<P> *bmass = nullptr; // mass to use for the boundary source

  // apply the mass matrix, if any
  if (tentry.num_chain < 0) {
    // member of a chain, can have unique mass matrix
    mass_md<P> const &tms = tentry.tmd.mass();
    if (tms and not tms[dim].is_identity()) {
      int const nrows = fm::ipow2(level); // needed number of rows
      if (tentry.mass[dim].nrows() != nrows) {
        build_raw_mass(dim, tms[dim], max_level, tentry.mass[dim]);
        tentry.mass[dim].spd_factorize(n);
      }
      bmass = &tentry.mass[dim];
    }
  } else if (mass[dim]) { // no chain (or last link), and there's global mass
    // global case, use the global mass matrices
    if (level == max_level) {
      bmass = &mass[dim];
    } else { // using lower level, construct lower mass matrix
      int const nrows = fm::ipow2(level); // needed number of rows
      if (lmass[dim].nrows() != nrows) {
        build_raw_mass(dim, mass_term[dim], max_level, lmass[dim]);
        lmass[dim].spd_factorize(n);
      }
      bmass = &lmass[dim];
    }
  }

  bool is_diag = t1d.is_diagonal();
  if (t1d.is_chain()) {
    rebuld_chain(tentry, dim, level, hier, bmass, is_diag, wraw_diag, wraw_tri);
  } else {
    build_raw_mat(tentry, dim, 0, level, hier, bmass, wraw_diag, wraw_tri);
  }

  // the build/rebuild put the result in raw_diag or raw_tri
  // if the term is identity, then there is no matrix, all the calls
  // above are needed to handle the boundary conditions
  if (t1d.is_identity()) {
    if (merge_with_interp)
      tentry.coeffs[dim] = interp.get_hier2wav();
  } else {
    if (is_diag) {
      if (merge_with_interp)
        tentry.coeffs[dim] = interp.mult_transform_h2w(hier, conn, wraw_diag, raw_diag0);
      else
        tentry.coeffs[dim] = hier.diag2hierarchical(wraw_diag, level, conn);
    } else {
      if (merge_with_interp)
        tentry.coeffs[dim] = interp.mult_transform_h2w(hier, conn, wraw_tri, raw_tri0);
      else
        tentry.coeffs[dim] = hier.tri2hierarchical(wraw_tri, level, conn);
    }
  }

  // the last interpolation stage (2wav) comes with a scaling factor
  // apply the scaling factor to the zeroth dimension
  if (merge_with_interp and dim == 0)
    tentry.coeffs[dim].scal(interp.wav_scale_h2w());

  #ifdef ASGARD_USE_GPU
  if (not tentry.coeffs[dim].empty()) { // load to the GPU
    compute->set_device(gpu::device{tentry.rec.device});
    #ifdef ASGARD_GPU_MEMGREEDY
    tentry.gpu_coeffs[dim] = tentry.coeffs[dim].data_vector();
    #else
    tentry.gpu_lcoeffs[dim].resize(level + 1);
    std::vector<P*> coeff_pntrs(level + 1, nullptr);
    for (int l = 0; l < level; l++) {
      tentry.gpu_lcoeffs[dim][l] = tentry.coeffs[dim].get_subpattern(l, conn).data_vector();
      coeff_pntrs[l] = tentry.gpu_lcoeffs[dim][l].data();
    }
    tentry.gpu_lcoeffs[dim][level] = tentry.coeffs[dim].data_vector();
    coeff_pntrs[level]             = tentry.gpu_lcoeffs[dim][level].data();

    tentry.gpu_coeffs[dim] = coeff_pntrs;
    #endif

    compute->set_device(gpu::device{0});
  }
  #endif

  // apply the mass matrices and convert to hierarchical form
  for (int b : indexrange{tentry.bc}) {
    boundary_entry<P> &bentry = bcs[b];
    if (not bentry.consts[dim].empty()) {
      // will be empty if non-flux direction and non-separable in time
      hier.transform(level, bentry.consts[dim]);
    }
  }
}

template<typename P>
void term_manager<P>::build_raw_mat(
    term_entry<P> &tentry, int d, int clink, int level,
    hierarchy_manipulator<P> const &hier,
    block_diag_matrix<P> const *bmass,
    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri)
{
  term_1d<P> &t1d = (tentry.tmd.dim(d).is_chain()) ? tentry.tmd.dim(d).chain_[clink] : tentry.tmd.dim(d);
  expect(not t1d.is_chain());

  switch (t1d.optype())
  {
    case operation_type::volume:
      switch (t1d.depends()) {
        case term_dependence::electric_field_only:
          if (t1d.rhs()) {
            // using w1 as workspaces, it probably has enough space already
            size_t const n = kwork.w1.size();
            t1d.rhs(moms.poisson_level(), kwork.w1);
            gen_diag_cmat_pwc<P>(basis, level, kwork.w1, raw_diag);
            kwork.w1.resize(n);
          } else {
            gen_diag_cmat_pwc<P>(basis, level, moms.poisson_level(), raw_diag);
          }
          break;
        case term_dependence::electric_field:
          throw std::runtime_error("el-field with position depend is not done (yet)");
          break;
        case term_dependence::moment_divided_by_density:
          gen_diag_mom_over_zero<P>(basis, level, t1d.rhs_const(),
                                    moms.get_cached_level(t1d.moment_ids()[0], hier),
                                    moms.get_cached_level(t1d.moment_ids()[1], hier),
                                    raw_diag);
          break;
        case term_dependence::lenard_bernstein_coll_theta:
          switch (moms.num_vel()) {
          case 1:
            moms.cache_levels(3, hier, t1d.mids_);
            gen_diag_lenard_bernstein_theta<P, 1>(basis, level, t1d.rhs_const(),
                                                  t1d.mids_, moms.get_cached_levels(),
                                                  raw_diag);
            break;
          case 2:
            moms.cache_levels(5, hier, t1d.mids_);
            gen_diag_lenard_bernstein_theta<P, 2>(basis, level, t1d.rhs_const(),
                                                  t1d.mids_, moms.get_cached_levels(),
                                                  raw_diag);
            break;
          case 3:
            moms.cache_levels(7, hier, t1d.mids_);
            gen_diag_lenard_bernstein_theta<P, 3>(basis, level, t1d.rhs_const(),
                                                  t1d.mids_, moms.get_cached_levels(),
                                                  raw_diag);
            break;
          default:
            // unreachable here
            break;
          };
          break;
        default:
          if (t1d.rhs()) {
            gen_volume_mat<P>(basis, xleft[d], xright[d], level, t1d.rhs(), raw_rhs, raw_diag);
          } else {
            gen_volume_mat<P>(basis, level, t1d.rhs_const(), raw_diag);
          }
          break;
      }
      break;
    case operation_type::div:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::div, rhs_type::is_func>
          (basis, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::div, rhs_type::is_const>
          (basis, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      }
      if (t1d.penalty() != 0) {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
          (basis, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      }
      break;
    case operation_type::grad:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::grad, rhs_type::is_func>
          (basis, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::grad, rhs_type::is_const>
          (basis, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      }
      if (t1d.penalty() != 0) {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
          (basis, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      }
      break;
    case operation_type::penalty:
      expect(not t1d.rhs());
      gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const>
        (basis, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      break;
    case operation_type::robin:
      expect(not t1d.rhs());
      gen_robin_cmat<P>(basis, xleft[d], xright[d], level, t1d.left_robin(), t1d.right_robin(), raw_diag);
      break;
    default: // case operation_type::identity:
      // identity, nothing to do for the matrix, but may have to do boundary conditions
      break;
  }

  if (bmass) {
    if (t1d.is_diagonal() and not t1d.is_identity())
      bmass->solve(basis.pdof, raw_diag);
    else
      bmass->solve(basis.pdof, raw_tri);
  }

  for (int b : indexrange(tentry.bc)) {
    // handle the non-separable in time, keep rhs values
    boundary_entry<P> &bentry = bcs[b];

    if (bentry.flux.chain_level(d) > clink) {
      expect(not bentry.consts[d].empty());
      if (t1d.is_diagonal()) {
        raw_diag.inplace_gemv(basis.pdof, bentry.consts[d], t1);
      } else {
        raw_tri.inplace_gemv(basis.pdof, bentry.consts[d], t1);
      }
    } else if (bentry.flux.chain_level(d) == clink) {
      // create a new entry
      if (tentry.flux_dim == d) {
        int const pdof = basis.pdof;

        int64_t const num_cells = fm::ipow2(level);
        int64_t const num_entries = pdof * num_cells;

        bentry.consts[d].resize(num_entries);

        P scale = P{1} / std::sqrt( (xright[d] - xleft[d]) / num_cells );
        if (t1d.is_penalty()) // penalty flips the sign of the boundary conditions
          scale = -scale;

        if (bentry.flux.is_left()) {
          P rhs_left  = (t1d.rhs()) ? raw_rhs.vals.front() : t1d.rhs_const();
          if (t1d.penalty() != 0)
            rhs_left *= P{1} + t1d.penalty();

          P const fc = bentry.flux.func().cdomain(d);
          if (fc == 0) { // non-separable in time
            // single-point value is always separable, so we can pre-compute in d-direction
            smmat::axpy(pdof, - rhs_left * scale, basis.leg_left, bentry.consts[d].data());
          } else {
            smmat::axpy(pdof, - rhs_left * scale * fc, basis.leg_left, bentry.consts[d].data());
          }
        }

        if (bentry.flux.is_right()) {
          P rhs_right = (t1d.rhs()) ? raw_rhs.vals.back()  : t1d.rhs_const();

          if (t1d.penalty() != 0)
            rhs_right *= P{1} - t1d.penalty();

          P const fc = bentry.flux.func().cdomain(d);
          if (fc == 0) { // non-separable in time
            // single-point value is always separable, so we can pre-compute in d-direction
            smmat::axpy(pdof, rhs_right * scale, basis.leg_right,
                        bentry.consts[d].data() + num_entries - pdof);
          } else {
            smmat::axpy(pdof, rhs_right * scale * fc, basis.leg_right,
                        bentry.consts[d].data() + num_entries - pdof);
          }
        }

        if (bmass)
          bmass->solve(pdof, bentry.consts[d]);

      } else {
        if (bentry.is_time_dependent()) // no constant components to pre-compute
          continue;

        P const dsqr = std::sqrt(xright[d] - xleft[d]);

        if (bentry.flux.func().is_const(d)) {
          if (t1d.rhs()) { // constant times spatially variable
            bentry.consts[d] = basis.project(t1d.is_diagonal(), level, dsqr,
                                             bentry.flux.func().cdomain(d), raw_rhs.vals);
          } else { // constant times a constant
            P const rconst = (t1d.is_identity()) ? 1 : t1d.rhs_const();
            bentry.consts[d] = basis.project(level, dsqr,
                                             bentry.flux.func().cdomain(d) * rconst);
          }
        } else {
          if (t1d.rhs()) { // product of non-consts
            std::vector<P> f(raw_rhs.pnts.size());
            bentry.flux.func().fdomain(d, raw_rhs.pnts, 0, f);
            bentry.consts[d] = basis.project(t1d.is_diagonal(), level, dsqr, f, raw_rhs.vals);
          } else {
            // need function values, rhs is a constant
            basis.interior_quad(xleft[d], xright[d], level, raw_rhs.pnts);
            raw_rhs.vals.resize(raw_rhs.pnts.size());
            bentry.flux.func().fdomain(d, raw_rhs.pnts, 0, raw_rhs.vals);
            bool constexpr use_interior = true;
            bentry.consts[d] = basis.project(use_interior, level, dsqr, t1d.rhs_const(), raw_rhs.vals);
          }
        }

        if (bmass)
          bmass->solve(basis.pdof, bentry.consts[d]);
      }
    } // if the bentry is associated with a higher link, then do nothing here
  }
}

template<typename P>
void term_manager<P>::build_raw_mass(int dim, term_1d<P> const &t1d, int level,
                                     block_diag_matrix<P> &raw_diag)
{
  expect(t1d.is_diagonal());
  expect(t1d.depends() == term_dependence::none);

  if (t1d.rhs()) {
    gen_volume_mat<P>(basis, xleft[dim], xright[dim], level, t1d.rhs(), raw_rhs, raw_diag);
  } else {
    gen_volume_mat<P>(basis, level, t1d.rhs_const(), raw_diag);
  }
}

template<typename P>
void term_manager<P>::rebuld_chain(
    term_entry<P> &tentry, int const d, int const level,
    hierarchy_manipulator<P> const &hier,
    block_diag_matrix<P> const *bmass,
    bool &is_diag, block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri)
{
  term_1d<P> &t1d = tentry.tmd.dim(d);
  expect(t1d.is_chain());
  int const num_chain = t1d.num_chain();
  expect(num_chain > 1);

  is_diag = true;
  for (int i : iindexof(num_chain)) {
    if (t1d[i].is_tri_diag()) {
      is_diag = false;
      break;
    }
  }

  if (is_diag) { // a bunch of diag matrices, easy case
    // raw_tri will not be referenced, it's just passed in
    // using raw_diag to make the intermediate matrices, until the last one
    // the last product has to be written to raw_diag
    block_diag_matrix<P> *diag0 = &raw_diag0;
    block_diag_matrix<P> *diag1 = &raw_diag1;
    build_raw_mat(tentry, d, num_chain - 1, level, hier, bmass, *diag0, raw_tri);
    for (int i = num_chain - 2; i > 0; i--) {
      build_raw_mat(tentry, d, i, level, hier, bmass, raw_diag, raw_tri);
      diag1->check_resize(raw_diag);
      gemm_block_diag(basis.pdof, raw_diag, *diag0, *diag1);
      std::swap(diag0, diag1);
    }
    build_raw_mat(tentry, d, 0, level, hier, bmass, *diag1, raw_tri);
    raw_diag.check_resize(*diag1);
    gemm_block_diag(basis.pdof, *diag1, *diag0, raw_diag);

    return;
  }

  // the final is always a tri-diagonal matrix
  // but we have to keep track of upper/lower and diagonal
  block_diag_matrix<P> *diag0 = &raw_diag0;
  block_diag_matrix<P> *diag1 = &raw_diag1;

  block_tri_matrix<P> *tri0 = &raw_tri0;
  block_tri_matrix<P> *tri1 = &raw_tri1;

  enum class fill {
    diag, tri
  };

  // here we start with either a diagonal or tri-diagonal matrix
  // and at each stage we multiply by diag/tri-matrix
  // if we start with a diagonal, we will switch to tri at some point

  fill current = (t1d.chain_.back().is_diagonal()) ? fill::diag : fill::tri;
  build_raw_mat(tentry, d, num_chain - 1, level, hier, bmass, *diag0, *tri0);

  for (int i = num_chain - 2; i > 0; i--)
  {
    build_raw_mat(tentry, d, i, level, hier, bmass, raw_diag, raw_tri);
    // the result is in either raw_diag or raw_tri and must be multiplied and put
    // into either diag1 or tri1, then those should swap with diag0 and tri0
    if (t1d[i].is_diagonal()) { // computed a diagonal fill
      if (current == fill::diag) { // diag-to-diag
        diag1->check_resize(raw_diag);
        gemm_block_diag(basis.pdof, raw_diag, *diag0, *diag1);
        std::swap(diag0, diag1);
      } else { // multiplying diag by tri-diag
        tri1->check_resize(raw_diag);
        gemm_diag_tri(basis.pdof, raw_diag, *tri0, *tri1);
        std::swap(tri0, tri1);
      }
    } else { // computed tri matrix (upper or lower diagonal)
      if (current == fill::diag ) { // tri times diag
        tri1->check_resize(raw_tri);
        gemm_tri_diag(basis.pdof, raw_tri, *diag0, *tri1);
        std::swap(tri0, tri1);
        current = fill::tri;
      } else {
        tri1->check_resize(raw_tri);
        gemm_block_tri(basis.pdof, raw_tri, *tri0, *tri1);
        std::swap(tri0, tri1);
        current = fill::tri;
      }
    }
  }

  // last term, compute in diag1/tri1 and multiply into raw_tri
  build_raw_mat(tentry, d, 0, level, hier, bmass, *diag1, *tri1);

  if (t1d[0].is_diagonal()) {
    // the rest must be a tri-diagonal matrix already
    // otherwise the whole chain would consist of only diagonal ones
    raw_tri.check_resize(*tri0);
    gemm_diag_tri(basis.pdof, *diag1, *tri0, raw_tri);
  } else {
    if (current == fill::diag) {
      raw_tri.check_resize(*tri1);
      gemm_tri_diag(basis.pdof, *tri1, *diag0, raw_tri);
    } else {
      raw_tri.check_resize(*tri1);
      gemm_block_tri(basis.pdof, *tri1, *tri0, raw_tri);
    }
  }

  // apply the penalty that is added to the whole chain
  if (t1d.penalty() != 0) {
    if (bmass) {
      gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const>
        (basis, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.chain_.back().flux(),
          t1d.chain_.back().boundary(), raw_rhs, *tri0);
      bmass->solve(basis.pdof, *tri0);
      raw_tri += *tri0;
    } else {
      // no need to worry about the mass, just add the penalty to the raw-tri
      gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
        (basis, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.chain_.back().flux(),
          t1d.chain_.back().boundary(), raw_rhs, raw_tri);
    }
  }

  if (t1d.left_robin() != 0 or t1d.right_robin() != 0) {
    if (bmass) {
      gen_robin_cmat<P>
        (basis, xleft[d], xright[d], level, t1d.left_robin(), t1d.right_robin(), *diag0);
      bmass->solve(basis.pdof, *diag0);
      raw_tri += *diag0;
    } else {
      gen_robin_cmat<P>
        (basis, xleft[d], xright[d], level, t1d.left_robin(), t1d.right_robin(), raw_tri);
    }
  }

  // the penalty may yield additional work for the boundary conditions
  // the robin added matrices do not affect the boundary terms (they don't chain)
  if (t1d.penalty() == 0)
    return;

  // handle the penalty component of the boundary conditions
  std::vector<P> penwork; // extra allocation, should be rare, when having mass + builtin penalty
  for (int b : indexrange(tentry.bc)) {
    // handle the non-separable in time, keep rhs values
    boundary_entry<P> &bentry = bcs[b];

    // apply only the conditions for the bottom link
    if (bentry.flux.chain_level(d) != num_chain - 1)
      continue;

    int const pdof = basis.pdof;

    int64_t const num_cells = fm::ipow2(level);
    int64_t const num_entries = pdof * num_cells;

    if (bmass)
      penwork.reserve(num_entries);

    // for no mass, write directly into consts, else must use scratch space to invert the matrix
    P *dest = (bmass) ? penwork.data() : bentry.consts[d].data();

    expect(bentry.consts[d].size() == static_cast<size_t>(num_entries));

    P const scale = -t1d.penalty() / std::sqrt( (xright[d] - xleft[d]) / num_cells );

    if (bentry.flux.is_left()) {
      P const fc = bentry.flux.func().cdomain(d);
      if (fc == 0) { // non-separable in time
        smmat::axpy(pdof, -scale, basis.leg_left, dest);
      } else {
        smmat::axpy(pdof, -scale * fc, basis.leg_left, dest);
      }
    }

    if (bentry.flux.is_right()) {
      P const fc = bentry.flux.func().cdomain(d);
      if (fc == 0) { // non-separable in time
        smmat::axpy(pdof, scale, basis.leg_right, dest + num_entries - pdof);
      } else {
        smmat::axpy(pdof, scale * fc, basis.leg_right, dest + num_entries - pdof);
      }
    }

    if (bmass) {
      bmass->solve(pdof, dest);
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < num_entries; i++)
        bentry.consts[d][i] += dest[i];
    }
  }
}

template<typename P>
void term_manager<P>::assign_compute_resources()
{
// if there's no MPI or GPU, then there's nothing to do
#ifdef ASGARD_MANAGED_RESOURCES
  // measuring work in units of 1D kron operations
  // a 3d term with 1 identity has weight 2, 2 identities is weight 1
  // interpolation term has weight 3 * num_dims + 1
  //    - the extra comes from the function evaluation
  // source term has lower weight, say 0.5
  // interpolatory source has weight 2 * num_dims + 1

  // (TODO) there is an optimization problem here ...

  float constexpr source_weight = 0.5f;
  float const iterm_weight   = 3.0f * num_dims + 1.0f;
  float const isource_weight = 2.0f * num_dims + 1.0f;

  struct work_amount {
    explicit work_amount(float v) : value(v) {}
    float value = 0;
  };

  struct balance_manager {
    std::vector<float> workload;
    void add(int id, work_amount work) {
      workload[id] += work.value;
    }
    int lowest() { // get the id with lowest load
      int im = 0, l = workload[0];
      for (size_t i = 1; i < workload.size(); i++) {
        if (workload[i] < l) {
          im = static_cast<int>(i);
          l = workload[i];
        }
      }
      return im;
    }
  };

  std::vector<int> ids;
  ids.reserve(std::max(terms.size(), sources.size()));
  std::vector<float> weights;
  weights.reserve(ids.capacity());

  auto get_weight = [&](term_entry<P> const &t)
    -> float {
      // count the number of 1D Kronecker operations
      return (t.is_separable()) ? t.perm.num_dimensions() : iterm_weight;
    };

  auto get_heaviest = [&]()
    -> int {
      // get the id of the heaviest unassigned term
      auto iw = std::max_element(weights.begin(), weights.end());
      if (*iw < 0) // all assigned
        return -1;
      else
        return static_cast<int>(std::distance(weights.begin(), iw));
    };

  enum class balance_mode {
    mpi_ranks, gpus
  };

  auto load_balance = [&](int gid, int num_workers, balance_mode mode)
    -> void {
      // case of 0 GPUs, all goes to the CPU
      // MPI always has at least 1 rank
      // consider cases: num_workers == 1 or num_workers > 1
      if (num_workers == 1) {
        // if using only one GPU, then assign all terms to that device
        // the mpi-ranks default to 0 anyway
        if (mode == balance_mode::gpus) {
          for (auto &t : terms)
            t.rec.device = 0;
        }
      } else if (num_workers > 1) {
        balance_manager balance;
        balance.workload.resize(num_workers);

        // device 0 handles the interpolation sources
        if (gid < 0 and sources_md[0])
          balance.add(0, work_amount{isource_weight});
        if (gid >= 0 and sources_md[gid])
          balance.add(0, work_amount{isource_weight});

        ids.resize(0);
        weights.resize(0);

        int id   = (gid < 0) ? 0 : term_groups[gid].begin();
        int iend = (gid < 0) ? static_cast<int>(terms.size()) : term_groups[gid].end();
        while(id < iend)
        {
          // skip terms assigned to other mpi ranks
          if (mode == balance_mode::gpus and not resources.owns(terms[id].rec)) {
            id += terms[id].num_chain;
            continue;
          }

          ids.push_back(id);

          int w = 0;
          for (int j = id; j < id + terms[id].num_chain; j++)
            w += get_weight(terms[j]);
          weights.push_back(w);

          id += terms[id].num_chain;
        }

        id = (weights.empty()) ? -1 : get_heaviest();
        while (id >= 0) {
          // get the heaviest term, add it to the lowest weigh group
          // then mark the term as "done" (remove the weight), move to next
          int const low = balance.lowest();
          if (mode == balance_mode::mpi_ranks)
            terms[ids[id]].rec.group = low;
          else
            terms[ids[id]].rec.device = low;
          balance.add(low, work_amount{weights[id]});
          weights[id] = -1;

          id = get_heaviest();
        }

        ids.resize(0);
        weights.resize(0);

        int ibegin = (gid < 0) ? 0                                : source_groups[gid].source_range.begin();
        iend       = (gid < 0) ? static_cast<int>(sources.size()) : source_groups[gid].source_range.end();
        for (int i = ibegin; i < iend; i++)
        {
          // skip terms assigned to other mpi ranks
          if (mode == balance_mode::gpus and not resources.owns(sources[i].rec))
            continue;
          ids.push_back(i);
        }

        for (auto i : ids) {
          int const low = balance.lowest();
          if (mode == balance_mode::mpi_ranks)
            sources[i].rec.group = low;
          else
            sources[i].rec.device = low;
          balance.add(low, work_amount{source_weight});
        }
      }
    };

  if (term_groups.empty()) {
    load_balance(-1, resources.num_ranks(), balance_mode::mpi_ranks);
    load_balance(-1, resources.num_gpus(), balance_mode::gpus);
  } else {
    for (int gid = 0; gid < static_cast<int>(term_groups.size()); gid++) {
      load_balance(gid, resources.num_ranks(), balance_mode::mpi_ranks);
      load_balance(gid, resources.num_gpus(), balance_mode::gpus);
    }
  }

  // mark all chains to make sure they go together
  // check whether there are any terms
  has_terms_ = false;
  {
    auto it = terms.begin();
    while (it < terms.end()) {
      if (resources.owns(it->rec))
        has_terms_ = true;

      if (it->num_chain > 1) {
        for (int i = 0; i < it->num_chain; i++)
          (it + i)->rec = it->rec;
      }
      it += it->num_chain;
    }
  }
  if (not has_terms_) {
    bool has_sources = false;
    for (auto const &s : sources)
      if (resources.owns(s.rec))
        has_sources = true;
    if (not terms.empty() and resources.num_ranks() > 1 and not has_sources) {
      // if the PDE has some terms, e.g., some testing PDEs don't,
      // and if there are multiple MPI ranks, yet some ranks have no terms
      // that means there are more ranks then terms and we should print a warning
      std::cerr << " -- warning: the number of MPI ranks exceeds the number of terms and sources,"
                << " the likely outcome is performance degradation" << std::endl;
    }
  }

  // if (mpi::is_world_rank(0)) {
    // for (auto const &t : terms)
    //   std::cout << " assigned to: " << t.rec.group << "  gpu: " << t.rec.device << " chain num = " << t.num_chain << '\n';
    //
    // for (auto const &s : sources)
    //   std::cout << " source to: " << s.rec.group << "  gpu: " << s.rec.device << '\n';
    //
    // std::cout << "rank 0 dep 0 moms = " << deps(0).num_moments << " dep 1 moms = " << deps(1).num_moments << std::endl;
  // }
#endif
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct term_entry<double>;
template struct term_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct term_entry<float>;
template struct term_manager<float>;
#endif

}
