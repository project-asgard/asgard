#include "asgard_term_manager.hpp"

#include "asgard_coefficients_mats.hpp" // also brings in small-mats module

#include "asgard_blas.hpp"

namespace asgard
{

template<typename P>
term_entry<P>::term_entry(term_md<P> tin)
  : tmd(std::move(tin))
{
  expect(not tmd.is_chain());
  if (tmd.is_interpolatory()) {
    deps[0] = {false, 0}; // set interpolation deps here
    return;
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

    deps[d] = get_deps(t1d);
  }

  perm = kronmult::permutes(active_dirs, flux_dir);
}

template<typename P>
mom_deps term_entry<P>::get_deps(term_1d<P> const &t1d) {
  auto process_dep = [](term_1d<P> const &single)
    -> mom_deps {
      switch (single.depends()) {
        case term_dependence::electric_field:
        case term_dependence::electric_field_only:
          // technically, el-field requires 1 moment, but it is a special case
          return {true, 0};
        case term_dependence::moment_divided_by_density:
          return {false, std::abs(single.moment())};
        case term_dependence::lenard_bernstein_coll_theta_1x1v:
          return {false, 3};
        case term_dependence::lenard_bernstein_coll_theta_1x2v:
          return {false, 5};
        case term_dependence::lenard_bernstein_coll_theta_1x3v:
          return {false, 7};
        default:
          return {};
      };
    };

  if (t1d.is_chain()) {
    mom_deps result;
    for (int i : iindexof(t1d.num_chain()))
      result += process_dep(t1d[i]);
    return result;
  } else {
    return process_dep(t1d);
  }
}

template<typename P>
term_manager<P>::term_manager(prog_opts const &options, pde_domain<P> const &domain,
                              pde_scheme<P> &pde, sparse_grid const &grid,
                              hierarchy_manipulator<P> const &hier,
                              connection_patterns const &conn)
  : num_dims(domain.num_dims()), max_level(options.max_level()), legendre(hier.degree())
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

  // compute the dependencies
  if (term_groups.empty()) {
    mom_deps deps;
    for (auto const &tentry : terms)
      for (int d : iindexof(num_dims))
        deps += tentry.deps[d];
    deps_.emplace_back(deps);
  } else {
    deps_.reserve(term_groups.size() + 1);
    for (auto const &tg : term_groups) {
      mom_deps deps;
      for (int tid : indexrange(tg))
        for (int d : iindexof(num_dims))
          deps += terms[tid].deps[d];
      deps_.emplace_back(deps);
    }
    mom_deps deps;
    for (auto const &dp : deps_)
      deps += dp;
    deps_.emplace_back(deps);
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
      t.is_interpolatory = t.tmd.is_interpolatory();
      if (t.is_interpolatory) {
        if (t.num_chain == 1) { // single entry
          t.interp_uses_ifield = true;
        } else if (t.is_chain_link() and
                   (i+1 == static_cast<int>(terms.size())
                    or not terms[i+1].is_chain_link())) {
          // if part of a chain and the next term is not from the current chain
          // i.e., this is the first link in the chain
          t.interp_uses_ifield = true;
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
      has_field_interp = has_field_interp or it->interp_uses_ifield;
      if (it->is_chain_start())
        has_field_interp = has_field_interp or (it + it->num_chain -1)->interp_uses_ifield;
      it += it->num_chain;
    }

    if (has_field_interp)
      ifield.resize(1);
  }
}

template<typename P>
void term_manager<P>::buld_term(
    int const tid, sparse_grid const &grid, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, precon_method precon, P alpha)
{
  if (terms[tid].tmd.is_interpolatory()) // skip interpolation terms
    return;

  expect(legendre.pdof == hier.degree() + 1);
  expect(not terms[tid].tmd.is_chain());

  auto &tmd = terms[tid];

  bool merging_with_interp = false;
  if ((tmd.is_chain_start() or tmd.is_chain_link())
       and (static_cast<size_t>(tid + 1) < terms.size())
        and terms[tid + 1].is_chain_link()
         and terms[tid + 1].is_interpolatory)
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
    terms[tid + 1].interp_stop_at_hierarchy = true;

    std::vector<int> id_dirs;
    id_dirs.reserve(num_dims);
    for (int d : iindexof(num_dims))
    {
      rebuld_term1d(terms[tid], d, max_level, conn, hier, precon, alpha, merge_with_interp);
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
    for (int d : iindexof(num_dims)) {
      auto const &t1d = tmd.tmd.dim(d);

      int level = grid.current_level(d); // required level

      // terms that don't change should be build only once
      if (t1d.change() == changes_with::none)
        level = max_level; // build up to the max

      rebuld_term1d(terms[tid], d, level, conn, hier, precon, alpha);
    } // move to next dimension d
  }
}

template<typename P>
void term_manager<P>::rebuld_term1d(
    term_entry<P> &tentry, int const dim, int level,
    connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
    precon_method precon, P alpha, bool merge_with_interp)
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

  bool is_diag = t1d.is_volume();
  if (t1d.is_chain()) {
    rebuld_chain(tentry, dim, level, bmass, is_diag, wraw_diag, wraw_tri);
  } else {
    build_raw_mat(tentry, dim, 0, level, bmass, wraw_diag, wraw_tri);
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
    tentry.gpu_lcoeffs[dim].resize(level + 1);
    std::vector<P*> coeff_pntrs(level + 1, nullptr);
    for (int l = 0; l < level; l++) {
      tentry.gpu_lcoeffs[dim][l] = tentry.coeffs[dim].get_subpattern(l, conn).data_vector();
      coeff_pntrs[l] = tentry.gpu_lcoeffs[dim][l].data();
    }
    tentry.gpu_lcoeffs[dim][level] = tentry.coeffs[dim].data_vector();
    coeff_pntrs[level]             = tentry.gpu_lcoeffs[dim][level].data();

    tentry.gpu_coeffs[dim] = coeff_pntrs;

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

  // build the ADI preconditioner here
  if (precon == precon_method::adi) {
    if (is_diag) {
      to_euler(legendre.pdof, alpha, wraw_diag);
      psedoinvert(legendre.pdof, wraw_diag, raw_diag0);
      tentry.adi[dim] = hier.diag2hierarchical(raw_diag0, level, conn);
    } else {
      to_euler(legendre.pdof, alpha, wraw_tri);
      psedoinvert(legendre.pdof, wraw_tri, raw_tri0);
      tentry.adi[dim] = hier.tri2hierarchical(raw_tri0, level, conn);
    }
  }
}

template<typename P>
void term_manager<P>::build_raw_mat(
    term_entry<P> &tentry, int d, int clink, int level,
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
            t1d.rhs(cdata.electric_field, kwork.w1);
            gen_diag_cmat_pwc<P>(legendre, level, kwork.w1, raw_diag);
            kwork.w1.resize(n);
          } else {
            gen_diag_cmat_pwc<P>(legendre, level, cdata.electric_field, raw_diag);
          }
          break;
        case term_dependence::electric_field:
          throw std::runtime_error("el-field with position depend is not done (yet)");
          break;
        case term_dependence::moment_divided_by_density:
          if (t1d.moment() > 0) {
            gen_diag_mom_cases<P, +1, term_dependence::moment_divided_by_density>
              (legendre, level, t1d.moment(), cdata.moments, raw_diag);
          } else {
            gen_diag_mom_cases<P, -1, term_dependence::moment_divided_by_density>
              (legendre, level, -t1d.moment(), cdata.moments, raw_diag);
          }
          break;
        case term_dependence::lenard_bernstein_coll_theta_1x1v:
          gen_diag_mom_cases<P, 1, term_dependence::lenard_bernstein_coll_theta_1x1v>
            (legendre, level, 0, cdata.moments, raw_diag);
          break;
        case term_dependence::lenard_bernstein_coll_theta_1x2v:
          gen_diag_mom_cases<P, 1, term_dependence::lenard_bernstein_coll_theta_1x2v>
            (legendre, level, 0, cdata.moments, raw_diag);
          break;
        case term_dependence::lenard_bernstein_coll_theta_1x3v:
          gen_diag_mom_cases<P, 1, term_dependence::lenard_bernstein_coll_theta_1x3v>
            (legendre, level, 0, cdata.moments, raw_diag);
          break;
        default:
          if (t1d.rhs()) {
            gen_diag_cmat<P, operation_type::volume>
              (legendre, xleft[d], xright[d], level, t1d.rhs(), raw_rhs, raw_diag);
          } else {
            gen_diag_cmat<P, operation_type::volume>
              (legendre, level, t1d.rhs_const(), raw_diag);
          }
          break;
      }
      break;
    case operation_type::div:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::div, rhs_type::is_func>
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::div, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      }
      if (t1d.penalty() != 0) {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      }
      break;
    case operation_type::grad:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::grad, rhs_type::is_func>
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::grad, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      }
      if (t1d.penalty() != 0) {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      }
      break;
    case operation_type::penalty:
      expect(not t1d.rhs());
      gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const>
        (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      break;
    default: // case operation_type::identity:
      // identity, nothing to do for the matrix, but may have to do boundary conditions
      break;
  }

  if (bmass) {
    if (t1d.optype() == operation_type::volume)
      bmass->solve(legendre.pdof, raw_diag);
    else
      bmass->solve(legendre.pdof, raw_tri);
  }

  for (int b : indexrange(tentry.bc)) {
    // handle the non-separable in time, keep rhs values
    boundary_entry<P> &bentry = bcs[b];

    if (bentry.flux.chain_level(d) > clink) {
      expect(not bentry.consts[d].empty());
      if (t1d.is_volume()) {
        raw_diag.inplace_gemv(legendre.pdof, bentry.consts[d], t1);
      } else {
        raw_tri.inplace_gemv(legendre.pdof, bentry.consts[d], t1);
      }
    } else if (bentry.flux.chain_level(d) == clink) {
      // create a new entry
      if (tentry.flux_dim == d) {
        int const pdof = legendre.pdof;

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
            smmat::axpy(pdof, - rhs_left * scale, legendre.leg_left, bentry.consts[d].data());
          } else {
            smmat::axpy(pdof, - rhs_left * scale * fc, legendre.leg_left, bentry.consts[d].data());
          }
        }

        if (bentry.flux.is_right()) {
          P rhs_right = (t1d.rhs()) ? raw_rhs.vals.back()  : t1d.rhs_const();

          if (t1d.penalty() != 0)
            rhs_right *= P{1} - t1d.penalty();

          P const fc = bentry.flux.func().cdomain(d);
          if (fc == 0) { // non-separable in time
            // single-point value is always separable, so we can pre-compute in d-direction
            smmat::axpy(pdof, rhs_right * scale, legendre.leg_right,
                        bentry.consts[d].data() + num_entries - pdof);
          } else {
            smmat::axpy(pdof, rhs_right * scale * fc, legendre.leg_right,
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
            bentry.consts[d] = legendre.project(t1d.is_volume(), level, dsqr,
                                                bentry.flux.func().cdomain(d), raw_rhs.vals);
          } else { // constant times a constant
            P const rconst = (t1d.is_identity()) ? 1 : t1d.rhs_const();
            bentry.consts[d] = legendre.project(level, dsqr,
                                                bentry.flux.func().cdomain(d) * rconst);
          }
        } else {
          if (t1d.rhs()) { // product of non-consts
            std::vector<P> f(raw_rhs.pnts.size());
            bentry.flux.func().fdomain(d, raw_rhs.pnts, 0, f);
            bentry.consts[d] = legendre.project(t1d.is_volume(), level, dsqr, f, raw_rhs.vals);
          } else {
            // need function values, rhs is a constant
            legendre.interior_quad(xleft[d], xright[d], level, raw_rhs.pnts);
            raw_rhs.vals.resize(raw_rhs.pnts.size());
            bentry.flux.func().fdomain(d, raw_rhs.pnts, 0, raw_rhs.vals);
            bool constexpr use_interior = true;
            bentry.consts[d] = legendre.project(use_interior, level, dsqr, t1d.rhs_const(), raw_rhs.vals);
          }
        }

        if (bmass)
          bmass->solve(legendre.pdof, bentry.consts[d]);
      }
    } // if the bentry is associated with a higher link, then do nothing here
  }
}

template<typename P>
void term_manager<P>::build_raw_mass(int dim, term_1d<P> const &t1d, int level,
                                     block_diag_matrix<P> &raw_diag)
{
  expect(t1d.is_volume());
  expect(t1d.depends() == term_dependence::none);

  if (t1d.rhs()) {
    gen_diag_cmat<P, operation_type::volume>
      (legendre, xleft[dim], xright[dim], level, t1d.rhs(), raw_rhs, raw_diag);
  } else {
    gen_diag_cmat<P, operation_type::volume>
      (legendre, level, t1d.rhs_const(), raw_diag);
  }
}

template<typename P>
void term_manager<P>::rebuld_chain(
    term_entry<P> &tentry, int const d, int const level,
    block_diag_matrix<P> const *bmass,
    bool &is_diag, block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri)
{
  term_1d<P> &t1d = tentry.tmd.dim(d);
  expect(t1d.is_chain());
  int const num_chain = t1d.num_chain();
  expect(num_chain > 1);

  is_diag = true;
  for (int i : iindexof(num_chain)) {
    if (not t1d[i].is_volume()) {
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
    build_raw_mat(tentry, d, num_chain - 1, level, bmass, *diag0, raw_tri);
    for (int i = num_chain - 2; i > 0; i--) {
      build_raw_mat(tentry, d, i, level, bmass, raw_diag, raw_tri);
      diag1->check_resize(raw_diag);
      gemm_block_diag(legendre.pdof, raw_diag, *diag0, *diag1);
      std::swap(diag0, diag1);
    }
    build_raw_mat(tentry, d, 0, level, bmass, *diag1, raw_tri);
    raw_diag.check_resize(*diag1);
    gemm_block_diag(legendre.pdof, *diag1, *diag0, raw_diag);

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

  fill current = (t1d.chain_.back().is_volume()) ? fill::diag : fill::tri;
  build_raw_mat(tentry, d, num_chain - 1, level, bmass, *diag0, *tri0);

  for (int i = num_chain - 2; i > 0; i--)
  {
    build_raw_mat(tentry, d, i, level, bmass, raw_diag, raw_tri);
    // the result is in either raw_diag or raw_tri and must be multiplied and put
    // into either diag1 or tri1, then those should swap with diag0 and tri0
    if (t1d[i].is_volume()) { // computed a diagonal fill
      if (current == fill::diag) { // diag-to-diag
        diag1->check_resize(raw_diag);
        gemm_block_diag(legendre.pdof, raw_diag, *diag0, *diag1);
        std::swap(diag0, diag1);
      } else { // multiplying diag by tri-diag
        tri1->check_resize(raw_diag);
        gemm_diag_tri(legendre.pdof, raw_diag, *tri0, *tri1);
        std::swap(tri0, tri1);
      }
    } else { // computed tri matrix (upper or lower diagonal)
      if (current == fill::diag ) { // tri times diag
        tri1->check_resize(raw_tri);
        gemm_tri_diag(legendre.pdof, raw_tri, *diag0, *tri1);
        std::swap(tri0, tri1);
        current = fill::tri;
      } else {
        tri1->check_resize(raw_tri);
        gemm_block_tri(legendre.pdof, raw_tri, *tri0, *tri1);
        std::swap(tri0, tri1);
        current = fill::tri;
      }
    }
  }

  // last term, compute in diag1/tri1 and multiply into raw_tri
  build_raw_mat(tentry, d, 0, level, bmass, *diag1, *tri1);

  if (t1d[0].is_volume()) {
    // the rest must be a tri-diagonal matrix already
    // otherwise the whole chain would consist of only diagonal ones
    raw_tri.check_resize(*tri0);
    gemm_diag_tri(legendre.pdof, *diag1, *tri0, raw_tri);
  } else {
    if (current == fill::diag) {
      raw_tri.check_resize(*tri1);
      gemm_tri_diag(legendre.pdof, *tri1, *diag0, raw_tri);
    } else {
      raw_tri.check_resize(*tri1);
      gemm_block_tri(legendre.pdof, *tri1, *tri0, raw_tri);
    }
  }

  // apply the penalty that is added to the whole chain
  if (t1d.penalty() == 0)
    return;

  if (bmass) {
    gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const>
      (legendre, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.chain_.back().flux(),
        t1d.chain_.back().boundary(), raw_rhs, *tri0);
    bmass->solve(legendre.pdof, *tri0);
    raw_tri += *tri0;
  } else {
    // no need to worry about the mass, just add the penalty to the raw-tri
    gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
      (legendre, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.chain_.back().flux(),
        t1d.chain_.back().boundary(), raw_rhs, raw_tri);
  }
  // handle the penalty component of the boundary conditions
  std::vector<P> penwork; // extra allocation, should be rare, when having mass + builtin penalty
  for (int b : indexrange(tentry.bc)) {
    // handle the non-separable in time, keep rhs values
    boundary_entry<P> &bentry = bcs[b];

    // apply only the conditions for the bottom link
    if (bentry.flux.chain_level(d) != num_chain - 1)
      continue;

    int const pdof = legendre.pdof;

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
        smmat::axpy(pdof, -scale, legendre.leg_left, dest);
      } else {
        smmat::axpy(pdof, -scale * fc, legendre.leg_left, dest);
      }
    }

    if (bentry.flux.is_right()) {
      P const fc = bentry.flux.func().cdomain(d);
      if (fc == 0) { // non-separable in time
        smmat::axpy(pdof, scale, legendre.leg_right, dest + num_entries - pdof);
      } else {
        smmat::axpy(pdof, scale * fc, legendre.leg_right, dest + num_entries - pdof);
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
void term_manager<P>::mass_apply(
    sparse_grid const &grid, connection_patterns const &conns,
    P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const
{
  if (beta == 0) {
    y.resize(x.size());
  } else {
    expect(y.size() == x.size());
  }
  if (mass_term) {
    block_cpu(legendre.pdof, grid, conns, mass_perm, mass_forward,
              alpha, x.data(), beta, y.data(), kwork);
  } else {
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < x.size(); i++)
      y[i] = alpha * x[i] + beta * y[i];
  }
}
template<typename P>
P term_manager<P>::normL2(
    sparse_grid const &grid, connection_patterns const &conns,
    std::vector<P> const &x) const
{
  if (mass_term) {
    mass_apply(grid, conns, 1, x, 0, t1);
    P nrm = 0;
    for (size_t i = 0; i < x.size(); i++)
      nrm += x[i] * t1[i];
    return std::sqrt(nrm);
  } else {
    P nrm = 0;
    for (size_t i = 0; i < x.size(); i++)
      nrm += x[i] * x[i];
    return std::sqrt(nrm);
  }
}

template<typename P>
template<typename vector_type_x, typename vector_type_y>
void term_manager<P>::apply_tmpl(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    P alpha, vector_type_x x, P beta, vector_type_y y) const
{
  bool constexpr using_vectors = std::is_same_v<vector_type_x, std::vector<P> const &>;

  if constexpr (using_vectors)
  {
    expect(x.size() == y.size());
    expect(x.size() == kwork.w1.size());
  }
  expect(-1 <= gid and gid < static_cast<int>(term_groups.size()));

  auto kterm = [&grid, &conns, this](term_entry<P> const &tme, P al, P const in[], P be, P out[])
    -> void {
      if (tme.is_interpolatory) {
        if (tme.interp_uses_ifield) {
          if (tme.interp_stop_at_hierarchy)
            interp.field2hier(grid, conns, 0, ifield, tme.tmd.interp(), out, kwork, it1);
          else
            interp.field2wav(grid, conns, 0, ifield, al, tme.tmd.interp(), be, out, kwork, it1, it2);
        } else { // no field
          if (tme.interp_stop_at_hierarchy)
            interp.wav2hier(grid, conns, 0, in, tme.tmd.interp(), out, kwork, it1, it2);
          else
            interp(grid, conns, 0, in, al, tme.tmd.interp(), be, out, kwork, it1, it2);
        }
      } else {
        block_cpu(legendre.pdof, grid, conns, tme.perm, tme.coeffs,
                  al, in, be, out, kwork);
      }
    };

  P b = beta; // on first iteration, overwrite y

  P const *px = [&]()
        -> P const * {
      if constexpr (using_vectors)
        return x.data();
      else
        return x;
      }();
  P *py = [&]()
        -> P * {
      if constexpr (using_vectors)
        return y.data();
      else
        return y;
      }();

  if (not ifield.empty()) // using interpolation and will need the field
    interp.wav2nodal(grid, conns, px, ifield, kwork);

  int icurrent   = (gid == -1) ? 0                              : term_groups[gid].begin();
  int const iend = (gid == -1) ? static_cast<int>(terms.size()) : term_groups[gid].end();
  while (icurrent < iend)
  {
    auto it = terms.begin() + icurrent;

    #ifdef ASGARD_USE_MPI
    if (not resources.owns(it->rec)) {
      icurrent += it->num_chain;
      continue;
    }
    #endif

    if (it->num_chain == 1) {
      kterm(*it, alpha, px, b, py);
      ++icurrent;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      kterm(*(it + num_chain - 1), 1, px, 0, t1.data());
      for (int i = num_chain - 2; i > 0; --i) {
        kterm(*(it + i), 1, t1.data(), 0, t2.data());
        std::swap(t1, t2);
      }
      kterm(*it, alpha, t1.data(), b, py);

      icurrent += num_chain;
    }

    b = 1; // next iteration appends on y
  }

  if (not has_terms_) {
    int64_t const num = grid.num_indexes() * fm::ipow(legendre.pdof, num_dims);
    if (beta == 0) {
      std::fill_n(py, num, 0);
    } else {
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < num; i++)
        py[i] *= beta;
    }
  }
}

#ifdef ASGARD_USE_FLOPCOUNTER
template<typename P>
int64_t term_manager<P>::flop_count(
    int gid, sparse_grid const &grid, connection_patterns const &conns, P alpha, P beta) const
{
  #ifdef ASGARD_USE_MPI
  if (not is_leader())
    return -1;
  #endif

  expect(-1 <= gid and gid < static_cast<int>(term_groups.size()));

  int const gidx = gid + 1;
  if (flop_info.size() <= static_cast<size_t>(gidx))
    flop_info.resize(gidx + 1);

  if (flop_info[gidx].grid_gen == grid.generation())
    return flop_info[gidx].flops;

  int64_t flops = 0;

  auto kterm = [&grid, &conns, &flops, this](term_entry<P> const &tme, P al, P be)
    -> void {
      if (not tme.tmd.is_interpolatory())
        flops += block_cpu(legendre.pdof, grid, conns, tme.perm, al, be, kwork);
    };

  P b = beta; // on first iteration, overwrite y

  int icurrent   = (gid == -1) ? 0                              : term_groups[gid].begin();
  int const iend = (gid == -1) ? static_cast<int>(terms.size()) : term_groups[gid].end();
  while (icurrent < iend)
  {
    auto it = terms.begin() + icurrent;

    if (it->num_chain == 1) {
      kterm(*it, alpha, b);
      ++icurrent;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      kterm(*(it + num_chain - 1), 1, 0);

      for (int i = num_chain - 2; i > 0; --i)
        kterm(*(it + i), 1, 0);

      kterm(*it, alpha, b);

      icurrent += num_chain;
    }

    b = 1; // next iteration appends on y
  }

  if (not has_terms_)
    flops += static_cast<int64_t>(kwork.w1.size());

  flop_info[gidx].grid_gen = grid.generation();
  flop_info[gidx].flops    = flops;

  return flops;
}
#endif

#ifdef ASGARD_USE_GPU
template<typename P>
void term_manager<P>::prapare_kron_workspace_gpu(int64_t num_entries)
{
  int const num_gpus = compute->num_gpus();

  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});
    // GPU 0 always uses gpu_t1[0] for scratch-space when collecting local data
    if ((not t1.empty() or g == 0) and gpu_t1[g].size() < num_entries)
      gpu_t1[g].resize(num_entries);
    if (not t2.empty() and gpu_t2[g].size() < num_entries)
      gpu_t2[g].resize(num_entries);
    kwork.gpu_w1[g].resize(num_entries);
    kwork.gpu_w2[g].resize(num_entries);

    if (interp) {
      cpu_it1[g].resize(num_entries);
      cpu_it2[g].resize(num_entries);
      gpu_it1[g].resize(num_entries);
      gpu_it2[g].resize(num_entries);
    }
  }
}

template<typename P>
template<typename vector_type_x, typename vector_type_y, compute_mode mode>
void term_manager<P>::apply_tmpl_gpu(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    P alpha, vector_type_x x, P beta, vector_type_y y) const
{
  bool constexpr using_cpu_vectors = std::is_same_v<vector_type_x, std::vector<P> const &>;
  bool constexpr using_gpu_vectors = std::is_same_v<vector_type_x, gpu::vector<P> const &>;

  bool constexpr using_vectors = using_cpu_vectors or using_gpu_vectors;

  // general idea
  // 1. If CPU -> mode data to GPU 0
  // 2. If GPU 0 (moved or provided there), distribute to all GPUs
  //    --- have multi-GPU scratch space, data for x and y on each device
  // 3. !!!! Fix the load of the coefficient matrices, currently doesn't respect the device
  // 4. Loop over all devices and perform kron only for the local terms
  //    --- respect the has-term per device
  // 5. Bring all data back to device 0 and add it up
  //    --- apply alpha on every device, do beta at the end
  //
  //  -- maybe have 2 versions, 1 GPU and multi-GPUs

  if constexpr (using_cpu_vectors)
    static_assert(mode == compute_mode::cpu, "std::vector requires compute_mode::cpu");
  if constexpr (using_gpu_vectors)
    static_assert(mode == compute_mode::gpu, "gpu::vector requires compute_mode::gpu");
  // no reasonable way to check if pointers are on the CPU or GPU, assume "mode" is set correctly

  if constexpr (using_vectors)
    expect(x.size() == y.size());

  int64_t const num_entries  = fm::ipow(legendre.pdof, grid.num_dims()) * grid.num_indexes();

  expect(-1 <= gid and gid < static_cast<int>(term_groups.size()));

  auto kterm = [&grid, &conns, this]
               (gpu::device dev, term_entry<P> const &tme, P al, P const in[], P be, P out[])
    -> void {
      if (tme.is_interpolatory) {
        if (tme.interp_uses_ifield) {
          if (tme.interp_stop_at_hierarchy)
            interp.field2hier(dev, grid, conns, 0, ifield, tme.tmd.interp(), out, kwork,
                              cpu_it1[dev.id], gpu_it1[dev.id]);
          else
            interp.field2wav(dev, grid, conns, 0, ifield, al, tme.tmd.interp(), be, out, kwork,
                             cpu_it1[dev.id], gpu_it1[dev.id], gpu_it2[dev.id]);
        } else {
          if (tme.interp_stop_at_hierarchy)
            interp.wav2hier(dev, grid, conns, 0, in, tme.tmd.interp(), out, kwork,
                            cpu_it1[dev.id], cpu_it2[dev.id], gpu_it1[dev.id]);
          else
            interp(dev, grid, conns, 0, in, al, tme.tmd.interp(), be, out, kwork,
                   cpu_it1[dev.id], cpu_it2[dev.id], gpu_it1[dev.id], gpu_it2[dev.id]);
        }
      } else {
        block_gpu(dev, legendre.pdof, grid, conns, tme.perm, tme.gpu_coeffs,
                  al, in, be, out, kwork, tme.coeffs);
      }
    };

  // if doing out-of-core, load data onto the device and sync across devices, device 0 is always the "root"
  if constexpr (mode == compute_mode::cpu) {
    compute->set_device(gpu::device{0});
    if constexpr (using_cpu_vectors) {
      gpu_x[0] = x;
      if (beta == 0) // allocate memory, no need to copy
        gpu_y[0].resize(num_entries);
      else
        gpu_y[0] = y;
    } else {
      gpu_x[0].resize(num_entries);
      gpu_x[0].copy_from_host(num_entries, x);
      gpu_y[0].resize(num_entries);
      gpu_y[0].copy_from_host(num_entries, y);
    }
  }

  int const num_gpus = compute->num_gpus();

  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});

    // effective x/y, either x or gpu_x[id]
    P const *xpntr = nullptr;
    P *ypntr = nullptr;

    if (g == 0) {
      if constexpr (mode == compute_mode::cpu) {
        xpntr = gpu_x[0].data();
        ypntr = gpu_y[0].data();
      } else {
        if constexpr (using_vectors) {
          xpntr = x.data();
          ypntr = y.data();
        } else {
          xpntr = x;
          ypntr = y;
        }
      }
    } else {
      gpu_x[g].resize(num_entries);
      gpu_y[g].resize(num_entries);
      if constexpr (mode == compute_mode::cpu) {
        gpu::mcopy(gpu::device{0}, gpu_x[0], gpu::device{g}, gpu_x[g]);
        gpu::mcopy(gpu::device{0}, gpu_y[0], gpu::device{g}, gpu_y[g]);
      } else {
        gpu::mcopy(gpu::device{0}, x, gpu::device{g}, gpu_x[g]);
        gpu::mcopy(gpu::device{0}, y, gpu::device{g}, gpu_y[g]);
      }
      xpntr = gpu_x[g].data();
      ypntr = gpu_y[g].data();
    }

    P b = (g == 0) ? beta : 0; // on first iteration, overwrite y

    if (not ifield.empty()) {
      interp.wav2nodal(gpu::device{0}, grid, conns, xpntr, gpu_it1[0].data(), kwork);
      gpu_it1[0].copy_to_host(ifield);
    }

    bool term_found = false; // does this GPU have at least 1 term

    int icurrent   = (gid == -1) ? 0                              : term_groups[gid].begin();
    int const iend = (gid == -1) ? static_cast<int>(terms.size()) : term_groups[gid].end();
    while (icurrent < iend)
    {
      auto it = terms.begin() + icurrent;

      // skip the terms associated with other MPI ranks or devices
      #ifdef ASGARD_USE_MPI
      if (not resources.owns(it->rec) or it->rec.device != g) {
        icurrent += it->num_chain;
        continue;
      }
      #else
      if (it->rec.device != g) {
        icurrent += it->num_chain;
        continue;
      }
      #endif

      if (it->num_chain == 1) {
        kterm(gpu::device{g}, *it, alpha, xpntr, b, ypntr);
      } else {
        // dealing with a chain
        int const num_chain = it->num_chain;

        kterm(gpu::device{g}, *(it + num_chain - 1), 1, xpntr, 0, gpu_t1[g].data());
        for (int i = num_chain - 2; i > 0; --i) {
          kterm(gpu::device{g}, *(it + i), 1, gpu_t1[g].data(), 0, gpu_t2[g].data());
          std::swap(gpu_t1[g], gpu_t2[g]);
        }
        kterm(gpu::device{g}, *it, alpha, gpu_t1[g].data(), b, ypntr);
      }

      icurrent += it->num_chain;

      term_found = true; // something got computed above
      b = 1; // next iteration appends on y
    }

    // handle the case when a GPU has no terms
    if (not term_found) {
      if (g == 0 and beta != 0) { // main GPU is expected to scale y
        compute->scal(num_entries, beta, ypntr);
      } else {
        // either scale by zero or no terms, so set to zero
        compute->fill_zeros(num_entries, ypntr);
      }
    }
    compute->device_synchronize();
  }

  // collect the data across the GPUs
  for (int g = 1; g < num_gpus; g++) {
    gpu::mcopy(num_entries, gpu::device{g}, gpu_y[g].data(), gpu::device{0}, gpu_t1[0].data());
    if constexpr (mode == compute_mode::cpu) {
      compute->axpy(num_entries, gpu_t1[0].data(), gpu_y[0].data());
    } else {
      if constexpr (using_gpu_vectors)
        compute->axpy(num_entries, gpu_t1[0].data(), y.data());
      else
        compute->axpy(num_entries, gpu_t1[0].data(), y);
    }
  }

  if constexpr (mode == compute_mode::cpu)// send back to the CPU
    gpu_y[0].copy_to_host(y);
}
#endif

template<typename P>
void term_manager<P>::apply_all_adi(
    sparse_grid const &grid, connection_patterns const &conns,
    P const x[], P y[]) const
{
  int64_t const n = grid.num_indexes() * fm::ipow(legendre.pdof, grid.num_dims());

  t1.resize(n);
  t2.resize(n);
  std::copy_n(x, n, t1.data());

  auto it = terms.begin();
  while (it < terms.end())
  {
    if (it->num_chain == 1) {
      kron_term_adi(grid, conns, *it, 1, t1.data(), 0, t2.data());
      std::swap(t1, t2);
      ++it;
    } else {
      // TODO: consider whether we should do this or not
      it += it->num_chain;
    }
  }
  std::copy_n(t1.data(), n, y);
}

template<typename P>
void term_manager<P>::make_jacobi(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    std::vector<P> &y) const
{
  int const block_size      = fm::ipow(legendre.pdof, grid.num_dims());
  int64_t const num_entries = block_size * grid.num_indexes();

  if (y.size() == 0)
    y.resize(num_entries);
  else {
    y.resize(num_entries);
    std::fill(y.begin(), y.end(), P{0});
  }

  kwork.w1.resize(num_entries);

  int icurrent   = (gid == -1) ? 0                              : term_groups[gid].begin();
  int const iend = (gid == -1) ? static_cast<int>(terms.size()) : term_groups[gid].end();
  while (icurrent < iend)
  {
    auto it = terms.begin() + icurrent;

    #ifdef ASGARD_USE_MPI
    if (not resources.owns(it->rec)) {
      icurrent += it->num_chain;
      continue;
    }
    #endif

    if (it->num_chain == 1) {
      kron_diag<data_mode::increment>(grid, conns, *it, block_size, y);
      icurrent++;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      std::fill(kwork.w1.begin(), kwork.w1.end(), P{0});

      kron_diag<data_mode::increment>(grid, conns, *(it + num_chain - 1),
                                      block_size, kwork.w1);

      for (int i = num_chain - 2; i >= 0; --i) {
        kron_diag<data_mode::multiply>(grid, conns, *(it + i),
                                       block_size, kwork.w1);
      }
ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < num_entries; i++)
        y[i] += kwork.w1[i];

      icurrent += num_chain;
    }
  }
}

template<typename P>
template<data_mode mode>
void term_manager<P>::kron_diag(
    sparse_grid const &grid, connection_patterns const &conn,
    term_entry<P> const &tme, int const block_size, std::vector<P> &y) const
{
  static_assert(mode == data_mode::increment or mode == data_mode::multiply);

#pragma omp parallel
  {
    std::array<P const *, max_num_dimensions> amats;

#pragma omp for
    for (int i = 0; i < grid.num_indexes(); i++) {
      for (int d : iindexof(num_dims))
        if (tme.coeffs[d].empty())
          amats[d] = nullptr;
        else
          amats[d] = tme.coeffs[d][conn[tme.coeffs[d]].row_diag(grid[i][d])];

      for (int t : iindexof(block_size)) {
        P a = 1;
        int tt = i;
        for (int d = num_dims - 1; d >= 0; --d)
        {
          if (amats[d] != nullptr) {
            int const rc = tt % legendre.pdof;
            a *= amats[d][rc * legendre.pdof + rc];
          }
          tt /= legendre.pdof;
        }
        if constexpr (mode == data_mode::increment)
          y[i * block_size + t] += a;
        else if constexpr (mode == data_mode::multiply)
          y[i * block_size + t] *= a;
      }
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

  std::vector<int> ranks;
  if (deps().poisson or deps().num_moments > 0)
    ranks.reserve(terms.size() + 1);

  #ifdef ASGARD_USE_MPI
  if (deps().poisson) {
    for (auto const &t : terms) {
      for (auto const &d : t.deps)
        if (d.poisson)
          ranks.push_back(t.rec.group);
    }
    expect(ranks.size() > 0);
    if (ranks.size() > 1) {
      ranks.push_back(0);
      std::sort(ranks.begin(), ranks.end());
      ranks.erase( std::unique(ranks.begin(), ranks.end()), ranks.end() );

      MPI_Comm cm = resources.new_comm_from_group(ranks);
      if (std::any_of(ranks.begin(), ranks.end(), [&](int r) -> bool { return (r == resources.rank()); }))
        resources.set_poisson_comm(cm);
    }
  }

  if (deps().num_moments > 0) {
    ranks.resize(0);
    for (auto const &t : terms) {
      for (auto const &d : t.deps)
        if (d.num_moments > 0)
          ranks.push_back(t.rec.group);
    }
    expect(ranks.size() > 0);
    if (ranks.size() > 1) {
      ranks.push_back(0);
      std::sort(ranks.begin(), ranks.end());
      ranks.erase( std::unique(ranks.begin(), ranks.end()), ranks.end() );
      MPI_Comm cm = resources.new_comm_from_group(ranks);
      if (std::any_of(ranks.begin(), ranks.end(), [&](int r) -> bool { return (r == resources.rank()); }))
        resources.set_moments_comm(cm);
    }
  }
  #endif // ASGARD_USE_MPI

  // if (mpi::is_world_rank(0)) {
  //   std::cout << term_groups.size() << "\n";
  //
  //   for (auto const &t : terms)
  //     std::cout << " assigned to: " << t.rec.group << "  gpu: " << t.rec.device << " chain num = " << t.num_chain << '\n';
  //
  //   for (auto const &s : sources)
  //     std::cout << " source to: " << s.rec.group << "  gpu: " << s.rec.device << '\n';
  // }
#endif
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct term_entry<double>;
template struct term_manager<double>;

template void term_manager<double>::kron_diag<data_mode::increment>(
    sparse_grid const &, connection_patterns const &,
    term_entry<double> const &, int const, std::vector<double> &) const;
template void term_manager<double>::kron_diag<data_mode::multiply>(
    sparse_grid const &, connection_patterns const &,
    term_entry<double> const &, int const, std::vector<double> &) const;

template void term_manager<double>::apply_tmpl<std::vector<double> const &, std::vector<double> &>(
    int, sparse_grid const &, connection_patterns const &, double,
    std::vector<double> const &, double, std::vector<double> &) const;
template void term_manager<double>::apply_tmpl<double const[], double[]>(
    int, sparse_grid const &, connection_patterns const &, double,
    double const[], double, double[]) const;

#ifdef ASGARD_USE_GPU
template void term_manager<double>::apply_tmpl_gpu<std::vector<double> const &, std::vector<double> &, compute_mode::cpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    double alpha, std::vector<double> const &x, double beta, std::vector<double> &y) const;
template void term_manager<double>::apply_tmpl_gpu<double const[], double[], compute_mode::cpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    double alpha, double const x[], double beta, double y[]) const;
template void term_manager<double>::apply_tmpl_gpu<gpu::vector<double> const &, gpu::vector<double> &, compute_mode::gpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    double alpha, gpu::vector<double> const &x, double beta, gpu::vector<double> &y) const;
template void term_manager<double>::apply_tmpl_gpu<double const[], double[], compute_mode::gpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    double alpha, double const x[], double beta, double y[]) const;
#endif

#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct term_entry<float>;
template struct term_manager<float>;

template void term_manager<float>::kron_diag<data_mode::increment>(
    sparse_grid const &, connection_patterns const &,
    term_entry<float> const &, int const, std::vector<float> &) const;
template void term_manager<float>::kron_diag<data_mode::multiply>(
    sparse_grid const &, connection_patterns const &,
    term_entry<float> const &, int const, std::vector<float> &) const;

template void term_manager<float>::apply_tmpl<std::vector<float> const &, std::vector<float> &>(
    int, sparse_grid const &, connection_patterns const &, float,
    std::vector<float> const &, float, std::vector<float> &) const;
template void term_manager<float>::apply_tmpl<float const[], float[]>(
    int, sparse_grid const &, connection_patterns const &, float,
    float const[], float, float[]) const;

#ifdef ASGARD_USE_GPU
template void term_manager<float>::apply_tmpl_gpu<std::vector<float> const &, std::vector<float> &, compute_mode::cpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    float alpha, std::vector<float> const &x, float beta, std::vector<float> &y) const;
template void term_manager<float>::apply_tmpl_gpu<float const[], float[], compute_mode::cpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    float alpha, float const x[], float beta, float y[]) const;
template void term_manager<float>::apply_tmpl_gpu<gpu::vector<float> const &, gpu::vector<float> &, compute_mode::gpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    float alpha, gpu::vector<float> const &x, float beta, gpu::vector<float> &y) const;
template void term_manager<float>::apply_tmpl_gpu<float const[], float[], compute_mode::gpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    float alpha, float const x[], float beta, float y[]) const;
#endif

#endif

}

