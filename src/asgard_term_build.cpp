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

  { // copy the terms from the pde_scheme
    // label chain-links and mark if interp or extra workspace is needed
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

  // link terms to boundary-condition terms
  for (auto &tt : terms) {
    int const n = static_cast<int>(tt.tmd.bc_flux_.size());
    tt.bc = indexrange(num_bc, num_bc + n);
    num_bc += n;
  }

  // form groups for the sources
  if (not term_groups.empty()) {
    int j = 0, bc_begin = 0, bc_end = 0; // index for the boundary conditions
    for (int groupid : iindexof(term_groups)) {
      for (int it : indexrange(term_groups[groupid]))
        bc_end += terms[it].bc.size();
      source_groups[j++].bc_range = irange(bc_begin, bc_end);
      bc_begin = bc_end;
    }
  }

  // copy the boundary fluxes, set the proper chain levels and mark as sep/const
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
      if (bcs.back().is_time_non_sep()) { // non-separable in time
        bcs_have_time_dep = true;
        for (int d : iindexof(num_dims)) {
          rassert(not tt.tmd.dim(d).is_chain(),
                  "cannot use non-separable in time boundary conditions with 1d-chains, "
                  "the purpose of the 1d chain is to pre-compute and cache entries but non-separable "
                  "data cannot be pre-computed, an md-chain must be used instead");
        }
      }
    }
  }

  // domain left/right bounds
  for (int d : iindexof(num_dims)) {
    xleft[d]  = pde.domain().xleft(d);
    xright[d] = pde.domain().xright(d);
  }

  // set the mass, needed for the sources below
  build_mass_matrices(hier, conn); // large, up to max-level
  rebuild_mass_matrices(grid); // small, up to the current level

  {// copy the separable sources, prepare the constant components
    std::vector<separable_func<P>> &sep = pde.sources_sep_;

    int num_sources = 0;
    for (auto const &s : sep) {
      int const dims = s.num_dims();
      rassert(dims == 0 or dims == num_dims, "incorrect dimension set for source");
      if (dims > 0) ++num_sources;
    }

    sources_md.resize(pde.sources_md_.size());
    for (size_t i = 0; i < pde.sources_md_.size(); i++)
      sources_md[i].func = std::move(pde.sources_md_[i]);

    sources.reserve(num_sources);

    for (auto &s : sep) {
      if (s.num_dims() == 0)
        continue;

      sources.emplace_back(std::move(s));

      if (sources.back().is_time_non_sep())
      {
        sources_have_time_dep = true;
      }
      else
      {
        expect(sources.back().is_time_const() or sources.back().is_time_sep());

        for (int d : iindexof(num_dims)) {
          if (sources.back().func.is_const(dimension_id{d})) {
            sources.back().consts[d]
                = hier.get_project1d_c(s.const_at(dimension_id{d}), mass[d], d, max_level);
          } else {
            expect(sources.back().func.is_fixed(dimension_id{d}));
            sources.back().consts[d] = hier.get_project1d_f(
                [&](std::vector<P> const &x, std::vector<P> &y)->
                  void {
                    sources.back().func.fixed_at(dimension_id{d})(x, y);
                  },
                mass[d], d, max_level);
          }
          #ifdef ASGARD_USE_GPU
          compute->set_device(gpu::device{sources.back().rec.device});
          sources.back().gpu_consts[d] = sources.back().consts[d];
          compute->set_device(gpu::device{0});
          #endif
        }
      }
    }
  }

  prapare_kron_workspace(grid); // setup kronmult workspace

  // reshuffle the terms and sources across MPI ranks and GPU devices
  has_terms_ = not terms.empty();
  assign_compute_resources();

  // prepare the workspaces for the sources
  // consider only sources that are associated with this MPI rank and not time-dependant
  // the time sources cannot use workspace to accelerate computations
  auto is_active_src = [&, this](source_entry<P> const &src) -> bool
    {
      if (not resources.owns(src.rec))
        return false;
      return (not src.is_time_non_sep());
    };
  auto is_active_bc = [&, this](boundary_entry<P> const &bc) -> bool
    {
      if (not resources.owns(terms[bc.term_index].rec))
        return false;
      return (not bc.is_time_non_sep());
    };

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
  // e.g., the requirements change if this MPI rank has no terms
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
        if (not t.tmd.get_interp_moments().empty()) {
          t.interplan.use_moments();
        }
        #ifdef ASGARD_USE_GPU
        if (t.tmd.is_gpu_interpolatory()) {
          t.interplan.use_gpu_func();
          if (t.interplan.uses_field() and resources.owns(t.rec))
            gpu_ifield.resize(1);
        }
        #else
        rassert(not t.tmd.is_gpu_interpolatory(), "cannot use GPU interpolation without CUDA or ROCM enabled");
        #endif
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
      if (it->is_chain_start()) {
        auto const itn = it + (it->num_chain -1); // first link of the chain
        // if using field from the CPU
        if (itn->interplan.uses_field() and not itn->interplan.uses_gpu_func())
          has_field_interp = true;
      } else {
        if (it->interplan.uses_field() and not it->interplan.uses_gpu_func())
          has_field_interp = true;
      }

      it += it->num_chain;
    }

    if (has_field_interp)
      ifield.resize(1);

    // handle the moment dependencies, identify regular and interp moments for each group
    // respect the MPI and GPU distributions
    auto insert = [](std::vector<moment_id> const &s, std::vector<moment_id> &dest)
      {
        if (s.empty()) return;
        dest.insert(dest.end(), s.begin(), s.end());
      };

    auto compar_id = [](moment_id id1, moment_id id2) -> bool { return (id1() < id2()); };

    auto remove_repeated = [&](std::vector<moment_id> &vec)
      {
        if (vec.empty()) return;

        std::sort(vec.begin(), vec.end(), compar_id);
        // std::unique shuffles the vector, so the entries that need to removed appear after unique
        auto last = std::unique(vec.begin(), vec.end());
        vec.erase(last, vec.end());
      };

    bool has_poisson = false; // are there any Poisson deps
    bool has_sep_mom = false; // are there any separable moments deps

    // indexes for the term and source groups, if no groups then using only 1 index
    auto igroups = (term_groups.empty()) ? indexrange(1) : indexrange(term_groups);
    expect(term_groups.size() == source_groups.size());

    #ifdef ASGARD_USE_GPU
    // have to clean the logic of skip-interp
    // gpu-moms should keep all moments, term1d moments -> cpu_raw
    // cpu_interp -> when we have interp from the CPU
    // all-interp -> CPU or GPU interp, if not there then it will be in skip-interp
    std::array<std::vector<std::vector<moment_id>>, max_num_gpus> gpu_moms;
    std::vector<std::vector<moment_id>> cpu_raw(std::max(term_groups.size(), size_t{1}));
    std::vector<std::vector<moment_id>> cpu_interp(cpu_raw.size());
    std::vector<std::vector<moment_id>> all_interp(cpu_raw.size());
    std::vector<std::vector<moment_id>> skip_interp(cpu_raw.size());
    for (auto &r : cpu_raw) r.reserve(250);
    for (auto &r : cpu_interp) r.reserve(250);
    for (auto &r : all_interp) r.reserve(250);
    for (auto &r : skip_interp) r.reserve(250);

    for (auto &gm : gpu_moms) {
      gm.resize(std::max(term_groups.size(), size_t{1}));
      for (auto &r : gm) r.reserve(250);
    }

    for (int gid : igroups) {
      auto this_group = (term_groups.empty()) ? indexrange(terms) : indexrange(term_groups[gid]);
      for (int tid : this_group) {
        auto const &tentry = terms[tid];
        if (not resources.owns(tentry.rec)) continue;

        if (tentry.is_separable()) { // only separable terms can have 1D moment deps
          for (int d : iindexof(num_dims)) {
            auto const &mids = tentry.tmd.dim(d).mids_;
            insert(mids, gpu_moms[tentry.rec.device][gid]);
            insert(mids, cpu_raw[gid]);
          }
        } else if (tentry.interplan.uses_moments()) {
          auto const &mids = tentry.tmd.mids_;
          insert(mids, gpu_moms[tentry.rec.device][gid]);
          insert(mids, all_interp[gid]);
          if (not tentry.interplan.uses_gpu_func())
            insert(mids, cpu_interp[gid]);
        }
      }
    }
    for (int gid : igroups) {
      auto const &src = sources_md[gid];
      if (not src.is_moment() or not resources.owns(src.rec)) continue;

      auto const &mids = pde.sources_moments_[gid];
      insert(mids, gpu_moms[src.rec.device][gid]);
      insert(mids, all_interp[gid]);
      if (not src.is_gpu())
        insert(mids, cpu_interp[gid]);
    }

    for (auto &gm : gpu_moms) for (auto &vec : gm) remove_repeated(vec);
    for (auto &vec : cpu_raw) remove_repeated(vec);
    for (auto &vec : cpu_interp) remove_repeated(vec);
    for (auto &vec : all_interp) remove_repeated(vec);

    auto missing = [&](std::vector<moment_id> const &vec, moment_id mid)
        -> bool {
        return not std::binary_search(vec.begin(), vec.end(), mid, compar_id);
      };

    for (int gid : igroups) {
      for (auto const &gm : gpu_moms) {
        for (moment_id mid : gm[gid]) {
          if (missing(all_interp[gid], mid)) // not being interpolated
            skip_interp[gid].push_back(mid);
        }
      }
      remove_repeated(skip_interp[gid]);
    }

    // if (mpi::is_world_rank(1)) {
    //   std::cout << " gpu-moms - num-groups: " << cpu_raw.size() << '\n';
    //   for (auto const &dev : gpu_moms) {
    //     std::cout << " -- dev --\n";
    //     for (auto const &grp : dev) {
    //       for (auto m : grp) {
    //         std::cout << m() << "    ";
    //       }
    //       std::cout << '\n';
    //     }
    //     std::cout << '\n';
    //   }
    //
    //   std::cout << " cpu_raw\n";
    //   for (auto const &grp : cpu_raw) {
    //     for (auto m : grp) {
    //       std::cout << m() << "    ";
    //     }
    //     std::cout << '\n';
    //   }
    //   std::cout << " cpu_interp\n";
    //   for (auto const &grp : cpu_interp) {
    //     for (auto m : grp) {
    //       std::cout << m() << "    ";
    //     }
    //     std::cout << '\n';
    //   }
    //   std::cout << " skip_interp\n";
    //   for (auto const &grp : skip_interp) {
    //     for (auto m : grp) {
    //       std::cout << m() << "    ";
    //     }
    //     std::cout << '\n';
    //   }
    // }
    moms.set_moment_distribution(gpu_moms, cpu_raw, cpu_interp, skip_interp);
    #endif

    // CPU logic here, have only regular and interp moments per group
    std::vector<std::vector<moment_id>> regular(std::max(term_groups.size(), size_t{1}));
    std::vector<std::vector<moment_id>> intp(regular.size());
    for (auto &r : regular) r.reserve(250);
    for (auto &r : intp) r.reserve(250);

    for (int gid : igroups) {
      auto this_group = (term_groups.empty()) ? indexrange(terms) : indexrange(term_groups[gid]);
      for (int tid : this_group) {
        auto const &tentry = terms[tid];
        if (not resources.owns(tentry.rec)) continue;

        has_poisson = has_poisson or tentry.has_poisson;
        if (tentry.is_separable()) { // only separable terms can have 1D moment deps
          for (int d : iindexof(num_dims)) {
            auto const &mids = tentry.tmd.dim(d).mids_;
            insert(mids, regular[gid]);
            has_sep_mom = has_sep_mom or not mids.empty();
          }
        } else if (tentry.interplan.uses_moments()) {
          auto const &mids = tentry.tmd.mids_;
          insert(mids, regular[gid]);
          insert(mids, intp[gid]);
        }
      }
    }
    for (int gid : igroups) {
      auto const &src = sources_md[gid];
      if (not src or not src.is_moment() or not resources.owns(src.rec)) continue;

      auto const &mids = pde.sources_moments_[gid];
      insert(mids, regular[gid]);
      insert(mids, intp[gid]);
    }
    // remove redundant entries
    for (auto &vec : regular) remove_repeated(vec);

    for (auto &vec : intp) remove_repeated(vec);

    moms.set_moment_types(regular, intp);

    if (has_poisson) {
      if (term_groups.empty())
        has_poisson_.resize(1, true); // one group, has Poisson
      else {
        // multiple groups, not all groups would have Poisson dep
        has_poisson_.resize(term_groups.size(), false);
        for (int gid : iindexof(term_groups)) {
          for (int tid : indexrange(term_groups[gid])) {
            if (not resources.owns(terms[tid].rec)) continue;

            if (terms[tid].has_poisson) {
              has_poisson_[gid] = true;
              break; // move to the next group
            }
          }
        }
      }
    }
    // There is a catch here. The has_poisson() logic is used to determine whether we need
    // to have a local Poisson solve and whether to update any Poisson terms at all.
    // The has_sep_moments() logic is used to determine when any of the terms change,
    // which would require updating the preconditioner even if the sparse grid is unchanged.
    // Thus, the Poisson logic considers separable and non-separable terms and excludes terms
    // not associated with this MPI rank, while the sep-mom logic considers only separable terms
    // but includes all MPI ranks, since building the preconditioner is a global MPI operation.
    // (maybe this should be "change-with-time" logic)
    if (has_sep_mom) {
      if (term_groups.empty())
        has_sep_moments_.resize(1, true);
      else {
        has_sep_moments_.resize(term_groups.size(), false); // will process groups below
        for (int gid : iindexof(term_groups)) {
          for (int tid : indexrange(term_groups[gid])) {
            if (terms[tid].is_separable()) {
              for (int d : iindexof(num_dims))
                if (terms[tid].tmd.dim(d).depends() != term_dependence::none) {
                  has_sep_moments_[gid] = true;
                  break;
                }
            }
          }
        }
      }
    }

  } // end of the moment dependencies logic

  #ifdef ASGARD_USE_GPU
  kwork.row_map.resize(max_num_gpus); // TODO: check if this is needed
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
      #ifdef ASGARD_USE_GPU
      compute->set_device(gpu::device{terms[bentry.term_index].rec.device});
      bentry.gpu_consts[dim] = bentry.consts[dim];
      compute->set_device(gpu::device{0});
      #endif
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

          P const fc = bentry.flux.func().const_at(dimension_id{d});
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

          P const fc = bentry.flux.func().const_at(dimension_id{d});
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
        if (bentry.is_time_non_sep()) // no constant components to pre-compute
          continue;

        P const dsqr = std::sqrt(xright[d] - xleft[d]);

        if (bentry.flux.func().is_const(dimension_id{d})) {
          if (t1d.rhs()) { // constant times spatially variable
            bentry.consts[d] = basis.project(t1d.is_diagonal(), level, dsqr,
                                             bentry.flux.func().const_at(dimension_id{d}), raw_rhs.vals);
          } else { // constant times a constant
            P const rconst = (t1d.is_identity()) ? 1 : t1d.rhs_const();
            bentry.consts[d] = basis.project(level, dsqr,
                                             bentry.flux.func().const_at(dimension_id{d}) * rconst);
          }
        } else {
          if (t1d.rhs()) { // product of non-consts
            std::vector<P> f(raw_rhs.pnts.size());
            bentry.flux.func().fixed_at(dimension_id{d})(raw_rhs.pnts, f);
            bentry.consts[d] = basis.project(t1d.is_diagonal(), level, dsqr, f, raw_rhs.vals);
          } else {
            // need function values, rhs is a constant
            basis.interior_quad(xleft[d], xright[d], level, raw_rhs.pnts);
            raw_rhs.vals.resize(raw_rhs.pnts.size());
            bentry.flux.func().fixed_at(dimension_id{d})(raw_rhs.pnts, raw_rhs.vals);
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
      P const fc = bentry.flux.func().const_at(dimension_id{d});
      if (fc == 0) { // non-separable in time
        smmat::axpy(pdof, -scale, basis.leg_left, dest);
      } else {
        smmat::axpy(pdof, -scale * fc, basis.leg_left, dest);
      }
    }

    if (bentry.flux.is_right()) {
      P const fc = bentry.flux.func().const_at(dimension_id{d});
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
  // measuring work in units of 1D lower/upper kron operations
  // assuming the cost is the same (it is near the same)
  // interpolation terms count all the steps and ignore the nodal-function
  // (the nodal function can be costly, especially in GPU context with data has to move)
  // terms take into account chaining

  struct work_amount {
    explicit work_amount(float v) : value(v) {}
    float value = 0;
  };

  struct work_item {
    work_item() = default;
    work_item(work_amount amount) : work(amount) {}
    work_item(work_amount amount, int id_in) : work(amount), id(id_in) {}
    work_amount work{0};
    int id = -1;
  };

  std::vector<work_item> work;
  work.reserve(terms.size() + sources.size());

  auto get_work = [&](term_entry<P> const &tentry)
    -> work_amount {
      if (tentry.is_separable()) {
        auto const &perm      = tentry.perm;
        int const active_dims = perm.num_dimensions();

        float w = 0;
        for (int64_t i = 0; i < perm.size(); i++) {
          for (int d : iindexof(active_dims))
            w += (perm(i, d).fill == conn_fill::both) ? 2 : 1;
        }
        return work_amount{w};
      } else {
        float const w = fm::ipow2(num_dims) + 2 * num_dims;
        return work_amount{w};
      }
    };

  struct balance_manager {
    std::vector<float> workload;
    balance_manager(int num_workers) : workload(num_workers, 0) {}
    void add(int id, work_amount work) {
      workload[id] += work.value;
    }
    int lowest() { // get the id with lowest load
      int im  = 0;
      float l = workload[0];
      for (int i = 1; i < static_cast<int>(workload.size()); i++) {
        if (workload[i] < l) {
          im = i;
          l = workload[i];
        }
      }
      return im;
    }
    int lowest_back() { // get the id with lowest load, but starts from the back
      int im  = static_cast<int>(workload.size() - 1);
      float l = workload.back();
      for (int i = static_cast<int>(workload.size()) - 2; i >= 0; i--) {
        if (workload[i] < l) {
          im = i;
          l = workload[i];
        }
      }
      return im;
    }
  };

  enum class balance_mode {
    mpi_ranks, gpus
  };

  auto compute_terms_work = [&](int gid, balance_mode mode)
    -> void {
      work.resize(0); // load the new work-items

      auto const tgroup = terms_group_range(group_id{gid});
      int icurrent = tgroup.ibegin();
      while (icurrent < tgroup.iend())
      {
        auto it = terms.begin() + icurrent;

        if (mode == balance_mode::gpus and not resources.owns(it->rec)) {
          icurrent += it->num_chain;
          continue;
        }

        work_item item{get_work(*it), icurrent};
        int const num_chain = it->num_chain;
        for (int i = 1; i < num_chain; i++)
          item.work.value += get_work(terms[icurrent + i]).value;

        work.push_back(item);
        icurrent += num_chain;
      }
    };

  auto compute_src_work = [&](int gid, balance_mode mode)
    -> void {
      work.resize(0); // load the new work-items

      work_amount interp_src{2.0f * num_dims};
      if (gid == -1) {
        for (int i : iindexof(sources_md)) {
          if (mode == balance_mode::gpus and not resources.owns(sources_md[i].rec))
            continue;
          if (sources_md[i])
            work.emplace_back(interp_src, -i - 1); // negative id for interp-sources
        }
      } else {
        if ((mode == balance_mode::mpi_ranks or resources.owns(sources_md[gid].rec))
            and sources_md[gid])
          work.emplace_back(interp_src, -gid - 1);
      }

      indexrange sgroup = (gid == -1) ? indexrange(sources)
                                      : source_groups[gid].source_range;

      for (auto is : sgroup) {
        auto const &src = sources[is];
        if (src.is_time_non_sep())
          work.emplace_back(work_amount{static_cast<float>(num_dims)}, is);
        else
          work.emplace_back(work_amount{0.1f}, is);
      }
    };

  auto load_balance_terms = [&](int gid, int num_workers, balance_mode mode)
    -> void {
      expect(num_workers >= 1);
      compute_terms_work(gid, mode);

      // consider cases: num_workers == 1 or num_workers > 1
      if (num_workers == 1) {
        // if using only 1 worker, put it all in one place regardless of the gid
        if (mode == balance_mode::gpus) {
          for (auto &item : work)
            terms[item.id].rec.device = 0;
        } else {
          for (auto &item : work)
            terms[item.id].rec.group = 0; // maybe redundant
        }
        return;
      }

      balance_manager load{num_workers};

      if (mode == balance_mode::gpus) {
        for (auto const &item : work) {
          int const worker = load.lowest();
          terms[item.id].rec.device = worker;
          load.add(worker, item.work);
        }
      } else {
        for (auto const &item : work) {
          int const worker = load.lowest();
          terms[item.id].rec.group = worker;
          load.add(worker, item.work);
        }
      }
    };

  auto load_balance_src = [&](int gid, int num_workers, balance_mode mode)
    -> void {
      expect(num_workers >= 1);
      compute_src_work(gid, mode);

      // consider cases: num_workers == 1 or num_workers > 1
      if (num_workers == 1) {
        // if using only 1 worker, put it all in one place regardless of the gid
        if (mode == balance_mode::gpus) {
          for (auto &item : work)
            if (item.id >= 0)
              sources[item.id].rec.device = 0;
            else
              sources_md[std::abs(item.id + 1)].rec.device = 0;
        } else {
          for (auto &item : work)
            if (item.id >= 0)
              sources[item.id].rec.group = 0;
            else
              sources_md[std::abs(item.id + 1)].rec.group = 0;
        }
        return;
      }

      balance_manager load{num_workers};

      if (mode == balance_mode::gpus) {
        for (auto const &item : work) {
          int const worker = load.lowest_back();
          if (item.id >= 0)
            sources[item.id].rec.device = worker;
          else
            sources_md[std::abs(item.id + 1)].rec.device = worker;
          load.add(worker, item.work);
        }
      } else {
        for (auto const &item : work) {
          int const worker = load.lowest_back();
          if (item.id >= 0)
            sources[item.id].rec.group = worker;
          else
            sources_md[std::abs(item.id + 1)].rec.group = worker;
          load.add(worker, item.work);
        }
      }
    };

  int const num_ranks = std::max(resources.num_ranks(), 1);
  int const num_gpus  = std::max(compute->num_gpus(), 1);

  if (term_groups.empty()) {
    load_balance_terms(-1, num_ranks, balance_mode::mpi_ranks);
    load_balance_terms(-1, num_gpus, balance_mode::gpus);
    load_balance_src(-1, num_ranks, balance_mode::mpi_ranks);
    load_balance_src(-1, num_gpus, balance_mode::gpus);
  } else {
    for (int gid = 0; gid < static_cast<int>(term_groups.size()); gid++) {
      load_balance_terms(gid, num_ranks, balance_mode::mpi_ranks);
      load_balance_terms(gid, num_gpus, balance_mode::gpus);
      load_balance_src(gid, num_ranks, balance_mode::mpi_ranks);
      load_balance_src(gid, num_gpus, balance_mode::gpus);
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

  bool has_sources = false;
  for (auto const &s : sources)
    if (resources.owns(s.rec))
      has_sources = true;
  for (auto const &s : sources_md)
    if (!!s and resources.owns(s.rec))
      has_sources = true;

  if (not terms.empty() and resources.num_ranks() > 1 and not has_terms_ and not has_sources) {
    // if the PDE has some terms, e.g., some testing PDEs don't,
    // and if there are multiple MPI ranks, yet some ranks have no terms
    // that means there are more ranks then terms and we should print a warning
    std::cerr << " -- warning: the number of MPI ranks exceeds the number of terms and sources,"
              << " the likely outcome is performance degradation" << std::endl;
  }

  bool constexpr print_dist = false;
  if constexpr (print_dist) {
    #ifdef ASGARD_USE_MPI
    if (mpi::is_world_rank(0))
    #else
    if (true)
    #endif
    {
      std::cout << "\n";
      for (auto const &t : terms)
        std::cout << " term to rank: " << t.rec.group << "  gpu: " << t.rec.device << "  chain num = " << t.num_chain << '\n';

      std::cout << "\n";
      for (auto const &s : sources)
        std::cout << " source to rank: " << s.rec.group << "  gpu: " << s.rec.device << '\n';

      std::cout << "\n";
      for (auto const &s : sources_md)
        if (s.func)
          std::cout << " source-md to rank: " << s.rec.group << "  gpu: " << s.rec.device << '\n';
        else
          std::cout << " source-md: inactive\n";

      std::cout << "\n";
    }
  }

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
