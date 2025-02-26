#include "asgard_term_manager.hpp"

#include "asgard_coefficients_mats.hpp" // also brings in small-mats module

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
        case pterm_dependence::electric_field:
        case pterm_dependence::electric_field_only:
          return {true, 1};
        case pterm_dependence::moment_divided_by_density:
          return {false, std::abs(single.moment())};
        case pterm_dependence::lenard_bernstein_coll_theta_1x1v:
          return {false, 3};
        case pterm_dependence::lenard_bernstein_coll_theta_1x2v:
          return {false, 5};
        case pterm_dependence::lenard_bernstein_coll_theta_1x3v:
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
term_manager<P>::term_manager(PDEv2<P> &pde, sparse_grid const &grid,
                              hierarchy_manipulator<P> const &hier)
  : num_dims(pde.num_dims()), max_level(pde.max_level()), legendre(pde.degree())
{
  if (num_dims == 0)
    return;

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

  terms.resize(num_terms);

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

      *ir = term_entry<P>(std::move(pde_terms[i].chain_[0]));
      ir++->num_chain = num_chain;
      for (int c = 1; c < num_chain; c++) {
        *ir = term_entry<P>(std::move(pde_terms[i].chain_[c]));
        ir++->num_chain = -1;
      }
    } else {
      *ir++ = term_entry<P>(std::move(pde_terms[i]));
    }
  }

  int num_bc = 0;

  // check if we need to keep the intermediate terms from matrix builds
  for (auto &tt : terms) {
    bool has_sep_dir = false;
    bool has_1d_chain = false;
    for (int d : iindexof(num_dims)) {
      if (tt.tmd.dim(d).num_sep_dirichlet() > 0)
        has_sep_dir = true;
      if (tt.tmd.dim(d).is_chain())
        has_1d_chain = true;
    }
    rassert(not (has_sep_dir and has_1d_chain),
            "1d chain terms cannot be coupled with separable boundary conditions");

    tt.bc.begin = num_bc;
    num_bc += static_cast<int>(tt.tmd.bc_flux_.size());
    tt.bc.end = num_bc;
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

  for (int d : iindexof(num_dims)) {
    xleft[d]  = pde.domain().xleft(d);
    xright[d] = pde.domain().xright(d);
  }

  build_mass_matrices(); // large, up to max-level
  rebuild_mass_matrices(grid); // small, up to the current level

  std::vector<separable_func<P>> &sep = pde.sources_sep_;

  int num_sources = 0;
  for (auto const &s : sep) {
    int const dims = s.num_dims();
    rassert(dims == 0 or dims == num_dims, "incorrect dimension set for source");
    if (dims > 0) ++num_sources;
  }

  num_interior_sources = num_sources;

  for (auto &tmd : terms) {
    for (int d : indexof(num_dims)) {
      if (tmd.tmd.dim(d).has_dirichlet())
        tmd.bc_source_id = num_sources++;
      num_sources += tmd.tmd.dim(d).num_sep_dirichlet();
    }
  }

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

  for (int tid : iindexof(terms)) {
    auto const &tmd = terms[tid];
    for (int d : indexof(num_dims)) {
      if (tmd.tmd.dim(d).has_dirichlet()) {
        sources.emplace_back(source_entry<P>::time_mode::boundary);
        sources.back().func = source_boundary_data<P>{{}, d, tid};
      }
    }
  }

  prapare_workspace(grid); // setup kronmult workspace
}

template<typename P>
mom_deps term_manager<P>::find_deps() const
{
  mom_deps deps;

  for (auto const &tentry : terms)
    for (int d : iindexof(num_dims))
      deps += tentry.deps[d];

  return deps;
}

template<typename P>
void term_manager<P>::update_const_sources(
    sparse_grid const &grid, connection_patterns const &conns,
    hierarchy_manipulator<P> const &hier)
{
  if (grid.generation() == sources_grid_gen)
    return;

  int const pdof = hier.degree() + 1;

  int64_t const block_size  = hier.block_size();
  int64_t const num_entries = grid.num_indexes() * block_size;

  // update the constant components
  for (auto &src : sources) {
    if (src.is_time_dependent() or src.is_edge_time())
      continue;

    std::cout << " ------- next src -------------- \n";
    for (size_t i = 0; i < src.consts[0].size(); i++) {
      std::cout << src.consts[0][i] << "    " << src.consts[1][i] << "\n";
    }

    int const dim = (src.is_boundary()) ? src.dim() : -1;

    // either an interior source or boundary with constant term
    if (dim < 0 or not src.consts[dim].empty()) {
      src.val.resize(num_entries);

      #pragma omp parallel
      {
        std::array<P const *, max_num_dimensions> data1d;

        #pragma omp for
        for (int64_t c = 0; c < grid.num_indexes(); c++) {
          P *proj = src.val.data() + c * block_size;

          int const *idx = grid[c];
          for (int d : iindexof(num_dims))
            data1d[d] = src.consts[d].data() + idx[d] * pdof;

          for (int i : iindexof(block_size))
          {
            int t   = i;
            proj[i] = 1;
            for (int d = num_dims - 1; d >= 0; d--) {
              proj[i] *= data1d[d][t % pdof];
              t /= pdof;
            }
          }
        }
      }
    }
    // handle the time-dependent (separable part of the boundary conditions)
    // and handle the chains, if using term_md chain
    if (not src.is_boundary() and not src.is_edge()) // nothing more to do for interior boundary
      continue;

    if (src.is_boundary())
      for (auto &tdata : src.time_boundary()) {
        std::vector<P> const &c1d = tdata.const_1d;

        std::vector<P> &md = tdata.const_md;

        md.resize(num_entries);

        #pragma omp parallel
        {
          std::array<P const *, max_num_dimensions> data1d;

          #pragma omp for
          for (int64_t c = 0; c < grid.num_indexes(); c++) {
            P *proj = md.data() + c * block_size;

            int const *idx = grid[c];
            for (int d = 0; d < dim; d++)
              data1d[d] = src.consts[d].data() + idx[d] * pdof;

            data1d[dim] = c1d.data() + idx[dim] * pdof;

            for (int d = dim + 1; d < num_dims; d++)
              data1d[d] = src.consts[d].data() + idx[d] * pdof;

            for (int i : iindexof(block_size))
            {
              int t   = i;
              proj[i] = 1;
              for (int d = num_dims - 1; d >= 0; d--) {
                proj[i] *= data1d[d][t % pdof];
                t /= pdof;
              }
            }
          }
        }
      } // done with time entries

    // not in a chain or last link, then nothing more to do
    if (terms[src.term_index()].num_chain > 0)
      continue;

    // otherwise we have to push the vectors through the term_md chain

    t1.resize(num_entries); // workspace

    bool keep_working = true;
    int tid = src.term_index();
    while (keep_working) {
      --tid;

      if (not src.val.empty()) {
        kron_term(grid, conns, terms[tid], 1, src.val, 0, t1);
        std::swap(src.val, t1);
      }
      if (src.is_boundary())
        for (auto &tdata : src.time_boundary()) {
          kron_term(grid, conns, terms[tid], 1, tdata.const_md, 0, t1);
          std::swap(tdata.const_md, t1);
        }

      keep_working = (terms[tid].num_chain < 0);
    }
  } // done with all sources

  if (sources_have_time_dep)
    rebuild_mass_matrices(grid);

  // make sure to handle the boundary conditions too
  update_bc(grid, conns, hier);

  sources_grid_gen = grid.generation();
}

template<typename P>
template<data_mode dmode>
void term_manager<P>::apply_sources(
    pde_domain<P> const &domain, sparse_grid const &grid, connection_patterns const &conns,
    hierarchy_manipulator<P> const &hier, P time, P alpha, P y[])
{
  update_const_sources(grid, conns, hier);

  int64_t const num_entries = grid.num_indexes() * hier.block_size();
  if constexpr (dmode == data_mode::replace or dmode == data_mode::scal_rep)
    std::fill_n(y, num_entries, P{0});

  // NOTE: when adding the "boundary" and "edge" sources, the sign is flipped

  for (auto const &src : sources) {
    switch (src.tmode) {
      case source_entry<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] += src.val[i];
        else
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] += alpha * src.val[i];
        break;
      case source_entry<P>::time_mode::edge_constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= src.val[i];
        else
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= alpha * src.val[i];
        break;
      case source_entry<P>::time_mode::separable:
      case source_entry<P>::time_mode::edge_separable: {
          P t = std::get<scalar_func<P>>(src.func)(time);
          if (src.tmode == source_entry<P>::time_mode::edge_separable)
            t = -t;
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            t *= alpha;
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] += t * src.val[i];
        }
        break;
      case source_entry<P>::time_mode::time_dependent:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          hier.template project_separable<data_mode::increment>
              (std::get<separable_func<P>>(src.func), domain, grid, lmass, time, alpha, y);
        else
          hier.template project_separable<data_mode::scal_inc>
              (std::get<separable_func<P>>(src.func), domain, grid, lmass, time, alpha, y);
        break;
      case source_entry<P>::time_mode::edge_time:
        throw std::runtime_error("not implemented yet");
        break;
      case source_entry<P>::time_mode::boundary: {
          if (not src.val.empty()) { // we have constant bc
            if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
              ASGARD_OMP_PARFOR_SIMD
              for (int64_t i = 0; i < num_entries; i++)
                y[i] -= src.val[i];
            else
              ASGARD_OMP_PARFOR_SIMD
              for (int64_t i = 0; i < num_entries; i++)
                y[i] -= alpha * src.val[i];
          }

          for (auto const &td : src.time_boundary()) {
            P t = td.time(time);
            if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
              t *= alpha;
            ASGARD_OMP_PARFOR_SIMD
            for (int64_t i = 0; i < num_entries; i++)
              y[i] -= t * td.const_md[i];
          }
        }
        break;
      default:
        // unreachable here
        break;
    }
  }

  if constexpr (dmode == data_mode::replace)
    apply_bc<data_mode::increment>(domain, grid, conns, hier, time, alpha, y);
  else if constexpr (dmode == data_mode::scal_rep)
    apply_bc<data_mode::scal_inc>(domain, grid, conns, hier, time, alpha, y);
  else
    apply_bc<dmode>(domain, grid, conns, hier, time, alpha, y);
}

template<typename P>
void term_manager<P>::update_bc(
    sparse_grid const &grid, connection_patterns const &conns,
    hierarchy_manipulator<P> const &hier)
{
  int const pdof = hier.degree() + 1;

  int64_t const block_size  = hier.block_size();
  int64_t const num_entries = grid.num_indexes() * block_size;

  // update the constant components
  for (auto &bc : bcs) {
    if (bc.is_time_dependent())
      continue;

    std::cout << " ------- next bc -------------- \n";
    for (size_t i = 0; i < bc.consts[0].size(); i++) {
      std::cout << bc.consts[0][i] << "    " << bc.consts[1][i] << "\n";
    }

    bc.val.resize(num_entries);

    #pragma omp parallel
    {
      std::array<P const *, max_num_dimensions> data1d;

      #pragma omp for
      for (int64_t c = 0; c < grid.num_indexes(); c++) {
        P *proj = bc.val.data() + c * block_size;

        int const *idx = grid[c];
        for (int d : iindexof(num_dims))
          data1d[d] = bc.consts[d].data() + idx[d] * pdof;

        for (int i : iindexof(block_size))
        {
          int t   = i;
          proj[i] = 1;
          for (int d = num_dims - 1; d >= 0; d--) {
            proj[i] *= data1d[d][t % pdof];
            t /= pdof;
          }
        }
      }
    }

    // not in a chain or last link, then nothing more to do
    if (terms[bc.term_index].num_chain > 0)
      continue;

    // otherwise we have to push the vectors through the term_md chain

    t1.resize(num_entries); // workspace

    bool keep_working = true;
    int tid = bc.term_index;
    while (keep_working) {
      --tid;

      kron_term(grid, conns, terms[tid], 1, bc.val, 0, t1);
      std::swap(bc.val, t1);

      keep_working = (terms[tid].num_chain < 0);
    }
  } // done with all sources
}

template<typename P>
template<data_mode dmode>
void term_manager<P>::apply_bc(
    pde_domain<P> const &, sparse_grid const &grid,
    connection_patterns const &, hierarchy_manipulator<P> const &hier,
    P time, P alpha, P y[])
{
  // update_bc(grid, conns, hier);

  int64_t const num_entries = grid.num_indexes() * hier.block_size();

  for (auto const &bc : bcs) {
    switch (bc.tmode) {
      case boundary_entry<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= bc.val[i];
        else
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= alpha * bc.val[i];
        break;
      case boundary_entry<P>::time_mode::separable: {
          std::cout << " time-sep bc\n";
          P t = bc.flux.func().ftime(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            t *= alpha;
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= t * bc.val[i];
        }
        break;
      case boundary_entry<P>::time_mode::time_dependent:
        std::cout << " time-dep bc\n";
        // TIME DEPENDANT mess
        // if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
        //   hier.template project_separable<data_mode::increment>
        //       (std::get<separable_func<P>>(src.func), domain, grid, lmass, time, alpha, y);
        // else
        //   hier.template project_separable<data_mode::scal_inc>
        //       (std::get<separable_func<P>>(src.func), domain, grid, lmass, time, alpha, y);
        break;
      default:
        // unreachable here
        break;
    }
  }
}

template<typename P>
void term_manager<P>::buld_term(
    int const tid, sparse_grid const &grid, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, precon_method precon, P alpha)
{
  expect(legendre.pdof == hier.degree() + 1);
  expect(not terms[tid].tmd.is_chain());
  expect(not terms[tid].tmd.is_interpolatory());

  auto &tmd = terms[tid];

  // if no boundary condition, use generic no_bc source entry
  // otherwise, start using the entry from the sources
  source_entry<P> no_bc;
  source_entry<P> &bc = (tmd.bc_source_id >= 0) ? sources[tmd.bc_source_id] : no_bc;

  bool has_sep_dir  = false;
  int dirichlet_dir = -1; // separable Dirichlet direction

  for (int d : iindexof(num_dims)) {
    auto const &t1d = tmd.tmd.dim(d);

    if (t1d.num_sep_dirichlet() > 0) {
      expect(dirichlet_dir == -1); // only one direction can have Dirichlet boundary
      has_sep_dir   = true;
      dirichlet_dir = d;
    }

    int level = grid.current_level(d); // required level

    // terms that don't change should be build only once
    if (t1d.change() == changes_with::none) {
      if (terms[tid].coeffs[d].empty())
        level = max_level; // build up to the max
      else
        continue; // already build, we can skip
    }

    rebuld_term1d(terms[tid], d, level, conn, hier, bc, precon, alpha);
  } // move to next dimension d

  // add the separable boundary sources
  if (not has_sep_dir)
    return;

  expect(dirichlet_dir != -1);

  // collect all the extras that we need
  // the int encodes left (negative) or right (positive)
  // as well as the link number, if part of a 1d chain
  std::vector<std::pair< int, separable_func<P>* >> seps;

  for (int d : iindexof(num_dims)) {
    auto &t1d = tmd.tmd.sep[d];
    if (t1d.is_chain()) {
      for (int c : iindexof(t1d.num_chain())) {
        dirichelt_boundary1d<P> &dir = t1d.chain_[c].dirichlet_;
        for (auto &s : dir.sep_left)
          seps.emplace_back(-c-2, &s);
        for (auto &s : dir.sep_right)
          seps.emplace_back(c+2, &s);
      }
      dirichelt_boundary1d<P> &dir = t1d.dirichlet_;
      for (auto &s : dir.sep_left)
        seps.emplace_back(-1, &s);
      for (auto &s : dir.sep_right)
        seps.emplace_back(+1, &s);
    } else {
      dirichelt_boundary1d<P> &dir = t1d.dirichlet_;
      for (auto &s : dir.sep_left)
        seps.emplace_back(-1, &s);
      for (auto &s : dir.sep_right)
        seps.emplace_back(+1, &s);
    }
  }

  int const pdof = legendre.pdof;

  int64_t const num_cells   = fm::ipow2(max_level);
  int64_t const num_entries = pdof * num_cells;

  P const scale0 = P{1} / std::sqrt( (xright[dirichlet_dir] - xleft[dirichlet_dir]) / num_cells );

  for (auto &sl : seps) {
    if (sl.second->num_dims() == 0) // empty function (zero)
      continue;

    separable_func<P> &s = *sl.second;

    expect(s.is_const(dirichlet_dir));

    // set the boundary value, multiplies by the values of the legendre basis
    P mag = s.cdomain(dirichlet_dir) * scale0;

    {
      auto const &tdir = tmd.tmd.sep[dirichlet_dir];

      if (tdir.is_penalty())
        mag = -mag;

      // if tdir has builtin penalty, then add the correction
      mag *= P{1} + ((sl.first < 0) ? tdir.penalty() : -tdir.penalty());
    }

    if (s.ignores_time() or s.ftime()) {
      if (s.ignores_time()) {
        sources.emplace_back(source_entry<P>::time_mode::edge_constant);
        sources.back().func = tid; // save the id, in case this is part of a md-chain
      } else {
        sources.emplace_back(source_entry<P>::time_mode::edge_separable);
        sources.back().func = source_edge_stime<P>{s.ftime(), tid};
      }

      auto get_mass = [&](int d)-> block_diag_matrix<P> const & {
          if (tmd.num_chain < 0) { // part of an md-chain
            return tmd.mass[d];
          } else {
            return mass[d];
          }
        };

      for (int d : iindexof(num_dims)) {
        if (d == dirichlet_dir) // skip the "special direction"
          continue;
        if (s.is_const(d)) {
          sources.back().consts[d]
              = hier.get_project1d_c(s.cdomain(d), get_mass(d), d, max_level);
        } else {
          sources.back().consts[d] = hier.get_project1d_f(
              [&](std::vector<P> const &x, std::vector<P> &y)-> void { s.fdomain(d, x, 0, y); },
              get_mass(d), d, max_level);
        }
      }

      // set the actual bc in direction dirichlet_dir
      {
        std::vector<P> &vec = sources.back().consts[dirichlet_dir];
        vec.resize(num_entries);

        if (sl.first < 0) // left boundary
          smmat::axpy(pdof, -mag, legendre.leg_left, vec.data());
        else
          smmat::axpy(pdof, mag, legendre.leg_right, vec.data() + num_entries - pdof);

        auto const &tmass = get_mass(dirichlet_dir);
        if (tmass)
          tmass.solve(pdof, vec);
        hier.project1d(max_level, vec);
      }
    } else { // not separable in time
      sources_have_time_dep = true;
      sources.emplace_back(source_entry<P>::time_mode::edge_time);
      // set left/right data
      if (sl.first < 0)
        sources.back().func = source_edge_data<P>(std::move(*sl.second), dirichlet_dir);
      else
        sources.back().func = source_edge_data<P>(dirichlet_dir, std::move(*sl.second));
      source_edge_data<P> &data = std::get<source_edge_data<P>>(sources.back().func);
      data.mag = (sl.first < 0) ? -mag : mag;
      data.term_index = tid;
    }
  }
}

template<typename P>
void term_manager<P>::rebuld_term1d(
    term_entry<P> &tentry, int const dim, int level,
    connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
    source_entry<P> &bc, precon_method precon, P alpha)
{
  int const n = hier.degree() + 1;
  auto &t1d   = tentry.tmd.dim(dim);

  // if (t1d.is_identity()) {
  //   // identity has only a simple case of boundary conditions
  //   if (bc.is_boundary()) {
  //     expect(bc.dim() != dim); // boundary must be in another direction
  //     // the assumption here is that mass[dim] is empty
  //     bc.consts[dim] = hier.get_project1d_c(1, mass[dim], dim, level);
  //   }
  //   if (tentry.bc.size() > 0) {
  //     std::vector<P> ones = hier.get_project1d_c(1, mass[dim], dim, level);
  //     for (int c = tentry.bc.begin; c < tentry.bc.end; c++)
  //       bcs[c].consts[dim] = ones;
  //       //bcs[c].consts[dim] = legendre.project(level, P{1});
  //   }
  //   return; // nothing to do about the matrix
  // }

  bool is_diag = t1d.is_mass();
  if (t1d.is_chain()) {
    rebuld_chain(tentry, dim, level, is_diag, wraw_diag, wraw_tri, bc);
  } else {
    if (bc.is_boundary() and bc.dim() != dim)
      bc.consts[dim].resize(n * fm::ipow2(level));
    build_raw_mat(tentry, dim, 0, level, wraw_diag, wraw_tri, bc);
  }

  block_diag_matrix<P> *bmass = nullptr; // mass to use for the boundary source

  // apply the mass matrix, if any
  if (tentry.num_chain < 0) {
    // member of a chain, can have unique mass matrix
    mass_md<P> const &tms = tentry.tmd.mass();
    if (tms and not tms[dim].is_identity()) {
      int const nrows = fm::ipow2(level); // needed number of rows
      if (tentry.mass[dim].nrows() != nrows) {
        build_raw_mass(dim, mass_term[dim], max_level, tentry.mass[dim]);
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

  // the build/rebuild put the result in raw_diag or raw_tri
  if (not t1d.is_identity()) {
    if (is_diag) {
      if (bmass)
        bmass->solve(n, wraw_diag);
      tentry.coeffs[dim] = hier.diag2hierarchical(wraw_diag, level, conn);
    } else {
      if (bmass)
        bmass->solve(n, wraw_tri);
      tentry.coeffs[dim] = hier.tri2hierarchical(wraw_tri, level, conn);
    }
  }

  if (bc.is_boundary()) {
    if (bc.dim() == dim) { // boundary direction
      // may have mass to apply
      if (bmass) {
        if (not bc.consts[dim].empty())
          bmass->solve(n, bc.consts[dim]);
        for (auto &tc : bc.time_boundary())
          bmass->solve(n, tc.const_1d);
      }
      // convert to hierarchical form
      if (not bc.consts[dim].empty())
        hier.project1d(level, bc.consts[dim]);
      for (auto &tc : bc.time_boundary())
        hier.project1d(level, tc.const_1d);
    } else { // non-boundary direction
      if (bmass)
        bmass->solve(n, bc.consts[dim]);
      hier.project1d(level, bc.consts[dim]);
    }
  }

  // apply the mass matrices and convert to hierarchical form
  for (int b = tentry.bc.begin; b < tentry.bc.end; b++) {
    boundary_entry<P> &bentry = bcs[b];
    if (not bentry.consts[dim].empty()) {
      // will be empty if non-flux direction and non-separable in time
      if (bmass)
        bmass->solve(n, bentry.consts[dim]);
      hier.project1d(level, bentry.consts[dim]);
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
    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri, source_entry<P> &bc)
{
  term_1d<P> &t1d = (tentry.tmd.dim(d).is_chain()) ? tentry.tmd.dim(d).chain_[clink] : tentry.tmd.dim(d);
  expect(not t1d.is_chain());

  switch (t1d.optype())
  {
    case operation_type::mass:
      switch (t1d.depends()) {
        case pterm_dependence::electric_field_only:
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
        case pterm_dependence::electric_field:
          throw std::runtime_error("el-field with position depend is not done (yet)");
          break;
        default:
          if (t1d.rhs()) {
            gen_diag_cmat<P, operation_type::mass>
              (legendre, xleft[d], xright[d], level, t1d.rhs(), nullptr, raw_diag);
          } else {
            gen_diag_cmat<P, operation_type::mass>
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
      std::cout << " added penalty\n";
      gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const>
        (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_rhs, raw_tri);
      break;
    default:
      // identity, nothing to do
      std::cout << " skipped identity\n";
      break;
  }

  if (bc.is_boundary() and bc.dim() == d) {
    int const pdof = legendre.pdof;

    // constant and time-dependent conditions
    std::vector<P> &cnt = bc.consts[d];

    // multiply the current set of bc by the matrix
    // this will happen when working with a 1d chain
    if (t1d.is_mass()) { // using diag-matrix
      if (not cnt.empty())
        raw_diag.inplace_gemv(pdof, cnt, t1);
      for (auto &st : bc.time_boundary())
        raw_diag.inplace_gemv(pdof, st.const_1d, t1);
    } else { // using tri-diag matrix
      if (not cnt.empty())
        raw_tri.inplace_gemv(pdof, cnt, t1);
      for (auto &st : bc.time_boundary())
        raw_tri.inplace_gemv(pdof, st.const_1d, t1);
    }

    add_dirichlet(t1d, level, t1d.dirichlet_, bc);

  } else if (bc.is_boundary() and not bc.consts[d].empty()) {
    // has Dirichlet in other directions and expecting to load the rhs
    // if using 1d-chain, this is handled externally
    if (t1d.rhs()) {
      bc.consts[d] = legendre.project(t1d.is_mass(), level, std::sqrt(xright[d] - xleft[d]), 1, raw_rhs.vals);
    } else { // using a constant
      if (legendre.pdof == 1) {
        std::fill(bc.consts[d].begin(), bc.consts[d].end(), t1d.rhs_const());
      } else {
        int const num_cells = fm::ipow2(level);
        #pragma omp parallel for
        for (int i = 0; i < num_cells; i++) {
          bc.consts[d][i * legendre.pdof] = t1d.rhs_const();
          std::fill_n(bc.consts[d].data() + i * legendre.pdof + 1, legendre.pdof - 1, P{0});
        }
      }
    }
  }

  // boundary conditions
  if (tentry.bc.size() == 0)
    return;

  for (int b = tentry.bc.begin; b < tentry.bc.end; b++) {
    // handle the non-separable in time, keep rhs values
    boundary_entry<P> &bentry = bcs[b];

    // std::cout << " b = " << b << " d = " << d << "  clink = " << clink << " chain-level = " << bentry.flux.chain_level(d) << "\n";

    if (bentry.flux.chain_level(d) > clink) {
      expect(not bentry.consts[d].empty());
      if (t1d.is_mass()) {
        raw_diag.inplace_gemv(legendre.pdof, bentry.consts[d], t1);
        // std::cout << " b = " << b << " chain mult diag at link = " << clink << "\n";
      } else {
        raw_tri.inplace_gemv(legendre.pdof, bentry.consts[d], t1);
        // std::cout << " b = " << b << " chain mult 3 at link = " << clink << "\n";
      }
    } else if (bentry.flux.chain_level(d) == clink) {
      // create a new entry
      // std::cout << " b = " << b << " make new dir = " << d << "\n";
      if (tentry.flux_dim == d) {
        // std::cout << "   d = " << d << " is flux dir\n";
        int const pdof = legendre.pdof;

        int64_t const num_cells = fm::ipow2(level);
        int64_t const num_entries = pdof * num_cells;

        bentry.consts[d].resize(num_entries);

        P scale = P{1} / std::sqrt( (xright[d] - xleft[d]) / num_cells );
        if (t1d.is_penalty()) // penalty flips the sign of the boundary conditions
          scale = -scale;

        if (bentry.flux.is_left()) {
          P rhs_left  = (t1d.rhs()) ? raw_rhs.vals.front() : t1d.rhs_const();
          std::cout << " b = " << b << " rhs-left = " << rhs_left << "\n";

          if (t1d.penalty() != 0)
            rhs_left *= P{1} + t1d.penalty();

          P const fc = bentry.flux.func().cdomain(d);
          std::cout << "            fc = " << fc << "  scale = " << scale << "\n";
          if (fc == 0) { // non-separable in time
            // single-point value is always separable, so we can pre-compute in d-direction
            smmat::axpy(pdof, - rhs_left * scale, legendre.leg_left, bentry.consts[d].data());
          } else {
            smmat::axpy(pdof, - rhs_left * scale * fc, legendre.leg_left, bentry.consts[d].data());
          }
          //std::cout << " setting left\n";
        }

        if (bentry.flux.is_right()) {
          P rhs_right = (t1d.rhs()) ? raw_rhs.vals.back()  : t1d.rhs_const();
          std::cout << " b = " << b << " rhs-right = " << rhs_right << "\n";

          if (t1d.penalty() != 0)
            rhs_right *= P{1} - t1d.penalty();

          P const fc = bentry.flux.func().cdomain(d);
          std::cout << "            fc = " << fc << "  scale = " << scale << "\n";
          if (fc == 0) { // non-separable in time
            // single-point value is always separable, so we can pre-compute in d-direction
            smmat::axpy(pdof, rhs_right * scale, legendre.leg_right,
                        bentry.consts[d].data() + num_entries - pdof);
          } else {
            smmat::axpy(pdof, rhs_right * scale * fc, legendre.leg_right,
                        bentry.consts[d].data() + num_entries - pdof);
          }
          // std::cout << " setting right\n";
        }
      } else {
        // std::cout << "   d = " << d << " is non-flux dir\n";
        // this is a rhs combined with a derivative in a different direction
        // build the legendre (not-hierarchical) entries, the mass matrix should be applied in rebuld_term1d
        // and then the hierarchy rebuild

        if (bentry.is_time_dependent()) // no constant components to pre-compute
          continue;

        P const dsqr = std::sqrt(xright[d] - xleft[d]);

        if (bentry.flux.func().is_const(d)) {
          if (t1d.rhs()) { // constant times spatially variable
            bentry.consts[d] = legendre.project(t1d.is_mass(), level, dsqr,
                                                bentry.flux.func().cdomain(d), raw_rhs.vals);
          } else { // constant times a constant
            bentry.consts[d] = legendre.project(level, dsqr, bentry.flux.func().cdomain(d) * t1d.rhs_const());
          }
        } else {
          if (t1d.rhs()) { // product of non-consts
            std::vector<P> f(raw_rhs.pnts.size());
            bentry.flux.func().fdomain(d, raw_rhs.pnts, 0, f);
            bentry.consts[d] = legendre.project(t1d.is_mass(), level, dsqr, f, raw_rhs.vals);
          } else {
            // need function values, rhs is a constant
            legendre.interior_quad(xleft[d], xright[d], level, raw_rhs.pnts);
            raw_rhs.vals.resize(raw_rhs.pnts.size());
            bentry.flux.func().fdomain(d, raw_rhs.pnts, 0, raw_rhs.vals);
            bool constexpr use_interior = true;
            std::cout << " FUNC EVAL\n";
            // for (size_t i = 0; i < raw_rhs.pnts.size(); i++)
            //   std::cout << raw_rhs.pnts[i] << "    " << raw_rhs.vals[i] << "\n";
            bentry.consts[d] = legendre.project(use_interior, level, dsqr, t1d.rhs_const(), raw_rhs.vals);
            for (auto q : bentry.consts[d])
              std::cout << q << "\n";
          }
        }
      }
    } // if the bentry is associated with a higher link, then do nothing here
  }
}

template<typename P>
void term_manager<P>::add_dirichlet(
    term_1d<P> const &t1d, int level, dirichelt_boundary1d<P> &dirichlet,
    source_entry<P> &bc) const
{
  expect(bc.is_boundary());
  if (not dirichlet.has_any()) // nothing to do
    return;

  int const d = bc.dim();
  std::vector<P> &cnt = bc.consts[d];
  std::vector<time_boundary_data<P>> &tdata = bc.time_boundary();

  int const pdof = legendre.pdof;

  int64_t const num_cells   = fm::ipow2(level);
  int64_t const num_entries = legendre.pdof * num_cells;

  P scale = P{1} / std::sqrt( (xright[d] - xleft[d]) / num_cells );
  if (t1d.is_penalty()) // penalty flips the sign of the boundary conditions
    scale = -scale;

  if (t1d.penalty() != 0 and t1d.is_chain()) // penalty added directly to the term
    scale *= -t1d.penalty();

  P rhs_left  = (t1d.rhs()) ? raw_rhs.vals.front() : t1d.rhs_const();
  P rhs_right = (t1d.rhs()) ? raw_rhs.vals.back()  : t1d.rhs_const();

  if (dirichlet.has_left()) {
    if (t1d.penalty() != 0 and not t1d.is_chain())
      rhs_left *= P{1} + t1d.penalty();

    if (dirichlet.left_t) { // time-dependant
      tdata.emplace_back(std::move(dirichlet.left_t));
      tdata.back().const_1d.resize(num_entries);
      smmat::axpy(pdof, - rhs_left * scale, legendre.leg_left, tdata.back().const_1d.data());
    } else {
      if (cnt.empty())
        cnt.resize(num_entries);
      smmat::axpy(pdof, - rhs_left * scale * dirichlet.const_left, legendre.leg_left, cnt.data());
    }
  }
  if (dirichlet.has_right()) {
    if (t1d.penalty() != 0 and not t1d.is_chain())
      rhs_right *= P{1} - t1d.penalty();

    if (dirichlet.right_t) { // time-dependant
      tdata.emplace_back(std::move(dirichlet.right_t));
      tdata.back().const_1d.resize(num_entries);
      smmat::axpy(pdof, rhs_right * scale, legendre.leg_right,
                  tdata.back().const_1d.data() + num_entries - pdof);
    } else {
      if (cnt.empty())
        cnt.resize(num_entries);
      smmat::axpy(pdof, rhs_right * scale * dirichlet.const_right,
                  legendre.leg_right, cnt.data() + num_entries - pdof);
    }
  }
}

template<typename P>
void term_manager<P>::build_raw_mass(int dim, term_1d<P> const &t1d, int level,
                                     block_diag_matrix<P> &raw_diag)
{
  expect(t1d.is_mass());
  expect(t1d.depends() == pterm_dependence::none);

  if (t1d.rhs()) {
    gen_diag_cmat<P, operation_type::mass>
      (legendre, xleft[dim], xright[dim], level, t1d.rhs(), nullptr, raw_diag);
  } else {
    gen_diag_cmat<P, operation_type::mass>
      (legendre, level, t1d.rhs_const(), raw_diag);
  }
}

template<typename P>
void term_manager<P>::rebuld_chain(
    term_entry<P> &tentry, int const d, int const level, bool &is_diag,
    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri,
    source_entry<P> &bc)
{
  term_1d<P> &t1d = tentry.tmd.dim(d);
  expect(t1d.is_chain());
  int const num_chain = t1d.num_chain();
  expect(num_chain > 1);

  is_diag = true;
  for (int i : iindexof(num_chain)) {
    if (not t1d[i].is_mass()) {
      is_diag = false;
      break;
    }
  }

  if (bc.is_boundary() and bc.dim() != d)
    expect(is_diag); // boundary implies flux, non-boundary direction can have only mass

  if (is_diag) { // a bunch of diag matrices, easy case
    bool const use_bc = (bc.is_boundary() and bc.dim() != d);
    std::vector<P> crhs;
    if (use_bc)
      bc.consts[d].resize(0);

    // raw_tri will not be referenced, it's just passed in
    // using raw_diag to make the intermediate matrices, until the last one
    // the last product has to be written to raw_diag
    block_diag_matrix<P> *diag0 = &raw_diag0;
    block_diag_matrix<P> *diag1 = &raw_diag1;
    build_raw_mat(tentry, d, num_chain - 1, level, *diag0, raw_tri, bc);
    crhs = raw_rhs.vals;
    for (int i = num_chain - 2; i > 0; i--) {
      build_raw_mat(tentry, d, i, level, raw_diag, raw_tri, bc);
      diag1->check_resize(raw_diag);
      gemm_block_diag(legendre.pdof, raw_diag, *diag0, *diag1);
      std::swap(diag0, diag1);
      if (use_bc) {
        ASGARD_OMP_PARFOR_SIMD
        for (size_t j = 0; j < raw_rhs.vals.size(); j++)
          crhs[j] *= raw_rhs.vals[j];
      }
    }
    build_raw_mat(tentry, d, 0, level, *diag1, raw_tri, bc);
    raw_diag.check_resize(*diag1);
    gemm_block_diag(legendre.pdof, *diag1, *diag0, raw_diag);
    if (use_bc) {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t j = 0; j < raw_rhs.vals.size(); j++)
        crhs[j] *= raw_rhs.vals[j];
      // bc.consts[d] = legendre.project(is_diag, level, 1, crhs);
    }
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

  fill current = (t1d.is_mass()) ? fill::diag : fill::tri;
  build_raw_mat(tentry, d, num_chain - 1, level, *diag0, *tri0, bc);

  for (int i = num_chain - 2; i > 0; i--)
  {
    build_raw_mat(tentry, d, i, level, raw_diag, raw_tri, bc);
    // the result is in either raw_diag or raw_tri and must be multiplied and put
    // into either diag1 or tri1, then those should swap with diag0 and tri0
    if (t1d.is_mass()) { // computed a diagonal fill
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
  build_raw_mat(tentry, d, 0, level, *diag1, *tri1, bc);

  if (t1d[0].is_mass()) {
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

  if (t1d.penalty() != 0) {
    gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
      (legendre, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.flux(),
       t1d.boundary(), raw_rhs, raw_tri);

    if (bc.is_boundary())
      add_dirichlet(t1d, level, t1d.dirichlet_, bc);
  }

  // check if we have a chain and penalty added in the end
  if (t1d.penalty() == 0 or tentry.bc.size() == 0)
    return;

  for (int b = tentry.bc.begin; b < tentry.bc.end; b++) {
    // handle the non-separable in time, keep rhs values
    boundary_entry<P> &bentry = bcs[b];

    // apply only the conditions for the bottom link
    if (bentry.flux.chain_level(d) != num_chain - 1)
      continue;

    int const pdof = legendre.pdof;

    int64_t const num_cells = fm::ipow2(level);
    int64_t const num_entries = pdof * num_cells;

    expect(bentry.consts[d].size() == static_cast<size_t>(num_entries));

    P const scale = -t1d.penalty() / std::sqrt( (xright[d] - xleft[d]) / num_cells );

    if (bentry.flux.is_left()) {
      P const fc = bentry.flux.func().cdomain(d);
      if (fc == 0) { // non-separable in time
        smmat::axpy(pdof, -scale, legendre.leg_left, bentry.consts[d].data());
      } else {
        smmat::axpy(pdof, -scale * fc, legendre.leg_left, bentry.consts[d].data());
      }
    }

    if (bentry.flux.is_right()) {
      P const fc = bentry.flux.func().cdomain(d);
      if (fc == 0) { // non-separable in time
        smmat::axpy(pdof, scale, legendre.leg_right,
                    bentry.consts[d].data() + num_entries - pdof);
      } else {
        smmat::axpy(pdof, scale * fc, legendre.leg_right,
                    bentry.consts[d].data() + num_entries - pdof);
      }
    }
  }
}


template<typename P>
void term_manager<P>::apply_all(
    sparse_grid const &grid, connection_patterns const &conns,
    P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const
{
  expect(x.size() == y.size());
  expect(x.size() == kwork.w1.size());

  P b = beta; // on first iteration, overwrite y

  auto it = terms.begin();
  while (it < terms.end())
  {
    if (it->num_chain == 1) {
      kron_term(grid, conns, *it, alpha, x, b, y);
      ++it;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      kron_term(grid, conns, *(it + num_chain - 1), 1, x, 0, t1);
      for (int i = num_chain - 2; i > 0; --i) {
        kron_term(grid, conns, *(it + i), 1, t1, 0, t2);
        std::swap(t1, t2);
      }
      kron_term(grid, conns, *it, alpha, t1, b, y);

      it += it->num_chain;
    }

    b = 1; // next iteration appends on y
  }
}
template<typename P>
void term_manager<P>::apply_all(
    sparse_grid const &grid, connection_patterns const &conns,
    P alpha, P const x[], P beta, P y[]) const
{
  P b = beta; // on first iteration, overwrite y

  auto it = terms.begin();
  while (it < terms.end())
  {
    if (it->num_chain == 1) {
      kron_term(grid, conns, *it, alpha, x, b, y);
      ++it;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      kron_term(grid, conns, *(it + num_chain - 1), 1, x, 0, t1.data());
      for (int i = num_chain - 2; i > 0; --i) {
        kron_term(grid, conns, *(it + i), 1, t1, 0, t2);
        std::swap(t1, t2);
      }
      kron_term(grid, conns, *it, alpha, t1.data(), b, y);

      it += it->num_chain;
    }

    b = 1; // next iteration appends on y
  }
}

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
    sparse_grid const &grid, connection_patterns const &conns,
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

  auto it = terms.begin();
  while (it < terms.end())
  {
    if (it->num_chain == 1) {
      kron_diag<data_mode::increment>(grid, conns, *it, block_size, y);
      ++it;
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

      it += it->num_chain;
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

#ifdef ASGARD_ENABLE_DOUBLE
template struct term_entry<double>;
template struct term_manager<double>;

template void term_manager<double>::kron_diag<data_mode::increment>(
    sparse_grid const &, connection_patterns const &,
    term_entry<double> const &, int const, std::vector<double> &) const;
template void term_manager<double>::kron_diag<data_mode::multiply>(
    sparse_grid const &, connection_patterns const &,
    term_entry<double> const &, int const, std::vector<double> &) const;

template void term_manager<double>::apply_sources<data_mode::replace>(
  pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::increment>(
  pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_inc>(
  pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_rep>(
  pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<double> const &, double, double, double[]);

// template void term_manager<double>::apply_bc<data_mode::replace>(
//   pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
//   hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_bc<data_mode::increment>(
  pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_bc<data_mode::scal_inc>(
  pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<double> const &, double, double, double[]);
// template void term_manager<double>::apply_bc<data_mode::scal_rep>(
//   pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
//   hierarchy_manipulator<double> const &, double, double, double[]);
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

template void term_manager<float>::apply_sources<data_mode::replace>(
  pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::increment>(
  pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_inc>(
  pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_rep>(
  pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<float> const &, float, float, float[]);

// template void term_manager<float>::apply_bc<data_mode::replace>(
//   pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
//   hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_bc<data_mode::increment>(
  pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_bc<data_mode::scal_inc>(
  pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
  hierarchy_manipulator<float> const &, float, float, float[]);
// template void term_manager<float>::apply_bc<data_mode::scal_rep>(
//   pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
//   hierarchy_manipulator<float> const &, float, float, float[]);
#endif

}
