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
      if (tmd.tmd.dim(d).dirichlet().has_any())
        tmd.bc_source_id = num_sources++;
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
        hier.project1d_f(
            [&](std::vector<P> const &x, std::vector<P> &y)-> void { s.fdomain(d, x, 0, y); },
            mass[d], d, max_level);
        sources.back().consts[d] = hier.get_projected1d(d);
      }

    } else {
      // non-separable in time
      sources_have_time_dep = true;
      sources.emplace_back(source_entry<P>::time_mode::time_dependent);
      sources.back().func = std::move(s);
    }
  }

  // extract the boundary sources and pre-compute as appropriate
  // always compute up to the max level and use only the needed parts
  int const num_cells = fm::ipow2(max_level);
  int const pdof      = legendre.pdof;

  for (int tid : iindexof(terms)) {
    term_entry<P> &tmd = terms[tid];
    for (int dim : iindexof(num_dims)) {
      dirichelt_boundary1d<P> &dirichlet = tmd.tmd.dim(dim).dirichlet_;
      if (not dirichlet.has_any())
        continue;

      sources.emplace_back(source_entry<P>::time_mode::boundary);

      source_entry<P> &src = sources.back();

      auto get_mmass = [&](int d) -> block_diag_matrix<P> const &{
          if (tmd.num_chain >= 1) { // using top-level mass
            return mass[d];
          } else { // using local mass, build in the term
            mass_md<P> const &tms = tmd.tmd.mass();
            if (tms and not tms[d].is_identity()) {
              if (tmd.mass[d].nrows() != num_cells) {
                build_raw_mat(d, tms[d], max_level, tmd.mass[d]);
                tmd.mass[d].spd_factorize(pdof);
              }
            }
            return tmd.mass[d];
          }
        };

      // build the constant terms along the boundary in other directions
      for (int d : iindexof(num_dims)) {
        if (d == dim)
          continue;

        block_diag_matrix<P> const &mmass = get_mmass(d);

        if (tmd.tmd.dim(d).rhs()) {
          hier.project1d_f(tmd.tmd.dim(d).rhs(), mmass, d, max_level);
          src.consts[d] = hier.get_projected1d(d);
        } else if (mmass) {
          hier.project1d_f(
              [&](std::vector<P> const &, std::vector<P> &y)-> void{
                std::fill(y.begin(), y.end(), tmd.tmd.dim(d).rhs_const());
              }, mmass, d, max_level);
          src.consts[d] = hier.get_projected1d(d);
        } else { // no mass and constant function
          src.consts[d].resize(pdof * num_cells);
          src.consts[d].front() = tmd.tmd.dim(d).rhs_const();
        }
      }

      // will be moved into the source-term func
      std::vector<P> const_r;

      block_diag_matrix<P> const &mmass = get_mmass(dim);

      // build the vectors with the actual boundary conditions
      P const scale = P{1} / std::sqrt( pde.domain().length(dim) / num_cells );

      // get the values of the term rhs
      term_1d<P> const &t1d = tmd.tmd.dim(dim);
      P rhs_left  = t1d.rhs_const();
      P rhs_right = t1d.rhs_const();
      if (t1d.rhs()) { // using variable rhs, get the correct values
        if (dirichlet.has_left() and dirichlet.has_right()) { // need both
          std::vector<P> const x = {pde.domain().xleft(dim), pde.domain().xright(dim)};
          std::vector<P> rhs(2);
          t1d.rhs(x, rhs);
          rhs_left  = rhs[0];
          rhs_right = rhs[1];
        } else if (dirichlet.has_left()) { // need left only
          std::vector<P> const x = {pde.domain().xleft(dim), };
          std::vector<P> rhs(1);
          t1d.rhs(x, rhs);
          rhs_left  = rhs.front();
        } else { // need right only
          std::vector<P> const x = {pde.domain().xright(dim), };
          std::vector<P> rhs(1);
          t1d.rhs(x, rhs);
          rhs_right = rhs.front();
        }
      }

      if (not dirichlet.left_t and not dirichlet.right_t) {
        // time-independent boundary, merge the two vectors into one
        src.consts[dim].resize(pdof * num_cells);
        if (dirichlet.has_left())
          smmat::axpy(pdof, -dirichlet.const_left * rhs_left * scale,
                      legendre.leg_left, src.consts[dim].data());
        if (dirichlet.has_right())
          smmat::axpy(pdof, dirichlet.const_right * rhs_right * scale,
                      legendre.leg_right, src.consts[dim].data() + src.consts[dim].size() - pdof);

      } else { // need to have separate left and/or right vectors
        if (dirichlet.has_left()) {
          src.consts[dim].resize(pdof * num_cells);
          if (dirichlet.left_t)
            smmat::axpy(pdof, - rhs_left * scale,
                        legendre.leg_left, src.consts[dim].data());
          else
            smmat::axpy(pdof, -dirichlet.const_left * rhs_left * scale,
                        legendre.leg_left, src.consts[dim].data());
        }
        if (dirichlet.has_right()) {
          const_r.resize(pdof * num_cells);
          if (dirichlet.right_t)
            smmat::axpy(pdof, rhs_right * scale,
                        legendre.leg_right, const_r.data() + const_r.size() - pdof);
          else
            smmat::axpy(pdof, dirichlet.const_right * rhs_right * scale,
                        legendre.leg_right, const_r.data() + const_r.size() - pdof);
        }
      }

      if (not src.consts[dim].empty()) {
        if (mmass)
          mmass.solve(pdof, src.consts[dim]);
        hier.project1d(max_level, src.consts[dim]);
      }
      if (not const_r.empty()) {
        if (mmass)
          mmass.solve(pdof, const_r);
        hier.project1d(max_level, const_r);
      }

      // move the time-functions into the
      src.func = source_boundary_data<P>{
          std::move(dirichlet.left_t), std::move(dirichlet.right_t),
          std::move(const_r), std::vector<P>{}, dim};

      if (tmd.num_chain < 0) // part of a chain
        std::get<source_boundary_data<P>>(src.func).term_index = tid;
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
  if (grid.generation() != sources_grid_gen) {

    int const pdof = hier.degree() + 1;

    int64_t const block_size  = hier.block_size();
    int64_t const num_entries = grid.num_indexes() * block_size;

    // update the constant components
    for (auto &src : sources) {
      if (src.tmode == source_entry<P>::time_mode::time_dependent)
        continue;

      int const dim = (src.tmode == source_entry<P>::time_mode::boundary)
                      ? std::get<source_boundary_data<P>>(src.func).dim : -1;

      if (dim < 0 or not src.consts[dim].empty()) { // left vector
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
      if (dim >= 0) {
        std::vector<P> const &const_r = std::get<source_boundary_data<P>>(src.func).consts_r;
        if (const_r.empty())
          continue;

        std::vector<P> &vals_r = std::get<source_boundary_data<P>>(src.func).vals_r;

        vals_r.resize(num_entries);

        #pragma omp parallel
        {
          std::array<P const *, max_num_dimensions> data1d;

          #pragma omp for
          for (int64_t c = 0; c < grid.num_indexes(); c++) {
            P *proj = vals_r.data() + c * block_size;

            int const *idx = grid[c];
            for (int d = 0; d < dim; d++)
              data1d[d] = src.consts[d].data() + idx[d] * pdof;

            data1d[dim] = const_r.data() + idx[dim] * pdof;

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
      }

      if (src.tmode == source_entry<P>::time_mode::boundary) {
        // recursively apply the previous terms in the chain
        source_boundary_data<P> &bnd = std::get<source_boundary_data<P>>(src.func);
        if (bnd.term_index < 0) // not chain, nothing to do
          continue;
        // this is a chain
        t1.resize(num_entries); // workspace

        bool keep_working = true;
        int tid = bnd.term_index - 1;
        while (keep_working) {

          if (not src.val.empty()) {
            kron_term(grid, conns, terms[tid], 1, src.consts[dim], 0, t1);
            std::swap(src.consts[dim], t1);
          }
          if (not bnd.vals_r.empty()) {
            kron_term(grid, conns, terms[tid], 1, bnd.vals_r, 0, t1);
            std::swap(bnd.vals_r, t1);
          }

          keep_working = (terms[--tid].num_chain < 0);
        }
      }
    }

    if (sources_have_time_dep)
      rebuild_mass_matrices(grid);

    sources_grid_gen = grid.generation();
  }
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
      case source_entry<P>::time_mode::separable: {
          P t = std::get<scalar_func<P>>(src.func)(time);
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
      case source_entry<P>::time_mode::boundary: {
          // NOTE: when adding the "boundary" sources, the sign is flipped
          source_boundary_data<P> const &bnd = std::get<source_boundary_data<P>>(src.func);

          if (not src.val.empty()) { // we have left bc
            if (bnd.left_f) { // time-dependant
              P t = bnd.left_f(time);
              if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
                t *= alpha;
              ASGARD_OMP_PARFOR_SIMD
              for (int64_t i = 0; i < num_entries; i++)
                y[i] -= t * src.val[i];
            } else {
              if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
                ASGARD_OMP_PARFOR_SIMD
                for (int64_t i = 0; i < num_entries; i++)
                  y[i] -= src.val[i];
              else
                ASGARD_OMP_PARFOR_SIMD
                for (int64_t i = 0; i < num_entries; i++)
                  y[i] -= alpha * src.val[i];
            }
          }

          if (not bnd.vals_r.empty()) { // we have right bc
            if (bnd.right_f) { // time-dependant
              P t = bnd.right_f(time);
              if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
                t *= alpha;
              ASGARD_OMP_PARFOR_SIMD
              for (int64_t i = 0; i < num_entries; i++)
                y[i] -= t * bnd.vals_r[i];
            } else {
              if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
                ASGARD_OMP_PARFOR_SIMD
                for (int64_t i = 0; i < num_entries; i++)
                  y[i] -= bnd.vals_r[i];
              else
                ASGARD_OMP_PARFOR_SIMD
                for (int64_t i = 0; i < num_entries; i++)
                  y[i] -= alpha * bnd.vals_r[i];
            }
          }
        }
        break;
      default:
        // unreachable here
        break;
    }
  }
}

template<typename P>
void term_manager<P>::rebuld_term(
    int const tid, sparse_grid const &grid, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, preconditioner_opts precon, P alpha)
{
  expect(legendre.pdof == hier.degree() + 1);
  expect(not terms[tid].tmd.is_chain());
  expect(not terms[tid].tmd.is_interpolatory());

  for (int d : iindexof(num_dims)) {
    auto const &t1d = terms[tid].tmd.dim(d);
    // do not build identity 1d terms
    if (t1d.is_identity())
      continue;

    int level = grid.current_level(d); // required level

    // terms that don't change should be build only once
    if (t1d.change() == changes_with::none) {
      if (terms[tid].coeffs[d].empty())
        level = max_level; // build up to the max
      else
        continue; // already build, we can skip
    }
    // terms that change only on level should be build only on level change
    if (t1d.change() == changes_with::level and terms[tid].level[d] == level)
      continue;

    rebuld_term1d(terms[tid], d, level, conn, hier, precon, alpha);
  } // move to next dimension d
}

template<typename P>
void term_manager<P>::rebuld_term1d(
    term_entry<P> &tentry, int const dim, int level,
    connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
    preconditioner_opts precon, P alpha)
{
  int const n = hier.degree() + 1;
  auto const &t1d = tentry.tmd.dim(dim);

  bool is_diag = t1d.is_mass();
  if (t1d.is_chain()) {
    rebuld_chain(dim, t1d, level, is_diag, wraw_diag, wraw_tri);
  } else {
    build_raw_mat(dim, t1d, level, wraw_diag, wraw_tri);
  }
  // apply the mass matrix, if any
  if (tentry.num_chain < 0) {
    // member of a chain, can have unique mass matrix
    mass_md<P> const &tms = tentry.tmd.mass();
    if (tms and not tms[dim].is_identity()) {
      int const nrows = fm::ipow2(level); // needed number of rows
      if (tentry.mass[dim].nrows() != nrows) {
        build_raw_mat(dim, mass_term[dim], max_level, tentry.mass[dim]);
        tentry.mass[dim].spd_factorize(n);
      }
      if (is_diag)
        tentry.mass[dim].solve(n, wraw_diag);
      else
        tentry.mass[dim].solve(n, wraw_tri);
    }
  } else if (mass[dim]) { // no chain (or last link), and there's global mass
    // global case, use the global mass matrices
    if (level == max_level) {
      if (is_diag)
        mass[dim].solve(n, wraw_diag);
      else
        mass[dim].solve(n, wraw_tri);
    } else { // using lower level, construct lower mass matrix
      int const nrows = fm::ipow2(level); // needed number of rows
      if (lmass[dim].nrows() != nrows) {
        build_raw_mat(dim, mass_term[dim], max_level, lmass[dim]);
        lmass[dim].spd_factorize(n);
      }
      if (is_diag)
        lmass[dim].solve(n, wraw_diag);
      else
        lmass[dim].solve(n, wraw_tri);
    }
  }

  // the build/rebuild put the result in raw_diag or raw_tri
  if (is_diag)
    tentry.coeffs[dim] = hier.diag2hierarchical(wraw_diag, level, conn);
  else
    tentry.coeffs[dim] = hier.tri2hierarchical(wraw_tri, level, conn);

  // build the ADI preconditioner here
  if (precon == preconditioner_opts::adi) {
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
    int d, term_1d<P> const &t1d, int level, block_diag_matrix<P> &raw_diag,
    block_tri_matrix<P> &raw_tri)
{
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
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::div, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_tri);
      }
      if (t1d.penalty() != 0) {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.flux(), t1d.boundary(), raw_tri);
      }
      break;
    case operation_type::grad:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::grad, rhs_type::is_func>
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::grad, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_tri);
      }
      if (t1d.penalty() != 0) {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const, data_mode::increment>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.flux(), t1d.boundary(), raw_tri);
      }
      break;
    case operation_type::penalty:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_func>
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_tri);
      }
      break;
    default:
      // must be unreachable
      break;
  }
}

template<typename P>
void term_manager<P>::build_raw_mat(int dim, term_1d<P> const &t1d, int level,
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
    int const d, term_1d<P> const &t1d, int const level, bool &is_diag,
    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri)
{
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

  if (is_diag) { // a bunch of diag matrices, easy case
    // raw_tri will not be referenced, it's just passed in
    // using raw_diag to make the intermediate matrices, until the last one
    // the last product has to be written to raw_diag
    block_diag_matrix<P> *diag0 = &raw_diag0;
    block_diag_matrix<P> *diag1 = &raw_diag1;
    build_raw_mat(d, t1d[num_chain - 1], level, *diag0, raw_tri);
    for (int i = num_chain - 2; i > 0; i--) {
      build_raw_mat(d, t1d[i], level, raw_diag, raw_tri);
      diag1->check_resize(raw_diag);
      gemm_block_diag(legendre.pdof, raw_diag, *diag0, *diag1);
      std::swap(diag0, diag1);
    }
    build_raw_mat(d, t1d[0], level, *diag1, raw_tri);
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

  fill current = (t1d.is_mass()) ? fill::diag : fill::tri;
  build_raw_mat(d, t1d[num_chain - 1], level, *diag0, *tri0);

  for (int i = num_chain - 2; i > 0; i--)
  {
    build_raw_mat(d, t1d[i], level, raw_diag, raw_tri);
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
  build_raw_mat(d, t1d[0], level, *diag1, *tri1);

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
      (legendre, xleft[d], xright[d], level, nullptr, t1d.penalty(), t1d.flux(), t1d.boundary(), raw_tri);
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
#endif

}
