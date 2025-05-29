#pragma once

#include "asgard_interp.hpp"

namespace asgard
{

/*!
 * \internal
 * \brief holds the points and values of a variable rhs function
 *
 * Used a both workspace/scratch-space and a way to avoid double-evals
 * of the rhs, e.g., once for the operator matrix and once for
 * the separable boundary condition.
 * \endinternal
 */
template<typename P>
struct rhs_raw_data {
  //! points in the domain, where rhs was evaluated
  std::vector<P> pnts;
  //! the values of the rhs
  std::vector<P> vals;
};

/*!
 * \internal
 * \brief Additional data for term coupling, e.g., Poisson electric field
 *
 * This just holds a bunch of vectors with data needed for the term coefficients,
 * the data depends on coupling, e.g., moments or Poisson solver, and thus
 * cannot be hard-coded in the PDE spec.
 *
 * \endinternal
 */
template<typename P>
struct coupled_term_data
{
  //! electic field from the Poisson solver
  std::vector<P> electric_field;
  //! max-absolute value of the electric field
  std::optional<P> electric_field_infnrm;
  //! number of computed moments
  int num_moments = 0;
  //! data for the computed moments
  std::vector<P> moments;
};

//! holds the moment dependencies in the current term set
struct mom_deps {
  //! requires an electric field and poisson solver
  bool poisson = false;
  //! number of required moments
  int num_moments = 0;
  //! set new minimum moments required
  void set_min(int n) { num_moments = std::max(num_moments, n); }
  //! combine with other deps
  void set_min(mom_deps const &dep) {
    poisson = (poisson or dep.poisson);
    set_min(dep.num_moments);
  }
  //! combine with other deps
  mom_deps &operator += (mom_deps const &dep) {
    set_min(dep);
    return *this;
  }
};

//! \brief Combines a term with data used for linear operations
template<typename P>
struct term_entry {
  //! make default entry, needs to be re-initialized
  term_entry() = default;
  //! initialize the entry with the given term
  term_entry(term_md<P> tin);
  //! resource (mpi-rank/gpu) that will own this term
  resource rec;
  //! the term, moved from the pde definition
  term_md<P> tmd;
  //! coefficient matrices for the term
  std::array<block_sparse_matrix<P>, max_num_dimensions> coeffs;
  //! ADI pseudoinverses of the coefficients
  std::array<block_sparse_matrix<P>, max_num_dimensions> adi;
  //! if the term has additional mass terms, term 0 will contain the mass-up-to current level
  std::array<block_diag_matrix<P>, max_num_dimensions> mass;
  //! current level that has been constructed
  std::array<int, max_num_dimensions> level = {{0}};
  //! kronmult operation permutations
  kronmult::permutes perm;
  //! dependencies on the moments
  std::array<mom_deps, max_num_dimensions> deps;
  //! indicates if this a single term or a chain, negative means member of a chain
  int num_chain = 1;
  //! left/right boundary conditions source index, if positive
  int bc_source_id = -1;
  //! returns true if the term is separable
  bool is_separable() const {
    return perm; // check if kronmult permutations have been set
  }

  //! returns the dependencies for a 1d term
  static mom_deps get_deps(term_1d<P> const &t1d);

  //! boundary conditions, start and end
  indexrange<int> bc;
  //! dimension holding a flux, -1 if no flux
  int flux_dim = -1;
};

//! holds data associated with with either a source term of boundary condition
template<typename P>
struct source_entry
{
  //! mode indicating when to recompute the coefficients
  enum class time_mode {
    //! interior source that is constant in time
    constant = 0,
    //! interior source that is separable in time, i.e., constant in space with time multiplier
    separable,
    //! interior source that is non-separable in time, still separable in space for fixed time
    time_dependent
  };
  //! default source entry, must be reinitialized before use
  source_entry() = default;
  //! create a new source entry
  source_entry(time_mode mode_in) : tmode(mode_in) {}

  //! when should we recompute the sources and when can we reuse existing data
  time_mode tmode = time_mode::constant;
  //! resource (GPU/MPI-rank) assigned to this source
  resource rec;

  bool is_constant() const { return tmode == time_mode::constant; }
  bool is_separable() const { return tmode == time_mode::separable; }
  bool is_time_dependent() const { return tmode == time_mode::time_dependent; }

  //! if the function is separable or time-dependent, handle the extra data
  std::variant<int, scalar_func<P>, separable_func<P>> func;

  //! vector for the current grid
  std::vector<P> val;
  //! constant components of the source vector
  std::array<std::vector<P>, max_num_dimensions> consts;
};

/*!
 * \brief Manages the terms and matrices, also holds the mass-matrices and kronmult-workspace
 *
 * This is the core of the spatial discretization of the terms.
 */
template<typename P>
struct boundary_entry {
  //! mode indicating when to recompute the coefficients
  enum class time_mode {
    //! boundary condition that is constant in time
    constant = 0,
    //! boundary condition that is separable in time, i.e., constant in space with time multiplier
    separable,
    //! boundary condition that is non-separable in time, still separable in space for fixed time
    time_dependent
  };
  //! default source entry, must be reinitialized before use
  boundary_entry() = default;
  //! create a new source entry
  boundary_entry(boundary_flux<P> f) : flux(std::move(f)) {}
  //! defines the flux, moved out of the term
  boundary_flux<P> flux;

  //! when should we recompute the sources and when can we reuse existing data
  time_mode tmode = time_mode::constant;

  bool is_constant() const { return tmode == time_mode::constant; }
  bool is_separable() const { return tmode == time_mode::separable; }
  bool is_time_dependent() const { return tmode == time_mode::time_dependent; }

  //! the term associated with this boundary entry
  int term_index = -1;
  //! vector for the current grid
  std::vector<P> val;
  //! constant components of the source vector
  std::array<std::vector<P>, max_num_dimensions> consts;
};

/*!
 * \brief Manages the terms and matrices, also holds the mass-matrices and kronmult-workspace
 *
 * This is the core of the spatial discretization of the terms.
 */
template<typename P>
struct term_manager
{
  //! create an empty manager, must reinitialize to use
  term_manager() = default;

  /*!
   * \brief Initializes the term manager with the terms of the pde
   *
   * The terms are moved out of the pde object and into the term-manager,
   * holds for both terms_md and the mass matrices.
   * The grid is needed to construct the initial mass matrices
   * and setup the kronmult workspace.
   *
   * Following the constructor, kronmult operations (e.g., interpolation) can be called
   * but none of the terms can be used until the call to build_matrices().
   * The delay is needed to resolve the fact that terms that depend on the moments
   * require the initial solution and the initial solution may require kronmult
   * and the interpolation strategy.
   * The alternative here would be to extract kronmult and interpolation into
   * a separate manager class, but that would be used only in the initial
   * conditions and then repeatedly passed into every single call here.
   */
  term_manager(prog_opts const &opts, pde_domain<P> const &domain,
               pde_scheme<P> &pde, sparse_grid const &grid,
               hierarchy_manipulator<P> const &hier,
               connection_patterns const &conn);

  int num_dims = 0;
  int max_level = 0;

  bool sources_have_time_dep = false;
  bool bcs_have_time_dep     = false;

  //! definition of the mass matrix, usually used in inverse
  mass_md<P> mass_term;
  //! loaded to the max_level, done once and not changed
  std::array<block_diag_matrix<P>, max_num_dimensions> mass;
  //! loaded to the current level, updated as needed
  std::array<block_diag_matrix<P>, max_num_dimensions> lmass;
  //! not factorized for direct application
  std::array<block_sparse_matrix<P>, max_num_dimensions> mass_forward;
  //! mass permutes for kronmult
  kronmult::permutes mass_perm;

  //! all terms, chains are serialized and marked
  std::vector<term_entry<P>> terms;

  //! all sources in the interior
  std::vector<source_entry<P>> sources;
  //! all boundary conditions
  std::vector<boundary_entry<P>> bcs;
  //! interpolatory sources
  std::vector<md_func<P>> sources_md;

  //! left end-point of the domain
  std::array<P, max_num_dimensions> xleft;
  //! right end-point of the domain
  std::array<P, max_num_dimensions> xright;

  //! handles basis manipulations
  legendre_basis<P> legendre;

  //! data for the coupling with moments and electric field
  coupled_term_data<P> cdata;
  //! interpolation data
  interpolation_manager<P> interp;

  mutable kronmult::workspace<P> kwork;
  mutable std::vector<P> t1, t2; // used when doing chains
  mutable std::vector<P> it1, it2; // used for interpolation

  //! term groups, chains are flattened
  std::vector<irange> term_groups;
  //! source groups, same as the PDE
  std::vector<irange> source_groups;

  //! dependencies for each term group, last entry is for all terms
  std::vector<mom_deps> deps_;

  //! resource set to use for the computations
  resource_set resources;

  #ifdef ASGARD_USE_MPI
  //! workspace for MPI
  mutable std::vector<P> mpiwork;
  #endif

  //! get the moment dependencies for all terms
  mom_deps const &deps() const { return deps_.back(); }
  //! get the moment dependencies for the given group
  mom_deps const &deps(int groupid) const { return deps_[groupid]; }

  //! update constant components of the sources
  void update_const_sources(sparse_grid const &grid, connection_patterns const &conn,
                            hierarchy_manipulator<P> const &hier);

  //! update constant components of the sources
  void update_bc(sparse_grid const &grid, connection_patterns const &conn,
                 hierarchy_manipulator<P> const &hier);

  //! rebuild all matrices
  void build_matrices(sparse_grid const &grid, connection_patterns const &conn,
                      hierarchy_manipulator<P> const &hier,
                      precon_method precon = precon_method::none,
                      P alpha = 0) {
    tools::time_event timing_("initial coefficients");
    for (int t : iindexof(terms)) {
      #ifdef ASGARD_USE_MPI
      if (not resources.owns(terms[t].rec))
        continue;
      #endif
      buld_term(t, grid, conn, hier, precon, alpha);
    }
  }
  //! build the large matrices to the max level
  void build_mass_matrices(hierarchy_manipulator<P> const &hier,
                           connection_patterns const &conn)
  {
    if (mass_term) {
      tools::time_event timing_("rebuild mass mats");
      std::vector<int> active_dirs;
      active_dirs.reserve(num_dims);
      for (int d : iindexof(num_dims))
        if (not mass_term[d].is_identity()) {
          build_raw_mass(d, mass_term[d], max_level, mass[d]);
          mass_forward[d] = hier.diag2hierarchical(mass[d], max_level, conn);
          mass[d].spd_factorize(legendre.pdof);
          active_dirs.push_back(d);
        }
      mass_perm = kronmult::permutes(active_dirs);
    }
  }
  //! rebuild the small matrices to the current level for the grid
  void rebuild_mass_matrices(sparse_grid const &grid)
  {
    if (mass_term) {
      tools::time_event timing_("rebuild mass mats");
      for (int d : iindexof(num_dims))
        if (not mass_term[d].is_identity()) {
          int const nrows = fm::ipow2(grid.current_level(d));
          if (lmass[d].nrows() != nrows) {
            build_raw_mass(d, mass_term[d], grid.current_level(d), lmass[d]);
            lmass[d].spd_factorize(legendre.pdof);
          }
        }
    }
  }

  //! rebuild the terms that depend on the Poisson electric field
  void rebuild_poisson(sparse_grid const &grid, connection_patterns const &conn,
                       hierarchy_manipulator<P> const &hier)
  {
    tools::time_event timing_("rebuild - poisson");
    for (auto &te : terms) {
      for (int d : indexof(num_dims))
        if (te.deps[d].poisson and resources.owns(te.rec))
          rebuld_term1d(te, d, grid.current_level(d), conn, hier);
    }
  }
  //! rebuild the terms that depend only on the moments
  void rebuild_moment_terms(sparse_grid const &grid, connection_patterns const &conn,
                            hierarchy_manipulator<P> const &hier)
  {
    tools::time_event timing_("rebuild - moments (all)");
    for (auto &te : terms) {
      for (int d : indexof(num_dims))
        if (te.deps[d].num_moments > 0 and resources.owns(te.rec))
          rebuld_term1d(te, d, grid.current_level(d), conn, hier);
    }
  }
  //! rebuild the terms for the given group
  void rebuild_moment_terms(int groupid, sparse_grid const &grid,
                            connection_patterns const &conn,
                            hierarchy_manipulator<P> const &hier)
  {
    tools::time_event timing_("rebuild - moments");
    expect(0 <= groupid and groupid < static_cast<int>(term_groups.size()));

    for (int it : indexrange(term_groups[groupid])) {
      auto &te = terms[it];
      for (int d : indexof(num_dims))
        if (te.deps[d].num_moments > 0)
          rebuld_term1d(te, d, grid.current_level(d), conn, hier);
    }
  }

  void prapare_workspace(sparse_grid const &grid) {
    if (workspace_grid_gen == grid.generation())
      return;

    int const block_size = fm::ipow(legendre.pdof, grid.num_dims());
    int64_t num_entries  = block_size * grid.num_indexes();

    kwork.w1.resize(num_entries);
    kwork.w2.resize(num_entries);

    if (not t1.empty())
      t1.resize(num_entries);
    if (not t2.empty())
      t2.resize(num_entries);

    if (interp) {
      it1.resize(num_entries);
      it2.resize(num_entries);
    }

    workspace_grid_gen = grid.generation();
  }

  //! returns whether the manager has any terms
  bool has_terms() const { return has_terms_; }

  //! apply the mass matrix
  void mass_apply(sparse_grid const &grid, connection_patterns const &conns,
                  P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const;
  //! compute the inner product < x, mass * x >
  P normL2(sparse_grid const &grid, connection_patterns const &conns,
           std::vector<P> const &x) const;
  //! y = sum(terms * x), applies all terms
  void apply_all(sparse_grid const &grid, connection_patterns const &conn,
                 P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const {
    apply_tmpl<std::vector<P> const &, std::vector<P> &>(-1, grid, conn, alpha, x, beta, y);
  }
  //! y = sum(terms * x), applies all terms
  void apply_all(sparse_grid const &grid, connection_patterns const &conn,
                 P alpha, P const x[], P beta, P y[]) const {
    apply_tmpl<P const[], P[]>(-1, grid, conn, alpha, x, beta, y);
  }
  //! y = sum(terms * x), applies all terms
  void apply_group(int gid, sparse_grid const &grid, connection_patterns const &conn,
                   P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const {
    apply_tmpl<std::vector<P> const &, std::vector<P> &>(gid, grid, conn, alpha, x, beta, y);
  }
  //! y = sum(terms * x), applies all terms
  void apply_group(int gid, sparse_grid const &grid, connection_patterns const &conn,
                   P alpha, P const x[], P beta, P y[]) const {
    apply_tmpl<P const[], P[]>(gid, grid, conn, alpha, x, beta, y);
  }
  //! y = prod(terms_adi * x), applies the ADI preconditioning to all terms
  void apply_all_adi(sparse_grid const &grid, connection_patterns const &conns,
                     P const x[], P y[]) const;

  //! construct term diagonal
  void make_jacobi(int groupid, sparse_grid const &grid, connection_patterns const &conns,
                   std::vector<P> &y) const;
  //! construct term diagonal
  void make_jacobi(sparse_grid const &grid, connection_patterns const &conns,
                   std::vector<P> &y) const {
    make_jacobi(-1, grid, conns, y);
  }

  //! y = alpha * tme * x + beta * y, assumes workspace has been set
  void kron_term(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, P alpha, std::vector<P> const &x, P beta,
                 std::vector<P> &y) const
  {
    if (tme.tmd.is_interpolatory()) {
      interp(grid, conns, 0, x, alpha, tme.tmd.interp(), beta, y, kwork, it1, it2);
    } else {
      block_cpu(legendre.pdof, grid, conns, tme.perm, tme.coeffs,
                alpha, x.data(), beta, y.data(), kwork);
    }
  }
  //! y = alpha * tme * x + beta * y, assumes workspace has been set and x/y have proper size
  void kron_term(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, P alpha, P const x[], P beta, P y[]) const
  {
    if (tme.tmd.is_interpolatory()) {
      interp(grid, conns, 0, x, alpha, tme.tmd.interp(), beta, y, kwork, it1, it2);
    } else {
      block_cpu(legendre.pdof, grid, conns, tme.perm, tme.coeffs,
                alpha, x, beta, y, kwork);
    }
  }
  void kron_term_adi(sparse_grid const &grid, connection_patterns const &conns,
                     term_entry<P> const &tme, P alpha, P const x[], P beta,
                     P y[]) const
  {
    block_cpu(legendre.pdof, grid, conns, tme.perm, tme.adi, alpha, x, beta, y, kwork);
  }

  template<data_mode mode>
  void kron_diag(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, int const block_size, std::vector<P> &y) const;

  //! process the sources and store the result into pre-allocated vector
  template<data_mode dmode>
  void apply_sources(int groupid, pde_domain<P> const &domain, sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, P y[]);

  template<data_mode dmode>
  void apply_sources(pde_domain<P> const &domain, sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, P y[]) {
    apply_sources<dmode>(-1, domain, grid, conns, hier, time, alpha, y);
  }

  template<data_mode dmode>
  void apply_sources(int groupid, pde_domain<P> const &domain, sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, std::vector<P> &y)
  {
    expect(static_cast<int64_t>(y.size()) == hier.block_size() * grid.num_indexes());
    apply_sources<dmode>(groupid, domain, grid, conns, hier, time, alpha, y.data());
  }
  template<data_mode dmode>
  void apply_sources(pde_domain<P> const &domain, sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, std::vector<P> &y)
  {
    expect(static_cast<int64_t>(y.size()) == hier.block_size() * grid.num_indexes());
    apply_sources<dmode>(-1, domain, grid, conns, hier, time, alpha, y.data());
  }

protected:
  //! process the boundary conditions and store the result into pre-allocated vector
  template<data_mode dmode>
  void apply_bc(int groupid, pde_domain<P> const &domain, sparse_grid const &grid,
                connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                P time, P alpha, P y[]);

  //! remember which grid was cached for the workspace
  int workspace_grid_gen = -1;
  //! remember which grid was cached for the sources
  int sources_grid_gen = -1;

  //! rebuild term[tid], loops over all dimensions
  void buld_term(int const tid, sparse_grid const &grid, connection_patterns const &conn,
                 hierarchy_manipulator<P> const &hier,
                 precon_method precon = precon_method::none, P alpha = 0);
  //! rebuild term[tmd][t1d], assumes non-identity
  void rebuld_term1d(term_entry<P> &tentry, int const dim, int level,
                     connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                     precon_method precon = precon_method::none, P alpha = 0);
  //! rebuild the 1d term chain to the given level
  void rebuld_chain(term_entry<P> &tentry, int const dim, int const level,
                    block_diag_matrix<P> const *bmass, bool &is_diag,
                    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri);

  //! helper method, build the matrix corresponding to the term
  void build_raw_mat(term_entry<P> &tentry, int dim, int clink, int level,
                     block_diag_matrix<P> const *bmass,
                     block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri);
  //! helper method, build a mass matrix with no dependencies
  void build_raw_mass(int dim, term_1d<P> const &t1d, int level,
                      block_diag_matrix<P> &raw_diag);
  //! single point implementation for all variations of apply
  template<typename vector_type_x, typename vector_type_y>
  void apply_tmpl(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    P alpha, vector_type_x x, P beta, vector_type_y y) const;

  //! helper method, converts the data on quad
  template<data_mode mode>
  void raw2cells(bool is_diag, int level, std::vector<P> &out);
  //! assign compute resources to the terms
  void assign_compute_resources();

private:
  // workspace and workspace matrices
  bool has_terms_ = false;
  rhs_raw_data<P> raw_rhs;

  block_diag_matrix<P> raw_mass;

  block_diag_matrix<P> wraw_diag;
  block_tri_matrix<P> wraw_tri;

  block_diag_matrix<P> raw_diag0, raw_diag1;
  block_tri_matrix<P> raw_tri0, raw_tri1;
};

} // namespace asgard
