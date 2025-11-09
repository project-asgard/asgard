#pragma once

#include "asgard_term_build.hpp"

namespace asgard
{

/*!
 * \internal
 * \brief Manages the terms and matrices, also holds the mass-matrices and kronmult-workspace
 *
 * The terms, sources and boundary conditions have a close interplay with each other,
 * building term entries and using the coefficient functions also leads to updates
 * of the boundary conditions. After construction, boundary conditions work
 * much like sources; however, both sources and boundary conditions also use mass-matrices
 * and kronmult operations in case of chaining or using non-separable sources.
 * The term matrices also have two stages, first is the build and then the application,
 * where both have to account for 1d and multi-d chaining, separable and interpolatory
 * operations.
 *
 * Managing a large amount of functionality is challenging while breaking it into separate
 * modules, components or classes will create artificial API walls and even more overall
 * complexity, e.g., more dependencies for each function call, grant access with fiend classes,
 * setter/getter methods, or incur computational cost by recomputing the same result more
 * than once. The solution here is to keep relevant data together but have the methods
 * split across multiple files.
 *
 * asgard_term_sources.hpp
 * asgard_term_build.hpp -> includes asgard_term_sources.hpp
 * asgard_term_manager.hpp -> includes asgard_term_build.hpp
 * asgard_term_manager.cpp -> all .cpp files include asgard_term_manager.hpp
 * asgard_term_build.cpp
 * asgard_term_sources.cpp
 *
 * The implementation is grouped:
 * - build methods, e.g., constructing matrices and sources, and assigning work
 *   to MPI ranks and GPU devices
 * - apply method, e.g., perform matrix-vector operations on groups of terms and sources
 *   associated with the current MPI-rank
 * - source methods, e.g., update the sources based on the current grid and add the vectors
 *
 * \endinternal
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
  //! number of dimensions, quick access
  int num_dims = 0;
  //! the max level, determines the highest level for operators
  int max_level = 0;
  //! indicates if there are time dependent sources, build extra data-structures
  bool sources_have_time_dep = false;
  //! indicates if there are time dependent boundary conditions, build extra data-structures
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

  //! term groups, chains are flattened
  std::vector<irange> term_groups;
  //! source groups for boundary and regular sources
  std::vector<group_combo> source_groups;
  //! number of all sources lumped into a gemv
  int num_lumped = 0;

  //! left end-point of the domain
  std::array<P, max_num_dimensions> xleft;
  //! right end-point of the domain
  std::array<P, max_num_dimensions> xright;

  //! handles basis manipulations
  legendre_basis<P> basis;

  //! storage for the moments
  momentset<P> momset;
  //! manages the moments operations, interplays with the mass
  moment_manager<P> moms;
  //! storage for the interpolated moments
  momentset<P> momset_interp;
  //! interpolation data
  interpolation_manager<P> interp;
  //! values for the interpolation field, allows reuse for several interp ops
  mutable std::vector<P> ifield;

  mutable kronmult::workspace<P> kwork;
  mutable std::vector<P> t1, t2; // used when doing chains
  mutable std::vector<P> it1, it2; // used for interpolation
  mutable std::vector<P> swork, sweights; // source workspace and time weights
  #ifdef ASGARD_USE_GPU
  mutable std::array<gpu::vector<P>, max_num_gpus> gpu_t1, gpu_t2;
  mutable std::array<gpu::vector<P>, max_num_gpus> gpu_x, gpu_y; // for out-of-core evals
  // for both multi-gpu support and interpolation evals on the CPU
  mutable std::array<std::vector<P>, max_num_gpus> cpu_it1, cpu_it2;
  mutable std::array<gpu::vector<P>, max_num_gpus> gpu_it1, gpu_it2;
  #endif

  //! dependencies for each term group, last entry is for all terms, use has_poisson() not this directly
  std::vector<bool> has_poisson_;
  //! has Poisson solver for the given group
  bool has_poisson(int groupid) const { return (not has_poisson_.empty() and has_poisson_[groupid]); }
  //! has Poisson solver for any group
  bool has_poisson() const { return (not has_poisson_.empty()); }

  //! resource set to use for the computations
  resource_set resources;

  #ifdef ASGARD_USE_MPI
  //! workspace for MPI
  mutable std::vector<P> mpiwork;
  #endif

  //! return the range for the given group, returns full range for group -1
  indexrange<int> terms_group_range(int groupid) const {
    return (groupid == all_groups) ? indexrange<int>(terms) : indexrange<int>(term_groups[groupid]);
  }

  //! rebuild all matrices
  void build_matrices(sparse_grid const &grid, connection_patterns const &conn,
                      hierarchy_manipulator<P> const &hier,
                      precon_method precon = precon_method::none,
                      P alpha = 0)
  {
    tools::time_event timing_("initial coefficients");
    for (int t : iindexof(terms)) {
      #ifdef ASGARD_USE_MPI
      if (not resources.owns(terms[t].rec))
        continue;
      #endif
      build_const_terms(t, grid, conn, hier, precon, alpha);
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
          mass[d].spd_factorize(basis.pdof);
          active_dirs.push_back(d);
          if (moms) {
            if (mass_term[d].rhs()) {
              // the constant will be ignored here
              moms.set_mass(d, xleft[d], xright[d], max_level, basis, hier, 1, raw_rhs);
            } else {
              raw_rhs.vals.resize(0); // no variable coefficient, will fill this with a constant
              moms.set_mass(d, xleft[d], xright[d], max_level, basis, hier,
                            mass_term[d].rhs_const(), raw_rhs);
            }
          }
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
            lmass[d].spd_factorize(basis.pdof);
          }
        }
    }
  }

  //! rebuild the terms that depend only on the moments
  void rebuild_moment_terms(sparse_grid const &grid, connection_patterns const &conn,
                               hierarchy_manipulator<P> const &hier)
  {
    rebuild_moment_terms(all_groups, grid, conn, hier);
  }
  //! rebuild the terms that depend only on the moments
  void rebuild_moment_terms(int groupid, sparse_grid const &grid, connection_patterns const &conn,
                               hierarchy_manipulator<P> const &hier)
  {
    tools::time_event timing_("rebuild moment terms (" + ((groupid == -1) ? std::string("all") : std::to_string(groupid)) + ")");
    expect(-1 <= groupid and groupid < static_cast<int>(term_groups.size()));
    for (int it : terms_group_range(groupid)) {
      auto &te = terms[it];
      for (int d : indexof(num_dims))
        if (resources.owns(te.rec) and te.tmd.dim(d).depends() != term_dependence::none)
          rebuild_term1d(te, d, grid.current_level(d), conn, hier);
    }
  }
  //! prepares the kronmult workspace
  void prapare_kron_workspace(sparse_grid const &grid) {
    if (workspace_grid_gen == grid.generation())
      return;

    int const block_size = fm::ipow(basis.pdof, grid.num_dims());
    int64_t num_entries  = block_size * grid.num_indexes();

    kwork.w1.resize(num_entries);
    kwork.w2.resize(num_entries);

    t1.resize(num_entries);
    t2.resize(num_entries);

    if (interp) {
      it1.resize(num_entries);
      it2.resize(num_entries);
    }

    #ifdef ASGARD_USE_GPU
    prapare_kron_workspace_gpu(num_entries);
    #endif

    workspace_grid_gen = grid.generation();
  }
  #ifdef ASGARD_USE_GPU
  //! prepares the kronmult workspace for the GPU
  void prapare_kron_workspace_gpu(int64_t num_entries);
  #endif

  //! returns whether the manager has any terms
  bool has_terms() const { return has_terms_; }

  //! apply the mass matrix
  void mass_apply(sparse_grid const &grid, connection_patterns const &conns,
                  P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const;
  //! compute the inner product < x, mass * x >
  P normL2(sparse_grid const &grid, connection_patterns const &conns,
           std::vector<P> const &x) const;
  //! y = sum(terms * x), applies all terms
  void apply(sparse_grid const &grid, connection_patterns const &conn,
             P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const {
    #ifdef ASGARD_USE_GPU
    apply_tmpl_gpu<std::vector<P> const &, std::vector<P> &, compute_mode::cpu>(all_groups, grid, conn, alpha, x, beta, y);
    #else
    apply_tmpl<std::vector<P> const &, std::vector<P> &>(all_groups, grid, conn, alpha, x, beta, y);
    #endif
  }
  //! y = sum(terms * x), applies all terms
  void apply(sparse_grid const &grid, connection_patterns const &conn,
             P alpha, P const x[], P beta, P y[]) const {
    #ifdef ASGARD_USE_GPU
    apply_tmpl_gpu<P const[], P[], compute_mode::cpu>(all_groups, grid, conn, alpha, x, beta, y);
    #else
    apply_tmpl<P const[], P[]>(all_groups, grid, conn, alpha, x, beta, y);
    #endif
  }
  //! y = sum(terms * x), applies all terms
  void apply(int gid, sparse_grid const &grid, connection_patterns const &conn,
             P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const {
    #ifdef ASGARD_USE_GPU
    apply_tmpl_gpu<std::vector<P> const &, std::vector<P> &, compute_mode::cpu>(gid, grid, conn, alpha, x, beta, y);
    #else
    apply_tmpl<std::vector<P> const &, std::vector<P> &>(gid, grid, conn, alpha, x, beta, y);
    #endif
  }
  //! y = sum(terms * x), applies all terms
  void apply(int gid, sparse_grid const &grid, connection_patterns const &conn,
             P alpha, P const x[], P beta, P y[]) const {
    #ifdef ASGARD_USE_GPU
    apply_tmpl_gpu<P const[], P[], compute_mode::cpu>(gid, grid, conn, alpha, x, beta, y);
    #else
    apply_tmpl<P const[], P[]>(gid, grid, conn, alpha, x, beta, y);
    #endif
  }
  #ifdef ASGARD_USE_GPU
  //! y = sum(terms * x), applies all terms, input is on the GPU
  void apply_gpu(sparse_grid const &grid, connection_patterns const &conn,
                 P alpha, P const x[], P beta, P y[]) const {
    apply_tmpl_gpu<P const[], P[], compute_mode::gpu>(all_groups, grid, conn, alpha, x, beta, y);
  }
  //! y = sum(terms * x), applies all terms for the group, input is on the GPU
  void apply_gpu(int gid, sparse_grid const &grid, connection_patterns const &conn,
                 P alpha, P const x[], P beta, P y[]) const {
    apply_tmpl_gpu<P const[], P[], compute_mode::gpu>(gid, grid, conn, alpha, x, beta, y);
  }
  #endif
  #ifdef ASGARD_USE_FLOPCOUNTER
  //! count flops for the application of the specified group
  int64_t flop_count(
    int gid, sparse_grid const &grid, connection_patterns const &conns) const;
  #endif

  //! construct term diagonal
  void make_jacobi(int groupid, sparse_grid const &grid, connection_patterns const &conns,
                   std::vector<P> &y) const;
  //! construct term diagonal
  void make_jacobi(sparse_grid const &grid, connection_patterns const &conns,
                   std::vector<P> &y) const {
    make_jacobi(all_groups, grid, conns, y);
  }

  //! y = alpha * tme * x + beta * y, assumes workspace has been set (used for boundary conditions)
  void kron_term(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, P alpha, std::vector<P> const &x, P beta,
                 std::vector<P> &y) const
  {
    if (tme.is_interpolatory()) {
      interp(tme.interplan, grid, conns, momset, 0, x.data(), {},
             alpha, tme.tmd, beta, y.data(), kwork, it1, it2);
    } else {
      block_cpu(basis.pdof, grid, conns, tme.perm, tme.coeffs,
                alpha, x.data(), beta, y.data(), kwork);
    }
  }
  //! y = alpha * tme * x + beta * y, assumes workspace has been set (used for boundary conditions)
  void kron_term(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, P alpha, P const x[], P beta, P y[]) const
  {
    if (tme.is_interpolatory()) {
      interp(tme.interplan, grid, conns, momset, 0, x, {},
             alpha, tme.tmd, beta, y, kwork, it1, it2);
    } else {
      block_cpu(basis.pdof, grid, conns, tme.perm, tme.coeffs,
                alpha, x, beta, y, kwork);
    }
  }

  //! build the diagonal preconditioner
  template<data_mode mode>
  void kron_diag(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, int const block_size, std::vector<P> &y) const;

  //! process the source group and store the result into pre-allocated vector
  template<data_mode dmode>
  void apply_sources(int groupid, sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, P y[]);
  //! process all the sources and store the result into pre-allocated vector
  template<data_mode dmode>
  void apply_sources(sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, P y[]) {
    apply_sources<dmode>(all_groups, grid, conns, hier, time, alpha, y);
  }
  //! process the sources in the group and apply the dmode operation to y
  template<data_mode dmode>
  void apply_sources(int groupid, sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, std::vector<P> &y)
  {
    expect(static_cast<int64_t>(y.size()) == hier.block_size() * grid.num_indexes());
    apply_sources<dmode>(groupid, grid, conns, hier, time, alpha, y.data());
  }
  //! process all sources and apply the dmode operation to y
  template<data_mode dmode>
  void apply_sources(sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, std::vector<P> &y)
  {
    expect(static_cast<int64_t>(y.size()) == hier.block_size() * grid.num_indexes());
    apply_sources<dmode>(all_groups, grid, conns, hier, time, alpha, y.data());
  }

  //! indicates the use of all groups
  static constexpr int all_groups = -1;

protected:
  //! remember which grid was cached for the workspace
  int workspace_grid_gen = -1;
  //! remember which grid was cached for the sources
  int sources_grid_gen = -1;

  //! rebuild term[tid], loops over all dimensions
  void build_const_terms(int const tid, sparse_grid const &grid, connection_patterns const &conn,
                         hierarchy_manipulator<P> const &hier,
                         precon_method precon = precon_method::none, P alpha = 0);
  //! rebuild term[tmd][t1d], assumes non-identity
  void rebuild_term1d(term_entry<P> &tentry, int const dim, int level,
                      connection_patterns const &conn, hierarchy_manipulator<P> const &hier,
                      precon_method precon = precon_method::none, P alpha = 0,
                      bool merge_with_interp = false);
  //! rebuild the 1d term chain to the given level
  void rebuld_chain(term_entry<P> &tentry, int const dim, int const level,
                    hierarchy_manipulator<P> const &hier,
                    block_diag_matrix<P> const *bmass, bool &is_diag,
                    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri);

  //! helper method, build the matrix corresponding to the term
  void build_raw_mat(term_entry<P> &tentry, int dim, int clink, int level,
                     hierarchy_manipulator<P> const &hier,
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

  #ifdef ASGARD_USE_GPU
  //! single point implementation for all variations of apply, uses the GPU the data can come from the CPU or GPU
  template<typename vector_type_x, typename vector_type_y, compute_mode mode>
  void apply_tmpl_gpu(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    P alpha, vector_type_x x, P beta, vector_type_y y) const;
  #endif

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

  block_diag_matrix<P> raw_diag0, raw_diag1; // workspace for 1D chains
  block_tri_matrix<P> raw_tri0, raw_tri1;

  #ifdef ASGARD_USE_FLOPCOUNTER
  struct flop_info_entry {
    int grid_gen = -1;
    int64_t flops = 0;
  };
  mutable std::vector<flop_info_entry> flop_info;
  #endif
};

} // namespace asgard
