#pragma once

#include "asgard_transformations.hpp"

namespace asgard
{

//! holds the moment dependencies in the current term set
struct mom_deps {
  //! requires an electric field and poisson solver
  bool poisson    = false;
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
  bool is_separable() {
    return perm; // check if kronmult permutations have been set
  }

  //! returns the dependencies for a 1d term
  static mom_deps get_deps(term_1d<P> const &t1d);
};

/*!
 * \brief Contains the data for a time-dependant boundary source
 */
template<typename P>
struct time_boundary_data {
  //! make empty data-entry
  time_boundary_data() = default;
  //! set a new entry with the given time
  time_boundary_data(scalar_func<P> sf) : time(std::move(sf)) {}
  //! time scalar function
  scalar_func<P> time;
  //! constant component in 1d
  std::vector<P> const_1d;
  //! constant component in multi-dimensions
  std::vector<P> const_md;
};

//! holds the extra data, if using boundary condition with different left/right components
template<typename P>
struct source_boundary_data {
  //! time components of the source
  std::vector<time_boundary_data<P>> time_entries;
  //! dimension where the boundary is applied
  int dim = -1;
  //! if the source entry is associated with term in a chain
  int term_index = -1;
};

//! holds the case for edge case when time is separable
template<typename P>
struct source_edge_stime {
  //! the separable time function
  scalar_func<P> time;
  //! the index of the term
  int term_index = -1;
};

//! holds the extra data for an edge case when everything depends on time
template<typename P>
class source_edge_data {
public:
  //! create a left-edge data
  source_edge_data(separable_func<P> f, int d)
    : sep_(std::move(f)), dim_(-d - 1)
  {
    expect(0 <= d and d < max_num_dimensions);
  }
  //! create a right-edge data
  source_edge_data(int d, separable_func<P> f)
    : sep_(std::move(f)), dim_(d + 1)
  {
    expect(0 <= d and d < max_num_dimensions);
  }
  //! returns the dimension for the edge
  int dim() const { return std::abs(dim_) - 1; }
  //! returns if the edge is left
  bool left() const { return (dim_ < 0); }
  //! returns if the edge is right
  bool right() const { return (dim_ > 0); }
  //! returns the separable function
  separable_func<P> const &sep() const { return sep_; }

  //! if the source entry is associated with term in a chain
  int term_index = -1;
  //! scaled value of the boundary condition
  P mag = 0;

private:
  //! separable function
  separable_func<P> sep_;
  //! dimension holding the edge direction, negative implies left side
  int dim_ = 0;
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
    time_dependent,
    //! boundary source that is constant across the wall
    boundary,
    //! boundary source that is constant in time but non-constant on the edge/boundary
    edge_constant,
    //! boundary source that is separable in time
    edge_separable,
    //! boundary source that is non-separable in time
    edge_time
  };
  //! default source entry, must be reinitialized before use
  source_entry() = default;
  //! create a new source entry
  source_entry(time_mode mode_in) : tmode(mode_in) {}

  //! when should we recompute the sources and when can we reuse existing data
  time_mode tmode = time_mode::constant;

  bool is_constant() const { return tmode == time_mode::constant; }
  bool is_separable() const { return tmode == time_mode::separable; }
  bool is_time_dependent() const { return tmode == time_mode::time_dependent; }
  bool is_boundary() const { return tmode == time_mode::boundary; }

  //! separable term, but one component is an edge component
  bool is_edge() const {
    return (tmode == time_mode::edge_constant or tmode == time_mode::edge_separable or
            tmode == time_mode::edge_time);
  }

  bool is_edge_constant() const { return tmode == time_mode::edge_constant; }
  bool is_edge_separable() const { return tmode == time_mode::edge_separable; }
  bool is_edge_time() const { return tmode == time_mode::edge_time; }

  //! if the function is separable or time-dependent, handle the extra data
  std::variant<int, scalar_func<P>, separable_func<P>, source_boundary_data<P>,
               source_edge_stime<P>, source_edge_data<P>> func;

  int edge_term_index() const {
    expect(is_edge());
    switch (tmode) {
      case time_mode::edge_constant:
        return std::get<int>(func);
      case time_mode::edge_separable:
        return std::get<source_edge_stime<P>>(func).term_index;
      default:
        return std::get<source_edge_data<P>>(func).term_index;
    }
  }

  //! quick access to the source-edge data, edge-time case only
  source_edge_data<P> const &source_edge() const {
    expect(tmode == time_mode::edge_time);
    return std::get<source_edge_data<P>>(func);
  }

  //! quick access to the boundary data
  int dim() const {
    expect(tmode == time_mode::boundary);
    return std::get<source_boundary_data<P>>(func).dim;
  }
  //! quick access to the boundary data
  int term_index() const {
    switch (tmode) {
      case time_mode::boundary:
        return std::get<source_boundary_data<P>>(func).term_index;
      case time_mode::edge_constant:
        return std::get<int>(func);
      case time_mode::edge_separable:
        return std::get<source_edge_stime<P>>(func).term_index;
      case time_mode::edge_time:
        return std::get<source_edge_data<P>>(func).term_index;
      default:
        expect(is_boundary() or is_edge());
        return 0;
    }
  }
  //! quick access to the boundary data
  std::vector<time_boundary_data<P>> const &time_boundary() const {
    expect(tmode == time_mode::boundary);
    return std::get<source_boundary_data<P>>(func).time_entries;
  }
  //! quick access to the boundary data
  std::vector<time_boundary_data<P>> &time_boundary() {
    expect(tmode == time_mode::boundary);
    return std::get<source_boundary_data<P>>(func).time_entries;
  }

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
  term_manager(PDEv2<P> &pde, sparse_grid const &grid,
               hierarchy_manipulator<P> const &hier);

  int num_dims = 0;
  int max_level = 0;

  bool sources_have_time_dep = false;

  mass_md<P> mass_term;
  // loaded to the max_level, done once and not changed
  std::array<block_diag_matrix<P>, max_num_dimensions> mass;
  // loaded to the current level, updated as needed
  std::array<block_diag_matrix<P>, max_num_dimensions> lmass;

  //! all terms, chains are serialized and marked
  std::vector<term_entry<P>> terms;
  //! number of sources not associated with boundary conditions
  int num_interior_sources = 0;
  //! all sources, interior and boundary conditions
  std::vector<source_entry<P>> sources;

  std::array<P, max_num_dimensions> xleft;
  std::array<P, max_num_dimensions> xright;

  legendre_basis<P> legendre;

  //! data for the coupling with moments and electric field
  coupled_term_data<P> cdata;

  // interpolation<P> interp; // must be rebuild as a module

  mutable kronmult::block_global_workspace<P> kwork;
  mutable std::vector<P> t1, t2; // used when doing chains

  mutable vector2d<P> inodes;

  //! find the dependencies of the current term set
  mom_deps find_deps() const;

  //! update constant components of the sources
  void update_const_sources(sparse_grid const &grid, connection_patterns const &conn,
                            hierarchy_manipulator<P> const &hier);

  //! rebuild all matrices
  void build_matrices(sparse_grid const &grid, connection_patterns const &conn,
                      hierarchy_manipulator<P> const &hier,
                      precon_method precon = precon_method::none,
                      P alpha = 0) {
    tools::time_event timing_("initial coefficients");
    for (int t : iindexof(terms))
      buld_term(t, grid, conn, hier, precon, alpha);
  }
  //! build the large matrices to the max level
  void build_mass_matrices()
  {
    if (mass_term) {
      tools::time_event timing_("rebuild mass mats");
      for (int d : iindexof(num_dims))
        if (not mass_term[d].is_identity()) {
          build_raw_mass(d, mass_term[d], max_level, mass[d]);
          mass[d].spd_factorize(legendre.pdof);
        }
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
    source_entry<P> no_bc;
    for (auto &te : terms) {
      for (int d : indexof(num_dims))
        if (te.deps[d].poisson)
          rebuld_term1d(te, d, grid.current_level(d), conn, hier, no_bc);
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

    workspace_grid_gen = grid.generation();
  }

  //! y = sum(terms * x), applies all terms
  void apply_all(sparse_grid const &grid, connection_patterns const &conns,
                 P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const;
  //! y = sum(terms * x), applies all terms
  void apply_all(sparse_grid const &grid, connection_patterns const &conns,
                 P alpha, P const x[], P beta, P y[]) const;

  //! y = prod(terms_adi * x), applies the ADI preconditioning to all terms
  void apply_all_adi(sparse_grid const &grid, connection_patterns const &conns,
                     P const x[], P y[]) const;

  //! construct term diagonal
  void make_jacobi(sparse_grid const &grid, connection_patterns const &conns,
                   std::vector<P> &y) const;

  //! y = alpha * tme * x + beta * y, assumes workspace has been set
  void kron_term(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, P alpha, std::vector<P> const &x, P beta,
                 std::vector<P> &y) const
  {
    block_cpu(legendre.pdof, grid, conns, tme.perm, tme.coeffs,
              alpha, x.data(), beta, y.data(), kwork);
  }
  //! y = alpha * tme * x + beta * y, assumes workspace has been set and x/y have proper size
  void kron_term(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, P alpha, P const x[], P beta, P y[]) const
  {
    block_cpu(legendre.pdof, grid, conns, tme.perm, tme.coeffs,
              alpha, x, beta, y, kwork);
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
  void apply_sources(pde_domain<P> const &domain, sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, P y[]);

  template<data_mode dmode>
  void apply_sources(pde_domain<P> const &domain, sparse_grid const &grid,
                     connection_patterns const &conns, hierarchy_manipulator<P> const &hier,
                     P time, P alpha, std::vector<P> &y)
  {
    expect(static_cast<int64_t>(y.size()) == hier.block_size() * grid.num_indexes());
    apply_sources<dmode>(domain, grid, conns, hier, time, alpha, y.data());
  }

protected:
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
                     source_entry<P> &bc,
                     precon_method precon = precon_method::none, P alpha = 0);
  //! rebuild the 1d term chain to the given level
  void rebuld_chain(int const dim, term_1d<P> &t1d, int const level, bool &is_diag,
                    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri,
                    source_entry<P> &bc);

  //! helper method, build the matrix corresponding to the term
  void build_raw_mat(int dim, term_1d<P> &t1d, int level,
                     block_diag_matrix<P> &raw_diag,
                     block_tri_matrix<P> &raw_tri, source_entry<P> &bc);
  //! helper method, build a mass matrix with no dependencies
  void build_raw_mass(int dim, term_1d<P> const &t1d, int level,
                      block_diag_matrix<P> &raw_diag);
  //! helper method, converts the data on quad
  template<data_mode mode>
  void raw2cells(bool is_diag, int level, std::vector<P> &out);
  //! add Dirichlet boundary conditions to the source term
  void add_dirichlet(term_1d<P> const &t1d, int level, dirichelt_boundary1d<P> &dirichlet,
                     source_entry<P> &bc) const;

private:
  // workspace and workspace matrices
  rhs_raw_data<P> raw_rhs;

  block_diag_matrix<P> raw_mass;

  block_diag_matrix<P> wraw_diag;
  block_tri_matrix<P> wraw_tri;

  block_diag_matrix<P> raw_diag0, raw_diag1;
  block_tri_matrix<P> raw_tri0, raw_tri1;
};

} // namespace asgard
