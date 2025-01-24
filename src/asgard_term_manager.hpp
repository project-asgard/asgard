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
  //! current level that has been constructed
  std::array<int, max_num_dimensions> level = {{0}};
  //! kronmult operation permutations
  kronmult::permutes perm;
  //! dependencies on the moments
  std::array<mom_deps, max_num_dimensions> deps;
  //! indicates if this a single term or a chain
  int num_chain = 1;
  //! returns true if the term is separable
  bool is_separable() {
    return perm; // check if kronmult permutations have been set
  }
  //! returns the dependencies for a 1d term
  static mom_deps get_deps(term_1d<P> const &t1d);
};

/*!
 * \brief Manges the terms and matrices
 *
 */
template<typename P>
struct term_manager
{
  term_manager() = default;

  term_manager(PDEv2<P> &pde);

  int num_dims = 0;
  int max_level = 0;

  std::vector<term_entry<P>> terms;

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

  //! rebuild all matrices
  void build_matrices(sparse_grid const &grid, connection_patterns const &conn,
                      hierarchy_manipulator<P> const &hier,
                      preconditioner_opts precon = preconditioner_opts::none,
                      P alpha = 0) {
    tools::time_event timing_("initial coefficients");
    for (int t : iindexof(terms))
      rebuld_term(t, grid, conn, hier, precon, alpha);
  }
  //! rebuild the terms that depend on the Poisson electric field
  void rebuild_poisson(sparse_grid const &grid, connection_patterns const &conn,
                      hierarchy_manipulator<P> const &hier)
  {
    for (auto &te : terms) {
      for (int d : indexof(num_dims))
        if (te.deps[d].poisson)
          rebuld_term1d(te, d, grid.current_level(d), conn, hier);
    }
  }

  void prapare_workspace(sparse_grid const &grid) {
    if (grid_gen == grid.generation())
      return;

    int const block_size = fm::ipow(legendre.pdof, grid.num_dims());
    int64_t num_entries  = block_size * grid.num_indexes();

    kwork.w1.resize(num_entries);
    kwork.w2.resize(num_entries);

    if (not t1.empty())
      t1.resize(num_entries);
    if (not t2.empty())
      t2.resize(num_entries);

    grid_gen = grid.generation();
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

protected:
  //! remember which grid was cached for the workspace
  int grid_gen = -1;

  //! rebuild term[tid], loops over all dimensions
  void rebuld_term(int const tid, sparse_grid const &grid, connection_patterns const &conn,
                   hierarchy_manipulator<P> const &hier,
                   preconditioner_opts precon = preconditioner_opts::none, P alpha = 0);
  //! rebuild term[tmd][t1d], assumes non-identity
  void rebuld_term1d(term_entry<P> &tentry, int const t1d,
                     int level, connection_patterns const &conn,
                     hierarchy_manipulator<P> const &hier,
                     preconditioner_opts precon = preconditioner_opts::none, P alpha = 0);
  //! rebuild the 1d term chain to the given level
  void rebuld_chain(int const dim, term_1d<P> const &t1d, int const level, bool &is_diag,
                    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri);

  //! helper method, build the matrix corresponding to the term, also inverts the mass matrix
  void build_raw_mat(int dim, term_1d<P> const &t1d, int level,
                     block_diag_matrix<P> &raw_diag,
                     block_tri_matrix<P> &raw_tri);

private:
  // workspace matrices
  block_diag_matrix<P> raw_mass;

  block_diag_matrix<P> wraw_diag;
  block_tri_matrix<P> wraw_tri;

  block_diag_matrix<P> raw_diag0, raw_diag1;
  block_tri_matrix<P> raw_tri0, raw_tri1;
};

} // namespace asgard
