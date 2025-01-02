#pragma once

#include "asgard_transformations.hpp"

namespace asgard
{

//! \brief Combines a term with data used for linear operations
template<typename P>
struct term_entry {
  //! the term, moved from the pde definition
  term_md<P> tmd;
  //! coefficient matrices for the term
  std::array<block_sparse_matrix<P>, max_num_dimensions> coeffs;
  //! current level that has been constructed
  std::array<int, max_num_dimensions> level = {{0}};
  //! kronmult operation permutations
  kronmult::permutes perm;
  //! indicates if this a single term or a chain
  int num_chain = 1;
  //! setup the perms
  void set_perms();
  //! returns true if the term is separable
  bool is_separable() {
    return perm; // check if perm has been set
  }
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

  // interpolation<P> interp; // must be rebuild as a module

  mutable kronmult::block_global_workspace<P> kwork;
  mutable std::vector<P> t1, t2; // used when doing chains

  mutable vector2d<P> inodes;
  //! rebuild all matrices
  void build_matrices(sparse_grid const &grid, connection_patterns const &conn,
                      hierarchy_manipulator<P> const &hier) {
    tools::time_event timing_("initial coefficients");
    for (int t : iindexof(terms))
      rebuld_term(t, grid, conn, hier);
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

  //! y += tme * x, assumes workspace has been set
  void kron_term(sparse_grid const &grid, connection_patterns const &conns,
                 term_entry<P> const &tme, P alpha, std::vector<P> const &x, P beta,
                 std::vector<P> &y) const
  {
    block_cpu(legendre.pdof, grid, conns, tme.perm, tme.coeffs,
              alpha, x.data(), beta, y.data(), kwork);
  }

protected:
  //! remember which grid was cached for the workspace
  int grid_gen = -1;

  //! rebuild term[tid], loops over all dimensions
  void rebuld_term(int const tid, sparse_grid const &grid, connection_patterns const &conn,
                   hierarchy_manipulator<P> const &hier);
  //! rebuild the 1d term chain to the given level
  void rebuld_chain(int const dim, term_1d<P> const &t1d, int const level, bool &is_diag,
                    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri);

  //! helper method, build the matrix corresponding to the term, also inverts the mass matrix
  void build_raw_mat(int dim, term_1d<P> const &t1d, int level,
                     block_diag_matrix<P> &raw_diag,
                     block_tri_matrix<P> &raw_tri);
};

} // namespace asgard
