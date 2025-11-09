#pragma once

#include "asgard_term_sources.hpp"

namespace asgard
{

/*!
 * \internal
 * \brief Combines a term with data used for linear operations
 *
 * Each term_md entry from the asgard::pde_scheme definition will yield a term entry,
 * for chain of term_md only the internal links will be added with the last
 * link marked by setting num_chians > 1 and the chain links marked with num_chains == 1.
 * For a regular non-chain term, num_chains == 1.
 *
 * The entry will store the term_md in tmd together with interpolation and term_1d
 * coefficient functions. In addition, the entry will hold the coefficient matrices
 * for the separable 1-D terms together with their mirrors in GPU memory.
 *
 * The kronmult permutations are also here, reflecting the inactive (identity)
 * directions and flux matrices.
 *
 * The resource rec component holds meta data about the associated MPI rank
 * and GPU devices, other data is also cached for quick access, such as shortcuts
 * in the interpolation stages or dependencies on the Poisson solver.
 * Finally, there is an index associated with the appropriate boundary condition,
 * which have to be updated when the coefficient matrices change.
 *
 * \endinternal
 */
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
  #ifdef ASGARD_USE_GPU
  #ifdef ASGARD_GPU_MEMGREEDY
  //! gpu coefficient matrices for each dimension
  std::array<gpu::vector<P>, max_num_dimensions> gpu_coeffs;
  #else
  //! gpu coefficient matrices for different levels
  std::array<std::vector<gpu::vector<P>>, max_num_dimensions> gpu_lcoeffs;
  //! pointers to gpu matrices
  std::array<gpu::vector<P*>, max_num_dimensions> gpu_coeffs;
  #endif
  #endif
  //! ADI pseudoinverses of the coefficients
  std::array<block_sparse_matrix<P>, max_num_dimensions> adi;
  //! if the term has additional mass terms, term 0 will contain the mass-up-to current level
  std::array<block_diag_matrix<P>, max_num_dimensions> mass;
  //! kronmult operation permutations
  kronmult::permutes perm;
  //! dependencies on the Poisson solver
  bool has_poisson = false;
  //! indicates if this a single term or a chain, negative means member of a chain
  int num_chain = 1;
  //! left/right boundary conditions source index, if positive
  int bc_source_id = -1;

  //! check if the 1d term needs a Poisson solver
  static bool has_needs_poisson(term_1d<P> const &t1d);

  //! boundary conditions, start and end
  indexrange<int> bc;
  //! dimension holding a flux, -1 if no flux
  int flux_dim = -1;

  //! returns true if this is the beginning of a chain with at least one more term
  bool is_chain_start() const { return (num_chain > 1); }
  //! returns true if this is a link in a chain, false if stand-alone or first link
  bool is_chain_link() const { return (num_chain < 0); }
  //! mark the entry as being part of a chain
  void mark_as_chain_link() { num_chain = -1; }
  //! retrun true if the term is separable
  bool is_separable() const { return (not is_interpolatory()); }

  //! plan for the interpolation options
  interpolation_plan interplan;
  //! indicates whether the term is interpolatory
  bool is_interpolatory() const { return interplan.is_enabled(); }
};

}
