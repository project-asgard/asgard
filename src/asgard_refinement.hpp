#pragma once
#include "asgard_reconstruct.hpp"
#include "asgard_solver.hpp"

/*!
 * \internal
 * \file asgard_refinement.hpp
 * \brief Defines the grid refinement strategy
 * \author The ASGarD Team
 * \ingroup asgard_discretization
 *
 * \endinternal
 */

namespace asgard
{

/*!
 * \brief Manages refinement criteria
 *
 * This is in a separate class from the asgard::sparse_grid so that the refinement
 * can utilize interpolation and other operator properties that are not known
 * so high up in the header inclusion tree.
 */
template<typename P>
class refinement_manager
{
private:
  using istatus  = sparse_grid::istatus;
  using strategy = sparse_grid::strategy;

public:
  //! create a new manager, no default refinement criteria
  refinement_manager() = default;
  //! initialize the manager and set the refinement criteria
  refinement_manager(prog_opts const &options, pde_scheme<P> &pde);

  //! refine the sparse_grid
  void refine(connection_patterns const &conns, term_manager<P> const &terms,
              std::vector<P> const &state, strategy mode, sparse_grid &grid) const
  {
    if (atol != -1)
      refine_(conns, terms, state, mode, grid);
  }

  //! returns true if a refinement tolerance has been set
  operator bool() const { return (atol > 0 or rtol > 0); }

private:
  //! if no-refinement is set, the public method will have an inline if-statement
  void refine_(connection_patterns const &conns, term_manager<P> const &terms,
               std::vector<P> const &state, strategy mode, sparse_grid &grid) const;

  P atol = -1;
  P rtol = -1;

  md_func_f<P> finterp_;
  md_mom_func_f<P> finterp_mom_;
  std::vector<moment_id> moments_;

  mutable std::vector<istatus> stats;
  mutable std::vector<P> weights;
};

} // namespace asgard
