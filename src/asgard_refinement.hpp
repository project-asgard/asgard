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
  //! using enums from the sparse grid class
  using istatus  = sparse_grid::istatus;
  //! using enums from the sparse grid class
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

  //! absolute tolerance, -1 indicates not using refinement
  P atol = -1;
  //! relative tolerance, -1 indicates not using refinement
  P rtol = -1;

  /*!
   * \brief Holds the information for additional interpolation weights
   *
   * The weights are defined via interpolation functions and match the call
   * convention used by asgard::term_md to easily work with interpolation.
   */
  struct interp_weights {
    //! interpolation weights using only the field
    void interp(P t, vector2d<P> const &x, std::vector<P> const &f, std::vector<P> &vals) const {
      expect(!!interp_);
      interp_(t, x, f, vals);
    }
    //! interpolation weights using the field and moments
    void interp(P t, vector2d<P> const &x, momentset<P> const &moments,
                std::vector<P> const &f, std::vector<P> &vals) const {
      expect(!!interp_mom_);
      interp_mom_(t, x, moments, f, vals);
    }
    operator bool () const { return (interp_ or interp_mom_); }
    md_func_f<P> interp_;
    md_mom_func_f<P> interp_mom_;
  };

  interp_weights weights_;

  std::vector<moment_id> moments_;

  mutable interpolation_plan iplan;

  mutable std::vector<istatus> stats;
  mutable std::vector<P> weights;
};

} // namespace asgard
