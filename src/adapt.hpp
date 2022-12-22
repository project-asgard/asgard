#pragma once
#include "basis.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "program_options.hpp"

//#include "asgard_pde_system.hpp"

namespace asgard::adapt
{
// this class bundles
// 1) the element table (set of active elements and their coordinates) and
// 2) the distribution plan that maps ranks to the active elements whose
// coefficients the rank must compute coefficients for.

// the active elements can be viewed as a 2d grid where grid(i,j) refers to the
// connection from the ith element to the jth element. the grid is square - we
// assume full connectivity. the element table and distribution plan together
// represent a "distributed grid" - each rank is assigned some subgrid of the 2d
// element grid.

// the purpose of this class is to adapt the set of the active elements given
// the coefficients in the initial condition/solution vector x. during gemv that
// drives explicit time advance, for each term, each element connection reads
// from a deg^dim segment x beginning at x[i*deg^dim] and writes deg^dim
// coefficients to y[j*deg^dim].

// elements responsible for coefficients with low absolute value may be removed
// from the grid (grid coarsening) elements responsible for coefficients with
// large absolute value may have child elements added to grid (grid refinement)
// the limits for refinement/coarsening are set by the user using command line
// options.

// this class relies on distribution component functions to communicate changes
// in element table and distribution plan between ranks after
// refinement/coarsening.
template<typename P>
class distributed_grid
{
public:
  distributed_grid(options const &cli_options,
                   std::vector<dimension<P>> const &dims);

  distributed_grid(PDE<P> const &pde, options const &cli_options)
      : distributed_grid(cli_options, pde.get_dimensions())
  {}

  // driver routines
  fk::vector<P> get_initial_condition(
      PDE<P> &pde,
      basis::wavelet_transform<P, resource::host> const &transformer,
      options const &cli_opts)
  {
    return this->get_initial_condition(
        pde.get_dimensions(),
        pde.has_analytic_soln ? pde.exact_time(0.0) : static_cast<P>(1.0),
        pde.num_terms, pde.get_terms(), transformer, cli_opts);
  }

  fk::vector<P> get_initial_condition(
      std::vector<dimension<P>> &dims, P const mult, int const num_terms,
      std::vector<std::vector<term<P>>> &terms,
      basis::wavelet_transform<P, resource::host> const &transformer,
      options const &cli_opts);

  void get_initial_condition(
      options const &cli_opts, std::vector<dimension<P>> &dims,
      std::vector<vector_func<P>> const &v_functions, P const mult,
      basis::wavelet_transform<P, resource::host> const &transformer,
      fk::vector<P, mem_type::view> result);

  fk::vector<P> coarsen_solution(PDE<P> &pde, fk::vector<P> const &x,
                                 options const &cli_opts);
  fk::vector<P>
  refine_solution(PDE<P> &pde, fk::vector<P> const &x, options const &cli_opts);

  fk::vector<P> redistribute_solution(fk::vector<P> const &x,
                                      distribution_plan const &old_plan,
                                      int const old_size);

  // adaptivity routines, meant to be invoked from driver routines
  // (conceptually private, exposed for testing)

  // the underlying distribution routines for adapt may rely on elements
  // not being "reshuffled", i.e., elements only deleted (coarsening) with
  // left shift to fill deleted segments of the element grid, or added
  // (refinement) to the end of the element grid
  fk::vector<P> refine(fk::vector<P> const &x, options const &cli_opts);
  fk::vector<P> coarsen(fk::vector<P> const &x, options const &cli_opts);

  distributed_grid(distributed_grid const &) = delete;
  distributed_grid &operator=(distributed_grid const &) = delete;
  // -- move constr./assignment op. implicitly deleted --

  distribution_plan const &get_distrib_plan() const { return plan_; }
  element_subgrid const &get_subgrid(int const rank) const
  {
    assert(rank >= 0);
    assert(rank < static_cast<int>(plan_.size()));
    return plan_.at(rank);
  }

  elements::table const &get_table() const { return table_; }
  int64_t size() const { return table_.size(); }

private:
  fk::vector<P> refine_elements(std::vector<int64_t> const &indices_to_refine,
                                options const &opts, fk::vector<P> const &x);
  fk::vector<P> remove_elements(std::vector<int64_t> const &indices_to_remove,
                                fk::vector<P> const &x);

  // remap element ranges after deletion/addition of elements
  // returns a mapping from new element indices -> old regions
  static std::map<grid_limits, grid_limits>
  remap_elements(std::vector<int64_t> const &deleted_indices,
                 int64_t const new_num_elems);

  // select elements from table given condition and solution vector
  template<typename F>
  std::vector<int64_t>
  filter_elements(F const condition, fk::vector<P> const &x)
  {
    auto const my_subgrid = this->get_subgrid(get_rank());
    assert(x.size() % my_subgrid.ncols() == 0);
    auto const element_dof = x.size() / my_subgrid.ncols();

    // check each of my rank's assigned elements against a condition
    std::vector<int64_t> matching_elements;
    for (int64_t i = 0; i < my_subgrid.ncols(); ++i)
    {
      auto const elem_start = i * element_dof;
      auto const elem_stop  = (i + 1) * element_dof - 1;
      fk::vector<P, mem_type::const_view> const element_x(x, elem_start,
                                                          elem_stop);
      auto const elem_index = my_subgrid.to_global_col(i);
      if (condition(elem_index, element_x))
      {
        matching_elements.push_back(elem_index);
      }
    }
    return matching_elements;
  }

  elements::table table_;
  distribution_plan plan_;
};

} // namespace asgard::adapt
