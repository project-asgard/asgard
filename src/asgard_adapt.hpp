#pragma once
#include "asgard_basis.hpp"
#include "asgard_distribution.hpp"

namespace asgard::adapt
{
// helper to find new levels for each dimension after adapting table
inline std::vector<int>
get_levels(elements::table const &adapted_table, int const num_dims)
{
  assert(num_dims > 0);
  auto const flat_table = adapted_table.get_active_table();
  auto const coord_size = num_dims * 2;
  std::vector<int> max_levels(num_dims, 0);
  for (int64_t element = 0; element < adapted_table.size(); ++element)
  {
    fk::vector<int, mem_type::const_view> coords(
        flat_table, element * coord_size, (element + 1) * coord_size - 1);
    for (auto i = 0; i < num_dims; ++i)
    {
      max_levels[i] = std::max(coords(i), max_levels[i]);
    }
  }
  return max_levels;
}

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
// from a n^dim segment x beginning at x[i*n^dim] and writes n^dim
// coefficients to y[j*n^dim], where n = degree + 1.

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
  distributed_grid(int max_level, prog_opts const &options);

  distributed_grid(PDE<P> const &pde)
      : distributed_grid(pde.max_level(), pde.options())
  {}

  fk::vector<P> coarsen_solution(PDE<P> &pde, fk::vector<P> const &x);
  fk::vector<P>
  refine_solution(PDE<P> &pde, fk::vector<P> const &x);

  fk::vector<P> redistribute_solution(fk::vector<P> const &x,
                                      distribution_plan const &old_plan,
                                      int const old_size);

  // adaptivity routines, meant to be invoked from driver routines
  // (conceptually private, exposed for testing)

  // the underlying distribution routines for adapt may rely on elements
  // not being "reshuffled", i.e., elements only deleted (coarsening) with
  // left shift to fill deleted segments of the element grid, or added
  // (refinement) to the end of the element grid
  fk::vector<P> refine(fk::vector<P> const &x, prog_opts const &cli_opts);
  fk::vector<P> coarsen(fk::vector<P> const &x, prog_opts const &cli_opts);

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

  void
  recreate_table(std::vector<int64_t> const &element_ids)
  {
    table_.recreate_from_elements(element_ids);

    plan_ = get_plan(get_num_ranks(), table_);
  }

  elements::table const &get_table() const { return table_; }
  int64_t size() const { return table_.size(); }

private:
  fk::vector<P> refine_elements(std::vector<int64_t> const &indices_to_refine,
                                std::vector<int> const &max_levels,
                                fk::vector<P> const &x);
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
  int max_level_;
};

} // namespace asgard::adapt
