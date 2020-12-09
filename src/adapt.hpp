#pragma once
#include "basis.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "program_options.hpp"

namespace adapt
{
template<typename P>
class distributed_grid
{
public:
  distributed_grid(PDE<P> const &pde, options const &cli_options);

  // driver routines
  fk::vector<P> get_initial_condition(
      PDE<P> &pde,
      basis::wavelet_transform<P, resource::host> const &transformer,
      options const &cli_opts);

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
  // returns a mapping new contiguous element index regions -> old regions
  static std::map<grid_limits, grid_limits>
  remap_elements(std::vector<int64_t> const &deleted_indices,
                 int64_t const new_num_elems);

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

} // namespace adapt
