#pragma once
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
  distributed_grid(options const &cli_opts, PDE<P> const &pde);

  fk::vector<P> refine(fk::vector<P> const &x, options const &cli_opts);
  fk::vector<P> coarsen(fk::vector<P> const &x, options const &cli_opts);

  distributed_grid(distributed_grid const &) = delete;
  distributed_grid &operator=(distributed_grid const &) = delete;
  // -- move constr./assignment op. implicitly deleted --

  // FIXME provide appropriate external interface rather than simple getter
  // once I figure out requirements

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
  elements::table table_;
  distribution_plan plan_;
};

} // namespace adapt
