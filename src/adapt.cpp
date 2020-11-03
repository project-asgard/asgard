#include "adapt.hpp"

namespace adapt
{
template<typename P>
distributed_grid<P>::distributed_grid(options const &cli_opts,
                                      PDE<P> const &pde)
    : table_(cli_opts, pde)
{
  plan_ = get_plan(get_num_ranks(), table_);
}

template<typename P>
fk::vector<P>
distributed_grid<P>::refine(fk::vector<P> const &x, options const &cli_opts)
{
  return x;
}

template<typename P>
fk::vector<P>
distributed_grid<P>::coarsen(fk::vector<P> const &x, options const &cli_opts)
{
  return x;
}

template class distributed_grid<float>;
template class distributed_grid<double>;

} // namespace adapt
