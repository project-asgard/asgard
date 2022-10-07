#include "moment.hpp"
#include "transformations.hpp"

template<typename P>
moment<P>::moment(std::vector<vector_func<P>> md_funcs_)
    : md_funcs(std::move(md_funcs_))
{}

// Creates the coefficients of the moment vector on each domain.
// No mass matrix inversion is done.
template<typename P>
void moment<P>::createFlist(PDE<P> const &pde, options const &opts)
{
  std::size_t num_md_funcs = this->md_funcs.size();

  auto const &dims     = pde.get_dimensions();
  std::size_t num_dims = dims.size();

  this->fList.resize(num_md_funcs);
  for (auto &elem : this->fList)
  {
    elem.resize(num_dims);
  }

  basis::wavelet_transform<P, resource::host> const transformer(opts, pde);

  for (std::size_t s = 0; s < num_md_funcs; ++s)
  {
    auto const &md_func = this->md_funcs[s];
    for (std::size_t d = 0; d < num_dims; ++d)
    {
      fList[s][d] = forward_transform<P>(
          dims[d], md_func, dims[d].volume_jacobian_dV, transformer);
    }
  }
}

// Actually contstructs the moment vector using fList.
// Calculate only if adapt is true or the vector field is empty
template<typename P>
void moment<P>::createMomentVector(parser const &opts,
                                   elements::table const &hash_table)
{
  if (this->vector.size() == 0 || opts.do_adapt_levels())
  {
    distribution_plan const plan = get_plan(get_num_ranks(), hash_table);
    auto rank = get_rank();
    this->vector = combine_dimensions(opts.get_degree(), hash_table, plan.at(rank).row_start, plan.at(rank).row_stop, this->fList[0]);
    auto num_md_funcs = md_funcs.size();
    for (std::size_t s = 1; s < num_md_funcs; ++s)
    {
      auto tmp = combine_dimensions(opts.get_degree(), hash_table, plan.at(rank).row_start, plan.at(rank).row_stop, this->fList[s]);
      std::transform(tmp.begin(), tmp.end(), this->vector.begin(), this->vector.begin(), std::plus<>{});
    }
  }
}

template class moment<float>;
template class moment<double>;
