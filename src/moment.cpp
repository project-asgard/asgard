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
      fList[s][d] = forward_transform<P>(dims[d], md_func, transformer);
    }
  }
}

template class moment<float>;
template class moment<double>;
