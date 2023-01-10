#include "moment.hpp"
#include "elements.hpp"
#include "transformations.hpp"

namespace asgard
{
template<typename P>
moment<P>::moment(std::vector<md_func_type<P>> md_funcs_)
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

  basis::wavelet_transform<P, resource::host> const transformer(opts, pde);

  for (std::size_t s = 0; s < num_md_funcs; ++s)
  {
    auto const &md_func = this->md_funcs[s];
    for (std::size_t d = 0; d < num_dims; ++d)
    {
      fList[s].push_back(forward_transform<P>(
          dims[d], md_func[d], dims[d].volume_jacobian_dV, transformer));
    }
  }
}

// Actually contstructs the moment vector using fList.
// Calculate only if adapt is true or the vector field is empty
template<typename P>
void moment<P>::createMomentVector(PDE<P> const &pde, parser const &opts,
                                   elements::table const &hash_table)
{
  // check that fList has been constructed
  expect(this->fList.size() > 0);

  if (this->vector.size() == 0 || opts.do_adapt_levels())
  {
    distribution_plan const plan = get_plan(get_num_ranks(), hash_table);
    auto rank                    = get_rank();
    int const degree             = pde.get_dimensions()[0].get_degree();
    auto tmp = combine_dimensions(degree, hash_table, plan.at(rank).row_start,
                                  plan.at(rank).row_stop, this->fList[0]);
    this->vector.resize(tmp.size());
    this->vector      = std::move(tmp);
    auto num_md_funcs = md_funcs.size();
    for (std::size_t s = 1; s < num_md_funcs; ++s)
    {
      tmp = combine_dimensions(degree, hash_table, plan.at(rank).row_start,
                               plan.at(rank).row_stop, this->fList[s]);
      std::transform(tmp.begin(), tmp.end(), this->vector.begin(),
                     this->vector.begin(), std::plus<>{});
    }
  }
}

// helpers for converting linear coordinates into operator matrix indices
inline fk::vector<int> linearize(fk::vector<int> const &coords)
{
  fk::vector<int> linear(coords.size() / 2);
  for (int i = 0; i < linear.size(); ++i)
  {
    linear(i) = elements::get_1d_index(coords(i), coords(i + linear.size()));
  }
  return linear;
}

template<typename P>
inline fk::vector<int>
linear_coords_to_indices(PDE<P> const &pde, int const degree,
                         fk::vector<int> const &coords)
{
  fk::vector<int> indices(coords.size());
  for (int d = 0; d < pde.num_dims; ++d)
  {
    indices(d) = coords(d) * degree;
  }
  return indices;
}

template<typename P>
void moment<P>::createMomentReducedMatrix(PDE<P> const &pde,
                                          elements::table const &hash_table)
{
  int const num_ele = hash_table.size();

  int const moment_idx = 0;
  int const x_dim      = 0; // hardcoded for now, needs to change
  int const v_dim      = 1;

  expect(static_cast<int>(this->fList.size()) > moment_idx);
  expect(this->fList[moment_idx].size() >= v_dim);
  auto g_vec = this->fList[moment_idx][v_dim];

  expect(pde.get_dimensions().size() >= v_dim);
  int const n = std::pow(pde.get_dimensions()[v_dim].get_degree(), 2) * num_ele;
  int const rows = std::pow(2, pde.get_dimensions()[x_dim].get_level()) *
                   pde.get_dimensions()[x_dim].get_degree();

  this->moment_matrix.clear_and_resize(rows, n);

  auto deg = pde.get_dimensions()[v_dim].get_degree();

  // TODO: this should be refactored into a sparse matrix
  for (int i = 0; i < num_ele; i++)
  {
    fk::vector<int> const coords       = hash_table.get_coords(i);
    fk::vector<int> const elem_indices = linearize(coords);

    for (int j = 0; j < deg; j++)
    {
      int const g_vec_index = elem_indices(v_dim) * deg;
      int const ind_i       = elem_indices(x_dim) * deg + j;
      int const ind_j       = i * std::pow(deg, 2) + j * deg;

      for (int d = 0; d < deg; d++)
      {
        moment_matrix(ind_i, ind_j + d) = g_vec(g_vec_index + d);
      }
    }
  }
}

template class moment<float>;
template class moment<double>;

} // namespace asgard
